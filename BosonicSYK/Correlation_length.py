#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation-length (distance-resolved) zero-mode analysis GUI for RAW HDF5 ensemble databases.

This is a spin-off of the distance-resolved zero-mode tool, simplified to compute and plot ONLY
the disorder-averaged absolute zero-mode correlation

    C^{ab}(r) = (1/N_r) * sum_{d(x,y)=r}  (1/M_xy) * sum_{s=1..M_xy} | G_{xy}^{ab,(s)}(i nu_0) |,

where:
- r is the distance between sites x and y on a chain of length L
- d(x,y) can be either open distance |x-y|, or periodic minimal distance min(|x-y|, L-|x-y|)
- M_xy is the number of samples for which the pair (x,y) is available in the raw database
- N_r is the number of directed pairs (x,y) with distance r that appear in the raw database.

No "typical" statistics are computed here.

Database schema (expected)
--------------------------
attrs: schema='raw_pairs_v1', L, M, nrep, beta, Nsamp_done
datasets:
  nu              shape (M,)
  sample_offsets  shape (Nsamp_done+1,)
  pairs_i_all     shape (Nrec,)
  pairs_j_all     shape (Nrec,)
  G_iw_all        shape (Nrec, M, nrep, nrep) complex

Notes
-----
- This tool uses log-log axes by default. Since log(r) is undefined at r=0, r=0 is omitted from the plot
  (even if it is included in the computation).
- Diagnostics are provided to verify scaling and rule out unintended extra normalizations.
"""

import json
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Tk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# HDF5
import h5py

# Matplotlib
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    # Allows importing this module in headless environments for testing.
    pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ---------------- helpers ----------------

def _as_int(s: str, default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default

def _as_float(s: str, default: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default

def _dist_open(x: int, y: int) -> int:
    return abs(int(x) - int(y))

def _dist_periodic_min(x: int, y: int, L: int) -> int:
    d = abs(int(x) - int(y))
    return min(d, int(L) - d)


# ---------------- raw DB ----------------

@dataclass
class RawDB:
    path: str
    h5: h5py.File
    L: int
    M: int
    nrep: int
    beta: float
    Ns: int
    nu: np.ndarray
    offsets: np.ndarray
    pairs_i_ds: h5py.Dataset
    pairs_j_ds: h5py.Dataset
    G_ds: h5py.Dataset


def load_raw_db(path: str) -> RawDB:
    h5 = h5py.File(path, "r")
    schema = str(h5.attrs.get("schema", ""))
    if schema and schema != "raw_pairs_v1":
        # still allow, but warn
        pass

    L = int(h5.attrs["L"])
    M = int(h5.attrs["M"])
    nrep = int(h5.attrs["nrep"])
    beta = float(h5.attrs["beta"])
    Ns = int(h5.attrs.get("Nsamp_done", h5.attrs.get("Ns", 0)))

    nu = np.array(h5["nu"][:], dtype=float)
    offsets = np.array(h5["sample_offsets"][:Ns + 1], dtype=np.int64)

    pairs_i_ds = h5["pairs_i_all"]
    pairs_j_ds = h5["pairs_j_all"]
    G_ds = h5["G_iw_all"]

    return RawDB(
        path=path,
        h5=h5,
        L=L,
        M=M,
        nrep=nrep,
        beta=beta,
        Ns=Ns,
        nu=nu,
        offsets=offsets,
        pairs_i_ds=pairs_i_ds,
        pairs_j_ds=pairs_j_ds,
        G_ds=G_ds,
    )


# ---------------- core computation ----------------

def compute_C_r_zero_mode_abs(
    db: RawDB,
    a: int,
    b: int,
    periodic: bool,
    include_r0: bool,
    progress_cb=None,
) -> dict:
    """
    Compute C^{ab}(r) as defined in the module docstring.

    Returns a dict with:
      r_vals, C_r, n_pairs, avg_Mxy, global_min_abs, global_mean_abs, global_max_abs, global_cnt_abs, nu0
    """
    if not (0 <= a < db.nrep and 0 <= b < db.nrep):
        raise ValueError(f"Replica indices out of range: a={a}, b={b}, nrep={db.nrep}")

    # Build pair -> list of record indices (one per sample occurrence)
    pair_lists: Dict[Tuple[int, int], List[int]] = {}

    Ns = db.Ns
    offs = db.offsets
    pairs_i_ds = db.pairs_i_ds
    pairs_j_ds = db.pairs_j_ds

    # Scan samples; progress is in [0, 40] portion
    for s in range(Ns):
        start = int(offs[s])
        end = int(offs[s + 1])
        if end <= start:
            continue
        pi = np.array(pairs_i_ds[start:end], dtype=np.int32)
        pj = np.array(pairs_j_ds[start:end], dtype=np.int32)
        for k in range(end - start):
            key = (int(pi[k]), int(pj[k]))
            pair_lists.setdefault(key, []).append(start + k)
        if progress_cb is not None and (s % max(1, Ns // 200) == 0 or s == Ns - 1):
            progress_cb(int(40 * (s + 1) / Ns))

    pair_to_recidx = {k: np.array(v, dtype=np.int64) for k, v in pair_lists.items()}
    pairs = list(pair_to_recidx.keys())

    if len(pairs) == 0:
        raise ValueError("No (x,y) records found in the database.")

    # r range
    if periodic:
        r_max = db.L // 2
        dist_fn = lambda x, y: _dist_periodic_min(x, y, db.L)
    else:
        r_max = db.L - 1
        dist_fn = _dist_open

    # Aggregators over r
    sum_pair_means = np.zeros((r_max + 1,), dtype=np.float64)
    n_pairs = np.zeros((r_max + 1,), dtype=np.int64)
    sum_Mxy = np.zeros((r_max + 1,), dtype=np.float64)
    min_pair_mean = np.full((r_max + 1,), np.inf, dtype=np.float64)
    max_pair_mean = np.full((r_max + 1,), -np.inf, dtype=np.float64)

    # Record-level stats for |G|
    gmin = np.inf
    gmax = -np.inf
    gsum = 0.0
    gcnt = 0

    # Compute over pairs; progress in [40, 100]
    G_ds = db.G_ds
    nu0 = float(db.nu[0])

    for idx_pair, (x, y) in enumerate(pairs):
        r = int(dist_fn(x, y))
        if r == 0 and not include_r0:
            continue
        if r < 0 or r > r_max:
            continue

        rec_idx = pair_to_recidx[(x, y)]
        Mxy = int(rec_idx.size)
        if Mxy <= 0:
            continue

        # Read only the zero-mode and replica component for all records
        # Shape: (Mxy,)
        vals = np.empty((Mxy,), dtype=np.float64)
        for t, ridx in enumerate(rec_idx):
            # G(ridx, 0, a, b) is complex
            gij = G_ds[int(ridx), 0, a, b]
            av = float(abs(gij))
            vals[t] = av

        # per-pair average across samples
        A_xy = float(vals.mean())

        # update global stats
        if vals.size:
            gmin = min(gmin, float(vals.min()))
            gmax = max(gmax, float(vals.max()))
            gsum += float(vals.sum())
            gcnt += int(vals.size)

        # accumulate over distance
        sum_pair_means[r] += A_xy
        n_pairs[r] += 1
        sum_Mxy[r] += Mxy
        min_pair_mean[r] = min(min_pair_mean[r], A_xy)
        max_pair_mean[r] = max(max_pair_mean[r], A_xy)

        if progress_cb is not None and (idx_pair % max(1, len(pairs) // 200) == 0 or idx_pair == len(pairs) - 1):
            progress_cb(40 + int(60 * (idx_pair + 1) / len(pairs)))

    # finalize
    C_r = np.full((r_max + 1,), np.nan, dtype=np.float64)
    avg_Mxy = np.full((r_max + 1,), np.nan, dtype=np.float64)
    for r in range(r_max + 1):
        if n_pairs[r] > 0:
            C_r[r] = sum_pair_means[r] / n_pairs[r]
            avg_Mxy[r] = sum_Mxy[r] / n_pairs[r]

    if gcnt <= 0:
        gmin = np.nan
        gmax = np.nan
        gmean = np.nan
    else:
        gmean = gsum / gcnt

    return dict(
        r_vals=np.arange(r_max + 1, dtype=int),
        C_r=C_r,
        n_pairs=n_pairs,
        avg_Mxy=avg_Mxy,
        min_pair_mean=min_pair_mean,
        max_pair_mean=max_pair_mean,
        global_min_abs=float(gmin),
        global_mean_abs=float(gmean),
        global_max_abs=float(gmax),
        global_cnt_abs=int(gcnt),
        a=int(a),
        b=int(b),
        L=int(db.L),
        nu0=float(nu0),
        periodic=bool(periodic),
        include_r0=bool(include_r0),
    )


# ---------------- GUI ----------------

class CorrelationLengthGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Correlation length C^{ab}(r) from RAW HDF5 ensemble")
        self.geometry("1280x860")

        # Matplotlib math style (no external LaTeX dependency)
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"

        self.db: Optional[RawDB] = None
        self._fig: Optional[Figure] = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._toolbar: Optional[NavigationToolbar2Tk] = None
        self._toolbar_frame: Optional[ttk.Frame] = None

        # diagnostics (filled after Plot)
        self._last_stats: Optional[dict] = None

        # threads
        self._compute_thread: Optional[threading.Thread] = None

        self._build_vars()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_vars(self):
        self.var_a = tk.IntVar(value=0)
        self.var_b = tk.IntVar(value=0)
        self.var_periodic = tk.BooleanVar(value=True)
        self.var_include_r0 = tk.BooleanVar(value=True)

        # style
        self.var_w = tk.StringVar(value="7.2")
        self.var_h = tk.StringVar(value="4.8")
        self.var_title_fs = tk.StringVar(value="13")
        self.var_label_fs = tk.StringVar(value="12")
        self.var_tick_fs = tk.StringVar(value="11")
        self.var_legend_fs = tk.StringVar(value="11")
        self.var_lw = tk.StringVar(value="2.0")

        # progress
        self.compute_prog_var = tk.IntVar(value=0)
        self.var_status = tk.StringVar(value="No file loaded.")

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # File
        box_file = ttk.LabelFrame(left, text="Database", padding=10)
        box_file.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(box_file, text="Load RAW .h5...", command=self._load_db).pack(fill=tk.X)
        self.lbl_file = ttk.Label(box_file, text="(none)", wraplength=320, justify="left")
        self.lbl_file.pack(fill=tk.X, pady=(6, 0))

        # Selection
        sel = ttk.LabelFrame(left, text="Selection", padding=10)
        sel.pack(fill=tk.X, pady=(0, 10))

        row_ab = ttk.Frame(sel)
        row_ab.pack(fill=tk.X)
        ttk.Label(row_ab, text="Replica a:").pack(side=tk.LEFT)
        ttk.Spinbox(row_ab, from_=0, to=999, textvariable=self.var_a, width=6).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row_ab, text="Replica b:").pack(side=tk.LEFT)
        ttk.Spinbox(row_ab, from_=0, to=999, textvariable=self.var_b, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row_dist = ttk.Frame(sel)
        row_dist.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(row_dist, text="Periodic minimal distance", variable=self.var_periodic).pack(anchor="w")
        ttk.Checkbutton(row_dist, text="Include r=0 in computation (r=0 not shown on log axes)", variable=self.var_include_r0).pack(anchor="w")

        # Style
        sty = ttk.LabelFrame(left, text="Plot styling", padding=10)
        sty.pack(fill=tk.X, pady=(0, 10))

        row_wh = ttk.Frame(sty)
        row_wh.pack(fill=tk.X)
        ttk.Label(row_wh, text="Fig W:").pack(side=tk.LEFT)
        ttk.Entry(row_wh, textvariable=self.var_w, width=6).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row_wh, text="Fig H:").pack(side=tk.LEFT)
        ttk.Entry(row_wh, textvariable=self.var_h, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row_fs = ttk.Frame(sty)
        row_fs.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row_fs, text="Title fs:").pack(side=tk.LEFT)
        ttk.Entry(row_fs, textvariable=self.var_title_fs, width=6).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row_fs, text="Label fs:").pack(side=tk.LEFT)
        ttk.Entry(row_fs, textvariable=self.var_label_fs, width=6).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row_fs, text="Tick fs:").pack(side=tk.LEFT)
        ttk.Entry(row_fs, textvariable=self.var_tick_fs, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row_fs2 = ttk.Frame(sty)
        row_fs2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row_fs2, text="Legend fs:").pack(side=tk.LEFT)
        ttk.Entry(row_fs2, textvariable=self.var_legend_fs, width=6).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row_fs2, text="Line width:").pack(side=tk.LEFT)
        ttk.Entry(row_fs2, textvariable=self.var_lw, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # Buttons
        btns = ttk.LabelFrame(left, text="Actions", padding=10)
        btns.pack(fill=tk.X, pady=(0, 10))

        self.btn_plot = ttk.Button(btns, text="Plot C^{ab}(r) [log-log]", command=self._plot, state=tk.DISABLED)
        self.btn_plot.pack(fill=tk.X)

        row_btn2 = ttk.Frame(btns)
        row_btn2.pack(fill=tk.X, pady=(8, 0))
        self.btn_save = ttk.Button(row_btn2, text="Save PDF...", command=self._save_pdf, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_diag = ttk.Button(row_btn2, text="Diagnostics...", command=self._show_diagnostics, state=tk.DISABLED)
        self.btn_diag.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        # Progress + status
        prog = ttk.LabelFrame(left, text="Progress", padding=10)
        prog.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(prog, text="Compute:").pack(anchor="w")
        self.pb_compute = ttk.Progressbar(prog, orient="horizontal", length=280, mode="determinate", maximum=100, variable=self.compute_prog_var)
        self.pb_compute.pack(fill=tk.X, pady=(4, 0))

        stat = ttk.LabelFrame(left, text="Status", padding=10)
        stat.pack(fill=tk.X)
        ttk.Label(stat, textvariable=self.var_status, wraplength=320, justify="left").pack(fill=tk.X)

        # Right plot area
        self.plot_area = ttk.Frame(root)
        self.plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._clear_plot_area()

    # ---------------- file loading ----------------

    def _load_db(self):
        fp = filedialog.askopenfilename(
            title="Select RAW HDF5 database",
            filetypes=[("HDF5 database", "*.h5 *.hdf5"), ("All files", "*")],
        )
        if not fp:
            return
        try:
            if self.db is not None:
                try:
                    self.db.h5.close()
                except Exception:
                    pass
            self.db = load_raw_db(fp)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.db = None
            return

        self.lbl_file.configure(text=fp)
        self.var_status.set(f"Loaded RAW DB: L={self.db.L}, M={self.db.M}, nrep={self.db.nrep}, Ns={self.db.Ns}")
        self.btn_plot.configure(state=tk.NORMAL)
        self.btn_save.configure(state=tk.DISABLED)
        self.btn_diag.configure(state=tk.DISABLED)
        self._clear_plot_area()
        self._last_stats = None

    # ---------------- plot + diagnostics ----------------

    def _plot(self):
        if self.db is None:
            return

        a = int(self.var_a.get())
        b = int(self.var_b.get())
        periodic = bool(self.var_periodic.get())
        include_r0 = bool(self.var_include_r0.get())

        fig_w = _as_float(self.var_w.get(), 7.2)
        fig_h = _as_float(self.var_h.get(), 4.8)
        title_fs = _as_float(self.var_title_fs.get(), 13)
        label_fs = _as_float(self.var_label_fs.get(), 12)
        tick_fs = _as_float(self.var_tick_fs.get(), 11)
        legend_fs = _as_float(self.var_legend_fs.get(), 11)
        lw = _as_float(self.var_lw.get(), 2.0)

        self.btn_plot.configure(state=tk.DISABLED)
        self.btn_save.configure(state=tk.DISABLED)
        self.btn_diag.configure(state=tk.DISABLED)
        self.compute_prog_var.set(0)
        self.var_status.set("Computing C^{ab}(r)...")

        def _progress(pct: int):
            pct = int(max(0, min(100, pct)))
            self.after(0, lambda: self.compute_prog_var.set(pct))

        def worker():
            try:
                stats = compute_C_r_zero_mode_abs(
                    db=self.db,
                    a=a,
                    b=b,
                    periodic=periodic,
                    include_r0=include_r0,
                    progress_cb=_progress,
                )
                self._last_stats = stats

                def _do_plot():
                    self._reset_plot_canvas(fig_w, fig_h)
                    ax = self._fig.add_subplot(111)

                    r_vals = stats["r_vals"]
                    C_r = stats["C_r"]
                    n_pairs = stats["n_pairs"]

                    # omit r=0 from plotting on log x
                    mask = (n_pairs > 0) & np.isfinite(C_r) & (C_r > 0)
                    if not include_r0:
                        mask &= (r_vals != 0)
                    else:
                        # even if included in computation, omit in log plot
                        mask &= (r_vals != 0)

                    rr = r_vals[mask]
                    yy = C_r[mask]

                    # Plot
                    ax.plot(rr, yy, marker="o", linewidth=lw, label=rf"$C^{{{a}{b}}}(r)$")

                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    ax.set_xlabel(r"$r$", fontsize=label_fs)
                    ax.set_ylabel(rf"$C^{{{a}{b}}}(r)$", fontsize=label_fs)

                    title = rf"Correlation length $C^{{ab}}(r)$ for $(a,b)=({a},{b})$"
                    ax.set_title(title, fontsize=title_fs)

                    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
                    ax.legend(fontsize=legend_fs, loc="best")
                    ax.grid(True, which="both", linestyle="--", alpha=0.35)

                    self._canvas.draw()
                    self.btn_plot.configure(state=tk.NORMAL)
                    self.btn_save.configure(state=tk.NORMAL)
                    self.btn_diag.configure(state=tk.NORMAL)

                    self.compute_prog_var.set(100)
                    mode = "periodic" if periodic else "open"
                    self.var_status.set(f"Done. Computed C^{a}{b}(r) ({mode} distance).")

                self.after(0, _do_plot)

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Plot error", str(e)))
                self.after(0, lambda: self.btn_plot.configure(state=tk.NORMAL))
                self.after(0, lambda: self.compute_prog_var.set(0))

        self._compute_thread = threading.Thread(target=worker, daemon=True)
        self._compute_thread.start()

    def _show_diagnostics(self):
        st = self._last_stats
        if st is None:
            messagebox.showinfo("Diagnostics", "No diagnostics available yet. Click Plot first.")
            return

        r_vals = st["r_vals"]
        n_pairs = st["n_pairs"]
        C_r = st["C_r"]
        avg_Mxy = st["avg_Mxy"]
        min_pm = st["min_pair_mean"]
        max_pm = st["max_pair_mean"]

        lines = []
        lines.append("Computation definitions used:\n"
                     "  For each pair (x,y):\n"
                     "    A_xy = (1/M_xy) Σ_s |G_xy^{ab,(s)}(iν_0)|\n"
                     "  For each distance r (over available pairs at that r):\n"
                     "    C^{ab}(r) = (1/N_r) Σ_{d(x,y)=r} A_xy\n"
                     "No other normalization (no division by β, M, or number of frequencies).")
        lines.append("")
        lines.append(f"Selected: a={st['a']}, b={st['b']}, L={st['L']}, nu0={st['nu0']:g}")
        lines.append(f"Record-level |G(iν0)| across all used records: min={st['global_min_abs']:.3e}, mean={st['global_mean_abs']:.3e}, max={st['global_max_abs']:.3e} (count={st['global_cnt_abs']})")
        lines.append("")
        lines.append("Per-distance summary (only r with N_r>0; r=0 omitted from log plot):")
        lines.append("  r   N_pairs   avg(M_xy)        C^{ab}(r)      min(A_xy)       max(A_xy)")
        for r in range(len(r_vals)):
            if n_pairs[r] <= 0:
                continue
            lines.append(f"{r:3d} {n_pairs[r]:9d} {avg_Mxy[r]:10.2f} {C_r[r]:13.3e} {min_pm[r]:13.3e} {max_pm[r]:13.3e}")

        txt = "\n".join(lines)

        win = tk.Toplevel(self)
        win.title("Diagnostics")
        win.geometry("900x700")
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        t = tk.Text(frm, wrap="none")
        t.insert("1.0", txt)
        t.configure(state="disabled")
        t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(frm, orient="vertical", command=t.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        t.configure(yscrollcommand=yscroll.set)
        xscroll = ttk.Scrollbar(win, orient="horizontal", command=t.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        t.configure(xscrollcommand=xscroll.set)

    # ---------------- save + close ----------------

    def _save_pdf(self):
        if self._fig is None:
            return
        fp = filedialog.asksaveasfilename(
            title="Save plot as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if not fp:
            return
        try:
            self._fig.savefig(fp, format="pdf", bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return
        messagebox.showinfo("Saved", f"Saved PDF:\n{fp}")

    def _on_close(self):
        try:
            if self.db is not None and getattr(self.db, "h5", None) is not None:
                try:
                    self.db.h5.close()
                except Exception:
                    pass
        finally:
            self.destroy()

    # ---------------- plot widget mgmt ----------------

    def _clear_plot_area(self):
        for child in list(self.plot_area.winfo_children()):
            child.destroy()
        self._placeholder = ttk.Label(self.plot_area, text="Load data and click Plot.")
        self._placeholder.grid(row=0, column=0, sticky="nsew")
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._toolbar_frame = None

    def _reset_plot_canvas(self, fig_w: float, fig_h: float):
        # Avoid mixing Tk geometry managers (grid/pack) in the same parent.
        # NavigationToolbar2Tk uses pack() internally, so we place it in its own frame.
        for child in list(self.plot_area.winfo_children()):
            child.destroy()

        self._fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self.plot_area)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._toolbar_frame = ttk.Frame(self.plot_area)
        self._toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, self._toolbar_frame)
        self._toolbar.update()

        self.plot_area.rowconfigure(0, weight=1)
        self.plot_area.rowconfigure(1, weight=0)
        self.plot_area.columnconfigure(0, weight=1)


def main():
    app = CorrelationLengthGUI()
    app.mainloop()


if __name__ == "__main__":
    main()