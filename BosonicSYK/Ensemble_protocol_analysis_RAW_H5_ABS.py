#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble analysis GUI (sparse directed pairs) for SDRG-on-SD protocol — RAW database mode.

CHANGES (2026-01-13)
--------------------
This GUI now expects the RAW HDF5 database produced by:
    Ensemble_protocol_miner.py  (RAW DB version)

The database stores per-sample restored Green's functions in Matsubara frequency:
    G_{xy}^{ab}(iν_m)  (complex)

This GUI computes, on demand, for the selected directed pair (x→y) and replica indices (a,b):

Matsubara mode
  Mean (real):
      (1/M_xy) Σ_s Re G^{(s)}(iν_m)
  Typical (real):
      sgn(mean) * exp( (1/M_xy) Σ_s log(|Re G^{(s)}(iν_m)| + eps) )
  and analogously for the imaginary part.

Imaginary-time mode
  For each sample, first compute:
      G^{(s)}(τ_j) = (M/β) IFFT[ G^{(s)}(iν_m) ]
  then apply the same mean/typical definitions to Re/Im in τ.

Notes on "typical"
------------------
Your specification uses log(Re G) and log(Im G). Since Re/Im can be negative in practice,
we implement a sign-preserving geometric mean:
    typ(x) := sgn(mean(x)) * exp(mean(log(|x| + eps)))
This avoids complex-valued logs and is the standard "typical" construction for signful data.

UI features preserved / added
-----------------------------
- Component selection: Real / Imag / Both / Abs
- Curve selection: Mean / Typical / Both
- Domain selection: Matsubara / Imaginary time
- Font/label/tick/legend sizes and line width
- Save current plot as PDF
- Progress bars:
    * Indexing progress (building available pair list)
    * Compute progress (scan + accumulate + optional iFFT)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as _e:
    raise RuntimeError("Tkinter is required to run this GUI.") from _e

try:
    import h5py
except Exception as _e:
    raise RuntimeError("h5py is required to read the RAW HDF5 database.") from _e

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # for rcParams only
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ---------- helpers ----------

def _as_int(x: str, default: int) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _as_float(x: str, default: float) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def tau_grid(beta: float, M: int) -> np.ndarray:
    return (beta / float(M)) * np.arange(M)


def iw_to_tau(beta: float, F_iw: np.ndarray) -> np.ndarray:
    """Discrete inverse (axis=0 is Matsubara index).

    Convention:
        F(iw_m) = (beta/M) * FFT[F(tau_j)]
        F(tau_j) = (M/beta) * IFFT[F(iw_m)]
    """
    M = F_iw.shape[0]
    return (M / float(beta)) * np.fft.ifft(F_iw, axis=0)


def typical_sign_preserving(x_sum: np.ndarray, logabs_sum: np.ndarray, count: int) -> np.ndarray:
    """Return sign-preserving geometric mean:
         sgn(mean) * exp(mean(log(|x|+eps)))
    where x_sum = Σ x, logabs_sum = Σ log(|x|+eps).
    """
    mean = x_sum / float(max(1, count))
    mag = np.exp(logabs_sum / float(max(1, count)))
    return np.sign(mean) * mag


@dataclass
class LoadedRaw:
    path: str
    h5: h5py.File
    beta: float
    nu: np.ndarray
    L: int
    nrep: int
    M: int
    Nsamp_requested: int
    Nsamp_done: int
    offsets: np.ndarray  # (Nsamp_done+1,)
    pairs_i_ds: h5py.Dataset
    pairs_j_ds: h5py.Dataset
    G_ds: h5py.Dataset


class AnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDRG-on-SD Protocol Analysis (Ensemble, RAW HDF5)")
        self.geometry("1380x900")

        # Matplotlib math style (no external LaTeX dependency)
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"

        self.data: Optional[LoadedRaw] = None

        # Pair list and counts
        self.pairs: List[Tuple[int, int]] = []
        self.counts: List[int] = []
        self.pair_counts: Dict[Tuple[int, int], int] = {}

        # Cache: selected pair -> list of record indices (one per sample where available)
        self._pair_to_record_indices: Dict[Tuple[int, int], np.ndarray] = {}

        # Matplotlib widgets
        self._fig: Optional[Figure] = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._toolbar: Optional[NavigationToolbar2Tk] = None
        self._toolbar_frame: Optional[ttk.Frame] = None

        self._build_vars()
        self._build_ui()

        self._index_thread: Optional[threading.Thread] = None
        self._compute_thread: Optional[threading.Thread] = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- vars ----------------

    def _build_vars(self):
        self.var_i = tk.IntVar(value=0)
        self.var_j = tk.IntVar(value=0)
        self.var_a = tk.IntVar(value=0)
        self.var_b = tk.IntVar(value=0)

        self.var_domain = tk.StringVar(value="Matsubara")  # Matsubara / Imaginary time
        self.var_component = tk.StringVar(value="Real")    # Real / Imag / Both / Abs
        self.var_curves = tk.StringVar(value="Both")       # Mean / Typical / Both

        self.var_filter = tk.StringVar(value="")

        # Typical epsilon (for log(|x|+eps))
        self.var_typ_eps = tk.StringVar(value="1e-30")

        # Figure/layout style
        self.var_w = tk.StringVar(value="7.4")
        self.var_h = tk.StringVar(value="4.8")
        self.var_title_fs = tk.StringVar(value="13")
        self.var_label_fs = tk.StringVar(value="12")
        self.var_tick_fs = tk.StringVar(value="11")
        self.var_legend_fs = tk.StringVar(value="11")
        self.var_lw = tk.StringVar(value="2.0")

        # Progress
        self.index_prog_var = tk.IntVar(value=0)
        self.compute_prog_var = tk.IntVar(value=0)
        self.var_status = tk.StringVar(value="No file loaded.")

    # ---------------- UI ----------------

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Load + actions
        ttk.Button(left, text="Load RAW .h5...", command=self._load).pack(fill=tk.X, pady=(0, 6))

        row_btn = ttk.Frame(left)
        row_btn.pack(fill=tk.X, pady=(0, 6))
        self.btn_plot = ttk.Button(row_btn, text="Plot", command=self._plot, state=tk.DISABLED)
        self.btn_plot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_save = ttk.Button(row_btn, text="Save PDF...", command=self._save_pdf, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bars
        box_prog = ttk.LabelFrame(left, text="Progress")
        box_prog.pack(fill=tk.X, pady=(0, 8), padx=0)

        ttk.Label(box_prog, textvariable=self.var_status, wraplength=520, justify="left").pack(fill=tk.X, padx=6, pady=(6, 6))

        row_idx = ttk.Frame(box_prog)
        row_idx.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(row_idx, text="Indexing:").pack(side=tk.LEFT)
        self.pb_index = ttk.Progressbar(row_idx, orient="horizontal", mode="determinate",
                                        maximum=100, variable=self.index_prog_var)
        self.pb_index.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        row_cmp = ttk.Frame(box_prog)
        row_cmp.pack(fill=tk.X, padx=6, pady=(0, 8))
        ttk.Label(row_cmp, text="Compute:").pack(side=tk.LEFT)
        self.pb_compute = ttk.Progressbar(row_cmp, orient="horizontal", mode="determinate",
                                          maximum=100, variable=self.compute_prog_var)
        self.pb_compute.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        # Available pairs
        box_pairs = ttk.LabelFrame(left, text="Available Green's functions (directed i→j) with counts")
        box_pairs.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        filter_row = ttk.Frame(box_pairs)
        filter_row.pack(fill=tk.X, padx=6, pady=(6, 4))
        ttk.Label(filter_row, text="Filter:").pack(side=tk.LEFT)
        ent = ttk.Entry(filter_row, textvariable=self.var_filter)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ent.bind("<KeyRelease>", lambda _e: self._refresh_pairs_list())

        list_row = ttk.Frame(box_pairs)
        list_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.list_pairs = tk.Listbox(list_row, height=16, exportselection=False)
        self.list_pairs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_row, orient=tk.VERTICAL, command=self.list_pairs.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_pairs.configure(yscrollcommand=sb.set)
        self.list_pairs.bind("<<ListboxSelect>>", self._on_pair_select)
        self.list_pairs.bind("<Double-Button-1>", lambda _e: self._plot())

        # Selection
        sel = ttk.LabelFrame(left, text="Selection", padding=8)
        sel.pack(fill=tk.X, pady=(0, 10))
        sel.columnconfigure(0, weight=0)
        sel.columnconfigure(1, weight=1)

        ttk.Label(sel, text="site i").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sel, textvariable=self.var_i, width=10).grid(row=0, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(sel, text="site j").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sel, textvariable=self.var_j, width=10).grid(row=1, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(sel, text="replica a").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sel, textvariable=self.var_a, width=10).grid(row=2, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(sel, text="replica b").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(sel, textvariable=self.var_b, width=10).grid(row=3, column=1, sticky="ew", padx=6, pady=3)

        # Options
        opt = ttk.LabelFrame(left, text="Plot options", padding=8)
        opt.pack(fill=tk.X, pady=(0, 10))

        row = ttk.Frame(opt)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Component:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_component, state="readonly",
                     values=["Real", "Imag", "Both", "Abs"], width=8).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row, text="Domain:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_domain, state="readonly",
                     values=["Matsubara", "Imaginary time"], width=14).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row, text="Curves:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_curves, state="readonly",
                     values=["Mean", "Typical", "Both"], width=10).pack(side=tk.LEFT, padx=(6, 0))

        row_eps = ttk.Frame(opt)
        row_eps.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row_eps, text="Typical eps:").pack(side=tk.LEFT)
        ttk.Entry(row_eps, textvariable=self.var_typ_eps, width=12).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(row_eps, text="(used in log(|x|+eps))").pack(side=tk.LEFT, padx=(10, 0))

        # Style
        sty = ttk.LabelFrame(left, text="Style (set before Plot)", padding=8)
        sty.pack(fill=tk.X, pady=(0, 10))
        sty.columnconfigure(0, weight=0)
        sty.columnconfigure(1, weight=1)

        fields = [
            ("Fig width (in)", self.var_w),
            ("Fig height (in)", self.var_h),
            ("Title size", self.var_title_fs),
            ("Label size", self.var_label_fs),
            ("Tick size", self.var_tick_fs),
            ("Legend size", self.var_legend_fs),
            ("Line width", self.var_lw),
        ]
        for r, (lab, var) in enumerate(fields):
            ttk.Label(sty, text=lab).grid(row=r, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(sty, textvariable=var, width=10).grid(row=r, column=1, sticky="ew", padx=6, pady=3)

        # Right: plot area
        self.plot_area = ttk.Frame(right)
        self.plot_area.grid(row=0, column=0, sticky="nsew")
        self.plot_area.rowconfigure(0, weight=1)
        self.plot_area.columnconfigure(0, weight=1)

        self._placeholder = ttk.Label(self.plot_area, text="Load data and click Plot.")
        self._placeholder.grid(row=0, column=0, sticky="nsew")

    # ---------------- load ----------------

    def _load(self):
        if self._index_thread is not None and self._index_thread.is_alive():
            messagebox.showwarning("Busy", "Indexing is still running.")
            return

        fp = filedialog.askopenfilename(
            title="Select RAW ensemble database (HDF5)",
            filetypes=[("HDF5 database", "*.h5 *.hdf5"), ("All files", "*")],
        )
        if not fp:
            return

        # Close previous file (important on Windows)
        try:
            if self.data is not None and getattr(self.data, "h5", None) is not None:
                self.data.h5.close()
        except Exception:
            pass

        # Reset state immediately (UI)
        self.data = None
        self.pairs = []
        self.counts = []
        self.pair_counts = {}
        self._pair_to_record_indices = {}
        self.list_pairs.delete(0, tk.END)
        self.btn_plot.configure(state=tk.DISABLED)
        self.btn_save.configure(state=tk.DISABLED)
        self._clear_plot_area()

        self.index_prog_var.set(0)
        self.compute_prog_var.set(0)
        self.var_status.set("Opening database...")

        self._index_thread = threading.Thread(target=self._index_worker, args=(fp,), daemon=True)
        self._index_thread.start()

    def _index_worker(self, fp: str):
        try:
            h5 = h5py.File(fp, "r")

            schema = str(h5.attrs.get("schema", ""))
            if schema != "raw_pairs_v1":
                # still allow if datasets exist, but warn
                pass

            beta = float(h5.attrs["beta"])
            L = int(h5.attrs["L"])
            nrep = int(h5.attrs["nrep"])
            M = int(h5.attrs["M"])
            Ns_req = int(h5.attrs.get("Nsamp_requested", 0))
            Ns_done = int(h5.attrs.get("Nsamp_done", Ns_req))

            nu = np.array(h5["nu"], dtype=float)
            offsets = np.array(h5["sample_offsets"][:Ns_done + 1], dtype=np.int64)

            pairs_i_ds = h5["pairs_i_all"]
            pairs_j_ds = h5["pairs_j_all"]
            G_ds = h5["G_iw_all"]

            # Build counts of available pairs (M_xy): number of samples where pair appears
            pair_counts: Dict[Tuple[int, int], int] = {}

            for s in range(Ns_done):
                start = int(offsets[s])
                end = int(offsets[s + 1])
                if end <= start:
                    # empty
                    if (s % 10) == 0:
                        self.after(0, lambda v=int(100 * (s + 1) / max(1, Ns_done)): self.index_prog_var.set(v))
                    continue

                pi = np.array(pairs_i_ds[start:end], dtype=np.int32)
                pj = np.array(pairs_j_ds[start:end], dtype=np.int32)

                # Ensure we count each pair at most once per sample
                seen = set(zip(pi.tolist(), pj.tolist()))
                for key in seen:
                    pair_counts[key] = pair_counts.get(key, 0) + 1

                if (s % max(1, Ns_done // 100)) == 0:
                    self.after(0, lambda v=int(100 * (s + 1) / max(1, Ns_done)): self.index_prog_var.set(v))

            # Sort pairs for display
            pairs_sorted = sorted(pair_counts.keys())
            counts_sorted = [pair_counts[k] for k in pairs_sorted]

            loaded = LoadedRaw(
                path=fp,
                h5=h5,
                beta=beta,
                nu=nu,
                L=L,
                nrep=nrep,
                M=M,
                Nsamp_requested=Ns_req,
                Nsamp_done=Ns_done,
                offsets=offsets,
                pairs_i_ds=pairs_i_ds,
                pairs_j_ds=pairs_j_ds,
                G_ds=G_ds,
            )

            def _finish():
                self.data = loaded
                self.pair_counts = pair_counts
                self.pairs = pairs_sorted
                self.counts = counts_sorted

                base = os.path.basename(fp)
                self.var_status.set(f"Loaded: {base} | L={L}, nrep={nrep}, M={M}, Ns={Ns_done}")
                self.index_prog_var.set(100)

                self._refresh_pairs_list()
                if len(self.pairs) > 0:
                    self.list_pairs.selection_set(0)
                    self._on_pair_select()
                    self.btn_plot.configure(state=tk.NORMAL)

            self.after(0, _finish)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Load error", str(e)))
            self.after(0, lambda: self.var_status.set("Load failed."))
            # Close file if opened
            try:
                h5.close()
            except Exception:
                pass

    def _refresh_pairs_list(self):
        self.list_pairs.delete(0, tk.END)
        if self.data is None:
            return
        f = self.var_filter.get().strip()
        for (i, j), c in zip(self.pairs, self.counts):
            label = f"{i} -> {j}   (M_xy={c})"
            if f and (f not in label):
                continue
            self.list_pairs.insert(tk.END, label)

    def _on_pair_select(self, _evt=None):
        if self.data is None:
            return
        sel = self.list_pairs.curselection()
        if not sel:
            return
        txt = self.list_pairs.get(sel[0])
        try:
            left = txt.split("(")[0].strip()  # "i -> j"
            i_s, j_s = [t.strip() for t in left.split("->")]
            self.var_i.set(int(i_s))
            self.var_j.set(int(j_s))
        except Exception:
            pass

    # ---------------- plotting ----------------

    def _plot(self):
        if self.data is None:
            return
        if self._compute_thread is not None and self._compute_thread.is_alive():
            messagebox.showwarning("Busy", "A computation is already running.")
            return

        self.compute_prog_var.set(0)
        self.btn_plot.configure(state=tk.DISABLED)
        self.btn_save.configure(state=tk.DISABLED)

        # snapshot parameters (thread-safe)
        i = int(self.var_i.get())
        j = int(self.var_j.get())
        a = int(self.var_a.get())
        b = int(self.var_b.get())
        domain = self.var_domain.get()
        component = self.var_component.get()
        curves = self.var_curves.get()

        fig_w = _as_float(self.var_w.get(), 7.4)
        fig_h = _as_float(self.var_h.get(), 4.8)
        title_fs = _as_float(self.var_title_fs.get(), 13.0)
        label_fs = _as_float(self.var_label_fs.get(), 12.0)
        tick_fs = _as_float(self.var_tick_fs.get(), 11.0)
        legend_fs = _as_float(self.var_legend_fs.get(), 11.0)
        lw = _as_float(self.var_lw.get(), 2.0)

        eps = _as_float(self.var_typ_eps.get(), 1e-30)
        if eps <= 0:
            eps = 1e-30

        args = (i, j, a, b, domain, component, curves, fig_w, fig_h, title_fs, label_fs, tick_fs, legend_fs, lw, eps)
        self._compute_thread = threading.Thread(target=self._compute_and_plot_worker, args=args, daemon=True)
        self._compute_thread.start()

    def _get_record_indices_for_pair(self, i: int, j: int) -> np.ndarray:
        """Return record indices (one per sample where available). Cached."""
        key = (int(i), int(j))
        if key in self._pair_to_record_indices:
            return self._pair_to_record_indices[key]

        d = self.data
        assert d is not None

        offsets = d.offsets
        Ns = d.Nsamp_done
        idx_list: List[int] = []

        for s in range(Ns):
            start = int(offsets[s])
            end = int(offsets[s + 1])
            if end <= start:
                if (s % max(1, Ns // 200)) == 0:
                    self.after(0, lambda v=int(30 * (s + 1) / max(1, Ns)): self.compute_prog_var.set(v))
                continue

            pi = np.array(d.pairs_i_ds[start:end], dtype=np.int32)
            pj = np.array(d.pairs_j_ds[start:end], dtype=np.int32)
            mask = (pi == i) & (pj == j)
            if np.any(mask):
                first = int(np.argmax(mask))
                idx_list.append(start + first)

            if (s % max(1, Ns // 200)) == 0:
                # first 30% reserved for this scan
                self.after(0, lambda v=int(30 * (s + 1) / max(1, Ns)): self.compute_prog_var.set(v))

        arr = np.array(idx_list, dtype=np.int64)
        self._pair_to_record_indices[key] = arr
        return arr

    def _compute_and_plot_worker(
        self,
        i: int, j: int, a: int, b: int,
        domain: str, component: str, curves: str,
        fig_w: float, fig_h: float,
        title_fs: float, label_fs: float, tick_fs: float, legend_fs: float, lw: float,
        eps: float,
    ):
        try:
            d = self.data
            if d is None:
                return

            if not (0 <= a < d.nrep and 0 <= b < d.nrep):
                raise ValueError(f"Replica indices must be in [0, {d.nrep-1}].")

            # Determine which records participate
            rec_idx = self._get_record_indices_for_pair(i, j)
            Mxy = int(rec_idx.shape[0])
            if Mxy == 0:
                raise ValueError(f"Pair ({i}->{j}) is not available in this dataset.")

            # x-axis & length
            if domain == "Matsubara":
                x = d.nu
                xlab = r"$\nu_k$"
                npts = d.M
            else:
                x = tau_grid(d.beta, d.M)
                xlab = r"$\tau$"
                npts = d.M

            # Accumulators (real and imag components separately)
            sum_re = np.zeros((npts,), dtype=np.float64)
            sum_im = np.zeros((npts,), dtype=np.float64)
            logabs_re = np.zeros((npts,), dtype=np.float64)
            logabs_im = np.zeros((npts,), dtype=np.float64)
            sum_abs = np.zeros((npts,), dtype=np.float64)
            logabs_abs = np.zeros((npts,), dtype=np.float64)

            # Compute loop (70% of bar reserved: 30..100)
            for t, ridx in enumerate(rec_idx):
                g_iw = np.array(d.G_ds[int(ridx), :, a, b], dtype=np.complex128)  # (M,)

                if domain == "Imaginary time":
                    g = iw_to_tau(d.beta, g_iw)
                else:
                    g = g_iw

                re = np.real(g).astype(np.float64)
                im = np.imag(g).astype(np.float64)
                absv = np.abs(g).astype(np.float64)

                sum_re += re
                sum_im += im
                logabs_re += np.log(np.abs(re) + eps)
                logabs_im += np.log(np.abs(im) + eps)
                sum_abs += absv
                logabs_abs += np.log(absv + eps)

                if (t % max(1, Mxy // 200)) == 0:
                    frac = (t + 1) / max(1, Mxy)
                    v = int(30 + 70 * frac)
                    self.after(0, lambda vv=v: self.compute_prog_var.set(vv))

            mean_re = sum_re / float(Mxy)
            mean_im = sum_im / float(Mxy)
            typ_re = typical_sign_preserving(sum_re, logabs_re, Mxy)
            typ_im = typical_sign_preserving(sum_im, logabs_im, Mxy)
            mean_abs = sum_abs / float(Mxy)
            typ_abs = np.exp(logabs_abs / float(Mxy))

            # Plot in UI thread
            def _do_plot():
                self._reset_plot_canvas(fig_w, fig_h)
                ax = self._fig.add_subplot(111)

                # Title
                title = rf"$G_{{{i}\,{j}}}^{{{a}\,{b}}}$   ($M_{{{i}{j}}}={Mxy}$)"
                ax.set_title(title, fontsize=title_fs)

                # Labels
                ax.set_xlabel(xlab, fontsize=label_fs)

                # Y label depends on component/domain
                if component == "Real":
                    ylab = r"$\Re\,G$"
                elif component == "Imag":
                    ylab = r"$\Im\,G$"
                elif component == "Abs":
                    ylab = r"$|G|$"
                else:
                    ylab = r"$G$ (components)"
                ax.set_ylabel(ylab, fontsize=label_fs)
                ax.tick_params(labelsize=tick_fs)

                did_any = False

                # Determine what to draw
                want_mean = curves in ("Mean", "Both")
                want_typ = curves in ("Typical", "Both")

                if component in ("Real", "Both"):
                    if want_mean:
                        ax.plot(x, mean_re, linewidth=lw, label=r"$\overline{\Re\,G}$")
                        did_any = True
                    if want_typ:
                        ax.plot(x, typ_re, linewidth=lw, label=r"$\Re\,G_{\mathrm{typ}}$")
                        did_any = True

                if component in ("Imag", "Both"):
                    if want_mean:
                        ax.plot(x, mean_im, linewidth=lw, label=r"$\overline{\Im\,G}$")
                        did_any = True
                    if want_typ:
                        ax.plot(x, typ_im, linewidth=lw, label=r"$\Im\,G_{\mathrm{typ}}$")
                        did_any = True

                if component == "Abs":
                    if want_mean:
                        ax.plot(x, mean_abs, linewidth=lw, label=r"$\overline{|G|}$")
                        did_any = True
                    if want_typ:
                        ax.plot(x, typ_abs, linewidth=lw, label=r"$|G|_{\mathrm{typ}}$")
                        did_any = True

                if did_any:
                    ax.legend(fontsize=legend_fs)
                ax.grid(True, linewidth=0.6, alpha=0.35)

                self._canvas.draw()
                self.btn_save.configure(state=tk.NORMAL)
                self.btn_plot.configure(state=tk.NORMAL)
                self.var_status.set(f"Plotted pair {i}->{j}, a={a}, b={b}, M_xy={Mxy}")
                self.compute_prog_var.set(100)

            self.after(0, _do_plot)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Plot error", str(e)))
            self.after(0, lambda: self.btn_plot.configure(state=tk.NORMAL))
            self.after(0, lambda: self.compute_prog_var.set(0))

    def _on_close(self):
        """Close the HDF5 handle to avoid Windows file-lock issues."""
        try:
            if self.data is not None and getattr(self.data, "h5", None) is not None:
                try:
                    self.data.h5.close()
                except Exception:
                    pass
        finally:
            self.destroy()


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
    app = AnalysisGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
