import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os


# ---------------------------------------------------------------------------
# Disorder sampling
# ---------------------------------------------------------------------------

def sample_couplings(L: int, M: int, dist_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an M x L array of positive couplings J_{s,i} according
    to a chosen distribution. Each row = one Hamiltonian; each column = bond (periodic).
    """
    if dist_name == "Uniform(0,1)":
        J = rng.random((M, L))

    elif dist_name == "Exponential(λ=1)":
        J = rng.exponential(scale=1.0, size=(M, L))

    elif dist_name == "Abs Gaussian N(0,1)":
        J = np.abs(rng.normal(loc=0.0, scale=1.0, size=(M, L)))

    elif dist_name == "Log-normal(μ=0,σ=1)":
        J = rng.lognormal(mean=0.0, sigma=1.0, size=(M, L))

    elif dist_name == "Gamma(k=2,θ=1)":
        # shape=2, scale=1 gives a simple positive distribution with some skew
        J = rng.gamma(shape=2.0, scale=1.0, size=(M, L))

    elif dist_name == "Fermi-Dirac-like":
        # Not a genuine many-body FD spectrum; just a positive, sigmoidal-like profile.
        beta = 1.0
        mu = 0.0
        eps = rng.normal(loc=0.0, scale=1.0, size=(M, L))
        J = 1.0 / (np.exp(beta * (eps - mu)) + 1.0)  # values in (0,1)

    else:
        J = rng.random((M, L))

    return J


# ---------------------------------------------------------------------------
# Single-chain SDRG with singlet tracking
# ---------------------------------------------------------------------------

def sdrg_single_chain(J_init, store_all_events=True):
    """
    Run Ma-Dasgupta SDRG on a single periodic AFM Heisenberg chain.

    Parameters
    ----------
    J_init : 1D array-like
        Initial couplings J_i > 0, length L (periodic).
    store_all_events : bool
        If True, returns full list of singlet formation events.
        If False, only the last event is kept (for gap).

    Returns
    -------
    gap : float
        Approximate finite-size excitation gap (minimal RG scale Ω encountered).
    events : list of (Omega, i_site, j_site)
        Singlet formation events. Each event gives RG scale Omega
        and the original sites i_site < j_site that form a singlet.
    L0 : int
        Original system size.
    """
    J_s = [float(x) for x in J_init]  # current couplings
    L0 = len(J_s)
    sites = list(range(L0))          # current site labels, by original index

    events = []
    min_Omega = np.inf

    # need at least 3 bonds for a nontrivial decimation
    while len(J_s) >= 3:
        n = len(J_s)

        # strongest bond index b
        arr = np.array(J_s, dtype=float)
        b = int(arr.argmax())
        Omega = float(J_s[b])

        left   = (b - 1) % n
        center = b
        right  = (b + 1) % n

        # bond "center" couples sites[center] and sites[right]
        i_site = int(sites[center])
        j_site = int(sites[right])
        if i_site > j_site:
            i_site, j_site = j_site, i_site

        if store_all_events:
            events.append((Omega, i_site, j_site))
        else:
            # keep only the last event
            events = [(Omega, i_site, j_site)]

        if Omega < min_Omega:
            min_Omega = Omega

        # Ma–Dasgupta effective coupling
        J_L = J_s[left]
        J_c = J_s[center]
        J_R = J_s[right]

        if J_c > 0.0:
            J_eff = J_L * J_R / (2.0 * J_c)
        else:
            J_eff = 0.0

        # Build new J_s: replace left by J_eff, remove center and right.
        killed = sorted([center, right])
        J_new = []
        new_sites = []
        for idx in range(n):
            if idx in killed:
                continue
            if idx == left:
                J_new.append(J_eff)
                new_sites.append(sites[left])
            else:
                J_new.append(J_s[idx])
                new_sites.append(sites[idx])

        J_s = J_new
        sites = new_sites

    gap = float(min_Omega) if np.isfinite(min_Omega) else 0.0
    return gap, events, L0


def compute_correlations_from_singlets(singlets_per_sample, L0):
    """
    From singlet data, compute:

      - C_avg(r): disorder-averaged correlations
      - C_per_sample[s, r]: per-sample correlations

    SDRG random-singlet approximation:
      C_s(r) ≈ -3/4 * (# singlets at distance r in sample s) / L0

    Parameters
    ----------
    singlets_per_sample : list of list of (i_site, j_site)
        For each sample s, list of singlet pairs.
    L0 : int
        System size.

    Returns
    -------
    C_avg : np.ndarray, shape (d_max+1,)
    C_per_sample : np.ndarray, shape (M, d_max+1)
    """
    M = len(singlets_per_sample)
    d_max = L0 // 2
    C_per_sample = np.zeros((M, d_max + 1), dtype=float)

    for s, singlets in enumerate(singlets_per_sample):
        counts_s = np.zeros(d_max + 1, dtype=int)
        for (i_site, j_site) in singlets:
            dx = abs(j_site - i_site)
            d = min(dx, L0 - dx)   # periodic distance
            if 0 <= d <= d_max:
                counts_s[d] += 1

        # For each sample: C_s(r) ~ -3/4 * (#singlets at r)/L0
        C_per_sample[s, :] = -0.75 * counts_s / float(L0)

    C_avg = np.mean(C_per_sample, axis=0)
    return C_avg, C_per_sample


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class SDRGGroundStateGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("SDRG Ground-State Viewer: correlations & gaps")

        # -------- state for single-shot mode --------
        self.L_single = None
        self.M_single = None
        self.singlets_per_sample = None  # list of lists (i,j)
        self.gaps_single = None          # shape (M,)
        self.C_avg = None                # shape (d_max+1,)
        self.C_per_sample = None         # shape (M, d_max+1)

        # -------- ground-state window --------
        self.gs_window = None
        self.gs_fig = None
        self.gs_ax = None
        self.gs_canvas = None
        self.gs_sample_var = tk.IntVar(value=0)

        # -------- state for batch mode --------
        self.batch_L_list = None         # list of sizes
        self.batch_gaps_list = None      # list of arrays (M,)

        # -------- gap scaling window --------
        self.gap_window = None
        self.gap_fig = None
        self.gap_ax = None
        self.gap_canvas = None

        # -------- build UI --------
        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # left: controls, right: plots
        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                              padx=5, pady=5)

        # ----- Mode selection at top -----
        mode_frame = ttk.LabelFrame(self.left_frame, text="Mode")
        mode_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.mode_var = tk.StringVar(value="single")
        rb_single = ttk.Radiobutton(mode_frame, text="Single shot",
                                    variable=self.mode_var, value="single",
                                    command=self.on_mode_changed)
        rb_batch = ttk.Radiobutton(mode_frame, text="Batch compute",
                                   variable=self.mode_var, value="batch",
                                   command=self.on_mode_changed)
        rb_single.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        rb_batch.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        # ----- Shared disorder controls (distribution, seed) -----
        shared_frame = ttk.LabelFrame(self.left_frame, text="Disorder & RNG")
        shared_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(shared_frame, text="Distribution:").grid(row=0, column=0,
                                                           sticky="w")
        self.combo_dist = ttk.Combobox(
            shared_frame,
            values=[
                "Uniform(0,1)",
                "Exponential(λ=1)",
                "Abs Gaussian N(0,1)",
                "Log-normal(μ=0,σ=1)",
                "Gamma(k=2,θ=1)",
                "Fermi-Dirac-like",
            ],
            width=22,
            state="readonly"
        )
        self.combo_dist.current(0)
        self.combo_dist.grid(row=0, column=1, columnspan=2, padx=4, pady=2,
                             sticky="w")

        ttk.Label(shared_frame, text="Seed:").grid(row=1, column=0,
                                                   sticky="w")
        self.entry_seed = ttk.Entry(shared_frame, width=10)
        self.entry_seed.insert(0, "0")
        self.entry_seed.grid(row=1, column=1, padx=4, pady=2, sticky="w")

        # ----- Single-shot controls -----
        self.single_frame = ttk.LabelFrame(self.left_frame,
                                           text="Single-shot parameters")
        self.single_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(self.single_frame, text="L (sites):").grid(row=0, column=0,
                                                             sticky="w")
        self.entry_L_single = ttk.Entry(self.single_frame, width=8)
        self.entry_L_single.insert(0, "64")
        self.entry_L_single.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(self.single_frame, text="M (samples):").grid(row=0, column=2,
                                                               sticky="w")
        self.entry_M_single = ttk.Entry(self.single_frame, width=8)
        self.entry_M_single.insert(0, "500")
        self.entry_M_single.grid(row=0, column=3, padx=4, pady=2)

        self.btn_run_single = ttk.Button(self.single_frame, text="Run single shot",
                                         command=self.on_run_single)
        self.btn_run_single.grid(row=1, column=0, columnspan=2,
                                 padx=4, pady=4, sticky="ew")

        self.btn_show_gs = ttk.Button(self.single_frame,
                                      text="Open ground-state view",
                                      command=self.on_open_gs_window,
                                      state="disabled")
        self.btn_show_gs.grid(row=1, column=2, columnspan=2,
                              padx=4, pady=4, sticky="ew")

        ttk.Label(self.single_frame, text="Decimation progress:").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4
        )
        self.progress_single = ttk.Progressbar(self.single_frame,
                                               orient="horizontal",
                                               mode="determinate", length=200)
        self.progress_single.grid(row=3, column=0, columnspan=4,
                                  padx=4, pady=(0, 4), sticky="ew")

        # NEW: Save plots for the main figure
        self.btn_save_main = ttk.Button(self.single_frame, text="Save plots",
                                        command=self.on_save_plots_main)
        self.btn_save_main.grid(row=4, column=0, columnspan=4,
                                padx=4, pady=(2, 4), sticky="ew")

        # ----- Batch compute controls -----
        self.batch_frame = ttk.LabelFrame(self.left_frame,
                                          text="Batch compute parameters")
        self.batch_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(self.batch_frame, text="L_min:").grid(row=0, column=0,
                                                        sticky="w")
        self.entry_L_min = ttk.Entry(self.batch_frame, width=8)
        self.entry_L_min.insert(0, "16")
        self.entry_L_min.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(self.batch_frame, text="L_max:").grid(row=0, column=2,
                                                        sticky="w")
        self.entry_L_max = ttk.Entry(self.batch_frame, width=8)
        self.entry_L_max.insert(0, "128")
        self.entry_L_max.grid(row=0, column=3, padx=4, pady=2)

        ttk.Label(self.batch_frame, text="L_step:").grid(row=1, column=0,
                                                         sticky="w")
        self.entry_L_step = ttk.Entry(self.batch_frame, width=8)
        self.entry_L_step.insert(0, "16")
        self.entry_L_step.grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(self.batch_frame, text="M (samples):").grid(row=1, column=2,
                                                              sticky="w")
        self.entry_M_batch = ttk.Entry(self.batch_frame, width=8)
        self.entry_M_batch.insert(0, "500")
        self.entry_M_batch.grid(row=1, column=3, padx=4, pady=2)

        self.btn_run_batch = ttk.Button(self.batch_frame,
                                        text="Run batch",
                                        command=self.on_run_batch)
        self.btn_run_batch.grid(row=2, column=0, columnspan=4,
                                padx=4, pady=4, sticky="ew")

        ttk.Label(self.batch_frame, text="Batch progress:").grid(
            row=3, column=0, columnspan=4, sticky="w", padx=4
        )
        self.progress_batch = ttk.Progressbar(self.batch_frame,
                                              orient="horizontal",
                                              mode="determinate", length=200)
        self.progress_batch.grid(row=4, column=0, columnspan=4,
                                 padx=4, pady=(0, 4), sticky="ew")

        # ----- Main figure: correlations & gaps -----
        self.fig_main = plt.Figure(figsize=(8, 6))
        gs = self.fig_main.add_gridspec(
            2, 2,
            height_ratios=[2, 1],
            width_ratios=[1, 1],
            hspace=0.4,
            wspace=0.3
        )

        self.ax_corr_avg = self.fig_main.add_subplot(gs[0, 0])
        self.ax_corr_typ = self.fig_main.add_subplot(gs[0, 1])
        self.ax_gap = self.fig_main.add_subplot(gs[1, :])

        self.canvas_main = FigureCanvasTkAgg(self.fig_main,
                                             master=self.right_frame)
        self.canvas_main_widget = self.canvas_main.get_tk_widget()
        self.canvas_main_widget.pack(fill=tk.BOTH, expand=True)

        self._draw_empty_main()
        self.on_mode_changed()

    # ------------------------------------------------------------------
    # Save plots helpers
    # ------------------------------------------------------------------

    def _save_figure_via_dialog(self, fig, title="Save figure", initialfile="figure.png"):
        if fig is None:
            messagebox.showinfo("Save plots", "No figure available to save.")
            return None

        filetypes = [
            ("PNG image", "*.png"),
            ("PDF file", "*.pdf"),
            ("SVG file", "*.svg"),
            ("All files", "*.*"),
        ]

        filename = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            initialfile=initialfile,
            filetypes=filetypes,
        )
        if not filename:
            return None

        try:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save plots", f"Failed to save figure:\n{e}")
            return None

        return filename

    def on_save_plots_main(self):
        """
        Save the main (right-panel) figure containing:
          - average correlation
          - typical correlation
          - gap distribution
        """
        saved = self._save_figure_via_dialog(
            self.fig_main,
            title="Save main plots (correlations & gaps)",
            initialfile="main_plots.png",
        )
        if saved:
            messagebox.showinfo("Save plots", f"Saved:\n{saved}")

    def on_save_plots_gs(self):
        """
        Save the ground-state singlet-circle figure (if open).
        """
        if self.gs_fig is None:
            messagebox.showinfo("Save plots", "Ground-state view is not open (no figure to save).")
            return
        saved = self._save_figure_via_dialog(
            self.gs_fig,
            title="Save ground-state singlet plot",
            initialfile="ground_state.png",
        )
        if saved:
            messagebox.showinfo("Save plots", f"Saved:\n{saved}")

    def on_save_plots_gap_scaling(self):
        """
        Save the batch gap-scaling figure (if open).
        """
        if self.gap_fig is None:
            messagebox.showinfo("Save plots", "Gap-scaling window is not open (no figure to save).")
            return
        saved = self._save_figure_via_dialog(
            self.gap_fig,
            title="Save gap-scaling plot",
            initialfile="gap_scaling.png",
        )
        if saved:
            messagebox.showinfo("Save plots", f"Saved:\n{saved}")

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def on_mode_changed(self):
        mode = self.mode_var.get()

        single_widgets = [
            self.entry_L_single,
            self.entry_M_single,
            self.btn_run_single,
            self.btn_show_gs,
            self.btn_save_main,
        ]
        batch_widgets = [
            self.entry_L_min,
            self.entry_L_max,
            self.entry_L_step,
            self.entry_M_batch,
            self.btn_run_batch,
        ]

        if mode == "single":
            for w in single_widgets:
                try:
                    w.configure(state="normal")
                except tk.TclError:
                    pass
            for w in batch_widgets:
                try:
                    w.configure(state="disabled")
                except tk.TclError:
                    pass
        else:  # batch
            for w in single_widgets:
                try:
                    w.configure(state="disabled")
                except tk.TclError:
                    pass
            for w in batch_widgets:
                try:
                    w.configure(state="normal")
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------
    # Single-shot run
    # ------------------------------------------------------------------

    def on_run_single(self):
        try:
            L = int(self.entry_L_single.get())
            M = int(self.entry_M_single.get())
            if L < 4 or M <= 0:
                raise ValueError
        except ValueError:
            print("Invalid L or M. Need L >= 4, M > 0.")
            return

        try:
            seed = int(self.entry_seed.get())
        except ValueError:
            seed = 0

        dist_name = self.combo_dist.get()
        rng = np.random.default_rng(seed)

        print(f"[Single-shot] Sampling {M} chains of length L={L} "
              f"with distribution {dist_name}...")

        J_ensemble = sample_couplings(L, M, dist_name, rng)

        self.progress_single["value"] = 0.0
        self.progress_single.update_idletasks()

        gaps = np.empty(M, dtype=float)
        singlets_per_sample = []

        for s in range(M):
            J_init = J_ensemble[s, :]
            gap, events, L0 = sdrg_single_chain(J_init, store_all_events=True)
            gaps[s] = gap
            # Store singlet endpoints only (ignore Omega)
            singlets = [(i_site, j_site) for (_, i_site, j_site) in events]
            singlets_per_sample.append(singlets)

            # progress in % of samples processed
            frac = (s + 1) / M
            self.progress_single["value"] = 100.0 * frac
            self.progress_single.update_idletasks()

        self.L_single = L0
        self.M_single = M
        self.gaps_single = gaps
        self.singlets_per_sample = singlets_per_sample

        # Compute average & per-sample correlations
        self.C_avg, self.C_per_sample = compute_correlations_from_singlets(
            self.singlets_per_sample, self.L_single
        )

        print("[Single-shot] Done. Updating plots.")
        self.btn_show_gs.configure(state="normal")
        self.update_main_plots()

    # ------------------------------------------------------------------
    # Batch run (gaps vs sqrt(L))
    # ------------------------------------------------------------------

    def on_run_batch(self):
        try:
            L_min = int(self.entry_L_min.get())
            L_max = int(self.entry_L_max.get())
            L_step = int(self.entry_L_step.get())
            M = int(self.entry_M_batch.get())
            if L_min < 4 or L_max < L_min or L_step <= 0 or M <= 0:
                raise ValueError
        except ValueError:
            print("Invalid batch parameters.")
            return

        try:
            seed = int(self.entry_seed.get())
        except ValueError:
            seed = 0

        dist_name = self.combo_dist.get()
        rng = np.random.default_rng(seed)

        L_list = list(range(L_min, L_max + 1, L_step))
        nL = len(L_list)
        gaps_list = []

        print(f"[Batch] Running SDRG for L in {L_list} with M={M} samples...")

        self.progress_batch["value"] = 0.0
        self.progress_batch.update_idletasks()

        for idx_L, L in enumerate(L_list):
            print(f"  L = {L} ...")
            J_ensemble = sample_couplings(L, M, dist_name, rng)
            gaps = np.empty(M, dtype=float)

            for s in range(M):
                J_init = J_ensemble[s, :]
                gap, _, _ = sdrg_single_chain(J_init, store_all_events=False)
                gaps[s] = gap

            gaps_list.append(gaps)

            frac = (idx_L + 1) / nL
            self.progress_batch["value"] = 100.0 * frac
            self.progress_batch.update_idletasks()

        self.batch_L_list = L_list
        self.batch_gaps_list = gaps_list

        print("[Batch] Finished. Opening gap-scaling window.")
        self.open_gap_scaling_window()

    # ------------------------------------------------------------------
    # Main plots: average & typical correlations, gap histogram
    # ------------------------------------------------------------------

    def _draw_empty_main(self):
        self.ax_corr_avg.clear()
        self.ax_corr_typ.clear()
        self.ax_gap.clear()

        self.ax_corr_avg.set_title("Average correlation (no data)")
        self.ax_corr_typ.set_title("Typical correlation (no data)")
        self.ax_gap.set_title("Gap distribution (no data)")

        self.fig_main.tight_layout()
        self.canvas_main.draw_idle()

    def update_main_plots(self):
        self.ax_corr_avg.clear()
        self.ax_corr_typ.clear()
        self.ax_gap.clear()

        if self.C_avg is None or self.C_per_sample is None:
            self._draw_empty_main()
            return

        C_avg = self.C_avg
        d_max = len(C_avg) - 1
        r_arr = np.arange(d_max + 1)

        # ---------- Average correlation: C_avg(r) ----------
        # We plot -C_avg(r) so that the curve is positive (AFM correlations)
        r_plot = r_arr[1:]          # skip r=0
        C_plot = -C_avg[1:]

        # mask out zeros, so you only see distances where correlations are nonzero
        mask = C_plot > 0.0
        r_plot = r_plot[mask]
        C_plot = C_plot[mask]

        if r_plot.size > 0:
            # numerical data
            self.ax_corr_avg.loglog(r_plot, C_plot, "o-", markersize=4,
                                    label="average")

            # reference ~ r^{-2}, rescaled to pass through first data point
            r_ref = r_plot
            r0 = r_ref[0]
            A_avg = C_plot[0] * (r0**2)     # so that A / r0^2 = C_plot[0]
            C_ref = A_avg / (r_ref**2)
            self.ax_corr_avg.loglog(
                r_ref,
                C_ref,
                "--",
                linewidth=1.2,
                label=r"ref $\propto r^{-2}$"
            )

        self.ax_corr_avg.set_xlabel("distance r")
        self.ax_corr_avg.set_ylabel(r"$-C^{\mathrm{avg}}(r)$")
        self.ax_corr_avg.set_title("Average correlation (log–log)")
        self.ax_corr_avg.grid(True, which="both", alpha=0.2)
        if r_plot.size > 0:
            self.ax_corr_avg.legend(fontsize=8, loc="best")

        # ---------- Typical correlation: C^{typ}(r) ----------
        # C_typ(r) = exp( over_s ln |C_s(r)| ).
        C_per_sample = self.C_per_sample
        M, n_r = C_per_sample.shape
        eps_floor = 1e-12

        C_typ_arr = np.zeros(n_r, dtype=float)
        for r_idx in range(1, n_r):   # skip r=0
            vals = C_per_sample[:, r_idx]
            abs_vals = np.abs(vals)

            # avoid log(0) by flooring to eps_floor
            abs_safe = np.where(abs_vals > 0.0, abs_vals, eps_floor)
            mean_log = np.mean(np.log(abs_safe))
            C_typ_arr[r_idx] = np.exp(mean_log)

        r_plot_typ = r_arr[1:]
        C_plot_typ = C_typ_arr[1:]

        # Distances where all samples had C_s(r)=0 give C_typ ≈ eps_floor.
        # Treat those as truly zero and remove them from the plot.
        mask_typ = C_plot_typ > 2 * eps_floor
        r_plot_typ = r_plot_typ[mask_typ]
        C_plot_typ = C_plot_typ[mask_typ]

        if r_plot_typ.size > 0:
            # numerical data
            self.ax_corr_typ.loglog(
                r_plot_typ,
                C_plot_typ,
                "o-",
                markersize=4,
                label=r"$C^{\mathrm{typ}}(r)=\exp\overline{\ln|C(r)|}$"
            )

            # Fit ln C_typ(r) ≈ ln A - alpha * sqrt(r).
            try:
                x_fit = np.sqrt(r_plot_typ)
                y_fit = np.log(C_plot_typ)

                if x_fit.size >= 2:
                    # ---- Weighted fit: bias toward small distances ----
                    r_for_w = r_plot_typ.astype(float)
                    weights = 1.0 / np.sqrt(r_for_w)

                    # Weighted linear fit
                    m, b = np.polyfit(x_fit, y_fit, 1, w=weights)
                    alpha = -m
                    A = np.exp(b)

                    C_ref_typ = A * np.exp(-alpha * x_fit)

                    print(f"[Typical weighted fit] A = {A:.6e}, alpha = {alpha:.6f}")

                    self.ax_corr_typ.loglog(
                        r_plot_typ,
                        C_ref_typ,
                        "--",
                        linewidth=1.2,
                        label=fr"fit $A e^{{-\alpha\sqrt{{r}}}}$, "
                              fr"$A={A:.2e}$, $\alpha={alpha:.3f}$"
                    )
            except Exception as e:
                print("Warning: typical reference fit failed:", e)

        self.ax_corr_typ.set_xlabel("distance r")
        self.ax_corr_typ.set_ylabel(r"$C^{\mathrm{typ}}(r)$")
        self.ax_corr_typ.set_title("Typical correlation (log–log)")
        self.ax_corr_typ.grid(True, which="both", alpha=0.2)
        if r_plot_typ.size > 0:
            self.ax_corr_typ.legend(fontsize=8, loc="best")

        # ---------- Gap histogram ----------
        gaps = self.gaps_single
        if gaps is not None and np.any(gaps > 0.0):
            good = gaps[gaps > 0.0]
            log_gaps = -np.log(good)

            self.ax_gap.hist(
                log_gaps,
                bins=40,
                density=True,
                alpha=0.7,
                edgecolor="black"
            )
            self.ax_gap.set_xlabel(r"$-\log \Delta$")
            self.ax_gap.set_ylabel("probability density")
            self.ax_gap.set_title("Excitation gap distribution")
            self.ax_gap.grid(True, alpha=0.2)
        else:
            self.ax_gap.set_title("Excitation gap distribution (no gaps > 0)")

        self.fig_main.tight_layout()
        self.canvas_main.draw_idle()

    # ------------------------------------------------------------------
    # Gap scaling window (batch mode)
    # ------------------------------------------------------------------

    def open_gap_scaling_window(self):
        if self.batch_L_list is None or self.batch_gaps_list is None:
            return

        L_list = np.array(self.batch_L_list, dtype=float)
        gaps_list = self.batch_gaps_list

        # Compute average -log(gap) for each L
        x_vals = []
        y_vals = []

        for L, gaps in zip(L_list, gaps_list):
            good = gaps[gaps > 0.0]
            if good.size == 0:
                continue
            avg_loggap = np.mean(-np.log(good))
            x_vals.append(np.sqrt(L))
            y_vals.append(avg_loggap)

        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        top = tk.Toplevel(self.master)
        top.title("Fisher activated scaling: -log Δ vs sqrt(L)")
        self.gap_window = top

        self.gap_fig = plt.Figure(figsize=(6, 4))
        self.gap_ax = self.gap_fig.add_subplot(111)
        ax = self.gap_ax

        if x_vals.size > 0:
            ax.scatter(x_vals, y_vals, color="k", label="data")

            # Fit y = a x + b
            coef = np.polyfit(x_vals, y_vals, 1)
            a, b = coef
            xfit = np.linspace(0, x_vals.max() * 1.05, 200)
            yfit = a * xfit + b
            ax.plot(xfit, yfit, "--r", label=f"fit slope = {a:.3f}")

            ax.set_xlabel(r"$\sqrt{L}$")
            ax.set_ylabel(r"$-\log \Delta_L$")
            ax.set_title("Fisher activated scaling of the excitation gap")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title("No valid (L, gap) data to plot.")

        self.gap_canvas = FigureCanvasTkAgg(self.gap_fig, master=top)
        self.gap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # NEW: save button in the gap scaling window
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btn_frame, text="Save plot", command=self.on_save_plots_gap_scaling).pack(side=tk.LEFT)

        self.gap_fig.tight_layout()
        self.gap_canvas.draw_idle()

        top.protocol("WM_DELETE_WINDOW", self.on_close_gap_window)

    def on_close_gap_window(self):
        try:
            if self.gap_window is not None:
                self.gap_window.destroy()
        except Exception:
            pass
        self.gap_window = None
        self.gap_fig = None
        self.gap_ax = None
        self.gap_canvas = None

    # ------------------------------------------------------------------
    # Ground-state visualization window
    # ------------------------------------------------------------------

    def on_open_gs_window(self):
        if self.singlets_per_sample is None or self.L_single is None:
            print("No single-shot data available for ground-state view.")
            return

        if self.gs_window is not None:
            # already open, just raise it
            try:
                self.gs_window.lift()
            except Exception:
                pass
            return

        self.gs_window = tk.Toplevel(self.master)
        self.gs_window.title("Ground-state singlet structure")

        # Figure for the circle + singlets
        self.gs_fig = plt.Figure(figsize=(4, 4))
        self.gs_ax = self.gs_fig.add_subplot(111)
        our_ax = self.gs_ax
        our_ax.set_aspect("equal")
        our_ax.axis("off")

        self.gs_canvas = FigureCanvasTkAgg(self.gs_fig, master=self.gs_window)
        self.gs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Controls row: slider + save button
        slider_frame = ttk.Frame(self.gs_window)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(slider_frame, text="Sample index:").pack(side=tk.LEFT)

        M = self.M_single if self.M_single is not None else 1
        self.gs_sample_var.set(0)
        self.gs_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=max(0, M - 1),
            orient="horizontal",
            command=self.on_gs_slider_changed
        )
        self.gs_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.gs_sample_label = ttk.Label(slider_frame, text="0 / 0")
        self.gs_sample_label.pack(side=tk.LEFT, padx=5)

        # NEW: save button for GS plot
        ttk.Button(slider_frame, text="Save plot", command=self.on_save_plots_gs).pack(side=tk.RIGHT)

        self.gs_window.protocol("WM_DELETE_WINDOW", self.on_close_gs_window)

        self.update_gs_plot()

    def on_close_gs_window(self):
        try:
            if self.gs_window is not None:
                self.gs_window.destroy()
        except Exception:
            pass
        self.gs_window = None
        self.gs_fig = None
        self.gs_ax = None
        self.gs_canvas = None

    def on_gs_slider_changed(self, event=None):
        self.update_gs_plot()

    def update_gs_plot(self):
        if (self.gs_ax is None or self.gs_canvas is None or
                self.singlets_per_sample is None or
                self.L_single is None):
            return

        M = self.M_single if self.M_single is not None else len(self.singlets_per_sample)
        if M <= 0:
            return

        # current sample index
        s_idx = int(round(self.gs_slider.get()))
        s_idx = max(0, min(s_idx, M - 1))
        self.gs_sample_var.set(s_idx)

        ax = self.gs_ax
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        L = self.L_single
        singlets = self.singlets_per_sample[s_idx]

        # site positions on a unit circle
        angles = np.linspace(0, 2.0 * np.pi, L, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # draw sites
        ax.scatter(x, y, s=20, color="black", zorder=3)
        for i in range(L):
            ax.text(x[i] * 1.08, y[i] * 1.08, str(i),
                    ha="center", va="center", fontsize=7)

        # draw singlets as straight chords
        for (i_site, j_site) in singlets:
            xi, yi = x[i_site], y[i_site]
            xj, yj = x[j_site], y[j_site]
            ax.plot([xi, xj], [yi, yj], color="C0", alpha=0.6)

        ax.set_title(f"Sample {s_idx + 1} / {M}")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)

        self.gs_sample_label.config(text=f"{s_idx + 1} / {M}")

        self.gs_fig.tight_layout()
        self.gs_canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = SDRGGroundStateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
