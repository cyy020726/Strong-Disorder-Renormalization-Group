import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# Sampling distributions (same variants as before + gamma, FD)
# ============================================================

def sample_fermi_dirac(shape, rng: np.random.Generator):
    """
    Rejection sampling for a simple Fermi–Dirac–like positive J.
    This just produces a broad-tailed positive distribution.
    """
    size = int(np.prod(shape))
    out = np.empty(size, dtype=float)
    i = 0
    while i < size:
        x = rng.exponential(scale=1.0)
        u = rng.random()
        # "Fermi-like" acceptance probability in x
        p_accept = np.exp(x) / (np.exp(x) + 1.0)
        if u < p_accept:
            out[i] = x
            i += 1
    return out.reshape(shape)


def sample_couplings(L: int, M: int, dist_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an M x L array of positive couplings J_{s,i} according
    to a chosen distribution. Each row = one chain; each column = bond (periodic).
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
        J = rng.gamma(shape=2.0, scale=1.0, size=(M, L))
    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        J = sample_fermi_dirac((M, L), rng)
    else:
        J = rng.random((M, L))
    return J


# ============================================================
# Main GUI class
# ============================================================

class SDRGGroundStateGUI:
    def __init__(self, master):
        self.master = master
        master.title("SDRG ground-state viewer — correlations & gaps")

        # ---------------- State ----------------
        self.mode_var = tk.StringVar(value="single")  # "single" or "batch"

        self.J_init = None               # single-shot ensemble J (M x L)
        self.singlets_per_sample = None  # list of lists of (i,j) per sample
        self.C_avg = None                # averaged correlation C(r)
        self.C_per_sample = None         # per-sample correlation profiles
        self.gaps = None                 # array of gaps per sample
        self.L = None
        self.M = None

        # batch data: dict[L] -> {"gaps","singlets","C_avg","C_per_sample","M","typical_indices"}
        self.batch_data = {}
        self.batch_L_list = []           # sorted list of L's
        self.batch_current_L_index = 0   # index into batch_L_list

        # typical sample set (indices into 0..M-1), based on sampling ensemble
        self.typical_indices = None

        # progress tracking
        self.decim_total = 1
        self.decim_done = 0
        self.batch_total_sizes = 1

        # Typical-set controls
        self.typical_metric_var = tk.StringVar(value="L2")        # "L2","L1","Chi2","KL"
        self.typical_mode_var = tk.StringVar(value="fraction")    # "fraction" or "threshold"

        # Ground-state (singlet pattern) viewer window
        self.show_gs = tk.BooleanVar(value=False)
        self.gs_top = None
        self.fig_gs = None
        self.ax_gs = None
        self.canvas_gs = None
        self.gs_sample_slider = None
        self.gs_sample_label = None
        self.gs_sample_index = 0   # actual sample index in 0..M-1
        self.gs_slider_index = 0   # position on slider (0..n_avail-1)
        self.typical_only_var = tk.BooleanVar(value=False)  # slider over typical samples only?

        # Gap-scaling fit window
        self.scaling_top = None
        self.scaling_fig = None
        self.scaling_canvas = None

        # Size slider (on main window) state
        self.size_index = 0
        self.size_slider_main = None
        self.size_label_main = None

        # ---------------- Layout ----------------
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ------- Controls (left, top) -------
        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Mode selection
        ttk.Label(control_frame, text="Mode:").grid(row=0, column=0, sticky="w")
        rb_single = ttk.Radiobutton(
            control_frame, text="Single shot",
            variable=self.mode_var, value="single",
            command=self.on_mode_change
        )
        rb_single.grid(row=0, column=1, sticky="w")
        rb_batch = ttk.Radiobutton(
            control_frame, text="Batch",
            variable=self.mode_var, value="batch",
            command=self.on_mode_change
        )
        rb_batch.grid(row=0, column=2, sticky="w")

        # Single-shot L, M
        ttk.Label(control_frame, text="Chain length L (single):").grid(row=1, column=0, sticky="w")
        self.entry_L = ttk.Entry(control_frame, width=8)
        self.entry_L.insert(0, "32")
        self.entry_L.grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="# samples M:").grid(row=1, column=2, sticky="w")
        self.entry_M = ttk.Entry(control_frame, width=8)
        self.entry_M.insert(0, "64")
        self.entry_M.grid(row=1, column=3, padx=4, pady=2)

        # Batch L-range
        ttk.Label(control_frame, text="L_min (batch):").grid(row=2, column=0, sticky="w")
        self.entry_Lmin = ttk.Entry(control_frame, width=8)
        self.entry_Lmin.insert(0, "16")
        self.entry_Lmin.grid(row=2, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="L_max (batch):").grid(row=2, column=2, sticky="w")
        self.entry_Lmax = ttk.Entry(control_frame, width=8)
        self.entry_Lmax.insert(0, "64")
        self.entry_Lmax.grid(row=2, column=3, padx=4, pady=2)

        ttk.Label(control_frame, text="L_step:").grid(row=2, column=4, sticky="w")
        self.entry_Lstep = ttk.Entry(control_frame, width=6)
        self.entry_Lstep.insert(0, "16")
        self.entry_Lstep.grid(row=2, column=5, padx=4, pady=2)

        # Distribution & seed
        ttk.Label(control_frame, text="Distribution:").grid(row=3, column=0, sticky="w")
        self.combo_dist = ttk.Combobox(
            control_frame,
            values=[
                "Uniform(0,1)",
                "Exponential(λ=1)",
                "Abs Gaussian N(0,1)",
                "Log-normal(μ=0,σ=1)",
                "Gamma(k=2,θ=1)",
                "Fermi-Dirac(μ=0,T=1)"
            ],
            width=20,
            state="readonly"
        )
        self.combo_dist.current(0)
        self.combo_dist.grid(row=3, column=1, columnspan=3, padx=4, pady=2, sticky="w")

        ttk.Label(control_frame, text="Seed:").grid(row=3, column=4, sticky="w")
        self.entry_seed = ttk.Entry(control_frame, width=8)
        self.entry_seed.insert(0, "0")
        self.entry_seed.grid(row=3, column=5, padx=4, pady=2)

        # Histogram bins for gap distribution
        ttk.Label(control_frame, text="# bins (gap hist):").grid(row=4, column=0, sticky="w")
        self.entry_bins = ttk.Entry(control_frame, width=6)
        self.entry_bins.insert(0, "20")
        self.entry_bins.grid(row=4, column=1, padx=4, pady=2)

        # Ground-state viewer toggle
        self.check_gs = ttk.Checkbutton(
            control_frame,
            text="Show singlet viewer",
            variable=self.show_gs,
            command=self.on_toggle_gs
        )
        self.check_gs.grid(row=4, column=2, columnspan=2, sticky="w", pady=(4, 2))

        # Typical-set controls
        ttk.Label(control_frame, text="Typical metric:").grid(row=5, column=0, sticky="w")
        self.combo_typ_metric = ttk.Combobox(
            control_frame,
            values=["L2", "L1", "Chi2", "KL"],
            width=8,
            state="readonly",
            textvariable=self.typical_metric_var
        )
        self.combo_typ_metric.grid(row=5, column=1, padx=4, pady=2, sticky="w")

        ttk.Label(control_frame, text="Typical mode:").grid(row=5, column=2, sticky="w")
        self.combo_typ_mode = ttk.Combobox(
            control_frame,
            values=["fraction", "threshold"],
            width=10,
            state="readonly",
            textvariable=self.typical_mode_var
        )
        self.combo_typ_mode.grid(row=5, column=3, padx=4, pady=2, sticky="w")

        ttk.Label(control_frame, text="Param (f or ε):").grid(row=5, column=4, sticky="w")
        self.entry_typical_param = ttk.Entry(control_frame, width=6)
        self.entry_typical_param.insert(0, "0.3")  # default: central 30% if fraction
        self.entry_typical_param.grid(row=5, column=5, padx=4, pady=2)

        # Buttons
        self.button_sample = ttk.Button(control_frame, text="Sample ensemble (single)", command=self.on_sample)
        self.button_sample.grid(row=6, column=0, padx=2, pady=4, sticky="ew")

        self.button_run = ttk.Button(control_frame, text="Run SDRG (compute)", command=self.on_run_sdrg)
        self.button_run.grid(row=6, column=1, padx=2, pady=4, sticky="ew")

        # ------- Progress + size slider -------
        self.status_label = ttk.Label(self.left_frame, text="Status: ready")
        self.status_label.pack(side=tk.TOP, anchor="w", pady=(4, 2))

        self.decim_label = ttk.Label(self.left_frame, text="Decimation progress: 0 / 0 bonds")
        self.decim_label.pack(side=tk.TOP, anchor="w")
        self.progressbar = ttk.Progressbar(self.left_frame, orient="horizontal", length=260, mode="determinate")
        self.progressbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 5))

        self.batch_label = ttk.Label(self.left_frame, text="Batch progress: 0 / 0 sizes")
        self.batch_label.pack(side=tk.TOP, anchor="w")
        self.batch_progressbar = ttk.Progressbar(self.left_frame, orient="horizontal", length=260, mode="determinate")
        self.batch_progressbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 5))

        # Size slider on main window (for batch)
        self.size_label_main = ttk.Label(self.left_frame, text="Size index (batch): 0")
        self.size_label_main.pack(side=tk.TOP, anchor="w", pady=(4, 0))
        self.size_slider_main = tk.Scale(
            self.left_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            command=self.on_size_slider_main_change
        )
        self.size_slider_main.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        # ------- Correlation & gap plots (right) -------
        self.fig_corr = plt.Figure(figsize=(10, 6))
        gs = self.fig_corr.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

        self.ax_corr_avg = self.fig_corr.add_subplot(gs[0, 0])   # ensemble-averaged C(r)
        self.ax_corr_typ = self.fig_corr.add_subplot(gs[0, 1])   # "typical" C(r)
        self.ax_corr_hist = self.fig_corr.add_subplot(gs[1, 0])  # full gap distribution
        self.ax_gap_typ = self.fig_corr.add_subplot(gs[1, 1])    # typical gap subset

        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, master=self.right_frame)
        self.canvas_corr_widget = self.canvas_corr.get_tk_widget()
        self.canvas_corr_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initial state
        self.on_mode_change()
        self.update_corr_plots()

    # ====================================================
    # Mode switching / sampling / running SDRG
    # ====================================================

    def on_mode_change(self):
        """Enable/disable controls depending on mode."""
        mode = self.mode_var.get()
        if mode == "single":
            self.status_label.config(text="Status: single-shot mode selected")

            self.entry_L.config(state="normal")
            self.button_sample.config(state="normal")

            self.entry_Lmin.config(state="disabled")
            self.entry_Lmax.config(state="disabled")
            self.entry_Lstep.config(state="disabled")
        else:
            self.status_label.config(text="Status: batch mode selected (sampling is done inside 'Run SDRG')")

            self.entry_L.config(state="disabled")
            self.button_sample.config(state="disabled")

            self.entry_Lmin.config(state="normal")
            self.entry_Lmax.config(state="normal")
            self.entry_Lstep.config(state="normal")

        self.update_size_slider_range()
        self.update_sample_slider_range()
        self.update_gs_viewer()

    def on_sample(self):
        """Sample an ensemble in single-shot mode."""
        if self.mode_var.get() == "batch":
            self.status_label.config(text="Status: in batch mode, sampling is done inside 'Run SDRG (compute)'")
            return

        try:
            L = int(self.entry_L.get())
            M = int(self.entry_M.get())
            seed = int(self.entry_seed.get())
            if L <= 1 or M <= 0 or L % 2 != 0:
                raise ValueError
        except ValueError:
            self.status_label.config(text="Status: invalid L or M (L must be even, L>1, M>0)")
            return

        self.L = L
        self.M = M
        dist_name = self.combo_dist.get()
        rng = np.random.default_rng(seed)
        self.J_init = sample_couplings(L, M, dist_name, rng)

        # Reset SDRG-related data
        self.singlets_per_sample = None
        self.C_avg = None
        self.C_per_sample = None
        self.gaps = None
        self.typical_indices = None

        # Reset progress bars
        self.decim_total = 0
        self.decim_done = 0
        self.progressbar['mode'] = 'determinate'
        self.progressbar['maximum'] = 1
        self.progressbar['value'] = 0
        self.decim_label.config(text="Decimation progress: 0 / 0 bonds")

        self.batch_total_sizes = 0
        self.batch_progressbar['mode'] = 'determinate'
        self.batch_progressbar['maximum'] = 1
        self.batch_progressbar['value'] = 0
        self.batch_label.config(text="Batch progress: 0 / 0 sizes")

        # Size slider (single-shot: disabled; just show L)
        self.size_index = 0
        self.update_size_slider_range()

        self.status_label.config(text="Status: ensemble sampled (single-shot, no SDRG yet)")
        self.update_corr_plots()
        self.update_gs_viewer()

    def on_run_sdrg(self):
        if self.mode_var.get() == "single":
            self.on_run_sdrg_single()
        else:
            self.on_run_sdrg_batch()

    def on_run_sdrg_single(self):
        """Run SDRG once for the current single-shot ensemble."""
        if self.J_init is None:
            self.status_label.config(text="Status: sample ensemble first (single-shot)")
            return
        L = self.L
        M = self.M
        if L is None or M is None:
            self.status_label.config(text="Status: internal error (L,M)")
            return

        self.status_label.config(text="Status: running SDRG (single-shot)...")
        self.master.update_idletasks()

        # Run SDRG using initial couplings
        singlets_all, gaps_all = self.run_sdrg_ensemble(self.J_init, L)
        self.singlets_per_sample = singlets_all
        self.gaps = gaps_all

        # Compute ensemble-average C(r) and per-sample C_s(r)
        self.C_avg, self.C_per_sample = self.compute_C_from_singlets(singlets_all, L, M)
        # Compute sampling-typical sample set based on J_init (not on gaps)
        self.typical_indices = self.compute_typical_indices_from_J(self.J_init)

        self.status_label.config(text="Status: SDRG complete (single-shot)")

        self.update_corr_plots()
        self.update_size_slider_range()
        self.update_sample_slider_range()
        self.update_gs_viewer()

    def on_run_sdrg_batch(self):
        """Run SDRG for many sizes L and build scaling plot."""
        try:
            Lmin = int(self.entry_Lmin.get())
            Lmax = int(self.entry_Lmax.get())
            Lstep = int(self.entry_Lstep.get())
            M = int(self.entry_M.get())
            seed = int(self.entry_seed.get())
            if Lmin < 2 or Lmax < 2 or Lmax < Lmin or Lstep <= 0 or M <= 0:
                raise ValueError
        except ValueError:
            self.status_label.config(text="Status: invalid batch parameters")
            return

        dist_name = self.combo_dist.get()

        # Build list of even sizes
        L_list = []
        L_cur = Lmin
        while L_cur <= Lmax:
            if L_cur % 2 != 0:
                L_cur += 1
            if L_cur < 2:
                L_cur = 2
            L_list.append(L_cur)
            L_cur += Lstep
        L_list = sorted(set(L_list))
        if not L_list:
            self.status_label.config(text="Status: no valid system sizes for batch")
            return

        self.status_label.config(text="Status: running SDRG (batch mode)...")
        self.master.update_idletasks()

        self.batch_data = {}
        self.batch_L_list = list(L_list)
        self.batch_current_L_index = 0

        # Batch progress bar
        self.batch_total_sizes = len(L_list)
        self.batch_progressbar['mode'] = 'determinate'
        self.batch_progressbar['maximum'] = self.batch_total_sizes
        self.batch_progressbar['value'] = 0
        self.batch_label.config(text=f"Batch progress: 0 / {self.batch_total_sizes} sizes")

        rng = np.random.default_rng(seed)
        scaling_x = []
        scaling_y = []

        for idx_L, L in enumerate(L_list):
            # Sample ensemble for this L
            J_ensemble = sample_couplings(L, M, dist_name, rng)
            singlets_all, gaps_all = self.run_sdrg_ensemble(J_ensemble, L)
            C_avg_L, C_per_sample_L = self.compute_C_from_singlets(singlets_all, L, M)
            # sampling-typical indices for this L
            typical_idx_L = self.compute_typical_indices_from_J(J_ensemble)

            self.batch_data[L] = {
                "gaps": gaps_all,
                "singlets": singlets_all,
                "C_avg": C_avg_L,
                "C_per_sample": C_per_sample_L,
                "M": M,
                "typical_indices": typical_idx_L,
            }

            # For gap scaling plot: average -log gap over positive gaps
            gaps_pos = gaps_all[gaps_all > 0]
            if gaps_pos.size > 0:
                avg_loggap = np.mean(-np.log(gaps_pos))
                scaling_x.append(np.sqrt(L))
                scaling_y.append(avg_loggap)

            self.batch_progressbar['value'] = idx_L + 1
            self.batch_label.config(
                text=f"Batch progress: {idx_L + 1} / {self.batch_total_sizes} sizes"
            )
            self.master.update_idletasks()

        # After batch computation: enable size slider and set to index 0
        self.size_index = 0
        self.size_slider_main.config(state="normal")
        self.size_slider_main.config(from_=0, to=len(self.batch_L_list) - 1)
        self.size_slider_main.set(0)
        self.update_size_slider_range()
        # This will also set active size, update plots and GS viewer
        self.on_size_slider_main_change("0")

        self.status_label.config(text="Status: batch SDRG complete")

        # Open scaling window
        if len(scaling_x) >= 2:
            self.create_scaling_window(np.array(scaling_x), np.array(scaling_y))

    # ====================================================
    # SDRG core routines
    # ====================================================

    def run_sdrg_ensemble(self, J_ensemble, L):
        """
        Run SDRG for each row in J_ensemble (shape M x L).
        Return list of singlet lists and array of gaps.
        """
        M = J_ensemble.shape[0]
        singlets_all = []
        gaps_all = np.empty(M, dtype=float)

        # Total decimations in all chains: each chain produces L/2 singlets
        self.decim_total = M * (L // 2)
        if self.decim_total <= 0:
            self.decim_total = 1
        self.decim_done = 0
        self.progressbar['mode'] = 'determinate'
        self.progressbar['maximum'] = self.decim_total
        self.progressbar['value'] = 0
        self.decim_label.config(
            text=f"Decimation progress: 0 / {self.decim_total} bonds"
        )

        for s in range(M):
            J_row = J_ensemble[s, :]
            singlets_s, gap_s = self.run_sdrg_single_chain(J_row, L)
            singlets_all.append(singlets_s)
            gaps_all[s] = gap_s

            self.decim_done += len(singlets_s)
            if self.decim_done > self.decim_total:
                self.decim_done = self.decim_total
            self.progressbar['value'] = self.decim_done
            self.decim_label.config(
                text=f"Decimation progress: {self.decim_done} / {self.decim_total} bonds"
            )
            self.master.update_idletasks()

        return singlets_all, gaps_all

    def run_sdrg_single_chain(self, J_row, L):
        """
        Graph-like SDRG for one chain:
        - Bonds connect sites i and (i+1) mod L.
        - At each step, pick max J, record a singlet between its sites,
          generate an effective coupling between one neighbor of i and one of j,
          then remove i and j from the graph.

        Returns:
          singlets: list of (i,j) original site indices
          gap: minimum RG scale (smallest Omega seen)
        """
        # Build initial edges and neighbor lists
        edges = {}
        neighbors = {i: set() for i in range(L)}

        # Bonds i-(i+1)
        for i in range(L):
            j = (i + 1) % L
            Jval = float(J_row[i])
            if Jval <= 0:
                continue
            key = frozenset((i, j))
            edges[key] = edges.get(key, 0.0) + Jval
            neighbors[i].add(j)
            neighbors[j].add(i)

        singlets = []
        min_Omega = np.inf

        # Keep decimating until no edges remain
        while edges:
            # Find strongest bond
            key_max, J_max = max(edges.items(), key=lambda kv: kv[1])
            i, j = tuple(key_max)
            Omega = float(J_max)

            # Record singlet
            singlets.append((i, j))
            if Omega < min_Omega:
                min_Omega = Omega

            # Neighbors of i and j (excluding each other)
            nei_i = set(neighbors.get(i, set()))
            if j in nei_i:
                nei_i.remove(j)
            nei_j = set(neighbors.get(j, set()))
            if i in nei_j:
                nei_j.remove(i)

            # Generate an effective coupling between some neighbor of i and some neighbor of j
            if len(nei_i) > 0 and len(nei_j) > 0:
                k = next(iter(nei_i))
                l = next(iter(nei_j))
                if k != l:
                    J_ik = edges.get(frozenset((i, k)), 0.0)
                    J_jl = edges.get(frozenset((j, l)), 0.0)
                    if J_max != 0:
                        J_eff = (J_ik * J_jl) / (2.0 * J_max)
                    else:
                        J_eff = 0.0
                    if J_eff != 0.0:
                        key_kl = frozenset((k, l))
                        edges[key_kl] = edges.get(key_kl, 0.0) + J_eff
                        neighbors.setdefault(k, set()).add(l)
                        neighbors.setdefault(l, set()).add(k)

            # Remove all edges involving i or j
            for n in list(neighbors.get(i, set())):
                key = frozenset((i, n))
                edges.pop(key, None)
                neighbors[n].discard(i)
            for n in list(neighbors.get(j, set())):
                key = frozenset((j, n))
                edges.pop(key, None)
                neighbors[n].discard(j)
            neighbors.pop(i, None)
            neighbors.pop(j, None)

        gap = float(min_Omega) if np.isfinite(min_Omega) else 0.0
        return singlets, gap

    # ====================================================
    # Derived observables: C(r) and typical indices
    # ====================================================

    def compute_C_from_singlets(self, singlets_per_sample, L, M):
        """
        From singlets (i,j) in each sample, compute

          C_s(r) for each sample s, and
          C_avg(r) = (1/M) sum_s C_s(r).

        We define

          C_s(r) = -3/4 * [ (# singlets at distance r in sample s ) / L ].

        This matches C_avg(r) = -3/4 * (#all singlets at r)/(M*L).
        """
        if singlets_per_sample is None or L is None or M is None:
            return None, None

        d_max = L // 2
        if d_max < 1:
            return None, None

        C_per_sample = np.zeros((M, d_max + 1), dtype=float)

        for s, singlets in enumerate(singlets_per_sample):
            counts_s = np.zeros(d_max + 1, dtype=int)
            for (i, j) in singlets:
                dx = abs(i - j)
                d = min(dx, L - dx)
                if 0 <= d <= d_max:
                    counts_s[d] += 1
            C_per_sample[s, :] = -0.75 * counts_s / float(L)

        C_avg = np.mean(C_per_sample, axis=0)
        return C_avg, C_per_sample

    def get_hist_bins(self):
        """Number of bins for gap histogram (also reused for 'typical set' histogram)."""
        try:
            nb = int(self.entry_bins.get())
            if nb < 1:
                nb = 1
        except ValueError:
            nb = 20
        return nb

    def compute_typical_indices_from_J(self, J_ensemble, nbins=None):
        """
        Define 'typical' samples as those whose empirical J-distribution
        is closest to the global J-distribution (in histogram space).

        Steps:
          - Build a global histogram P_global(J) from all bonds.
          - For each chain s, build P_s(J) over its L bonds (same bin edges).
          - Compute a distance d_s between P_s and P_global, using a metric
            chosen in the GUI ('L2','L1','Chi2','KL').
          - Depending on 'Typical mode':
              * 'fraction': choose the central fraction f of samples with
                smallest d_s (f from GUI).
              * 'threshold': choose all samples with d_s <= ε (ε from GUI).
        """
        if J_ensemble is None:
            return None
        M, L = J_ensemble.shape
        if M == 0 or L == 0:
            return None

        if nbins is None:
            nbins = self.get_hist_bins()

        # Flatten all bonds and build global histogram
        all_J = J_ensemble.reshape(-1)
        if all_J.size == 0:
            return list(range(M))

        counts_global, edges = np.histogram(all_J, bins=nbins, density=True)
        if counts_global.sum() <= 0:
            # If everything is zero, treat all samples as typical
            return list(range(M))
        P_global = counts_global / counts_global.sum()
        eps = 1e-12

        metric = self.typical_metric_var.get()

        # Helper: distance between P_s and P_global
        def distance(P_s, P_g):
            if metric == "L1":
                return float(np.sum(np.abs(P_s - P_g)))
            elif metric == "Chi2":
                denom = P_g + eps
                return float(np.sum((P_s - P_g) ** 2 / denom))
            elif metric == "KL":
                P_s_safe = P_s + eps
                P_g_safe = P_g + eps
                return float(np.sum(P_s_safe * np.log(P_s_safe / P_g_safe)))
            else:  # "L2" or unknown -> default to L2
                return float(np.linalg.norm(P_s - P_g))

        # Per-sample distances
        d_list = []
        for s in range(M):
            J_s = J_ensemble[s, :]
            counts_s, _ = np.histogram(J_s, bins=edges, density=True)
            if counts_s.sum() <= 0:
                d = np.inf
            else:
                P_s = counts_s / counts_s.sum()
                d = distance(P_s, P_global)
            d_list.append(d)
        d_list = np.array(d_list)

        idx_all = np.arange(M)
        finite_mask = np.isfinite(d_list)
        if not finite_mask.any():
            return idx_all.tolist()

        idx_valid = idx_all[finite_mask]
        d_valid = d_list[finite_mask]

        order = np.argsort(d_valid)
        idx_sorted = idx_valid[order]
        d_sorted = d_valid[order]

        if len(idx_sorted) <= 3:
            return idx_sorted.tolist()

        # Read user parameter (fraction f or threshold ε)
        try:
            param = float(self.entry_typical_param.get())
        except ValueError:
            param = None

        mode = self.typical_mode_var.get()

        chosen = None

        if mode == "threshold" and param is not None and param > 0.0:
            epsilon = param
            mask_thr = d_sorted <= epsilon
            if mask_thr.any():
                chosen = idx_sorted[mask_thr]

        # If threshold mode gave nothing (or invalid epsilon), fall back to fraction
        if chosen is None or chosen.size == 0:
            # Fraction mode
            if param is None or param <= 0.0 or param > 1.0:
                frac = 0.3  # default central 30%
            else:
                frac = param
            n_typ = max(1, int(round(frac * len(idx_sorted))))
            if n_typ >= len(idx_sorted):
                chosen = idx_sorted
            else:
                offset = (len(idx_sorted) - n_typ) // 2
                chosen = idx_sorted[offset: offset + n_typ]

        return sorted(chosen.tolist())

    # ====================================================
    # Correlation & gap plots (avg, typical)
    # ====================================================

    def update_corr_plots(self):
        """Update the correlation plots (avg & typical) and gap histograms."""
        self.ax_corr_avg.clear()
        self.ax_corr_typ.clear()
        self.ax_corr_hist.clear()
        self.ax_gap_typ.clear()

        if self.C_avg is None or self.L is None or self.gaps is None:
            # No data yet
            self.ax_corr_avg.set_title("Singlet-based correlation profile (average, no data)")
            self.ax_corr_avg.set_xlabel("distance r")
            self.ax_corr_avg.set_ylabel("-C(r)")

            self.ax_corr_typ.set_title("Typical correlation profile (no data)")
            self.ax_corr_typ.set_xlabel("distance r")
            self.ax_corr_typ.set_ylabel("|C(r)|")

            self.ax_corr_hist.set_title("Gap distribution (no data)")
            self.ax_corr_hist.set_xlabel(r"$-\log \Delta$")
            self.ax_corr_hist.set_ylabel("probability density")

            self.ax_gap_typ.set_title("Typical gap subset (no data)")
            self.ax_gap_typ.set_xlabel(r"$-\log \Delta_{\mathrm{typ}}$")
            self.ax_gap_typ.set_ylabel("probability density")

            self.fig_corr.tight_layout()
            self.canvas_corr.draw_idle()
            return

        # ===== Average correlation (top-left) =====
        L = self.L
        d_max = L // 2
        distances = np.arange(0, d_max + 1)
        y = self.C_avg

        r_plot = distances[1:]          # skip r=0
        C_plot = -y[1:]                 # >0 for plotting
        mask = C_plot > 0
        r_plot = r_plot[mask]
        C_plot = C_plot[mask]

        if r_plot.size > 0:
            self.ax_corr_avg.loglog(r_plot, C_plot, "o-", label="SDRG avg", markersize=4)
            theory = 1.0 / (r_plot ** 2)
            scale_factor = C_plot[0] / theory[0]
            theory_scaled = theory * scale_factor
            self.ax_corr_avg.loglog(r_plot, theory_scaled, "--", label="∝ 1/r^2 (rescaled)")
            self.ax_corr_avg.set_xlabel("distance r")
            self.ax_corr_avg.set_ylabel("-C(r)")
            self.ax_corr_avg.set_title("Log–log average spin correlations from SDRG")
            self.ax_corr_avg.legend()
        else:
            self.ax_corr_avg.set_title("Singlet-based correlation profile (no nonzero C(r))")
            self.ax_corr_avg.set_xlabel("distance r")
            self.ax_corr_avg.set_ylabel("-C(r)")

        # ===== Typical correlation (top-right) =====
        if self.C_per_sample is not None and self.C_per_sample.size > 0:
            M = self.C_per_sample.shape[0]
            if self.typical_indices and len(self.typical_indices) > 0:
                idxs = np.array(self.typical_indices, dtype=int)
                idxs = idxs[(0 <= idxs) & (idxs < M)]
                if idxs.size == 0:
                    idxs = np.arange(M)
            else:
                idxs = np.arange(M)

            # "Typical" magnitude: median over sampling-typical chains of |C_s(r)|
            C_sub = self.C_per_sample[idxs, :]
            C_mag = np.abs(C_sub[:, 1:])        # skip r=0
            if C_mag.size > 0:
                C_typ = np.median(C_mag, axis=0)
                r_typ = distances[1:]
                mask2 = C_typ > 0
                r_typ = r_typ[mask2]
                C_typ = C_typ[mask2]
                if r_typ.size > 0:
                    self.ax_corr_typ.loglog(r_typ, C_typ, "o-", markersize=4, label="typical |C(r)|")
                    self.ax_corr_typ.set_xlabel("distance r")
                    self.ax_corr_typ.set_ylabel("|C(r)|")
                    self.ax_corr_typ.set_title("Log–log typical correlations (sampling-typical chains)")
                    self.ax_corr_typ.legend()
                else:
                    self.ax_corr_typ.set_title("Typical correlations (all zero)")
                    self.ax_corr_typ.set_xlabel("distance r")
                    self.ax_corr_typ.set_ylabel("|C(r)|")
            else:
                self.ax_corr_typ.set_title("Typical correlations (no data)")
                self.ax_corr_typ.set_xlabel("distance r")
                self.ax_corr_typ.set_ylabel("|C(r)|")
        else:
            self.ax_corr_typ.set_title("Typical correlations (no data)")
            self.ax_corr_typ.set_xlabel("distance r")
            self.ax_corr_typ.set_ylabel("|C(r)|")

        # ===== Gap distribution (bottom-left, full ensemble) =====
        gaps = self.gaps
        gaps_pos = gaps[gaps > 0]
        nb = self.get_hist_bins()

        if gaps_pos.size == 0:
            self.ax_corr_hist.set_title("Gap distribution (no positive gaps)")
            self.ax_corr_hist.set_xlabel(r"$-\log \Delta$")
            self.ax_corr_hist.set_ylabel("probability density")
        else:
            g_log = -np.log(gaps_pos)
            g_min = float(g_log.min())
            g_max = float(g_log.max())
            if g_max == g_min:
                g_min = g_min - 0.5 if g_min != 0 else -0.5
                g_max = g_max + 0.5 if g_max != 0 else 0.5

            self.ax_corr_hist.hist(
                g_log,
                bins=nb,
                range=(g_min, g_max),
                density=True,          # empirical density
                edgecolor='black',
                alpha=0.7
            )
            self.ax_corr_hist.set_xlabel(r"$-\log \Delta$")
            self.ax_corr_hist.set_ylabel("probability density")
            self.ax_corr_hist.set_yscale("log")
            self.ax_corr_hist.set_title(r"Gap distribution across ensemble")

            mean_gap = float(np.mean(gaps_pos))
            self.ax_corr_hist.text(
                0.98, 0.95,
                f"⟨Δ⟩ ≈ {mean_gap:.3g}",
                transform=self.ax_corr_hist.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )

        # ===== Typical gap subset (bottom-right, only sampling-typical samples) =====
        if self.typical_indices and len(self.typical_indices) > 0:
            idxs = np.array(self.typical_indices, dtype=int)
            idxs = idxs[(idxs >= 0) & (idxs < len(self.gaps))]
            gaps_typ = self.gaps[idxs]
            gaps_typ_pos = gaps_typ[gaps_typ > 0]
        else:
            gaps_typ_pos = np.array([])

        if gaps_typ_pos.size == 0:
            self.ax_gap_typ.set_title("Typical gap subset (no positive gaps)")
            self.ax_gap_typ.set_xlabel(r"$-\log \Delta_{\mathrm{typ}}$")
            self.ax_gap_typ.set_ylabel("probability density")
        else:
            g_log_typ = -np.log(gaps_typ_pos)
            gl_min = float(g_log_typ.min())
            gl_max = float(g_log_typ.max())
            if gl_max == gl_min:
                gl_min = gl_min - 0.5 if gl_min != 0 else -0.5
                gl_max = gl_max + 0.5 if gl_max != 0 else 0.5

            self.ax_gap_typ.hist(
                g_log_typ,
                bins=nb,
                range=(gl_min, gl_max),
                density=True,
                edgecolor='black',
                alpha=0.7,
                color='C1'
            )
            self.ax_gap_typ.set_xlabel(r"$-\log \Delta_{\mathrm{typ}}$")
            self.ax_gap_typ.set_ylabel("probability density")
            self.ax_gap_typ.set_yscale("log")
            self.ax_gap_typ.set_title("Gap distribution restricted to sampling-typical chains")

        self.fig_corr.tight_layout()
        self.canvas_corr.draw_idle()

    # ====================================================
    # Gap scaling window (batch mode)
    # ====================================================

    def create_scaling_window(self, x_vals, y_vals):
        """
        x_vals ~ sqrt(L), y_vals ~ ⟨-log(gap)⟩_disorder.
        Plot and fit y = a x + b.
        """
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x = x_vals[mask]
        y = y_vals[mask]
        if x.size < 2:
            return

        a, b = np.polyfit(x, y, 1)
        xfit = np.linspace(0, float(x.max()), 200)
        yfit = a * xfit + b

        if self.scaling_top is not None:
            try:
                self.scaling_top.destroy()
            except Exception:
                pass

        self.scaling_top = tk.Toplevel(self.master)
        self.scaling_top.title("Gap scaling: ⟨-log Δ_L⟩ vs √L")

        self.scaling_fig = plt.Figure(figsize=(6, 4))
        ax = self.scaling_fig.add_subplot(111)

        ax.scatter(x, y, color='k', label="data")
        ax.plot(xfit, yfit, '--r', label=f"fit slope = {a:.3f}")
        ax.set_xlabel(r"$\sqrt{L}$")
        ax.set_ylabel(r"$\langle -\log \Delta_L \rangle$")
        ax.set_title("Fisher-like activated scaling of the excitation gap")
        ax.legend()

        self.scaling_fig.tight_layout()
        self.scaling_canvas = FigureCanvasTkAgg(self.scaling_fig, master=self.scaling_top)
        widget = self.scaling_canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scaling_canvas.draw_idle()

    # ====================================================
    # Ground-state (singlet) viewer window
    # ====================================================

    def on_toggle_gs(self):
        """Show/hide the ground-state singlet viewer window."""
        if self.show_gs.get():
            if self.gs_top is None:
                self.create_gs_window()
            self.update_sample_slider_range()
            self.update_gs_viewer()
        else:
            if self.gs_top is not None:
                self.on_close_gs()

    def create_gs_window(self):
        """Create the singlet viewer window (with sample slider and typical toggle)."""
        if self.gs_top is not None:
            return

        self.gs_top = tk.Toplevel(self.master)
        self.gs_top.title("SDRG ground-state singlet patterns (per sample / size)")

        fig_frame = ttk.Frame(self.gs_top)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_gs = plt.Figure(figsize=(6, 6))
        self.ax_gs = self.fig_gs.add_subplot(111)
        self.ax_gs.set_aspect('equal', adjustable='box')
        self.ax_gs.axis('off')

        self.canvas_gs = FigureCanvasTkAgg(self.fig_gs, master=fig_frame)
        self.canvas_gs_widget = self.canvas_gs.get_tk_widget()
        self.canvas_gs_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sliders + toggles
        slider_frame = ttk.Frame(self.gs_top)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        slider_frame.columnconfigure(1, weight=1)

        ttk.Label(slider_frame, text="Sample index:").grid(row=0, column=0, sticky="w")
        self.gs_sample_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            command=self.on_gs_sample_slider_change
        )
        self.gs_sample_slider.grid(row=0, column=1, sticky="ew", padx=5)

        # Toggle for "typical ground states only"
        self.typical_only_check = ttk.Checkbutton(
            slider_frame,
            text="Typical only (sampling ensemble)",
            variable=self.typical_only_var,
            command=self.on_toggle_typical_only
        )
        self.typical_only_check.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.gs_sample_label = ttk.Label(slider_frame, text="(no data)")
        self.gs_sample_label.grid(row=1, column=1, sticky="e")

        self.gs_top.protocol("WM_DELETE_WINDOW", self.on_close_gs)

    def on_close_gs(self):
        """Handle closing the singlet viewer window."""
        try:
            if self.gs_top is not None:
                self.gs_top.destroy()
        except Exception:
            pass

        self.gs_top = None
        self.fig_gs = None
        self.ax_gs = None
        self.canvas_gs = None
        self.gs_sample_slider = None
        self.gs_sample_label = None
        self.typical_only_check = None
        self.show_gs.set(False)

    # ---- Size slider (main window) and sample slider (GS window) ----

    def update_size_slider_range(self):
        """Configure main-window size slider depending on mode and data."""
        if self.size_slider_main is None:
            return

        mode = self.mode_var.get()
        if mode == "batch" and self.batch_L_list:
            max_idx = len(self.batch_L_list) - 1
            self.size_slider_main.config(state="normal", from_=0, to=max_idx)
            idx = max(0, min(self.size_index, max_idx))
            self.size_index = idx
            self.size_slider_main.set(idx)
            L = self.batch_L_list[idx]
            self.size_label_main.config(text=f"Size index (batch): {idx} (L={L})")
        elif mode == "single" and self.L is not None:
            # Single-shot: slider disabled, but show L
            self.size_slider_main.config(state="disabled", from_=0, to=0)
            self.size_slider_main.set(0)
            self.size_index = 0
            self.size_label_main.config(text=f"Size index (single): 0 (L={self.L})")
        else:
            # No data yet
            self.size_slider_main.config(state="disabled", from_=0, to=0)
            self.size_slider_main.set(0)
            self.size_index = 0
            self.size_label_main.config(text="Size index: 0")

    def update_sample_slider_range(self):
        """Configure the sample slider in the GS window (honouring 'typical only')."""
        if self.gs_sample_slider is None:
            return

        if self.M is None or self.M <= 0:
            self.gs_sample_slider.config(state="disabled", from_=0, to=0)
            self.gs_sample_slider.set(0)
            self.gs_slider_index = 0
            self.gs_sample_index = 0
            if self.gs_sample_label is not None:
                self.gs_sample_label.config(text="(no data)")
            return

        # Decide how many samples are available depending on typical-only toggle
        if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
            n_avail = len(self.typical_indices)
        else:
            n_avail = self.M

        if n_avail <= 0:
            self.gs_sample_slider.config(state="disabled", from_=0, to=0)
            self.gs_sample_slider.set(0)
            self.gs_slider_index = 0
            self.gs_sample_index = 0
            if self.gs_sample_label is not None:
                self.gs_sample_label.config(text="(no data)")
            return

        self.gs_sample_slider.config(state="normal", from_=0, to=n_avail - 1)
        # Reset slider to 0 in new regime
        self.gs_slider_index = 0
        if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
            self.gs_sample_index = self.typical_indices[0]
        else:
            self.gs_sample_index = 0
        self.gs_sample_slider.set(0)

        # Update label
        if self.gs_sample_label is not None:
            if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
                self.gs_sample_label.config(
                    text=f"typ slider 0 (sample {self.gs_sample_index} of {self.M - 1})"
                )
            else:
                self.gs_sample_label.config(
                    text=f"sample 0/{self.M - 1}"
                )

    def on_size_slider_main_change(self, value):
        """When user moves the size slider in batch mode, switch to that L."""
        try:
            idx = int(float(value))
        except ValueError:
            idx = 0
        self.size_index = idx

        if self.mode_var.get() == "batch":
            self.set_active_size_index(idx)
            self.update_corr_plots()
            self.update_sample_slider_range()
            self.update_gs_viewer()
        else:
            # In single-shot mode, the size slider is disabled and this shouldn't be triggered
            pass

    def on_gs_sample_slider_change(self, value):
        """When user moves the sample slider, map slider index to actual sample index."""
        try:
            slider_idx = int(float(value))
        except ValueError:
            slider_idx = 0

        if self.M is None or self.M <= 0:
            self.gs_slider_index = 0
            self.gs_sample_index = 0
            return

        # clamp slider index to available
        if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
            n_avail = len(self.typical_indices)
        else:
            n_avail = self.M

        if n_avail <= 0:
            self.gs_slider_index = 0
            self.gs_sample_index = 0
            return

        slider_idx = max(0, min(slider_idx, n_avail - 1))
        self.gs_slider_index = slider_idx

        if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
            self.gs_sample_index = self.typical_indices[slider_idx]
        else:
            self.gs_sample_index = slider_idx

        self.update_gs_viewer()

    def on_toggle_typical_only(self):
        """User toggled 'typical only' for ground-state slider."""
        self.update_sample_slider_range()
        self.update_gs_viewer()

    def set_active_size_index(self, idx):
        """
        In batch mode: pick which L (index in batch_L_list) is active.
        This sets L, M, singlets_per_sample, C_avg, C_per_sample, gaps, typical_indices
        so the main plots and GS viewer can use them.
        """
        if self.mode_var.get() != "batch":
            return
        if not self.batch_L_list:
            return

        idx = max(0, min(idx, len(self.batch_L_list) - 1))
        self.batch_current_L_index = idx

        L = self.batch_L_list[idx]
        data = self.batch_data.get(L)
        if data is None:
            return

        self.L = L
        self.M = data.get("M", len(data["singlets"]))
        self.singlets_per_sample = data["singlets"]
        self.C_avg = data["C_avg"]
        self.C_per_sample = data.get("C_per_sample")
        self.gaps = data["gaps"]
        self.typical_indices = data.get("typical_indices")

    def update_gs_viewer(self):
        """Redraw the singlet picture (if the GS window is open)."""
        if not self.show_gs.get() or self.gs_top is None or self.ax_gs is None:
            return

        self.ax_gs.clear()
        self.ax_gs.set_aspect('equal', adjustable='box')
        self.ax_gs.axis('off')

        if self.L is None:
            self.fig_gs.tight_layout()
            if self.canvas_gs is not None:
                self.canvas_gs.draw_idle()
            return

        L = self.L
        theta = np.linspace(0, 2 * np.pi, L, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        # Draw sites on circle
        self.ax_gs.scatter(x, y, s=20, c='k')
        for i in range(L):
            self.ax_gs.text(
                x[i] * 1.08, y[i] * 1.08,
                str(i),
                ha='center',
                va='center',
                fontsize=8
            )

        if self.singlets_per_sample is None or self.M is None:
            self.ax_gs.set_title("No ground-state singlets computed yet.")
        else:
            # Actual sample index we are showing
            s_idx = self.gs_sample_index
            if s_idx < 0 or s_idx >= self.M:
                s_idx = 0
                self.gs_sample_index = 0

            pairs = self.singlets_per_sample[s_idx]

            # Draw singlet chords
            for (i, j) in pairs:
                xi, yi = x[i], y[i]
                xj, yj = x[j], y[j]
                self.ax_gs.plot(
                    [xi, xj],
                    [yi, yj],
                    color='C0',
                    linewidth=1.2,
                    alpha=0.8
                )

            mode = self.mode_var.get()
            if mode == "batch" and self.batch_L_list:
                size_info = f"L = {L}, size idx = {self.batch_current_L_index}"
            else:
                size_info = f"L = {L}"

            # slider info, depending on typical-only
            if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
                slider_info = f"typ slider {self.gs_slider_index} (sample {s_idx} of {self.M - 1})"
            else:
                slider_info = f"sample {s_idx}/{self.M - 1}"

            self.ax_gs.set_title(f"{size_info} | {slider_info}")

        self.ax_gs.set_xlim(-1.3, 1.3)
        self.ax_gs.set_ylim(-1.3, 1.3)

        if self.gs_sample_label is not None and self.M is not None:
            if self.typical_only_var.get() and self.typical_indices and len(self.typical_indices) > 0:
                self.gs_sample_label.config(
                    text=f"typical slider {self.gs_slider_index} (sample {self.gs_sample_index} of {self.M - 1})"
                )
            else:
                self.gs_sample_label.config(
                    text=f"sample {self.gs_sample_index}/{self.M - 1}"
                )

        self.fig_gs.tight_layout()
        if self.canvas_gs is not None:
            self.canvas_gs.draw_idle()


# ============================================================
# Main
# ============================================================

def main():
    root = tk.Tk()
    app = SDRGGroundStateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
