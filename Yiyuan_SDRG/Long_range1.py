import numpy as np
import heapq
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Sampling helpers: positive inter-site spacings
# =============================================================================

def sample_fermi_dirac_positive(shape, rng: np.random.Generator) -> np.ndarray:
    """
    Sample X >= 0 from a Fermi-Dirac-like target density t(x) ∝ 1/(exp(x)+1).

    Rejection sampling with envelope Exp(1): g(x)=exp(-x).
    Accept with probability t/g = exp(x)/(exp(x)+1) = 1/(1+exp(-x)).

    Vectorized batching for speed.
    """
    size = int(np.prod(shape))
    if size <= 0:
        return np.zeros(shape, dtype=float)

    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        batch = max(2048, 2 * (size - filled))
        x = rng.exponential(scale=1.0, size=batch)
        u = rng.random(batch)
        accept = u < (1.0 / (1.0 + np.exp(-x)))  # sigmoid(x)
        if not np.any(accept):
            continue
        acc = x[accept]
        take = min(acc.size, size - filled)
        out[filled:filled + take] = acc[:take]
        filled += take

    return out.reshape(shape)


def sample_gaps(num_gaps: int,
                dist_name: str,
                rng: np.random.Generator,
                scale: float = 1.0,
                min_gap: float = 1e-3) -> np.ndarray:
    """
    Sample positive inter-site distances ("gaps") with a strict minimum separation.

    We treat min_gap as a *hard-core shift*:
        gap = min_gap + scale * X, with X >= 0 from the chosen base distribution.

    This keeps the shape of the base distribution (up to affine rescaling) while
    enforcing support on [min_gap, ∞).
    """
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    try:
        min_gap = float(min_gap)
    except Exception:
        min_gap = 0.0

    scale = max(scale, 0.0)
    min_gap = max(min_gap, 0.0)

    if num_gaps <= 0:
        return np.zeros((0,), dtype=float)

    if dist_name == "Regular lattice (constant)":
        gaps = np.ones(num_gaps, dtype=float)

    elif dist_name == "Uniform(0,1)":
        gaps = rng.random(num_gaps)

    elif dist_name == "Exponential(λ=1)":
        gaps = rng.exponential(scale=1.0, size=num_gaps)

    elif dist_name == "Abs Gaussian N(0,1)":
        gaps = np.abs(rng.normal(loc=0.0, scale=1.0, size=num_gaps))

    elif dist_name == "Log-normal(μ=0,σ=1)":
        gaps = rng.lognormal(mean=0.0, sigma=1.0, size=num_gaps)

    elif dist_name == "Gamma(k=2,θ=1)":
        gaps = rng.gamma(shape=2.0, scale=1.0, size=num_gaps)

    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        gaps = sample_fermi_dirac_positive((num_gaps,), rng)

    else:
        gaps = rng.random(num_gaps)

    gaps = np.asarray(gaps, dtype=float)
    gaps = min_gap + scale * gaps
    return gaps


def sample_positions_from_gaps(L: int,
                              M: int,
                              dist_name: str,
                              rng: np.random.Generator,
                              scale: float,
                              min_gap: float) -> np.ndarray:
    """
    Return positions array of shape (M, L) for open chains.
    Site 0 starts at x=0; x_i = sum_{k < i} gap_k.

    Vectorized across M for speed.
    """
    if L < 2:
        raise ValueError("Need L >= 2.")
    if M <= 0:
        return np.zeros((0, L), dtype=float)

    # Robustify inputs
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    try:
        min_gap = float(min_gap)
    except Exception:
        min_gap = 0.0
    scale = max(scale, 0.0)
    min_gap = max(min_gap, 0.0)

    n_gaps = L - 1

    if dist_name == "Regular lattice (constant)":
        gaps = np.ones((M, n_gaps), dtype=float)

    elif dist_name == "Uniform(0,1)":
        gaps = rng.random((M, n_gaps))

    elif dist_name == "Exponential(λ=1)":
        gaps = rng.exponential(scale=1.0, size=(M, n_gaps))

    elif dist_name == "Abs Gaussian N(0,1)":
        gaps = np.abs(rng.normal(loc=0.0, scale=1.0, size=(M, n_gaps)))

    elif dist_name == "Log-normal(μ=0,σ=1)":
        gaps = rng.lognormal(mean=0.0, sigma=1.0, size=(M, n_gaps))

    elif dist_name == "Gamma(k=2,θ=1)":
        gaps = rng.gamma(shape=2.0, scale=1.0, size=(M, n_gaps))

    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        gaps = sample_fermi_dirac_positive((M, n_gaps), rng)

    else:
        gaps = rng.random((M, n_gaps))

    gaps = min_gap + scale * np.asarray(gaps, dtype=float)
    pos = np.zeros((M, L), dtype=float)
    pos[:, 1:] = np.cumsum(gaps, axis=1)
    return pos


# =============================================================================
# SDRG core: distance renormalization rules
# =============================================================================

def _safe_power_ratio(a: float, b: float, alpha: float) -> float:
    """Compute (a/b)^alpha robustly for positive a,b."""
    a = max(float(a), 1e-300)
    b = max(float(b), 1e-300)
    return (a / b) ** float(alpha)


def renormalized_distance_alltoall(
    r_lm: float,
    rho: float,
    r_li: float,
    r_lj: float,
    r_im: float,
    r_jm: float,
    alpha: float,
    gamma: float,
) -> float:
    """
    Paper Eq. (5):

        r~ = r * { 1 + (1/(1+γ)) * ((r * ρ)/(r_li * r_jm))^α
                   * [1 - (r_li/r_lj)^α] * [1 - (r_jm/r_im)^α] }^{-1/α}

    All distances must be positive.
    """
    r = max(float(r_lm), 1e-300)
    rho = max(float(rho), 1e-300)
    r_li = max(float(r_li), 1e-300)
    r_lj = max(float(r_lj), 1e-300)
    r_im = max(float(r_im), 1e-300)
    r_jm = max(float(r_jm), 1e-300)

    alpha = float(alpha)
    gamma = float(gamma)

    term_geom = ((r * rho) / (r_li * r_jm)) ** alpha
    f1 = 1.0 - _safe_power_ratio(r_li, r_lj, alpha)
    f2 = 1.0 - _safe_power_ratio(r_jm, r_im, alpha)

    A = 1.0 + (1.0 / (1.0 + gamma)) * term_geom * f1 * f2
    A = max(A, 1e-12)  # guard
    return float(r * (A ** (-1.0 / alpha)))


def renormalized_distance_nn(
    rho: float,
    r_li: float,
    r_jm: float,
    alpha: float,
    gamma: float,
) -> float:
    """
    NN case (paper Eq. (7)/(8) specialized to J^x = 1/r^α):

        r~ = (1+γ)^{1/α} * (r_li * r_jm / rho)
    """
    rho = max(float(rho), 1e-300)
    r_li = max(float(r_li), 1e-300)
    r_jm = max(float(r_jm), 1e-300)
    alpha = float(alpha)
    gamma = float(gamma)
    return float(((1.0 + gamma) ** (1.0 / alpha)) * (r_li * r_jm / rho))


def gamma_renormalize(gamma: float) -> float:
    """Paper Eq. (6):  γ~ = (γ^2 (1+γ))/2"""
    g = float(gamma)
    return 0.5 * g * g * (1.0 + g)


def _recenter_global(x: np.ndarray):
    """Translate all coordinates so the global left boundary is at x=0."""
    if x.size == 0:
        return
    x -= float(np.min(x))


# =============================================================================
# Data model for one sample (with a heap for the closest alive pair)
# =============================================================================

class SampleState:
    """
    One disordered sample with L0 original sites.

    We maintain:
      - x: coordinate for every original site id (for visualization),
      - alive: which original sites remain in the effective chain,
      - prev/next: linked-list pointers for alive sites,
      - heap: min-heap of (gap, left_id, right_id) for adjacent alive gaps.

    The strongest bond is always between the closest pair of sites, which in 1D
    is necessarily adjacent among the alive sites; hence the heap over adjacent
    gaps is sufficient.
    """
    def __init__(self, x0: np.ndarray, gamma0: float):
        self.x = np.array(x0, dtype=float)
        self.alive = np.ones_like(self.x, dtype=bool)
        self.gamma = float(gamma0)

        self.frozen_pairs: list[tuple[int, int]] = []
        self.frozen_steps: list[int] = []
        self.frozen_omegas: list[float] = []

        self.L0 = float(np.max(self.x) - np.min(self.x)) if self.x.size else 0.0

        # alive linked list
        self.prev = np.full(self.x.size, -1, dtype=int)
        self.next = np.full(self.x.size, -1, dtype=int)

        # min-heap of adjacent gaps
        self._heap: list[tuple[float, int, int]] = []

        self.rebuild_links_and_heap()

    def rebuild_links_and_heap(self):
        """Rebuild prev/next pointers and heap from alive/x (O(L))."""
        self.prev.fill(-1)
        self.next.fill(-1)
        self._heap.clear()

        alive_idx = np.flatnonzero(self.alive)
        if alive_idx.size == 0:
            return

        # link alive indices
        for a, b in zip(alive_idx[:-1], alive_idx[1:]):
            self.next[a] = b
            self.prev[b] = a
            gap = float(abs(self.x[b] - self.x[a]))
            heapq.heappush(self._heap, (gap, int(a), int(b)))

    def current_length(self) -> float:
        if self.x.size == 0:
            return 0.0
        return float(np.max(self.x) - np.min(self.x))

    def length_ratio(self) -> float:
        if self.L0 <= 0:
            return 0.0
        return self.current_length() / self.L0

    def alive_indices(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    def strongest_pair(self, alpha: float) -> tuple[int, int, float, float] | None:
        """
        Return (i, j, rho, J) for the current strongest bond (closest alive adjacent pair),
        or None if fewer than 2 alive sites remain.

        Uses lazy cleanup of stale heap entries.
        """
        alpha = float(alpha)
        x = self.x
        alive = self.alive
        prev = self.prev
        nxt = self.next

        while self._heap:
            rho, i, j = self._heap[0]

            if (not alive[i]) or (not alive[j]):
                heapq.heappop(self._heap)
                continue
            if nxt[i] != j or prev[j] != i:
                heapq.heappop(self._heap)
                continue

            rho_now = float(abs(x[j] - x[i]))
            if abs(rho_now - rho) > 1e-10 * max(1.0, rho_now):
                heapq.heappop(self._heap)
                continue

            if rho_now <= 0.0 or not np.isfinite(rho_now):
                heapq.heappop(self._heap)
                continue

            J = rho_now ** (-alpha)
            return int(i), int(j), rho_now, float(J)

        return None

    def push_gap(self, i: int, j: int):
        """Push the current gap between alive neighbors i->j onto the heap."""
        if i < 0 or j < 0:
            return
        if (not self.alive[i]) or (not self.alive[j]):
            return
        gap = float(abs(self.x[j] - self.x[i]))
        if gap <= 0 or not np.isfinite(gap):
            return
        heapq.heappush(self._heap, (gap, int(i), int(j)))


# =============================================================================
# GUI
# =============================================================================

class LongRangeSDRGGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Long-range anisotropic Heisenberg SDRG (geometric)")

        # -------------------- global state --------------------
        self.samples: list[SampleState] | None = None
        self.L = None
        self.M = None

        # RG control
        self.Omega0 = None
        self.current_Omega_high = None
        self.current_Omega_low = None
        self.shell_index = 0

        # mode
        self.mode_var = tk.StringVar(value="animation")
        self.anim_running = False
        self.frame_delay_ms = 250

        # cached run params (set on sample)
        self._run_alpha = None
        self._run_dOmega = None
        self._run_conn = None

        # computation-mode storage
        self.comp_snapshots = None
        self.comp_index = 0

        # visualization window state
        self.sample_window = None
        self.sample_fig = None
        self.sample_ax = None
        self.sample_canvas = None
        self.sample_slider = None
        self.sample_idx_var = tk.IntVar(value=0)
        self.max_arcs_var = tk.IntVar(value=150)

        # hist figure
        self.fig_hist = None
        self.ax_r = None
        self.ax_J = None
        self.canvas_hist = None

        # build layout
        self._build_layout()

    # -------------------------------------------------------------------------
    # UI layout
    # -------------------------------------------------------------------------

    def _build_layout(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left = ttk.Frame(main_frame)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        self.right = ttk.Frame(main_frame)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------------- controls ----------------
        ctrl = ttk.LabelFrame(self.left, text="Model & sampling")
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        ttk.Label(ctrl, text="L (sites):").grid(row=0, column=0, sticky="w")
        self.entry_L = ttk.Entry(ctrl, width=8)
        self.entry_L.insert(0, "32")
        self.entry_L.grid(row=0, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="M (samples):").grid(row=0, column=2, sticky="w")
        self.entry_M = ttk.Entry(ctrl, width=8)
        self.entry_M.insert(0, "32")
        self.entry_M.grid(row=0, column=3, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="α (power):").grid(row=1, column=0, sticky="w")
        self.entry_alpha = ttk.Entry(ctrl, width=8)
        self.entry_alpha.insert(0, "2.0")
        self.entry_alpha.grid(row=1, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="γ (anisotropy):").grid(row=1, column=2, sticky="w")
        self.entry_gamma = ttk.Entry(ctrl, width=8)
        self.entry_gamma.insert(0, "1.0")
        self.entry_gamma.grid(row=1, column=3, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="Connectivity:").grid(row=2, column=0, sticky="w")
        self.combo_conn = ttk.Combobox(
            ctrl,
            values=["Nearest neighbor", "All-to-all"],
            width=16,
            state="readonly"
        )
        self.combo_conn.current(0)
        self.combo_conn.grid(row=2, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="Distance distribution:").grid(row=3, column=0, sticky="w")
        self.combo_dist = ttk.Combobox(
            ctrl,
            values=[
                "Regular lattice (constant)",
                "Uniform(0,1)",
                "Exponential(λ=1)",
                "Abs Gaussian N(0,1)",
                "Log-normal(μ=0,σ=1)",
                "Gamma(k=2,θ=1)",
                "Fermi-Dirac(μ=0,T=1)",
            ],
            width=22,
            state="readonly"
        )
        self.combo_dist.current(2)
        self.combo_dist.grid(row=3, column=1, columnspan=3, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="Distance scale:").grid(row=4, column=0, sticky="w")
        self.entry_scale = ttk.Entry(ctrl, width=8)
        self.entry_scale.insert(0, "1.0")
        self.entry_scale.grid(row=4, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="Min gap:").grid(row=4, column=2, sticky="w")
        self.entry_min_gap = ttk.Entry(ctrl, width=8)
        self.entry_min_gap.insert(0, "0.05")
        self.entry_min_gap.grid(row=4, column=3, padx=3, pady=2, sticky="w")

        ttk.Label(ctrl, text="Seed:").grid(row=5, column=0, sticky="w")
        self.entry_seed = ttk.Entry(ctrl, width=8)
        self.entry_seed.insert(0, "0")
        self.entry_seed.grid(row=5, column=1, padx=3, pady=2, sticky="w")

        # ---------------- RG controls ----------------
        rg = ttk.LabelFrame(self.left, text="RG control")
        rg.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        ttk.Label(rg, text="dΩ:").grid(row=0, column=0, sticky="w")
        self.entry_dOmega = ttk.Entry(rg, width=10)
        self.entry_dOmega.insert(0, "0.5")
        self.entry_dOmega.grid(row=0, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(rg, text="# bins:").grid(row=0, column=2, sticky="w")
        self.entry_bins = ttk.Entry(rg, width=8)
        self.entry_bins.insert(0, "40")
        self.entry_bins.grid(row=0, column=3, padx=3, pady=2, sticky="w")

        ttk.Label(rg, text="Mode:").grid(row=1, column=0, sticky="w")
        rb_anim = ttk.Radiobutton(rg, text="Animation", value="animation",
                                  variable=self.mode_var, command=self.on_mode_change)
        rb_comp = ttk.Radiobutton(rg, text="Computation", value="computation",
                                  variable=self.mode_var, command=self.on_mode_change)
        rb_anim.grid(row=1, column=1, sticky="w")
        rb_comp.grid(row=1, column=2, sticky="w")

        # Buttons row
        btns = ttk.Frame(rg)
        btns.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        for c in range(4):
            btns.columnconfigure(c, weight=1)

        self.btn_sample = ttk.Button(btns, text="Sample", command=self.on_sample)
        self.btn_sample.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_play = ttk.Button(btns, text="Play", command=self.on_play)
        self.btn_play.grid(row=0, column=1, padx=2, sticky="ew")

        self.btn_pause = ttk.Button(btns, text="Pause", command=self.on_pause)
        self.btn_pause.grid(row=0, column=2, padx=2, sticky="ew")

        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop)
        self.btn_stop.grid(row=0, column=3, padx=2, sticky="ew")

        btns2 = ttk.Frame(rg)
        btns2.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(2, 2))
        for c in range(3):
            btns2.columnconfigure(c, weight=1)

        self.btn_open_sample = ttk.Button(btns2, text="Open sample view",
                                          command=self.on_open_sample_window,
                                          state="disabled")
        self.btn_open_sample.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_compute = ttk.Button(btns2, text="Compute (batch)",
                                      command=self.on_compute)
        self.btn_compute.grid(row=0, column=1, padx=2, sticky="ew")

        self.btn_save = ttk.Button(btns2, text="Save plots",
                                   command=self.on_save_plots)
        self.btn_save.grid(row=0, column=2, padx=2, sticky="ew")

        # Batch settings
        ttk.Label(rg, text="# shells (batch):").grid(row=4, column=0, sticky="w")
        self.entry_shells = ttk.Entry(rg, width=10)
        self.entry_shells.insert(0, "80")
        self.entry_shells.grid(row=4, column=1, padx=3, pady=2, sticky="w")

        ttk.Label(rg, text="Batch progress:").grid(row=5, column=0, columnspan=4, sticky="w")
        self.progress_shells = ttk.Progressbar(rg, orient="horizontal", length=240, mode="determinate")
        self.progress_shells.grid(row=6, column=0, columnspan=4, sticky="ew", padx=2, pady=(2, 4))

        # Slider for computation mode
        ttk.Label(rg, text="View state (slider):").grid(row=7, column=0, columnspan=4, sticky="w")
        self.state_slider = tk.Scale(rg, from_=0, to=0, orient=tk.HORIZONTAL,
                                     resolution=1, showvalue=False,
                                     command=self.on_state_slider)
        self.state_slider.configure(state="disabled")
        self.state_slider.grid(row=8, column=0, columnspan=4, sticky="ew", padx=2, pady=(2, 2))

        self.info_label = ttk.Label(self.left, text="No data yet.")
        self.info_label.pack(side=tk.TOP, anchor="w", pady=(4, 2))

        # ---------------- right: histograms ----------------
        self.fig_hist = plt.Figure(figsize=(8, 7))
        gs = self.fig_hist.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
        self.ax_r = self.fig_hist.add_subplot(gs[0, 0])  # real space FIRST
        self.ax_J = self.fig_hist.add_subplot(gs[1, 0])  # J-space SECOND

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.right)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_empty()
        self.on_mode_change()

    # -------------------------------------------------------------------------
    # UI helpers
    # -------------------------------------------------------------------------

    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "animation":
            self.entry_shells.configure(state="disabled")
            self.btn_compute.configure(state="disabled")
            self.state_slider.configure(state="disabled")
        else:
            self.entry_shells.configure(state="normal")
            self.btn_compute.configure(state="normal")

    def get_params(self):
        try:
            L = int(self.entry_L.get())
            M = int(self.entry_M.get())
            if L < 2 or M <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Invalid L or M. Need L>=2 and M>0.")

        try:
            alpha = float(self.entry_alpha.get())
            if alpha <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Invalid α. Need α>0.")

        try:
            gamma0 = float(self.entry_gamma.get())
            if gamma0 <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Invalid γ. Need γ>0.")

        try:
            scale = float(self.entry_scale.get())
            if scale <= 0:
                raise ValueError
        except ValueError:
            scale = 1.0

        try:
            min_gap = float(self.entry_min_gap.get())
            if min_gap <= 0:
                raise ValueError
        except ValueError:
            min_gap = 1e-3

        try:
            seed = int(self.entry_seed.get())
        except ValueError:
            seed = 0

        try:
            dOmega = float(self.entry_dOmega.get())
            if dOmega <= 0:
                raise ValueError
        except ValueError:
            dOmega = 0.5

        try:
            nbins = int(self.entry_bins.get())
            if nbins < 1:
                raise ValueError
        except ValueError:
            nbins = 40

        conn = self.combo_conn.get()
        dist = self.combo_dist.get()

        return L, M, alpha, gamma0, dist, scale, min_gap, seed, dOmega, nbins, conn

    def _draw_empty(self):
        self.ax_r.clear()
        self.ax_J.clear()
        self.ax_r.set_title("Real-space distance distribution (no data)")
        self.ax_r.set_xlabel("log r")
        self.ax_r.set_ylabel("probability density")
        self.ax_J.set_title("J-space distribution (no data)")
        self.ax_J.set_xlabel("J")
        self.ax_J.set_ylabel("probability density")
        self.fig_hist.tight_layout()
        self.canvas_hist.draw_idle()

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def on_sample(self):
        try:
            L, M, alpha, gamma0, dist, scale, min_gap, seed, dOmega, nbins, conn = self.get_params()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        rng = np.random.default_rng(seed)
        pos = sample_positions_from_gaps(L, M, dist, rng, scale=scale, min_gap=min_gap)
        self.samples = [SampleState(pos[s, :], gamma0) for s in range(M)]
        self.L = L
        self.M = M

        # Cache run parameters (avoid re-parsing GUI fields in inner loops)
        self._run_alpha = float(alpha)
        self._run_dOmega = float(dOmega)
        self._run_conn = str(conn)

        # Initial global Ω0 from the strongest adjacent bond across the ensemble
        Omega0 = 0.0
        for s in self.samples:
            sp = s.strongest_pair(alpha)
            if sp is None:
                continue
            _, _, _, J = sp
            if np.isfinite(J) and J > Omega0:
                Omega0 = float(J)

        self.Omega0 = Omega0
        self.shell_index = 0
        self.current_Omega_high = Omega0
        self.current_Omega_low = max(0.0, Omega0 - dOmega)

        self.anim_running = False
        self.comp_snapshots = None
        self.comp_index = 0
        self.state_slider.configure(state="disabled", from_=0, to=0)
        self.state_slider.set(0)

        self.btn_open_sample.configure(state="normal")
        self.info_label.config(text=f"Sampled M={M}, L={L}. Ω0={Omega0:.4g}")

        self.update_histograms()

    # -------------------------------------------------------------------------
    # Plot saving
    # -------------------------------------------------------------------------

    def on_save_plots(self):
        if self.fig_hist is None:
            messagebox.showinfo("Save plots", "No histogram figure to save.")
            return

        default_name = "sdrg_histograms.png"
        filetypes = [
            ("PNG image", "*.png"),
            ("PDF file", "*.pdf"),
            ("SVG file", "*.svg"),
            ("All files", "*.*"),
        ]

        filename = filedialog.asksaveasfilename(
            title="Save histogram figure",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=filetypes,
        )
        if not filename:
            return

        saved = []
        try:
            self.fig_hist.savefig(filename, dpi=300, bbox_inches="tight")
            saved.append(filename)
        except Exception as e:
            messagebox.showerror("Save plots", f"Failed to save histogram figure:\n{e}")
            return

        if self.sample_fig is not None:
            import os
            base, ext = os.path.splitext(filename)
            fname2 = base + "_sample" + (ext if ext else ".png")
            try:
                self.sample_fig.savefig(fname2, dpi=300, bbox_inches="tight")
                saved.append(fname2)
            except Exception as e:
                messagebox.showerror("Save plots", f"Histograms saved.\nFailed to save sample view:\n{e}")
                return

        messagebox.showinfo("Save plots", "Saved:\n" + "\n".join(saved))

    # -------------------------------------------------------------------------
    # Histogram computation (adjacent distances only; paper-consistent P(r,Ω))
    # -------------------------------------------------------------------------

    def _collect_adjacent_distances(self) -> np.ndarray:
        """
        Collect *adjacent-alive* distances r for histogramming.

        In the paper, P(r,Ω) is defined for distances between adjacent spins in
        the effective chain; non-adjacent distances should not be counted in P.
        """
        if self.samples is None:
            return np.zeros((0,), dtype=float)

        dists = []
        for s in self.samples:
            alive_idx = s.alive_indices()
            if alive_idx.size < 2:
                continue
            xs = s.x[alive_idx]
            gaps = xs[1:] - xs[:-1]
            gaps = gaps[gaps > 0]
            if gaps.size:
                dists.append(gaps)

        if not dists:
            return np.zeros((0,), dtype=float)
        return np.concatenate(dists)

    def update_histograms(self):
        self.ax_r.clear()
        self.ax_J.clear()

        if self.samples is None:
            self._draw_empty()
            return

        try:
            _, _, alpha, _, _, _, _, _, _, nbins, _ = self.get_params()
        except ValueError:
            return

        r_vals = self._collect_adjacent_distances()
        r_vals = r_vals[np.isfinite(r_vals) & (r_vals > 0)]
        if r_vals.size == 0:
            self.ax_r.set_title("Real-space distance distribution (no data)")
            self.ax_J.set_title("J-space distribution (no data)")
            self.fig_hist.tight_layout()
            self.canvas_hist.draw_idle()
            return

        # --- Real-space histogram in log r ---
        log_r = np.log(r_vals)
        lo = float(np.min(log_r))
        hi = float(np.max(log_r))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0

        edges_logr = np.linspace(lo, hi, nbins + 1)
        self.ax_r.hist(log_r, bins=edges_logr, density=True, alpha=0.75, edgecolor="black")
        self.ax_r.set_xlabel("log r")
        self.ax_r.set_ylabel("probability density")
        self.ax_r.set_title("Real-space adjacent-distance distribution")

        # --- J histogram via J = 1/r^α ---
        J_vals = r_vals ** (-float(alpha))
        J_vals = J_vals[np.isfinite(J_vals)]
        if J_vals.size == 0:
            self.ax_J.set_title("J-space distribution (no finite J)")
        else:
            J_max = float(np.max(J_vals))
            edges_J = np.linspace(0.0, J_max, nbins + 1)
            self.ax_J.hist(J_vals, bins=edges_J, density=True, alpha=0.75, edgecolor="black")
            self.ax_J.set_xlabel("J")
            self.ax_J.set_ylabel("probability density")
            self.ax_J.set_title("J-space distribution (J = r^{-α})")

        # Info box
        Omega_hi = self.current_Omega_high if self.current_Omega_high is not None else np.nan
        Omega_lo = self.current_Omega_low if self.current_Omega_low is not None else np.nan
        self.ax_r.text(
            0.98, 0.95,
            f"shell n={self.shell_index}\nΩ∈[{Omega_lo:.3g}, {Omega_hi:.3g}]\nγ≈{self._ensemble_gamma_mean():.3g}",
            transform=self.ax_r.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
        )

        self.fig_hist.tight_layout()
        self.canvas_hist.draw_idle()

    def _ensemble_gamma_mean(self) -> float:
        if self.samples is None or len(self.samples) == 0:
            return 0.0
        return float(np.mean([s.gamma for s in self.samples]))

    # -------------------------------------------------------------------------
    # Sample visualization window
    # -------------------------------------------------------------------------

    def on_open_sample_window(self):
        if self.samples is None:
            return
        if self.sample_window is not None:
            try:
                self.sample_window.lift()
            except Exception:
                pass
            return

        top = tk.Toplevel(self.master)
        top.title("Real-space sample view (sites + arcs)")

        self.sample_window = top
        self.sample_fig = plt.Figure(figsize=(8, 3.6))
        self.sample_ax = self.sample_fig.add_subplot(111)
        self.sample_canvas = FigureCanvasTkAgg(self.sample_fig, master=top)
        self.sample_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(controls, text="Sample index:").pack(side=tk.LEFT)
        self.sample_slider = ttk.Scale(
            controls, from_=0, to=max(0, (self.M or 1) - 1),
            orient="horizontal", command=self.on_sample_slider
        )
        self.sample_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.sample_idx_var.set(0)

        ttk.Label(controls, text="Max arcs (all-to-all):").pack(side=tk.LEFT, padx=(6, 2))
        entry_arcs = ttk.Entry(controls, width=6, textvariable=self.max_arcs_var)
        entry_arcs.pack(side=tk.LEFT)
        btn_redraw = ttk.Button(controls, text="Redraw", command=self.update_sample_view)
        btn_redraw.pack(side=tk.LEFT, padx=6)

        top.protocol("WM_DELETE_WINDOW", self.on_close_sample_window)
        self.update_sample_view()

    def on_close_sample_window(self):
        try:
            if self.sample_window is not None:
                self.sample_window.destroy()
        except Exception:
            pass
        self.sample_window = None
        self.sample_fig = None
        self.sample_ax = None
        self.sample_canvas = None
        self.sample_slider = None

    def on_sample_slider(self, event=None):
        self.update_sample_view()

    def update_sample_view(self):
        if self.sample_ax is None or self.sample_canvas is None or self.samples is None:
            return

        try:
            s_idx = int(round(float(self.sample_slider.get()))) if self.sample_slider is not None else 0
        except Exception:
            s_idx = 0
        s_idx = max(0, min(s_idx, len(self.samples) - 1))
        self.sample_idx_var.set(s_idx)

        sample = self.samples[s_idx]

        ax = self.sample_ax
        ax.clear()

        x_all = sample.x.copy()
        xmin = float(np.min(x_all))
        xmax = float(np.max(x_all))
        span = max(xmax - xmin, 1e-12)
        x_norm = (x_all - xmin) / span

        # Baseline
        ax.plot([0.0, 1.0], [0.0, 0.0], linewidth=2)

        # Sites: alive black, frozen red
        alive = sample.alive
        frozen = ~alive

        ax.scatter(x_norm[alive], np.zeros(np.sum(alive)), s=30, color="black", zorder=4)
        if np.any(frozen):
            ax.scatter(x_norm[frozen], np.zeros(np.sum(frozen)), s=30, color="red", zorder=3)

        # Labels for small systems
        L0 = x_all.size
        if L0 <= 48:
            for i in range(L0):
                ax.text(x_norm[i], 0.02, str(i), ha="center", va="bottom", fontsize=8)

        conn = self.combo_conn.get()
        alpha = float(self.entry_alpha.get())

        # Active couplings as arcs ABOVE the line
        if conn == "Nearest neighbor":
            alive_idx = sample.alive_indices()
            for a, b in zip(alive_idx[:-1], alive_idx[1:]):
                self._draw_arc(ax, x_norm[a], x_norm[b], height=0.25, above=True, color="C0", alpha=0.7)
        else:
            # All-to-all: draw only strongest bonds up to a cap
            try:
                kmax = int(self.max_arcs_var.get())
                kmax = max(0, kmax)
            except Exception:
                kmax = 150

            alive_idx = sample.alive_indices()
            xs = sample.x[alive_idx]
            n = xs.size
            if n >= 2 and kmax > 0:
                diffs = np.abs(xs[:, None] - xs[None, :])
                iu = np.triu_indices(n, k=1)
                d = diffs[iu]
                J = d ** (-alpha)
                if J.size > 0:
                    k = min(kmax, J.size)
                    top_idx = np.argpartition(-J, kth=k - 1)[:k]
                    top_idx = top_idx[np.argsort(-J[top_idx])]
                    ii = iu[0][top_idx]
                    jj = iu[1][top_idx]
                    for a0, b0 in zip(ii, jj):
                        i = int(alive_idx[a0])
                        j = int(alive_idx[b0])
                        self._draw_arc(ax, x_norm[i], x_norm[j], height=0.25, above=True, color="C0", alpha=0.25)

        # Frozen singlets as arcs BELOW the line
        if sample.frozen_pairs:
            for (i, j) in sample.frozen_pairs:
                self._draw_arc(ax, x_norm[i], x_norm[j], height=0.20, above=False, color="red", alpha=0.8)

        ax.text(
            0.02, 0.92,
            f"sample {s_idx+1}/{len(self.samples)}\n"
            f"L/L0 = {sample.length_ratio():.3f}\n"
            f"alive = {int(np.sum(sample.alive))}/{L0}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.45, 0.45)
        ax.axis("off")
        self.sample_fig.tight_layout()
        self.sample_canvas.draw_idle()

    @staticmethod
    def _draw_arc(ax, x1: float, x2: float, height: float, above: bool, color: str, alpha: float):
        x1, x2 = float(x1), float(x2)
        if x2 < x1:
            x1, x2 = x2, x1
        xm = 0.5 * (x1 + x2)
        h = float(height) * (1.0 if above else -1.0)
        t = np.linspace(0.0, 1.0, 60)
        x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * xm + t ** 2 * x2
        y = 2 * (1 - t) * t * h
        ax.plot(x, y, color=color, alpha=alpha, linewidth=1.2, zorder=2)

    # -------------------------------------------------------------------------
    # Animation mode
    # -------------------------------------------------------------------------

    def on_play(self):
        if self.mode_var.get() != "animation":
            messagebox.showinfo("Mode", "Play is only available in animation mode.")
            return
        if self.samples is None:
            messagebox.showinfo("Data", "Sample an ensemble first.")
            return
        if not self.anim_running:
            self.anim_running = True
            self.master.after(self.frame_delay_ms, self._animate_tick)

    def on_pause(self):
        self.anim_running = False

    def on_stop(self):
        self.anim_running = False
        self.shell_index = 0
        self.comp_snapshots = None
        self.comp_index = 0
        self.update_histograms()
        self.update_sample_view()

    def _animate_tick(self):
        if not self.anim_running or self.samples is None:
            return

        if self.Omega0 is None or self.Omega0 <= 0:
            self.anim_running = False
            return

        alpha = self._run_alpha
        dOmega = self._run_dOmega
        conn = self._run_conn
        if alpha is None or dOmega is None or conn is None:
            try:
                _, _, alpha, _, _, _, _, _, dOmega, _, conn = self.get_params()
            except ValueError:
                self.anim_running = False
                return

        if all((s.alive.sum() < 2) for s in self.samples):
            self.anim_running = False
            self.update_histograms()
            self.update_sample_view()
            return

        Omega_high = self.Omega0 - self.shell_index * dOmega
        Omega_low = Omega_high - dOmega
        if Omega_high <= 0:
            self.anim_running = False
            return
        if Omega_low < 0:
            Omega_low = 0.0

        self.current_Omega_high = float(Omega_high)
        self.current_Omega_low = float(Omega_low)

        best = None  # (J, sample_index, i, j, rho)
        for s_idx, s in enumerate(self.samples):
            sp = s.strongest_pair(alpha)
            if sp is None:
                continue
            i, j, rho, J = sp
            if J < Omega_low:
                continue
            if (best is None) or (J > best[0]):
                best = (J, s_idx, i, j, rho)

        if best is None:
            self.shell_index += 1
            self.update_histograms()
            self.update_sample_view()
            self.master.after(self.frame_delay_ms, self._animate_tick)
            return

        _, s_idx, i, j, rho = best
        self._decimate_pair(self.samples[s_idx], i, j, rho, alpha, conn,
                            Omega_high=Omega_high, Omega_low=Omega_low)

        self.update_histograms()
        self.update_sample_view()
        self.master.after(self.frame_delay_ms, self._animate_tick)

    # -------------------------------------------------------------------------
    # Computation mode (batch)
    # -------------------------------------------------------------------------

    def on_compute(self):
        if self.mode_var.get() != "computation":
            messagebox.showinfo("Mode", "Batch compute is only available in computation mode.")
            return
        if self.samples is None:
            messagebox.showinfo("Data", "Sample an ensemble first.")
            return

        try:
            _, _, alpha, _, _, _, _, _, dOmega, _, conn = self.get_params()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        try:
            max_shells = int(self.entry_shells.get())
            if max_shells <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error", "Invalid # shells (batch).")
            return

        base_samples: list[SampleState] = []
        for s in self.samples:
            s2 = SampleState(s.x.copy(), s.gamma)
            s2.alive = s.alive.copy()
            s2.frozen_pairs = list(s.frozen_pairs)
            s2.frozen_steps = list(s.frozen_steps)
            s2.frozen_omegas = list(s.frozen_omegas)
            s2.L0 = s.L0
            s2.rebuild_links_and_heap()
            base_samples.append(s2)

        snapshots = []
        self.progress_shells.configure(mode="determinate", maximum=max_shells, value=0)

        snapshots.append(self._snapshot_dict(base_samples, shell=0,
                                             Omega_high=self.Omega0,
                                             Omega_low=max(0.0, self.Omega0 - dOmega)))

        for n in range(max_shells):
            if all((s.alive.sum() < 2) for s in base_samples):
                break

            Omega_high = self.Omega0 - n * dOmega
            Omega_low = Omega_high - dOmega
            if Omega_high <= 0:
                break
            if Omega_low < 0:
                Omega_low = 0.0

            for s in base_samples:
                while True:
                    sp = s.strongest_pair(alpha)
                    if sp is None:
                        break
                    i, j, rho, J = sp
                    if J < Omega_low:
                        break
                    self._decimate_pair(s, i, j, rho, alpha, conn,
                                        Omega_high=Omega_high, Omega_low=Omega_low,
                                        step_override=(n + 1))

            snapshots.append(self._snapshot_dict(base_samples, shell=n + 1,
                                                 Omega_high=Omega_high, Omega_low=Omega_low))

            self.progress_shells["value"] = n + 1
            self.master.update_idletasks()

        self.comp_snapshots = snapshots
        self.comp_index = 0

        max_idx = len(snapshots) - 1
        self.state_slider.configure(state="normal", from_=0, to=max_idx)
        self.state_slider.set(0)

        self._load_snapshot(0)

        self.info_label.config(text=f"Computed {len(snapshots)} snapshots.")
        self.update_histograms()
        self.update_sample_view()

    def _snapshot_dict(self, samples: list[SampleState], shell: int, Omega_high: float, Omega_low: float) -> dict:
        pos = np.stack([s.x.copy() for s in samples], axis=0)
        alive = np.stack([s.alive.copy() for s in samples], axis=0)
        gammas = np.array([s.gamma for s in samples], dtype=float)
        return {
            "shell": int(shell),
            "Omega_high": float(Omega_high if Omega_high is not None else np.nan),
            "Omega_low": float(Omega_low if Omega_low is not None else np.nan),
            "pos": pos,
            "alive": alive,
            "gammas": gammas,
            "frozen_pairs": [list(s.frozen_pairs) for s in samples],
            "frozen_steps": [list(s.frozen_steps) for s in samples],
            "frozen_omegas": [list(s.frozen_omegas) for s in samples],
            "L0": np.array([s.L0 for s in samples], dtype=float),
        }

    def _load_snapshot(self, idx: int):
        if self.comp_snapshots is None or self.samples is None:
            return
        idx = int(idx)
        idx = max(0, min(idx, len(self.comp_snapshots) - 1))
        snap = self.comp_snapshots[idx]

        for s_idx, s in enumerate(self.samples):
            s.x = snap["pos"][s_idx].copy()
            s.alive = snap["alive"][s_idx].copy()
            s.gamma = float(snap["gammas"][s_idx])
            s.frozen_pairs = list(snap["frozen_pairs"][s_idx])
            s.frozen_steps = list(snap["frozen_steps"][s_idx])
            s.frozen_omegas = list(snap["frozen_omegas"][s_idx])
            s.L0 = float(snap["L0"][s_idx])
            s.rebuild_links_and_heap()

        self.shell_index = int(snap["shell"])
        self.current_Omega_high = float(snap["Omega_high"])
        self.current_Omega_low = float(snap["Omega_low"])

    def on_state_slider(self, value):
        if self.comp_snapshots is None:
            return
        try:
            idx = int(float(value))
        except Exception:
            return
        self.comp_index = idx
        self._load_snapshot(idx)
        self.info_label.config(text=f"Viewing snapshot {idx}/{len(self.comp_snapshots)-1}")
        self.update_histograms()
        self.update_sample_view()

    # -------------------------------------------------------------------------
    # Decimation primitive
    # -------------------------------------------------------------------------

    def _decimate_pair(self,
                      s: SampleState,
                      i: int, j: int,
                      rho: float,
                      alpha: float,
                      conn: str,
                      Omega_high: float,
                      Omega_low: float,
                      step_override: int | None = None):
        """
        Decimate an adjacent-alive pair (i,j) at distance rho.

        Geometric SDRG approximation:
          - remove i,j,
          - renormalize the distance between the new neighbors l and m
            using NN or all-to-all rule,
          - implement renormalization by shifting the "right block" so that
            r(l,m) becomes r_tilde.

        Enforce r_tilde >= rho*(1+eps) to prevent generating bonds larger than
        the eliminated bond.
        """
        i = int(i)
        j = int(j)
        alpha = float(alpha)

        if (not s.alive[i]) or (not s.alive[j]):
            return

        if s.next[i] != j:
            if s.next[j] == i:
                i, j = j, i
            else:
                return

        s.frozen_pairs.append((min(i, j), max(i, j)))
        s.frozen_steps.append(int(step_override if step_override is not None else self.shell_index))
        s.frozen_omegas.append(float(max(rho, 1e-300) ** (-alpha)))

        l = int(s.prev[i])
        m = int(s.next[j])

        s.alive[i] = False
        s.alive[j] = False
        s.prev[i] = s.next[i] = -1
        s.prev[j] = s.next[j] = -1

        if l != -1:
            s.next[l] = m
        if m != -1:
            s.prev[m] = l

        if l == -1 or m == -1:
            _recenter_global(s.x)
            s.gamma = gamma_renormalize(s.gamma)
            return

        x = s.x
        r_li = float(abs(x[i] - x[l]))
        r_jm = float(abs(x[m] - x[j]))
        r_lm = float(abs(x[m] - x[l]))
        rho = float(max(rho, 1e-12))
        r_li = max(r_li, 1e-12)
        r_jm = max(r_jm, 1e-12)
        r_lm = max(r_lm, 1e-12)

        gamma = float(s.gamma)

        if conn == "Nearest neighbor":
            r_tilde = renormalized_distance_nn(rho=rho, r_li=r_li, r_jm=r_jm, alpha=alpha, gamma=gamma)
        else:
            r_lj = float(abs(x[j] - x[l]))
            r_im = float(abs(x[m] - x[i]))
            r_lj = max(r_lj, 1e-12)
            r_im = max(r_im, 1e-12)
            r_tilde = renormalized_distance_alltoall(
                r_lm=r_lm, rho=rho,
                r_li=r_li, r_lj=r_lj, r_im=r_im, r_jm=r_jm,
                alpha=alpha, gamma=gamma
            )

        eps = 1e-9
        r_tilde = max(float(r_tilde), float(rho * (1.0 + eps)))

        delta = float(r_lm - r_tilde)
        if abs(delta) > 0.0:
            x[m:] = x[m:] - delta

        _recenter_global(x)

        s.push_gap(l, m)
        s.gamma = gamma_renormalize(s.gamma)


# =============================================================================
# Main
# =============================================================================

def main():
    root = tk.Tk()
    app = LongRangeSDRGGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
