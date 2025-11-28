import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Patch
from scipy.signal import fftconvolve


# ============================ Sampling helpers ============================

def sample_fermi_dirac(shape, rng: np.random.Generator):
    """
    Sample positive couplings J from a Fermi-Dirac–like distribution on (0, ∞):
        target (unnormalized) t(J) ∝ 1 / (exp(J) + 1),  J >= 0

    We use rejection sampling with envelope g(J) = exp(-J) (an Exp(1) distribution).
    For J > 0 we have t(J) <= g(J), so we can choose M = 1 and:
        accept with probability t(J)/g(J) = [1/(e^J+1)] / e^{-J} = e^J/(e^J+1).

    The resulting normalized pdf is ∝ 1/(e^J+1). This gives a Fermi-Dirac–shaped
    distribution (μ=0, T=1) restricted to positive J.
    """
    size = int(np.prod(shape))
    out = np.empty(size, dtype=float)
    i = 0
    while i < size:
        x = rng.exponential(scale=1.0)  # candidate ~ Exp(1)
        u = rng.random()
        p_accept = np.exp(x) / (np.exp(x) + 1.0)
        if u < p_accept:
            out[i] = x
            i += 1
    return out.reshape(shape)


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
        J = rng.gamma(shape=2.0, scale=1.0, size=(M, L))
    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        J = sample_fermi_dirac((M, L), rng)
    else:
        J = rng.random((M, L))
    return J


# ====================== Master equation helpers (ζ-space) =======================

def shift_left(P, dx, dGamma):
    """
    Exact left shift by dGamma on a uniform grid of spacing dx:
        P_shift(ζ) = P(ζ + dGamma),
    with P=0 outside the original domain (ζ < 0 or ζ > ζ_max).
    """
    N = len(P)
    x = np.arange(N) * dx  # ζ grid (0, dx, 2dx, ...)
    return np.interp(x + dGamma, x, P, left=0.0, right=0.0)


def evolve(P, x, dGamma):
    """
    Friend's evolution code, adapted with np.trapz instead of np.trapezoid.

    P:  current pdf on ζ-grid x
    x:  ζ-grid (uniform)
    dGamma: small RG "time" step in Γ
    """
    dx = x[1] - x[0]

    # -----------------------------
    # Exact left shift by dGamma
    # -----------------------------
    P_shift = shift_left(P, dx, dGamma)

    # -----------------------------
    # Convolution term
    # -----------------------------
    P_conv = fftconvolve(P, P)[:len(x)] * dx

    # Normalize P_conv
    integral_conv = np.trapz(P_conv, x)
    if integral_conv > 0:
        P_conv /= integral_conv

    # -----------------------------
    # Integral of P from 0 to dGamma
    # -----------------------------
    mask = x <= dGamma
    if np.any(mask):
        A0 = np.trapz(P[mask], x[mask])
    else:
        A0 = 0.0

    # -----------------------------
    # Updated pdf
    # -----------------------------
    P_new = P_shift + A0 * P_conv
    P_new = np.maximum(P_new, 0.0)

    norm = np.trapz(P_new, x)
    if norm > 0:
        P_new /= norm

    return P_new


# ====================== Single-chain SDRG (non-rotating) =======================

def decimate_chain_nonrotating(J_s: list, b: int):
    """
    Decimate bond index b in a periodic chain represented by list J_s (length n>=3).

    Ring bonds indexed 0..n-1 in order.
      left   = (b - 1) mod n
      center = b
      right  = (b + 1) mod n

    Ma–Dasgupta (for AFM Heisenberg):
      J_eff = J_left * J_right / (2 * J_center)

    We:
      - replace J_left by J_eff
      - remove bonds at indices center and right
      => chain length n-2, order otherwise unchanged.

    Returns:
      J_new: new list of couplings (len n-2)
      new_index: index in J_new where J_eff resides
    """
    n = len(J_s)
    assert n >= 3
    b = int(b)
    left = (b - 1) % n
    center = b
    right = (b + 1) % n

    J_L = J_s[left]
    J_c = J_s[center]
    J_R = J_s[right]

    if J_c > 0:
        J_eff = J_L * J_R / (2.0 * J_c)
    else:
        J_eff = 0.0

    killed = sorted([center, right])
    J_new = []
    new_index = None

    for i in range(n):
        if i in killed:
            continue
        if i == left:
            J_new.append(J_eff)
            new_index = len(J_new) - 1
        else:
            J_new.append(J_s[i])

    return J_new, new_index


# ================================ GUI Class ===================================

class SDRGEnsembleGUI:
    def __init__(self, master):
        self.master = master
        master.title("SDRG ensemble — Ω-shell RG for ρ_Ω(J), P_Γ(ζ) + optional 3D view")

        # ---------- State ----------
        self.J_init = None          # initial couplings: shape (M,L)
        self.chains = None          # list of lists, current per sample
        self.J_current = None       # current coupling matrix: (M, n_bonds)
        self.step_count = 0         # shell index / state index

        self.Omega0 = None          # global Ω0 from initial ensemble
        self.current_Omega = None   # Ω for current snapshot (histograms + 3D label)
        self.dOmega_cached = 0.001  # last parsed dΩ

        # Animation state (Ω-shell)
        self.anim_running = False
        self.frame_delay_ms = 400   # ms between animation frames
        self.shell_index_anim = 0   # which shell we are currently integrating
        self.anim_shell_phase = 0   # 0: highlight bond+neighbors; 1: show inserted bond
        # current_decimation = ("normal" or "final", sample_index, center_index)
        self.current_decimation = None

        # MODE: "animation" or "computation"
        self.mode_var = tk.StringVar(value="animation")

        # Computation mode storage (Ω-shell SDRG)
        self.comp_J_steps = None    # list of J matrices for each shell step
        self.comp_Omegas = None     # list of Ω_n values
        self.comp_index = 0         # slider-selected shell / state index

        # 3D highlighting (used in animation)
        self.last_decimated = []    # list of (sample, bond_index) for bonds being decimated
        self.neighbor_bonds = []    # list of (sample, bond_index) for neighbors
        self.inserted_bonds = []    # list of (sample, bond_index) for newly inserted bonds

        # Height scale for vertical bars (3D view)
        self.h_max = 1.0
        self.h_dir = np.array([0.0, 1.0])  # vertical direction (upward) in plotting coords

        # ---- Base geometry for 3D view ----
        self.base_origin = np.array([0.0, 0.0])
        L1 = 1.0   # length of sample axis
        L2 = 1.0   # length of bond axis
        theta = np.deg2rad(30.0)

        self.s_dir = np.array([L1, 0.0])                      # horizontal sample axis
        self.b_dir = np.array([L2 * np.cos(theta),
                               L2 * np.sin(theta)])           # 30° up for bond axis

        # "3D" window (2D pseudo-3D with bars)
        self.top3d = None
        self.fig3d = None
        self.ax3 = None
        self.canvas3d = None

        # Whether to show the 3D plot (toggle; default = False)
        self.show_3d = tk.BooleanVar(value=False)

        # --------- Master equation in ζ-space ----------
        self.zeta_grid = None      # 1D ζ-grid
        self.P0_zeta = None        # initial P(ζ, Γ=0)
        self.P_zeta = None         # current P(ζ, Γ)
        self.Gamma_current = 0.0   # current Γ
        # batch history
        self.comp_Pzeta_steps = None
        self.comp_Gammas = None

        # ======================= Layout: left/right =========================
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ----------------------- Controls (left, top) -----------------------
        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Row 0: L, M
        ttk.Label(control_frame, text="Chain length L:").grid(row=0, column=0, sticky="w")
        self.entry_L = ttk.Entry(control_frame, width=8)
        self.entry_L.insert(0, "16")
        self.entry_L.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="# samples M:").grid(row=0, column=2, sticky="w")
        self.entry_M = ttk.Entry(control_frame, width=8)
        self.entry_M.insert(0, "32")
        self.entry_M.grid(row=0, column=3, padx=4, pady=2)

        # Row 1: distribution, seed
        ttk.Label(control_frame, text="Distribution:").grid(row=1, column=0, sticky="w")
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
        self.combo_dist.grid(row=1, column=1, columnspan=3, padx=4, pady=2, sticky="w")

        ttk.Label(control_frame, text="Seed:").grid(row=1, column=4, sticky="w")
        self.entry_seed = ttk.Entry(control_frame, width=8)
        self.entry_seed.insert(0, "0")
        self.entry_seed.grid(row=1, column=5, padx=4, pady=2)

        # Row 2: dΩ and # J bins
        ttk.Label(control_frame, text="dΩ:").grid(row=2, column=0, sticky="w")
        self.entry_dOmega = ttk.Entry(control_frame, width=8)
        self.entry_dOmega.insert(0, "0.001")
        self.entry_dOmega.grid(row=2, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="# J bins:").grid(row=2, column=2, sticky="w")
        self.entry_bins = ttk.Entry(control_frame, width=6)
        self.entry_bins.insert(0, "40")
        self.entry_bins.grid(row=2, column=3, padx=4, pady=2)

        # Row 3: 3D toggle
        self.check_3d = ttk.Checkbutton(
            control_frame,
            text="Show 3D plot",
            variable=self.show_3d,
            command=self.on_toggle_3d
        )
        self.check_3d.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 2))

        # Row 4: mode selection
        ttk.Label(control_frame, text="Mode:").grid(row=4, column=0, sticky="w")
        rb_anim = ttk.Radiobutton(
            control_frame, text="Animation", value="animation",
            variable=self.mode_var, command=self.on_mode_change
        )
        rb_comp = ttk.Radiobutton(
            control_frame, text="Computation", value="computation",
            variable=self.mode_var, command=self.on_mode_change
        )
        rb_anim.grid(row=4, column=1, sticky="w", padx=2)
        rb_comp.grid(row=4, column=2, sticky="w", padx=2)

        # Row 5: animation buttons
        self.button_sample = ttk.Button(control_frame, text="Sample", command=self.on_sample_ensemble)
        self.button_sample.grid(row=5, column=0, padx=2, pady=4, sticky="ew")

        self.button_play = ttk.Button(control_frame, text="Play", command=self.on_play)
        self.button_play.grid(row=5, column=1, padx=2, pady=4, sticky="ew")

        self.button_pause = ttk.Button(control_frame, text="Pause", command=self.on_pause)
        self.button_pause.grid(row=5, column=2, padx=2, pady=4, sticky="ew")

        self.button_stop = ttk.Button(control_frame, text="Stop", command=self.on_stop)
        self.button_stop.grid(row=5, column=3, padx=2, pady=4, sticky="ew")

        # Current shell / state label
        self.step_label = ttk.Label(self.left_frame, text="Current shell (n): 0")
        self.step_label.pack(side=tk.TOP, anchor="w", pady=(4, 4))

        # -------------------- Computation mode controls ---------------------
        comp_frame = ttk.Frame(self.left_frame)
        comp_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Row 0: number of shells
        ttk.Label(comp_frame, text="# shells (batch):").grid(row=0, column=0, sticky="w")
        self.entry_comp_steps = ttk.Entry(comp_frame, width=8)
        self.entry_comp_steps.insert(0, "50")
        self.entry_comp_steps.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        self.button_compute = ttk.Button(comp_frame, text="Compute (batch)", command=self.on_compute)
        self.button_compute.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

        # Row 1–2: shells progress bar
        ttk.Label(comp_frame, text="Batch progress (shells):").grid(row=1, column=0, columnspan=3, sticky="w")
        self.progressbar_shells = ttk.Progressbar(
            comp_frame, orient="horizontal", length=240, mode="determinate"
        )
        self.progressbar_shells.grid(row=2, column=0, columnspan=3, padx=2, pady=(2, 4), sticky="ew")

        # Row 3–4: chains-in-shell progress bar
        ttk.Label(comp_frame, text="Chains done in this shell:").grid(row=3, column=0, columnspan=3, sticky="w")
        self.progressbar_chains = ttk.Progressbar(
            comp_frame, orient="horizontal", length=240, mode="determinate"
        )
        self.progressbar_chains.grid(row=4, column=0, columnspan=3, padx=2, pady=(2, 4), sticky="ew")

        # Row 5–7: slider and info
        ttk.Label(comp_frame, text="View shell / Ω (slider):").grid(row=5, column=0, columnspan=3, sticky="w")
        self.slider = tk.Scale(
            comp_frame, from_=0, to=0, orient=tk.HORIZONTAL,
            resolution=1, showvalue=False, command=self.on_slider_change
        )
        self.slider.configure(state='disabled')
        self.slider.grid(row=6, column=0, columnspan=3, sticky="ew", padx=2, pady=(2, 2))

        self.slider_info_label = ttk.Label(comp_frame, text="No precomputed data yet.")
        self.slider_info_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ------------------------ Histogram figure (right) -------------------------
        self.fig_hist = plt.Figure(figsize=(8, 8))
        gs = self.fig_hist.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
        self.ax_J = self.fig_hist.add_subplot(gs[0, 0])
        self.ax_zeta = self.fig_hist.add_subplot(gs[1, 0])

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.right_frame)
        self.canvas_hist_widget = self.canvas_hist.get_tk_widget()
        self.canvas_hist_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initial empty plots
        self.update_hist_and_3d()

        # Set initial mode UI state (locks shells in animation)
        self.on_mode_change()

    # ------------------------ Mode handling -----------------------------

    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "animation":
            # lock the number-of-shells input in animation mode
            self.entry_comp_steps.configure(state='disabled')
        else:
            # in computation mode, allow editing #shells
            self.entry_comp_steps.configure(state='normal')
            self.anim_running = False

    # ------------------------ 3D window helpers -----------------------------

    def create_3d_window(self):
        if self.top3d is not None:
            return

        self.top3d = tk.Toplevel(self.master)
        self.top3d.title("3D ensemble view (grid + bars)")

        self.fig3d = plt.Figure(figsize=(8, 6))
        self.ax3 = self.fig3d.add_subplot(111)
        self.ax3.set_aspect('equal', adjustable='box')
        self.ax3.axis('off')

        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self.top3d)
        self.canvas3d_widget = self.canvas3d.get_tk_widget()
        self.canvas3d_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.top3d.protocol("WM_DELETE_WINDOW", self.on_close_3d)

    def on_close_3d(self):
        try:
            if self.top3d is not None:
                self.top3d.destroy()
        except Exception:
            pass
        self.top3d = None
        self.fig3d = None
        self.ax3 = None
        self.canvas3d = None
        self.show_3d.set(False)

    def on_toggle_3d(self):
        if self.show_3d.get():
            if self.top3d is None:
                self.create_3d_window()
            self.update_hist_and_3d()
        else:
            if self.top3d is not None:
                self.on_close_3d()

    # ---------------------- Helpers -----------------------------

    def get_dOmega(self):
        try:
            dO = float(self.entry_dOmega.get())
            if dO <= 0:
                dO = 0.001
        except ValueError:
            dO = 0.001
        self.dOmega_cached = dO
        return dO

    def get_nbins(self):
        try:
            nb = int(self.entry_bins.get())
            if nb < 1:
                nb = 1
        except ValueError:
            nb = 40
        return nb

    def initialize_master_equation(self):
        """
        Build the initial ζ-grid and P_0(ζ) from the *continuous* bare J-distribution
        defined by the GUI choice, truncated to [0, Ω0].
        """
        if self.J_init is None or self.Omega0 is None or self.Omega0 <= 0:
            self.zeta_grid = None
            self.P0_zeta = None
            self.P_zeta = None
            self.Gamma_current = 0.0
            return

        Omega0 = self.Omega0
        dist_name = self.combo_dist.get()

        # ζ-grid (fixed for all Γ)
        N_z = 1024
        z_max = 10.0
        z = np.linspace(0.0, z_max, N_z)
        self.zeta_grid = z

        def pdf_J_unnorm(J):
            J = np.asarray(J)
            if dist_name == "Uniform(0,1)":
                return np.where((J >= 0.0) & (J <= 1.0), 1.0, 0.0)
            elif dist_name == "Exponential(λ=1)":
                return np.where(J >= 0.0, np.exp(-J), 0.0)
            elif dist_name == "Abs Gaussian N(0,1)":
                return np.where(J >= 0.0,
                                np.sqrt(2.0 / np.pi) * np.exp(-0.5 * J**2),
                                0.0)
            elif dist_name == "Log-normal(μ=0,σ=1)":
                out = np.zeros_like(J, dtype=float)
                mask = J > 0
                out[mask] = (1.0 / (J[mask] * np.sqrt(2.0 * np.pi))
                             * np.exp(-0.5 * (np.log(J[mask]))**2))
                return out
            elif dist_name == "Gamma(k=2,θ=1)":
                return np.where(J >= 0.0, J * np.exp(-J), 0.0)
            elif dist_name == "Fermi-Dirac(μ=0,T=1)":
                return np.where(J >= 0.0, 1.0 / (np.exp(J) + 1.0), 0.0)
            else:
                return np.where((J >= 0.0) & (J <= Omega0), 1.0, 0.0)

        # Normalize on [0, Ω0]
        J_grid_norm = np.linspace(0.0, Omega0, 2000)
        Z = np.trapz(pdf_J_unnorm(J_grid_norm), J_grid_norm)
        if Z <= 0:
            Z = 1.0

        # ρ_0(J) on J = Ω0 e^{-ζ}
        J_from_z = Omega0 * np.exp(-z)
        rho0 = pdf_J_unnorm(J_from_z) / Z

        # P_0(ζ) = ρ_0(J) * |dJ/dζ| = ρ_0(J) * J
        P0 = rho0 * J_from_z
        norm_P0 = np.trapz(P0, z)
        if norm_P0 > 0:
            P0 /= norm_P0

        self.P0_zeta = P0
        self.P_zeta = P0.copy()
        self.Gamma_current = 0.0

    def reset_from_initial(self):
        """
        Reset animation state to the initial ensemble (J_init), without resampling.
        Also resets the master-equation state (Γ=0, P_0).
        """
        if self.J_init is None:
            return

        M, L = self.J_init.shape
        self.chains = [list(self.J_init[s, :]) for s in range(M)]
        self.J_current = self.J_init.copy()

        self.step_count = 0
        self.shell_index_anim = 0
        self.anim_shell_phase = 0
        self.current_decimation = None
        self.anim_running = False

        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []

        # Ω0 from initial ensemble
        J_flat = self.J_init.ravel()
        J_pos = J_flat[J_flat > 0]
        if J_pos.size > 0:
            self.Omega0 = float(J_pos.max())
        else:
            self.Omega0 = 0.0
        self.current_Omega = self.Omega0

        # Reset master equation (Γ=0, P_0)
        self.initialize_master_equation()

    # ---------------------- Callbacks for controls -----------------------------

    def on_sample_ensemble(self):
        try:
            L = int(self.entry_L.get())
            M = int(self.entry_M.get())
            seed = int(self.entry_seed.get())
            if L <= 1 or M <= 0:
                raise ValueError
        except ValueError:
            print("Invalid L or M. Need L>1, M>0.")
            return

        dist_name = self.combo_dist.get()
        rng = np.random.default_rng(seed)

        # Sample raw couplings (no rescaling)
        J_raw = sample_couplings(L, M, dist_name, rng)
        self.J_init = J_raw.copy()

        # Reset animation + master equation
        self.reset_from_initial()

        # Reset computation state
        self.comp_J_steps = None
        self.comp_Omegas = None
        self.comp_Pzeta_steps = None
        self.comp_Gammas = None
        self.comp_index = 0
        self.slider.configure(state='disabled', from_=0, to=0)
        self.slider.set(0)
        self.slider_info_label.config(text="No precomputed data yet.")

        self.progressbar_shells['value'] = 0
        self.progressbar_chains['value'] = 0

        self.update_hist_and_3d()

    def on_play(self):
        if self.mode_var.get() != "animation":
            print("Play is only available in animation mode.")
            return
        if self.J_init is None:
            print("Sample an ensemble first.")
            return
        if not self.anim_running:
            self.anim_running = True
            self.anim_shell_phase = 0
            self.current_decimation = None
            self.master.after(self.frame_delay_ms, self.animate_shell)

    def on_pause(self):
        self.anim_running = False

    def on_stop(self):
        self.anim_running = False
        if self.mode_var.get() == "animation":
            self.reset_from_initial()
            self.update_hist_and_3d()

    # ------------------------ Computation mode (Ω-shell SDRG) ------------------------

    def on_compute(self):
        """
        Batch mode: Ω-shell SDRG with final collapse + master-equation evolution.
        """
        if self.mode_var.get() != "computation":
            print("Batch compute is only available in computation mode.")
            return
        if self.J_init is None:
            print("Sample an ensemble first.")
            return

        try:
            max_shells = int(self.entry_comp_steps.get())
            if max_shells <= 0:
                raise ValueError
        except ValueError:
            print("Invalid number of shells for computation mode.")
            return

        dOmega = self.get_dOmega()

        # Initialize chains from initial ensemble
        M, L0 = self.J_init.shape
        chains = [list(self.J_init[s, :]) for s in range(M)]

        # Ω0 from initial ensemble
        J_flat = self.J_init.ravel()
        J_pos = J_flat[J_flat > 0]
        if J_pos.size > 0:
            Omega0 = float(J_pos.max())
        else:
            Omega0 = 0.0
        self.Omega0 = Omega0
        self.current_Omega = Omega0

        # (Re)initialize master equation
        self.initialize_master_equation()
        if self.zeta_grid is not None and self.P_zeta is not None:
            comp_Pzeta = [self.P_zeta.copy()]
            comp_Gammas = [0.0]
        else:
            comp_Pzeta = None
            comp_Gammas = None

        # Step 0 snapshot
        J0 = np.zeros((M, L0), dtype=float)
        for s, Js in enumerate(chains):
            J0[s, :] = np.array(Js, dtype=float)

        comp_J_steps = [J0]
        comp_Omegas = [Omega0]

        # Reset highlights
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []

        # Progress bars setup
        self.progressbar_shells['mode'] = 'determinate'
        self.progressbar_shells['maximum'] = max_shells
        self.progressbar_shells['value'] = 0

        self.progressbar_chains['mode'] = 'determinate'
        self.progressbar_chains['maximum'] = M
        self.progressbar_chains['value'] = 0

        self.master.update_idletasks()

        # Shell loop
        for n in range(max_shells):
            Omega_high = Omega0 - n * dOmega
            Omega_low = Omega_high - dOmega

            if Omega_low <= 0 or Omega_high <= 0:
                break

            # Process each chain independently for this shell
            self.progressbar_chains['value'] = 0
            self.master.update_idletasks()

            for s in range(M):
                J_list = chains[s]
                # Standard decimations while len>=3 and Jmax >= Omega_low
                if len(J_list) >= 3:
                    while len(J_list) >= 3:
                        arr = np.array(J_list, dtype=float)
                        Jmax_chain = float(arr.max())
                        if Jmax_chain < Omega_low:
                            break
                        idx = int(arr.argmax())
                        J_list, _ = decimate_chain_nonrotating(J_list, idx)

                # Final collapse [J0, J1] -> [J0+J1] if still in this shell
                if len(J_list) == 2:
                    arr = np.array(J_list, dtype=float)
                    Jmax_chain = float(arr.max())
                    if Jmax_chain >= Omega_low:
                        J_total = J_list[0] + J_list[1]
                        J_list = [J_total]

                chains[s] = J_list

                # Update per-shell chain completion progress
                self.progressbar_chains['value'] = s + 1
                self.master.update_idletasks()

            # Build rectangular matrix for snapshot after shell n
            max_len = max((len(J_list) for J_list in chains), default=0)
            if max_len <= 0:
                Jmat = np.zeros((M, 1), dtype=float)
            else:
                Jmat = np.zeros((M, max_len), dtype=float)
                for s in range(M):
                    J_list = chains[s]
                    if len(J_list) > 0:
                        Jmat[s, :len(J_list)] = np.array(J_list, dtype=float)

            comp_J_steps.append(Jmat)
            comp_Omegas.append(Omega_low)

            # Master equation: evolve by dΓ_shell = ln(Ω_high/Ω_low)
            if comp_Pzeta is not None and comp_Gammas is not None and \
               self.zeta_grid is not None and self.P_zeta is not None:
                dGamma_shell = np.log(Omega_high / Omega_low)
                if dGamma_shell > 0:
                    self.P_zeta = evolve(self.P_zeta, self.zeta_grid, dGamma_shell)
                    self.Gamma_current = comp_Gammas[-1] + dGamma_shell
                comp_Pzeta.append(self.P_zeta.copy())
                comp_Gammas.append(self.Gamma_current)

            # Update shells progress bar
            self.progressbar_shells['value'] = n + 1
            self.master.update_idletasks()

        # Store computation results
        self.comp_J_steps = comp_J_steps
        self.comp_Omegas = comp_Omegas
        self.comp_Pzeta_steps = comp_Pzeta
        self.comp_Gammas = comp_Gammas
        self.comp_index = 0

        # Slider configuration
        max_index = len(comp_J_steps) - 1
        self.slider.configure(state='normal', from_=0, to=max_index)
        self.slider.set(0)

        # Set current snapshot to index 0 (initial state)
        self.J_current = comp_J_steps[0]
        self.current_Omega = comp_Omegas[0] if len(comp_Omegas) > 0 else None
        self.step_count = 0

        # Reset P_zeta, Gamma_current to the first snapshot
        if self.comp_Pzeta_steps is not None and len(self.comp_Pzeta_steps) > 0:
            self.P_zeta = self.comp_Pzeta_steps[0].copy()
        if self.comp_Gammas is not None and len(self.comp_Gammas) > 0:
            self.Gamma_current = self.comp_Gammas[0]
        else:
            self.Gamma_current = 0.0

        # Clear highlights
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []

        self.update_slider_info()
        self.update_hist_and_3d()

    def update_slider_info(self):
        if self.comp_J_steps is None or self.comp_Omegas is None:
            self.slider_info_label.config(text="No precomputed data yet.")
            return
        idx = self.comp_index
        if idx < 0 or idx >= len(self.comp_Omegas):
            self.slider_info_label.config(text="Slider index out of range.")
            return
        Omega_cur = self.comp_Omegas[idx]

        if self.comp_Gammas is not None and idx < len(self.comp_Gammas):
            Gamma = self.comp_Gammas[idx]
        elif self.Omega0 is not None and self.Omega0 > 0 and Omega_cur > 0:
            Gamma = np.log(self.Omega0 / Omega_cur)
        else:
            Gamma = None

        if Gamma is not None:
            self.slider_info_label.config(
                text=f"Shell/state index = {idx},  Ω = {Omega_cur:.3g},  Γ = {Gamma:.3g}"
            )
        else:
            self.slider_info_label.config(
                text=f"Shell/state index = {idx},  Ω = {Omega_cur:.3g}"
            )

    def on_slider_change(self, value):
        if self.comp_J_steps is None or self.comp_Omegas is None:
            return
        try:
            idx = int(float(value))
        except ValueError:
            return
        idx = max(0, min(idx, len(self.comp_J_steps) - 1))
        self.comp_index = idx

        # Update snapshot and Ω for this shell/state
        self.J_current = self.comp_J_steps[idx]
        self.current_Omega = self.comp_Omegas[idx]
        self.step_count = idx

        # Update master-equation state to this snapshot
        if self.comp_Pzeta_steps is not None and idx < len(self.comp_Pzeta_steps):
            self.P_zeta = self.comp_Pzeta_steps[idx].copy()
        if self.comp_Gammas is not None and idx < len(self.comp_Gammas):
            self.Gamma_current = self.comp_Gammas[idx]
        else:
            if self.Omega0 is not None and self.Omega0 > 0 and \
               self.current_Omega is not None and self.current_Omega > 0:
                self.Gamma_current = np.log(self.Omega0 / self.current_Omega)
            else:
                self.Gamma_current = 0.0

        # Clear any highlights
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []

        self.update_slider_info()
        self.update_hist_and_3d()

    # --------------------------- Animation (Ω-shell + highlights + final collapse) -----------

    def animate_shell(self):
        """
        Animation mode: each (pair of) frames corresponds to one decimation
        within the current shell [Ω_low, Ω_high]. Continues until all chains
        have at most one bond.
        """
        if not self.anim_running:
            return
        if self.mode_var.get() != "animation":
            return
        if self.chains is None or len(self.chains) == 0:
            self.anim_running = False
            return

        # Stop when all chains have at most one bond
        if all(len(J_s) <= 1 for J_s in self.chains):
            # Rebuild final rectangular matrix
            M = len(self.chains)
            max_len = max((len(J_s) for J_s in self.chains), default=0)
            if max_len <= 0:
                self.J_current = None
                self.current_Omega = 0.0
            else:
                Jmat = np.zeros((M, max_len), dtype=float)
                for s in range(M):
                    J_s = self.chains[s]
                    if len(J_s) > 0:
                        Jmat[s, :len(J_s)] = np.array(J_s, dtype=float)
                self.J_current = Jmat
                all_J = Jmat.ravel()
                J_pos = all_J[all_J > 0]
                self.current_Omega = float(J_pos.max()) if J_pos.size > 0 else 0.0

            self.last_decimated = []
            self.neighbor_bonds = []
            self.inserted_bonds = []
            self.update_hist_and_3d()
            self.anim_running = False
            return

        if self.Omega0 is None or self.Omega0 <= 0:
            self.anim_running = False
            return

        dOmega = self.get_dOmega()
        if dOmega <= 0:
            self.anim_running = False
            return

        n = self.shell_index_anim
        Omega_high = self.Omega0 - n * dOmega
        Omega_low = Omega_high - dOmega

        if Omega_high <= 0 or Omega_low <= 0:
            self.anim_running = False
            return

        # For labeling
        self.current_Omega = Omega_high
        self.step_count = n

        M = len(self.chains)

        if self.anim_shell_phase == 0:
            # Phase 0: choose a bond / pair to decimate/collapse in this shell
            candidate = None      # (mode, s, idx)
            candidate_val = None  # Jmax_chain

            for s, J_list in enumerate(self.chains):
                if len(J_list) == 0:
                    continue
                arr = np.array(J_list, float)
                Jmax_chain = float(arr.max())
                if Jmax_chain < Omega_low:
                    continue

                if len(J_list) >= 3:
                    idx = int(arr.argmax())
                    mode = "normal"
                elif len(J_list) == 2:
                    idx = 0  # dummy index for final collapse
                    mode = "final"
                else:
                    continue

                if (candidate is None) or (Jmax_chain > candidate_val):
                    candidate = (mode, s, idx)
                    candidate_val = Jmax_chain

            if candidate is None:
                # This shell is finished; evolve master equation for this shell
                dGamma_shell = np.log(Omega_high / Omega_low)
                if (self.zeta_grid is not None and self.P_zeta is not None
                        and dGamma_shell > 0):
                    self.P_zeta = evolve(self.P_zeta, self.zeta_grid, dGamma_shell)
                    self.Gamma_current += dGamma_shell

                # Move to next shell
                self.shell_index_anim += 1

                # Rebuild J_current from chains with zero padding
                max_len = max((len(J_list) for J_list in self.chains), default=0)
                if max_len <= 0:
                    self.J_current = None
                    self.current_Omega = 0.0
                else:
                    Jmat = np.zeros((M, max_len), dtype=float)
                    for s in range(M):
                        J_list = self.chains[s]
                        if len(J_list) > 0:
                            Jmat[s, :len(J_list)] = np.array(J_list, dtype=float)
                    self.J_current = Jmat
                    all_J = Jmat.ravel()
                    J_pos = all_J[all_J > 0]
                    self.current_Omega = float(J_pos.max()) if J_pos.size > 0 else Omega_low

                # Clear highlights
                self.last_decimated = []
                self.neighbor_bonds = []
                self.inserted_bonds = []

                self.update_hist_and_3d()
                self.master.after(self.frame_delay_ms, self.animate_shell)
                return

            mode, s, idx = candidate
            J_list = self.chains[s]

            # Build J_current (pre-decimation) with zero padding
            max_len = max((len(J_s) for J_s in self.chains), default=0)
            if max_len <= 0:
                self.J_current = None
            else:
                Jmat = np.zeros((M, max_len), dtype=float)
                for ss in range(M):
                    J_s = self.chains[ss]
                    if len(J_s) > 0:
                        Jmat[ss, :len(J_s)] = np.array(J_s, dtype=float)
                self.J_current = Jmat

            if mode == "normal":
                n_bonds = len(J_list)
                left = (idx - 1) % n_bonds
                right = (idx + 1) % n_bonds

                self.current_decimation = (mode, s, idx)
                self.last_decimated = [(s, idx)]
                self.neighbor_bonds = [(s, left), (s, right)]
                self.inserted_bonds = []

            else:  # mode == "final" for len==2
                # highlight the two bonds as "decimated"
                self.current_decimation = (mode, s, 0)
                self.last_decimated = [(s, 0), (s, 1)]
                self.neighbor_bonds = []
                self.inserted_bonds = []

            self.update_hist_and_3d()
            self.anim_shell_phase = 1
            self.master.after(self.frame_delay_ms, self.animate_shell)
            return

        else:
            # Phase 1: perform decimation/collapse and highlight inserted bond
            if self.current_decimation is None:
                self.anim_shell_phase = 0
                self.master.after(self.frame_delay_ms, self.animate_shell)
                return

            mode, s, idx = self.current_decimation
            J_list = self.chains[s]

            if mode == "normal":
                if len(J_list) >= 3:
                    new_J_list, new_idx = decimate_chain_nonrotating(J_list, idx)
                    self.chains[s] = new_J_list
                else:
                    new_idx = None

                # Rebuild J_current (post-decimation)
                M = len(self.chains)
                max_len = max((len(J_s) for J_s in self.chains), default=0)
                if max_len <= 0:
                    self.J_current = None
                else:
                    Jmat = np.zeros((M, max_len), dtype=float)
                    for ss in range(M):
                        J_s = self.chains[ss]
                        if len(J_s) > 0:
                            Jmat[ss, :len(J_s)] = np.array(J_s, dtype=float)
                    self.J_current = Jmat

                self.last_decimated = []
                self.neighbor_bonds = []
                if new_idx is not None:
                    self.inserted_bonds = [(s, new_idx)]
                else:
                    self.inserted_bonds = []

            else:  # final collapse mode
                if len(J_list) == 2:
                    J_total = J_list[0] + J_list[1]
                    new_J_list = [J_total]
                    self.chains[s] = new_J_list
                # Rebuild J_current (post-collapse)
                M = len(self.chains)
                max_len = max((len(J_s) for J_s in self.chains), default=0)
                if max_len <= 0:
                    self.J_current = None
                else:
                    Jmat = np.zeros((M, max_len), dtype=float)
                    for ss in range(M):
                        J_s = self.chains[ss]
                        if len(J_s) > 0:
                            Jmat[ss, :len(J_s)] = np.array(J_s, dtype=float)
                    self.J_current = Jmat

                self.last_decimated = []
                self.neighbor_bonds = []
                # New single bond is at index 0
                self.inserted_bonds = [(s, 0)]

            self.update_hist_and_3d()

            # Prepare for next decimation in the same shell
            self.current_decimation = None
            self.anim_shell_phase = 0
            self.master.after(self.frame_delay_ms, self.animate_shell)
            return

    # --------------------------- Plot updating --------------------------------

    def update_hist_and_3d(self):
        self.step_label.config(text=f"Current shell (n): {self.step_count}")

        self.ax_J.clear()
        self.ax_zeta.clear()

        if self.J_current is None:
            self.ax_J.set_title("All-bonds J distribution (no data)")
            self.ax_J.set_xlabel("J")
            self.ax_J.set_ylabel("probability density")

            self.ax_zeta.set_title("All-bonds ζ distribution (no data)")
            self.ax_zeta.set_xlabel(r'$\zeta = \ln(\Omega / J)$')
            self.ax_zeta.set_ylabel("probability density")

            self.fig_hist.tight_layout()
            self.canvas_hist.draw_idle()
        else:
            J_cur = self.J_current
            all_J = J_cur.ravel()
            J_pos = all_J[all_J > 0]

            if J_pos.size == 0:
                self.ax_J.set_title("All-bonds J distribution (no positive couplings)")
                self.ax_J.set_xlabel("J")
                self.ax_J.set_ylabel("probability density")

                self.ax_zeta.set_title("All-bonds ζ distribution (no positive couplings)")
                self.ax_zeta.set_xlabel(r'$\zeta = \ln(\Omega / J)$')
                self.ax_zeta.set_ylabel("probability density")

                self.fig_hist.tight_layout()
                self.canvas_hist.draw_idle()
            else:
                nbins = self.get_nbins()

                # Ω for current snapshot: use current_Omega if possible
                if self.current_Omega is not None and self.current_Omega > 0:
                    Omega_plot = float(self.current_Omega)
                else:
                    Omega_plot = float(J_pos.max())
                    self.current_Omega = Omega_plot

                # ------------------ J histogram (density) ------------------
                edges = np.linspace(0.0, Omega_plot, nbins + 1)
                widths = edges[1:] - edges[:-1]
                centers = 0.5 * (edges[:-1] + edges[1:])

                counts, _ = np.histogram(J_pos, bins=edges)
                N = counts.sum()
                if N > 0:
                    heights = counts / (N * widths)
                else:
                    heights = np.zeros_like(widths)

                # Main empirical histogram
                self.ax_J.bar(
                    centers,
                    heights,
                    width=widths,
                    align='center',
                    alpha=0.7,
                    color='C0',
                    edgecolor='black',
                    label='Empirical density (counts / (N·ΔJ))',
                    zorder=2.0,
                )

                self.ax_J.set_xlabel("J (all bonds)")
                self.ax_J.set_ylabel("probability density")
                self.ax_J.set_title(r"Histogram (density) vs. theoretical $\rho_\Omega(J)$")

                # Lock histogram-determined limits (current state only)
                xlim_J = self.ax_J.get_xlim()
                ylim_J = self.ax_J.get_ylim()

                # ------------------ ζ histogram (density) ------------------
                zeta = np.log(Omega_plot / J_pos)
                zeta = np.maximum(zeta, 0.0)
                zmax = float(zeta.max())
                if zmax <= 0:
                    zmax = 1.0
                edges_z = np.linspace(0.0, zmax, nbins + 1)
                widths_z = edges_z[1:] - edges_z[:-1]
                centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])

                counts_z, _ = np.histogram(zeta, bins=edges_z)
                N_z = counts_z.sum()
                if N_z > 0:
                    heights_z = counts_z / (N_z * widths_z)
                else:
                    heights_z = np.zeros_like(widths_z)

                self.ax_zeta.bar(
                    centers_z,
                    heights_z,
                    width=widths_z,
                    align='center',
                    alpha=0.7,
                    color='C1',
                    edgecolor='black',
                    label='Empirical density (counts / (N·Δζ))',
                    zorder=2.0,
                )

                self.ax_zeta.set_xlabel(r'$\zeta = \ln(\Omega / J)$ (all bonds)')
                self.ax_zeta.set_ylabel("probability density")
                self.ax_zeta.set_title(r"Histogram (density) vs. theoretical $P_\Gamma(\zeta)$")

                # Lock histogram-determined limits for ζ
                xlim_z = self.ax_zeta.get_xlim()
                ylim_z = self.ax_zeta.get_ylim()

                # ------------------ Theoretical asymptotic curves ------------------
                if self.Omega0 is not None and self.Omega0 > 0 and 0 < Omega_plot < self.Omega0:
                    Gamma_th = np.log(self.Omega0 / Omega_plot)
                    if Gamma_th > 1e-8:
                        # Theoretical ρ_Ω(J)
                        J_min_data = float(J_pos.min())
                        J_min_grid = max(J_min_data, Omega_plot * 1e-4)
                        J_grid = np.linspace(J_min_grid, Omega_plot, 400)
                        rho_th = (1.0 / (Omega_plot * Gamma_th)) * (Omega_plot / J_grid) ** (1.0 - 1.0 / Gamma_th)
                        self.ax_J.plot(
                            J_grid,
                            rho_th,
                            'r-',
                            lw=2,
                            label=r"Theory $\rho_\Omega(J)$ (density)"
                        )

                        # Theoretical P_Γ(ζ) = (1/Γ) e^{-ζ/Γ} on ζ ≥ 0
                        z_grid = np.linspace(0.0, zmax, 400)
                        P_th = (1.0 / Gamma_th) * np.exp(-z_grid / Gamma_th)
                        self.ax_zeta.plot(
                            z_grid,
                            P_th,
                            'r-',
                            lw=2,
                            label=r"Theory $P_\Gamma(\zeta)$ (density)"
                        )

                        info_text = f"$\\Omega$={Omega_plot:.3g}\n$\\Gamma$={Gamma_th:.3g}"
                        self.ax_J.text(
                            0.98, 0.95,
                            info_text,
                            transform=self.ax_J.transAxes,
                            ha='right',
                            va='top',
                            fontsize=9,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                        )
                        self.ax_zeta.text(
                            0.98, 0.95,
                            info_text,
                            transform=self.ax_zeta.transAxes,
                            ha='right',
                            va='top',
                            fontsize=9,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                        )

                # ------------------ Master-equation prediction overlay ------------------
                if self.zeta_grid is not None and self.P_zeta is not None and Omega_plot > 0:
                    z = self.zeta_grid
                    Pz = self.P_zeta

                    # ζ-space line (green) up to current zmax
                    mask_me_z = z <= zmax
                    if np.any(mask_me_z):
                        self.ax_zeta.plot(
                            z[mask_me_z],
                            Pz[mask_me_z],
                            color='g',
                            lw=2,
                            label="Master eq $P(\\zeta)$"
                        )

                    # J-space prediction ρ(J) from same P(ζ), with J = Ω e^{-ζ}
                    J_me = Omega_plot * np.exp(-z)
                    mask_me_J = (J_me > 0) & (J_me <= Omega_plot)
                    if np.any(mask_me_J):
                        J_line = J_me[mask_me_J]
                        rho_line = Pz[mask_me_J] / np.maximum(J_line, 1e-300)
                        order = np.argsort(J_line)
                        J_line = J_line[order]
                        rho_line = rho_line[order]
                        self.ax_J.plot(
                            J_line,
                            rho_line,
                            color='g',
                            lw=2,
                            label="Master eq $\\rho(J)$"
                        )

                # Restore histogram-based limits so theory/ME curves don't change axes
                self.ax_J.set_xlim(xlim_J)
                self.ax_J.set_ylim(ylim_J)
                self.ax_zeta.set_xlim(xlim_z)
                self.ax_zeta.set_ylim(ylim_z)

                self.ax_J.text(
                    0.02, 0.95,
                    f"Shell n = {self.step_count}",
                    transform=self.ax_J.transAxes,
                    ha='left',
                    va='top',
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )

                self.ax_J.legend(loc='best', fontsize=8)
                self.ax_zeta.legend(loc='best', fontsize=8)

                self.fig_hist.tight_layout()
                self.canvas_hist.draw_idle()

        # ----- 3D view -----
        if not self.show_3d.get():
            return

        if self.top3d is None:
            self.create_3d_window()
        if self.ax3 is None:
            return

        if self.J_current is not None:
            M_grid, L_grid = self.J_current.shape
            Jmat_3d = self.J_current
        else:
            try:
                L_grid = int(self.entry_L.get())
                M_grid = int(self.entry_M.get())
            except ValueError:
                L_grid, M_grid = 10, 10
            M_grid = max(M_grid, 1)
            L_grid = max(L_grid, 1)
            Jmat_3d = np.zeros((M_grid, L_grid))

        self.ax3.clear()
        self.ax3.set_aspect('equal', adjustable='box')
        self.ax3.axis('off')

        O = self.base_origin
        s_dir = self.s_dir
        b_dir = self.b_dir
        S = O + s_dir
        B = O + b_dir
        SB = S + (B - O)

        xs = np.array([O[0], S[0], SB[0], B[0]])
        ys = np.array([O[1], S[1], SB[1], B[1]])

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        margin_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
        margin_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

        Bv = (y_max - y_min) + 2.0 * margin_y
        self.h_max = 2.0 * Bv

        y_min_plot = y_min - margin_y
        y_max_plot = y_max + margin_y + self.h_max

        self.ax3.set_xlim(x_min - margin_x, x_max + margin_x)
        self.ax3.set_ylim(y_min_plot, y_max_plot)

        self.draw_parallelogram_grid(Jmat_3d.shape[0], Jmat_3d.shape[1])
        self.draw_bars(Jmat_3d, Jmat_3d.shape[0], Jmat_3d.shape[1],
                       x_min, x_max, y_min, y_max, margin_x, margin_y)

        self.canvas3d.draw_idle()

    # --------------------- Parallelogram grid drawing -------------------------

    def draw_parallelogram_grid(self, M: int, L: int):
        O = self.base_origin
        s_dir = self.s_dir
        b_dir = self.b_dir

        S = O + s_dir
        B = O + b_dir
        SB = S + (B - O)

        border_x = [O[0], S[0], SB[0], B[0], O[0]]
        border_y = [O[1], S[1], SB[1], B[1], O[1]]
        self.ax3.plot(border_x, border_y, color='black', linewidth=1.5, zorder=0)

        total_stripes = 2 * M
        for m in range(1, total_stripes):
            u = m / total_stripes
            P0 = O + u * s_dir
            P1 = P0 + b_dir
            self.ax3.plot(
                [P0[0], P1[0]], [P0[1], P1[1]],
                color='lightgray', linewidth=0.8, zorder=0
            )

        for l in range(1, L):
            v = l / L
            Q0 = O + v * b_dir
            Q1 = Q0 + s_dir
            self.ax3.plot(
                [Q0[0], Q1[0]], [Q0[1], Q1[1]],
                color='lightgray', linewidth=0.8, zorder=0
            )

    # -------------------------- 3D bar drawing ---------------------------

    def draw_bars(self, Jmat: np.ndarray, M: int, L: int,
                  x_min: float, x_max: float,
                  y_min: float, y_max: float,
                  margin_x: float, margin_y: float):
        O = self.base_origin
        s_dir = self.s_dir
        b_dir = self.b_dir
        h_dir = self.h_dir

        total_stripes = 2 * M

        Jmax = float(np.max(Jmat)) if Jmat.size > 0 else 1.0
        if Jmax <= 0:
            Jmax = 1.0

        # Normalize heights against initial Ω0 (no rescaling Ω0 to 1)
        if self.Omega0 is not None and self.Omega0 > 0:
            norm_scale = self.Omega0
        else:
            norm_scale = Jmax

        decimated_set = set(self.last_decimated) if self.last_decimated else set()
        neighbor_set = set(self.neighbor_bonds) if self.neighbor_bonds else set()
        inserted_set = set(self.inserted_bonds) if self.inserted_bonds else set()

        S = O + s_dir
        B = O + b_dir
        SB = S + (B - O)

        for j in range(L):
            v0 = (L - 1 - j) / L
            v1 = (L - j) / L

            for s in range(M):
                J_val = float(Jmat[s, j])
                if J_val <= 0:
                    continue

                col = 2 * s
                u0 = col / total_stripes
                u1 = (col + 1) / total_stripes

                p00 = O + u0 * s_dir + v0 * b_dir
                p10 = O + u1 * s_dir + v0 * b_dir
                p11 = O + u1 * s_dir + v1 * b_dir
                p01 = O + u0 * s_dir + v1 * b_dir

                h = (J_val / norm_scale) * self.h_max
                dh = h * h_dir

                p00_top = p00 + dh
                p10_top = p10 + dh
                p11_top = p11 + dh
                p01_top = p01 + dh

                # Color logic:
                # - inserted (green)
                # - highest bond (red)
                # - neighbors (orange)
                # - others (blue)
                if (s, j) in inserted_set:
                    base_color = (0.2, 0.8, 0.2)  # green
                elif (s, j) in decimated_set:
                    base_color = (1.0, 0.2, 0.2)  # red
                elif (s, j) in neighbor_set:
                    base_color = (1.0, 0.6, 0.1)  # orange
                else:
                    base_color = (0.2, 0.4, 1.0)  # blue

                side_poly = Polygon(
                    [p10, p11, p11_top, p10_top],
                    closed=True,
                    facecolor=tuple(min(1.0, c * 0.6) for c in base_color),
                    edgecolor='black',
                    linewidth=0.4,
                    alpha=1.0,
                    zorder=1
                )

                top_poly = Polygon(
                    [p00_top, p10_top, p11_top, p01_top],
                    closed=True,
                    facecolor=base_color,
                    edgecolor='black',
                    linewidth=0.4,
                    alpha=1.0,
                    zorder=1
                )

                front_poly = Polygon(
                    [p00, p10, p10_top, p00_top],
                    closed=True,
                    facecolor=tuple(min(1.0, c * 0.8) for c in base_color),
                    edgecolor='black',
                    linewidth=0.4,
                    alpha=1.0,
                    zorder=1
                )

                self.ax3.add_patch(side_poly)
                self.ax3.add_patch(top_poly)
                self.ax3.add_patch(front_poly)

        d = 0.6 * margin_y

        bottom_mid = 0.5 * (O + S)
        x_label_x = bottom_mid[0]
        x_label_y = bottom_mid[1] - d
        self.ax3.text(
            x_label_x,
            x_label_y,
            f"samples # = {M}",
            ha='center',
            va='top',
            fontsize=10
        )

        right_mid = 0.5 * (S + SB)

        b_hat = b_dir / np.linalg.norm(b_dir)
        n_candidate = np.array([b_hat[1], -b_hat[0]])
        if n_candidate[0] < 0:
            n_candidate = -n_candidate
        n_hat = n_candidate

        y_label_pos = right_mid + d * n_hat
        y_label_x, y_label_y = y_label_pos

        label_angle = np.degrees(np.arctan2(b_dir[1], b_dir[0]))

        # Estimate bonds per chain from Jmat row-wise (count positive entries)
        lengths = []
        for s in range(M):
            row = Jmat[s, :]
            length_s = int(np.count_nonzero(row > 0))
            lengths.append(length_s)
        if len(lengths) > 0:
            L_min = min(lengths)
            L_max = max(lengths)
            bonds_label_text = f"bonds per chain: {L_min}–{L_max}"
        else:
            bonds_label_text = f"bonds # = {L}"

        self.ax3.text(
            y_label_x,
            y_label_y,
            bonds_label_text,
            ha='left',
            va='center',
            rotation=label_angle,
            fontsize=10
        )

        # ---------- Legend ----------
        blue_patch = Patch(facecolor=(0.2, 0.4, 1.0), edgecolor='black',
                           label=r'Relative bond energy $J$')
        red_patch = Patch(facecolor=(1.0, 0.2, 0.2), edgecolor='black',
                          label='Highest bond')
        orange_patch = Patch(facecolor=(1.0, 0.6, 0.1), edgecolor='black',
                             label='Neighbor bonds (to be removed)')
        green_patch = Patch(facecolor=(0.2, 0.8, 0.2), edgecolor='black',
                            label='Inserted bonds')

        self.ax3.legend(handles=[blue_patch, red_patch, orange_patch, green_patch],
                        loc='upper left',
                        frameon=True,
                        fontsize=9)

        # ---------- Current Ω label (top-right corner) ----------
        Omega_disp = self.current_Omega if self.current_Omega is not None else 0.0
        self.ax3.text(
            0.98, 0.98,
            f"Ω = {Omega_disp:.3g}",
            transform=self.ax3.transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )


# =============================== Main program =================================

def main():
    root = tk.Tk()
    app = SDRGEnsembleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
