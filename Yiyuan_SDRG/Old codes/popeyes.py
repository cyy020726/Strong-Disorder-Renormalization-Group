import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Patch


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
        # J_candidate ~ Exp(1)
        x = rng.exponential(scale=1.0)
        u = rng.random()
        # Acceptance probability t(x)/g(x) = e^x/(e^x + 1) in (0,1)
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
        # Gamma(shape=k, scale=θ)
        J = rng.gamma(shape=2.0, scale=1.0, size=(M, L))
    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        J = sample_fermi_dirac((M, L), rng)
    else:
        J = rng.random((M, L))
    return J


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
        master.title("SDRG ensemble — RG flow Ω & ρ_Ω(J), P_Γ(ζ) + optional 3D view")

        # ---------- State ----------
        self.J_init = None          # initial couplings: shape (M,L)
        self.chains = None          # list of lists, current per sample (animation mode)
        self.J_current = None       # current coupling matrix: (M, n_bonds)
        self.step_count = 0

        # SDRG animation state (animation mode)
        self.anim_running = False
        self.frame_delay_ms = 400   # ms between frames
        self.phase = 0              # 0 = choose bonds (highlight), 1 = decimate
        self.pending_decimations = []  # list of (sample, bond_index) to decimate
        self.last_decimated = []       # highest bonds (to be decimated)
        self.neighbor_bonds = []       # neighbors of highest bonds
        self.inserted_bonds = []       # newly inserted effective bonds

        # RG scale
        self.Omega0 = None          # global Ω0 from initial ensemble
        self.rg_steps = []          # list of SDRG steps
        self.rg_omegas = []         # list of Ω values

        # MODE: "animation" or "computation"
        self.mode_var = tk.StringVar(value="animation")

        # Computation mode storage
        self.comp_J_steps = None    # list of J matrices for each precomputed step
        self.comp_Omegas = None     # list of Ω values for each precomputed step
        self.comp_index = 0         # slider-selected step index

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

        # Row 1: distribution (span), seed
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

        # Row 2: # J bins
        ttk.Label(control_frame, text="# J bins:").grid(row=2, column=0, sticky="w")
        self.entry_bins = ttk.Entry(control_frame, width=6)
        self_bins_default = 40
        self.entry_bins.insert(0, str(self_bins_default))
        self.entry_bins.grid(row=2, column=1, padx=4, pady=2)

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

        # Current SDRG step label
        self.step_label = ttk.Label(self.left_frame, text="Current SDRG step: 0")
        self.step_label.pack(side=tk.TOP, anchor="w", pady=(4, 4))

        # -------------------- Computation mode controls ---------------------
        comp_frame = ttk.Frame(self.left_frame)
        comp_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(comp_frame, text="# SDRG steps (batch):").grid(row=0, column=0, sticky="w")
        self.entry_comp_steps = ttk.Entry(comp_frame, width=8)
        self.entry_comp_steps.insert(0, "50")
        self.entry_comp_steps.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        self.button_compute = ttk.Button(comp_frame, text="Compute (batch)", command=self.on_compute)
        self.button_compute.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

        # Progress bar
        self.progressbar = ttk.Progressbar(comp_frame, orient="horizontal", length=240, mode="determinate")
        self.progressbar.grid(row=1, column=0, columnspan=3, padx=2, pady=(4, 2), sticky="ew")

        # Slider for viewing precomputed steps (i.e. different Ω & Γ)
        ttk.Label(comp_frame, text="View step / Ω (slider):").grid(row=2, column=0, columnspan=3, sticky="w")
        self.slider = tk.Scale(
            comp_frame, from_=0, to=0, orient=tk.HORIZONTAL,
            resolution=1, showvalue=False, command=self.on_slider_change
        )
        self.slider.configure(state='disabled')
        self.slider.grid(row=3, column=0, columnspan=3, sticky="ew", padx=2, pady=(2, 2))

        self.slider_info_label = ttk.Label(comp_frame, text="No precomputed data yet.")
        self.slider_info_label.grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ------------------------ RG flow figure (left, bottom) -------------------------
        self.fig_rg = plt.Figure(figsize=(3, 3))
        self.ax_rg = self.fig_rg.add_subplot(111)
        self.ax_rg.set_xlabel("SDRG step")
        self.ax_rg.set_ylabel(r'$\Omega$')
        self.ax_rg.set_title("RG flow (no data)")

        self.canvas_rg = FigureCanvasTkAgg(self.fig_rg, master=self.left_frame)
        self.canvas_rg_widget = self.canvas_rg.get_tk_widget()
        self.canvas_rg_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
        self.update_rg_plot()

    # ------------------------ Mode handling -----------------------------

    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "animation":
            # Re-enable 3D toggle in animation mode
            self.check_3d.configure(state='normal')
        else:
            # Switching to computation mode: stop animation, disable 3D
            self.anim_running = False
            if self.top3d is not None:
                self.on_close_3d()
            self.show_3d.set(False)
            self.check_3d.configure(state='disabled')

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
        # Do nothing in computation mode
        if self.mode_var.get() == "computation":
            self.show_3d.set(False)
            return

        if self.show_3d.get():
            if self.top3d is None:
                self.create_3d_window()
            self.update_hist_and_3d()
        else:
            if self.top3d is not None:
                self.on_close_3d()

    # ---------------------- SDRG state helpers (animation) -----------------------------

    def reset_sdrg_state(self):
        if self.J_init is None:
            return
        M, L = self.J_init.shape
        self.chains = [list(self.J_init[s, :]) for s in range(M)]
        self.J_current = self.J_init.copy()
        self.step_count = 0
        self.phase = 0
        self.pending_decimations = []
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []

        # Global Ω0 from initial ensemble (max over all bonds)
        J_init_flat = self.J_init.ravel()
        J_init_pos = J_init_flat[J_init_flat > 0]
        if J_init_pos.size > 0:
            self.Omega0 = float(J_init_pos.max())
        else:
            self.Omega0 = 0.0

        # Reset RG flow
        self.rg_steps = []
        self.rg_omegas = []
        if self.Omega0 > 0:
            self.rg_steps.append(0)
            self.rg_omegas.append(self.Omega0)

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

        self.J_init = sample_couplings(L, M, dist_name, rng)

        # Reset both animation and computation state
        self.anim_running = False
        self.reset_sdrg_state()

        self.comp_J_steps = None
        self.comp_Omegas = None
        self.comp_index = 0
        self.slider.configure(state='disabled', from_=0, to=0)
        self.slider.set(0)
        self.slider_info_label.config(text="No precomputed data yet.")
        self.progressbar['value'] = 0

        self.update_rg_plot()
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
            self.animate()

    def on_pause(self):
        self.anim_running = False

    def on_stop(self):
        self.anim_running = False
        if self.mode_var.get() == "animation":
            self.reset_sdrg_state()
            self.update_rg_plot()
            self.update_hist_and_3d()

    # ------------------------ Computation mode (batch) ------------------------

    def on_compute(self):
        if self.mode_var.get() != "computation":
            print("Batch compute is only available in computation mode.")
            return
        if self.J_init is None:
            print("Sample an ensemble first.")
            return

        try:
            max_steps = int(self.entry_comp_steps.get())
            if max_steps <= 0:
                raise ValueError
        except ValueError:
            print("Invalid number of SDRG steps for computation mode.")
            return

        # Initialize chains locally (do not touch animation chains)
        M, L0 = self.J_init.shape
        chains = [list(self.J_init[s, :]) for s in range(M)]

        # Progress bar setup
        self.progressbar['mode'] = 'determinate'
        self.progressbar['maximum'] = max_steps
        self.progressbar['value'] = 0
        self.master.update_idletasks()

        comp_J_steps = []
        comp_Omegas = []

        # Step 0: initial
        L_cur = len(chains[0])
        M_cur = len(chains)
        J0 = np.zeros((M_cur, L_cur), dtype=float)
        for s, Js in enumerate(chains):
            J0[s, :] = np.array(Js, dtype=float)

        comp_J_steps.append(J0)
        J_init_flat = J0.ravel()
        J_init_pos = J_init_flat[J_init_flat > 0]
        Omega0 = float(J_init_pos.max()) if J_init_pos.size > 0 else 0.0
        comp_Omegas.append(Omega0)

        steps_done = 0

        for k in range(1, max_steps + 1):
            # Check if any chain can still be decimated with the usual rule
            if all(len(Js) < 3 for Js in chains):
                break

            # Choose bonds to decimate
            decimations = []
            for s, Js in enumerate(chains):
                if len(Js) < 3:
                    continue
                arr = np.array(Js, float)
                idx = int(np.argmax(arr))
                decimations.append((s, idx))

            # Apply decimations
            for (s, idx) in decimations:
                Js = chains[s]
                if len(Js) < 3:
                    continue
                new_Js, _ = decimate_chain_nonrotating(Js, idx)
                chains[s] = new_Js

            # Rebuild rectangular matrix for this step
            L_cur = len(chains[0])
            M_cur = len(chains)
            Jmat = np.zeros((M_cur, L_cur), dtype=float)
            for s, Js in enumerate(chains):
                if len(Js) != L_cur:
                    if len(Js) < L_cur:
                        Js = Js + [0.0] * (L_cur - len(Js))
                    else:
                        Js = Js[:L_cur]
                Jmat[s, :] = np.array(Js, dtype=float)

            comp_J_steps.append(Jmat)
            all_J = Jmat.ravel()
            J_pos = all_J[all_J > 0]
            Omega_cur = float(J_pos.max()) if J_pos.size > 0 else 0.0
            comp_Omegas.append(Omega_cur)

            steps_done = k
            self.progressbar['value'] = k
            self.master.update_idletasks()

            if Omega_cur <= 0:
                break

        # Adjust progress bar max to actual steps performed
        self.progressbar['maximum'] = max(1, steps_done)

        # Store computation results
        self.comp_J_steps = comp_J_steps
        self.comp_Omegas = comp_Omegas
        self.comp_index = 0
        self.Omega0 = comp_Omegas[0] if comp_Omegas else None

        # RG flow (precomputed, independent of slider)
        self.rg_steps = list(range(len(comp_Omegas)))
        self.rg_omegas = comp_Omegas
        self.update_rg_plot()

        # Enable slider
        max_index = len(comp_J_steps) - 1
        self.slider.configure(state='normal', from_=0, to=max_index)
        self.slider.set(0)
        self.update_slider_info()

        # For viewing: set current J to step 0 and update plots
        self.J_current = comp_J_steps[0]
        self.step_count = 0
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []
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
        # Gamma defined via Omega0 and current Omega; slider controls both Ω and Γ
        if self.Omega0 is not None and self.Omega0 > 0 and Omega_cur > 0:
            Gamma = np.log(self.Omega0 / Omega_cur)
            self.slider_info_label.config(
                text=f"View step = {idx},  Ω = {Omega_cur:.3g},  Γ = {Gamma:.3g}"
            )
        else:
            self.slider_info_label.config(
                text=f"View step = {idx},  Ω = {Omega_cur:.3g}"
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
        # View this precomputed step (controls Ω and Γ via J_current)
        self.J_current = self.comp_J_steps[idx]
        self.step_count = idx
        self.last_decimated = []
        self.neighbor_bonds = []
        self.inserted_bonds = []
        self.update_slider_info()
        self.update_hist_and_3d()

    # ------------------------ Final collapse step (2 bonds -> 1 bond) ------------------------

    def final_collapse_step(self):
        """
        When all chains have at most 2 bonds left, we perform one last SDRG-like step:
        for any chain with exactly 2 bonds [J0, J1], we replace it by a single effective bond
        J_total = J0 + J1. Chains with 1 or 0 bonds are left unchanged.

        After this collapse, we rebuild J_current as a rectangular matrix and treat this as
        one additional SDRG step in the RG flow.

        The resulting single bonds are marked as 'inserted_bonds' so that they are
        drawn in green in the 3D plot.
        """
        if self.chains is None:
            return

        new_inserted = []  # (sample_index, bond_index) created in this collapse

        # Collapse chains of length 2 into length 1
        for s, J_s in enumerate(self.chains):
            if len(J_s) == 2:
                J_total = J_s[0] + J_s[1]
                self.chains[s] = [J_total]
                # After collapse there is exactly one bond at index 0 -> mark as inserted
                new_inserted.append((s, 0))
            elif len(J_s) <= 1:
                # leave as is
                continue
            else:
                # Should not happen if called only when max length <= 2
                pass

        # Rebuild J_current with new maximum length
        max_len = max((len(J_s) for J_s in self.chains), default=0)
        if max_len == 0:
            self.J_current = None
        else:
            M_cur = len(self.chains)
            Jmat = np.zeros((M_cur, max_len), dtype=float)
            for s, J_s in enumerate(self.chains):
                J_list = list(J_s)
                if len(J_list) < max_len:
                    J_list = J_list + [0.0] * (max_len - len(J_list))
                else:
                    J_list = J_list[:max_len]
                Jmat[s, :] = np.array(J_list, dtype=float)
            self.J_current = Jmat

        # Count this as one more SDRG step
        self.step_count += 1

        # Update Ω = max J over all chains
        if self.J_current is not None:
            all_J = self.J_current.ravel()
            J_pos = all_J[all_J > 0]
            Omega_cur = float(J_pos.max()) if J_pos.size > 0 else 0.0
        else:
            Omega_cur = 0.0

        if Omega_cur > 0:
            self.rg_steps.append(self.step_count)
            self.rg_omegas.append(Omega_cur)

        # Mark the new single bonds as inserted (green)
        self.inserted_bonds = new_inserted
        self.last_decimated = []
        self.neighbor_bonds = []

    # --------------------------- SDRG animation (animation mode) -----------------------------

    def animate(self):
        if not self.anim_running:
            return
        if self.mode_var.get() != "animation":
            return
        if self.chains is None:
            return

        lengths = [len(J_s) for J_s in self.chains]
        if len(lengths) == 0:
            self.anim_running = False
            return

        max_len = max(lengths)

        # If every chain has at most one bond left, we are truly done
        if max_len <= 1:
            self.anim_running = False
            self.phase = 0
            self.pending_decimations = []
            self.last_decimated = []
            self.neighbor_bonds = []
            self.inserted_bonds = []
            return

        # If all chains have length <= 2 but at least one has length 2,
        # perform one final collapse step (2 bonds -> 1 bond) for all such chains.
        if max_len <= 2 and all(l <= 2 for l in lengths):
            self.final_collapse_step()
            self.update_rg_plot()
            self.update_hist_and_3d()
            self.master.after(self.frame_delay_ms, self.animate)
            return

        # Otherwise, at least one chain has length >= 3: perform standard SDRG step
        if self.phase == 0:
            # Phase 0: choose bonds to decimate (highlight in red) and neighbors in orange
            self.pending_decimations = []
            self.last_decimated = []
            self.neighbor_bonds = []
            self.inserted_bonds = []
            for s, J_s in enumerate(self.chains):
                if len(J_s) < 3:
                    continue
                arr = np.array(J_s, dtype=float)
                idx = int(np.argmax(arr))
                left = (idx - 1) % len(J_s)
                right = (idx + 1) % len(J_s)
                self.pending_decimations.append((s, idx))
                self.last_decimated.append((s, idx))
                self.neighbor_bonds.append((s, left))
                self.neighbor_bonds.append((s, right))

            self.update_hist_and_3d()
            self.phase = 1

        else:
            # Phase 1: decimate and highlight inserted bonds in green
            self.inserted_bonds = []
            for s, idx in self.pending_decimations:
                J_s = self.chains[s]
                if len(J_s) < 3:
                    continue
                new_J_s, new_idx = decimate_chain_nonrotating(J_s, idx)
                self.chains[s] = new_J_s
                if new_idx is not None:
                    self.inserted_bonds.append((s, new_idx))

            # Rebuild rectangular matrix
            L_cur = len(self.chains[0])
            M_cur = len(self.chains)
            Jmat = np.zeros((M_cur, L_cur), dtype=float)
            for s, J_s in enumerate(self.chains):
                if len(J_s) != L_cur:
                    if len(J_s) < L_cur:
                        J_s = J_s + [0.0] * (L_cur - len(J_s))
                    else:
                        J_s = J_s[:L_cur]
                Jmat[s, :] = np.array(J_s, dtype=float)

            self.J_current = Jmat
            self.step_count += 1

            # Update Ω (max J over all bonds of all chains)
            all_J = self.J_current.ravel()
            J_pos = all_J[all_J > 0]
            if J_pos.size > 0:
                Omega_cur = float(J_pos.max())
            else:
                Omega_cur = 0.0

            if Omega_cur > 0:
                self.rg_steps.append(self.step_count)
                self.rg_omegas.append(Omega_cur)

            # Clear decimation sets except inserted_bonds
            self.last_decimated = []
            self.neighbor_bonds = []
            self.pending_decimations = []

            self.update_rg_plot()
            self.update_hist_and_3d()
            self.phase = 0

        self.master.after(self.frame_delay_ms, self.animate)

    # --------------------------- Plot updating --------------------------------

    def get_nbins(self):
        try:
            nb = int(self.entry_bins.get())
            if nb < 1:
                nb = 1
        except ValueError:
            nb = 40
        return nb

    def update_hist_and_3d(self):
        self.step_label.config(text=f"Current SDRG step: {self.step_count}")

        # Clear histogram axes
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
                Omega_cur = float(J_pos.max())

                # ------------------ J histogram (density) ------------------
                edges = np.linspace(0.0, Omega_cur, nbins + 1)
                counts, edges = np.histogram(J_pos, bins=edges)
                N = counts.sum()
                widths = edges[1:] - edges[:-1]
                centers = 0.5 * (edges[:-1] + edges[1:])

                if N > 0:
                    # Explicitly convert the empirical histogram into a *probability density*:
                    #   density[k] = counts[k] / (N * bin_width[k])
                    # so that sum_k density[k] * bin_width[k] = 1.
                    heights = counts / (N * widths)
                else:
                    heights = np.zeros_like(widths)

                self.ax_J.bar(
                    centers,
                    heights,
                    width=widths,
                    align='center',
                    alpha=0.7,
                    color='C0',
                    edgecolor='black',
                    label='Empirical density (counts / (N·ΔJ))'
                )

                self.ax_J.set_xlabel("J (all bonds)")
                self.ax_J.set_ylabel("probability density")
                self.ax_J.set_title(r"Histogram (density) vs. theoretical $\rho_\Omega(J)$")

                # ------------------ ζ histogram (density) ------------------
                # ζ = ln(Ω / J) for J>0; ζ >= 0
                zeta = np.log(Omega_cur / J_pos)
                zeta = np.maximum(zeta, 0.0)
                zmax = float(zeta.max())
                if zmax <= 0:
                    zmax = 1.0
                edges_z = np.linspace(0.0, zmax, nbins + 1)
                counts_z, edges_z = np.histogram(zeta, bins=edges_z)
                N_z = counts_z.sum()
                widths_z = edges_z[1:] - edges_z[:-1]
                centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])

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
                    label='Empirical density (counts / (N·Δζ))'
                )

                self.ax_zeta.set_xlabel(r'$\zeta = \ln(\Omega / J)$ (all bonds)')
                self.ax_zeta.set_ylabel("probability density")
                self.ax_zeta.set_title(r"Histogram (density) vs. theoretical $P_\Gamma(\zeta)$")

                # ------------------ Theoretical curves ------------------
                if self.Omega0 is not None and self.Omega0 > 0 and 0 < Omega_cur < self.Omega0:
                    Gamma = np.log(self.Omega0 / Omega_cur)
                    if Gamma > 1e-8:
                        # Theoretical ρ_Ω(J)
                        J_min_data = float(J_pos.min())
                        J_min_grid = max(J_min_data, Omega_cur * 1e-4)
                        J_grid = np.linspace(J_min_grid, Omega_cur, 400)
                        rho_th = (1.0 / (Omega_cur * Gamma)) * (Omega_cur / J_grid) ** (1.0 - 1.0 / Gamma)
                        self.ax_J.plot(
                            J_grid,
                            rho_th,
                            'r-',
                            lw=2,
                            label=r"Theory $\rho_\Omega(J)$ (density)"
                        )

                        # Theoretical P_Γ(ζ) = (1/Γ) e^{-ζ/Γ} on ζ ≥ 0
                        z_grid = np.linspace(0.0, zmax, 400)
                        P_th = (1.0 / Gamma) * np.exp(-z_grid / Gamma)
                        self.ax_zeta.plot(
                            z_grid,
                            P_th,
                            'r-',
                            lw=2,
                            label=r"Theory $P_\Gamma(\zeta)$ (density)"
                        )

                        # Annotate Ω and Γ (same in both plots)
                        info_text = f"$\\Omega$={Omega_cur:.3g}\n$\\Gamma$={Gamma:.3g}"
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

                self.ax_J.text(
                    0.02, 0.95,
                    f"SDRG step = {self.step_count}",
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

        # ----- 3D view (only if enabled and in animation mode) -----
        if self.mode_var.get() != "animation":
            return
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

    # --------------------- RG flow plot update -------------------------

    def update_rg_plot(self):
        self.ax_rg.clear()
        self.ax_rg.set_xlabel("SDRG step")
        self.ax_rg.set_ylabel(r'$\Omega$')
        if len(self.rg_steps) == 0:
            self.ax_rg.set_title("RG flow (no data)")
        else:
            self.ax_rg.plot(self.rg_steps, self.rg_omegas, marker='o', linestyle='-')
            self.ax_rg.set_title("RG flow")
        self.fig_rg.tight_layout()
        self.canvas_rg.draw_idle()

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

                h = (J_val / Jmax) * self.h_max
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

        self.ax3.text(
            y_label_x,
            y_label_y,
            f"bonds # = {L}",
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


# =============================== Main program =================================

def main():
    root = tk.Tk()
    app = SDRGEnsembleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
