import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Patch


# ============================ Sampling the ensemble ============================

def sample_couplings(L: int, M: int, dist_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an M x L array of positive couplings J_{s,i} according
    to a chosen distribution. Each row = one Hamiltonian; each column = bond (periodic)
    """
    if dist_name == "Uniform(0,1)":
        J = rng.random((M, L))
    elif dist_name == "Exponential(λ=1)":
        J = rng.exponential(scale=1.0, size=(M, L))
    elif dist_name == "Abs Gaussian N(0,1)":
        J = np.abs(rng.normal(loc=0.0, scale=1.0, size=(M, L)))
    elif dist_name == "Log-normal(μ=0,σ=1)":
        J = rng.lognormal(mean=0.0, sigma=1.0, size=(M, L))
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


def run_sdrg_with_logging(J_init):
    """
    Previous code lacks tracking of length scales required for correlation length as well as energy vs. length plots
    This code runs the sdrg scheme on a 1 d periodic heisenberg chain while keeping track of the bond lengths
    
    J_init: 1D numpy array of initial couplings for one chain (periodic)
    
    We:
    initialize a new list keeping track of all sites. At first site korresponds to site_index, however as we remove bonds and shorten the sites array we shift positions around.
    this allows us to track length seperations between singlet. Since we keep the exact sites that couple to each other we can technically even reconstrcut the groundstate.
    
    
    Returns:
      gap: float, approximate finite-size gap
      events: list of (Omega, i_site, j_site)
    """
    
    
    L0 = len(J_init)
    # current bonds as Python list
    J_s = list(float(x) for x in J_init)
    # current site labels, in the same order as J_s
    sites = list(range(L0))

    events = []

    while len(J_s) >= 3:
        # find strongest bond
        arr = np.array(J_s, dtype=float)
        b = int(arr.argmax())
        Omega = J_s[b]

        n = len(J_s)
        i_site = sites[b]
        j_site = sites[(b + 1) % n]

        # record event
        events.append((Omega, i_site, j_site))

        # perform Ma–Dasgupta decimation of that bond
        J_s, new_index = decimate_chain_nonrotating(J_s, b)

        # update sites: remove the two spins that formed the singlet
        kill_a = b
        kill_b = (b + 1) % n
        killed = sorted([kill_a, kill_b])

        new_sites = []
        for k in range(n):
            if k in killed:
                continue
            new_sites.append(sites[k])
        sites = new_sites

    # gap = smallest energy scale in RG
    Omegas = [e[0] for e in events]
    gap = min(Omegas) if Omegas else 0.0

    return gap, events, L0


# ================================ GUI Class ===================================

class SDRGEnsembleGUI:
    def __init__(self, master):
        self.master = master
        master.title("SDRG ensemble viewer — J, ζ, RG flow + optional 3D view")

        # State
        self.J_init = None          # initial couplings: shape (M,L)
        self.chains = None          # list of lists, current per sample
        self.J_current = None       # current coupling matrix: (M, n_bonds)
        self.step_count = 0

        # SDRG animation state
        self.anim_running = False
        self.frame_delay_ms = 400   # ms between frames
        self.phase = 0              # 0 = choose bonds (highlight red), 1 = decimate
        self.pending_decimations = []  # list of (sample, bond_index) to decimate
        self.last_decimated = []       # bonds currently highlighted in red

        # RG flow tracking: steps vs Omega
        self.rg_steps = []
        self.rg_omegas = []

        # Height scale for vertical bars (set in update_plots)
        self.h_max = 1.0
        self.h_dir = np.array([0.0, 1.0])  # vertical direction in plotting coords (upward)

        # ---- Base geometry in its own coordinate system ----
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
        self.entry_M.insert(0, "8")
        self.entry_M.grid(row=0, column=3, padx=4, pady=2)

        # Row 1: distribution (span), seed
        ttk.Label(control_frame, text="Distribution:").grid(row=1, column=0, sticky="w")
        self.combo_dist = ttk.Combobox(
            control_frame,
            values=[
                "Uniform(0,1)",
                "Exponential(λ=1)",
                "Abs Gaussian N(0,1)",
                "Log-normal(μ=0,σ=1)"
            ],
            width=18,
            state="readonly"
        )
        self.combo_dist.current(0)
        self.combo_dist.grid(row=1, column=1, columnspan=3, padx=4, pady=2, sticky="w")

        ttk.Label(control_frame, text="Seed:").grid(row=1, column=4, sticky="w")
        self.entry_seed = ttk.Entry(control_frame, width=8)
        self.entry_seed.insert(0, "0")
        self.entry_seed.grid(row=1, column=5, padx=4, pady=2)

        # Row 2: bond index, bins
        ttk.Label(control_frame, text="Bond index i (1-based):").grid(row=2, column=0, sticky="w")
        self.entry_i = ttk.Entry(control_frame, width=8)
        self.entry_i.insert(0, "1")
        self.entry_i.grid(row=2, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="# bins:").grid(row=2, column=2, sticky="w")
        self.entry_bins = ttk.Entry(control_frame, width=6)
        self.entry_bins.insert(0, "30")
        self.entry_bins.grid(row=2, column=3, padx=4, pady=2)

        # Row 3: 3D toggle
        self.check_3d = ttk.Checkbutton(
            control_frame,
            text="Show 3D plot",
            variable=self.show_3d,
            command=self.on_toggle_3d
        )
        self.check_3d.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 2))

        # Row 4: buttons
        self.button_sample = ttk.Button(control_frame, text="Sample", command=self.on_sample_ensemble)
        self.button_sample.grid(row=4, column=0, padx=2, pady=4, sticky="ew")

        self.button_play = ttk.Button(control_frame, text="Play", command=self.on_play)
        self.button_play.grid(row=4, column=1, padx=2, pady=4, sticky="ew")

        self.button_pause = ttk.Button(control_frame, text="Pause", command=self.on_pause)
        self.button_pause.grid(row=4, column=2, padx=2, pady=4, sticky="ew")

        self.button_stop = ttk.Button(control_frame, text="Stop", command=self.on_stop)
        self.button_stop.grid(row=4, column=3, padx=2, pady=4, sticky="ew")

        # Current SDRG step label (below controls)
        self.step_label = ttk.Label(self.left_frame, text="Current SDRG step: 0")
        self.step_label.pack(side=tk.TOP, anchor="w", pady=(0, 4))

        # ------------------------ RG flow figure (left, bottom) -------------------------
        self.fig_rg = plt.Figure(figsize=(3, 3))
        self.ax_rg = self.fig_rg.add_subplot(111)
        self.ax_rg.set_xlabel("SDRG step")
        self.ax_rg.set_ylabel(r'$\Omega = \max J$')
        self.ax_rg.set_title("RG flow")

        self.canvas_rg = FigureCanvasTkAgg(self.fig_rg, master=self.left_frame)
        self.canvas_rg_widget = self.canvas_rg.get_tk_widget()
        self.canvas_rg_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ------------------------ Histogram figure (right) -------------------------
        # 2x2 grid: top row J, bottom row ζ
        self.fig_hist = plt.Figure(figsize=(10, 8))
        gs = self.fig_hist.add_gridspec(
            2, 2,
            height_ratios=[1, 1],
            width_ratios=[1, 1],
            hspace=0.35,
            wspace=0.3
        )

        self.ax_J_per = self.fig_hist.add_subplot(gs[0, 0])
        self.ax_J_all = self.fig_hist.add_subplot(gs[0, 1])
        self.ax_zeta_per = self.fig_hist.add_subplot(gs[1, 0])
        self.ax_zeta_all = self.fig_hist.add_subplot(gs[1, 1])

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.right_frame)
        self.canvas_hist_widget = self.canvas_hist.get_tk_widget()
        self.canvas_hist_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Note: 3D window is NOT created by default (show_3d = False)
        self.update_plots()

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
        # If user closes the window manually, uncheck the toggle.
        try:
            if self.top3d is not None:
                self.top3d.destroy()
        except Exception:
            pass
        self.top3d = None
        self.fig3d = None
        self.ax3 = None
        self.canvas3d = None
        # Sync toggle with reality
        self.show_3d.set(False)

    def on_toggle_3d(self):
        """Handle turning 3D plot on/off."""
        if self.show_3d.get():
            # Turn ON: create window if needed and redraw
            if self.top3d is None:
                self.create_3d_window()
            self.update_plots()
        else:
            # Turn OFF: close window if it exists
            if self.top3d is not None:
                self.on_close_3d()

    # ---------------------- SDRG state helpers -----------------------------

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
        # reset RG flow
        self.rg_steps = []
        self.rg_omegas = []

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

        self.anim_running = False
        self.reset_sdrg_state()
        self.update_plots()

    def on_play(self):
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
        self.reset_sdrg_state()
        self.update_plots()

    # --------------------------- SDRG animation -----------------------------

    def animate(self):
        if not self.anim_running:
            return
        if self.chains is None:
            return

        # Stop when all chains are too short to decimate
        if all(len(J_s) < 3 for J_s in self.chains):
            self.anim_running = False
            self.phase = 0
            self.pending_decimations = []
            self.last_decimated = []
            return

        if self.phase == 0:
            # Phase 0: choose bonds to decimate (highlight in red)
            self.pending_decimations = []
            self.last_decimated = []
            for s, J_s in enumerate(self.chains):
                if len(J_s) < 3:
                    continue
                arr = np.array(J_s, dtype=float)
                idx = int(np.argmax(arr))
                self.pending_decimations.append((s, idx))
                self.last_decimated.append((s, idx))

            self.update_plots()
            self.phase = 1

        else:
            # Phase 1: actually decimate
            for s, idx in self.pending_decimations:
                J_s = self.chains[s]
                if len(J_s) < 3:
                    continue
                new_J_s, _ = decimate_chain_nonrotating(J_s, idx)
                self.chains[s] = new_J_s

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

            self.last_decimated = []
            self.pending_decimations = []
            self.update_plots()
            self.phase = 0

        self.master.after(self.frame_delay_ms, self.animate)

    # --------------------------- Plot updating --------------------------------

    def get_nbins(self):
        try:
            nb = int(self.entry_bins.get())
            if nb < 1:
                nb = 1
        except ValueError:
            nb = 30
        return nb

    def update_plots(self):
        # Update SDRG step label
        self.step_label.config(text=f"Current SDRG step: {self.step_count}")

        # ----- Clear all histogram axes -----
        for ax in [self.ax_J_per, self.ax_J_all, self.ax_zeta_per, self.ax_zeta_all]:
            ax.clear()

        Omega_cur = None  # current RG scale (max J)
        Omega_init = None

        if self.J_current is None:
            # No data yet
            self.ax_J_per.set_title("Per-bond J distribution (no data)")
            self.ax_J_all.set_title("All-bonds J distribution (no data)")
            self.ax_zeta_per.set_title("Per-bond ζ distribution (no data)")
            self.ax_zeta_all.set_title("All-bonds ζ distribution (no data)")
            self.fig_hist.subplots_adjust(left=0.08, right=0.98,
                                          bottom=0.08, top=0.92, wspace=0.3, hspace=0.35)
            self.canvas_hist.draw_idle()
        else:
            J_cur = self.J_current
            M_cur, n_bonds_cur = J_cur.shape
            nbins = self.get_nbins()

            # Flatten current J and define Omega_cur = max J > 0
            all_J_cur = J_cur.ravel()
            J_pos_cur = all_J_cur[all_J_cur > 0]
            if J_pos_cur.size > 0:
                Omega_cur = float(J_pos_cur.max())
            else:
                Omega_cur = 0.0

            # Initial ensemble reference (shadow)
            J_init = self.J_init
            if J_init is not None:
                M_init, L_init = J_init.shape
                all_J_init = J_init.ravel()
                J_pos_init = all_J_init[all_J_init > 0]
                if J_pos_init.size > 0:
                    Omega_init = float(J_pos_init.max())
                else:
                    Omega_init = 0.0
            else:
                M_init, L_init = 0, 0
                all_J_init = np.array([])
                J_pos_init = np.array([])

            # ---------------------- J histograms ----------------------
            # Per-bond J_i: current vs initial (shadow)
            try:
                i_in = int(self.entry_i.get())
            except ValueError:
                i_in = 1

            # current index
            i0_cur = (i_in - 1) % max(1, n_bonds_cur)
            J_i_cur = J_cur[:, i0_cur]

            # initial index
            if J_init is not None and L_init > 0:
                i0_init = (i_in - 1) % L_init
                J_i_init = J_init[:, i0_init]
            else:
                J_i_init = np.array([])

            # bin edges for J_i from INITIAL data only (so shadow never changes)
            if J_i_init.size > 0:
                data_i = J_i_init
            else:
                data_i = J_i_cur

            if data_i.size > 0:
                jmax = float(data_i.max())
                jmin = 0.0
                if jmax == jmin:
                    jmax = jmin + 1.0
                edges_i = np.linspace(jmin, jmax, nbins + 1)
            else:
                edges_i = np.linspace(0.0, 1.0, nbins + 1)

            centers_i = 0.5 * (edges_i[:-1] + edges_i[1:])
            widths_i = edges_i[1:] - edges_i[:-1]

            # initial (shadow) per-bond J_i
            if J_i_init.size > 0:
                counts_i_init, _ = np.histogram(J_i_init, bins=edges_i)
                N_i_init = counts_i_init.sum()
                if N_i_init > 0:
                    probs_i_init = counts_i_init / N_i_init
                else:
                    probs_i_init = np.zeros_like(counts_i_init, dtype=float)
            else:
                probs_i_init = np.zeros(nbins, dtype=float)

            # current per-bond J_i
            if J_i_cur.size > 0:
                counts_i_cur, _ = np.histogram(J_i_cur, bins=edges_i)
                N_i_cur = counts_i_cur.sum()
                if N_i_cur > 0:
                    probs_i_cur = counts_i_cur / N_i_cur
                else:
                    probs_i_cur = np.zeros_like(counts_i_cur, dtype=float)
            else:
                probs_i_cur = np.zeros(nbins, dtype=float)

            # plot initial shadow (RED)
            self.ax_J_per.bar(
                centers_i,
                probs_i_init,
                width=widths_i,
                align='center',
                alpha=0.3,
                color='red',
                edgecolor='none',
                label='initial',
                zorder=0.5
            )
            # plot current (BLUE)
            self.ax_J_per.bar(
                centers_i,
                probs_i_cur,
                width=widths_i,
                align='center',
                alpha=0.8,
                color='C0',
                edgecolor='black',
                label='current',
                zorder=1.0
            )
            norm_i = float(probs_i_cur.sum())
            self.ax_J_per.set_xlabel(r'$J_{i}$')
            self.ax_J_per.set_ylabel('probability')
            self.ax_J_per.set_title('Per-bond J distribution')
            self.ax_J_per.text(
                0.98, 0.95,
                f"norm = {norm_i:.3f}",
                transform=self.ax_J_per.transAxes,
                ha='right',
                va='top',
                fontsize=9
            )
            self.ax_J_per.legend(loc='best', fontsize=8)

            # All-bonds J: current vs initial
            if all_J_init.size > 0:
                data_all = all_J_init
            else:
                data_all = all_J_cur

            if data_all.size > 0:
                jmax_all = float(data_all.max())
                jmin_all = 0.0
                if jmax_all == jmin_all:
                    jmax_all = jmin_all + 1.0
                edges_all = np.linspace(jmin_all, jmax_all, nbins + 1)
            else:
                edges_all = np.linspace(0.0, 1.0, nbins + 1)

            centers_all = 0.5 * (edges_all[:-1] + edges_all[1:])
            widths_all = edges_all[1:] - edges_all[:-1]

            # initial all-bonds
            if all_J_init.size > 0:
                counts_all_init, _ = np.histogram(all_J_init, bins=edges_all)
                N_all_init = counts_all_init.sum()
                if N_all_init > 0:
                    probs_all_init = counts_all_init / N_all_init
                else:
                    probs_all_init = np.zeros_like(counts_all_init, dtype=float)
            else:
                probs_all_init = np.zeros(nbins, dtype=float)

            # current all-bonds
            if all_J_cur.size > 0:
                counts_all_cur, _ = np.histogram(all_J_cur, bins=edges_all)
                N_all_cur = counts_all_cur.sum()
                if N_all_cur > 0:
                    probs_all_cur = counts_all_cur / N_all_cur
                else:
                    probs_all_cur = np.zeros_like(counts_all_cur, dtype=float)
            else:
                probs_all_cur = np.zeros(nbins, dtype=float)

            self.ax_J_all.bar(
                centers_all,
                probs_all_init,
                width=widths_all,
                align='center',
                alpha=0.3,
                color='red',
                edgecolor='none',
                label='initial',
                zorder=0.5
            )
            self.ax_J_all.bar(
                centers_all,
                probs_all_cur,
                width=widths_all,
                align='center',
                alpha=0.8,
                color='C0',
                edgecolor='black',
                label='current',
                zorder=1.0
            )
            norm_all = float(probs_all_cur.sum())
            self.ax_J_all.set_xlabel(r'$J$ (all bonds)')
            self.ax_J_all.set_ylabel('probability')
            self.ax_J_all.set_title('All-bonds J distribution')
            self.ax_J_all.text(
                0.98, 0.95,
                f"norm = {norm_all:.3f}",
                transform=self.ax_J_all.transAxes,
                ha='right',
                va='top',
                fontsize=9
            )
            self.ax_J_all.legend(loc='best', fontsize=8)

            # ---------------------- ζ histograms ----------------------
            # Per-bond ζ_i: current vs initial
            zeta_i_init = np.array([])
            zeta_i_cur = np.array([])

            if J_i_init.size > 0 and Omega_init is not None and Omega_init > 0:
                J_i_init_pos = J_i_init[J_i_init > 0]
                if J_i_init_pos.size > 0:
                    zeta_i_init = np.log(Omega_init / J_i_init_pos)

            if J_i_cur.size > 0 and Omega_cur is not None and Omega_cur > 0:
                J_i_cur_pos = J_i_cur[J_i_cur > 0]
                if J_i_cur_pos.size > 0:
                    zeta_i_cur = np.log(Omega_cur / J_i_cur_pos)

            # bin edges for ζ_i from INITIAL ζ_i only (constant shadow)
            if zeta_i_init.size > 0:
                data_zi = zeta_i_init
            else:
                data_zi = zeta_i_cur

            if data_zi.size > 0:
                zmin_i = float(data_zi.min())
                zmax_i = float(data_zi.max())
                if zmin_i == zmax_i:
                    zmin_i -= 0.5
                    zmax_i += 0.5
                edges_zi = np.linspace(zmin_i, zmax_i, nbins + 1)
            else:
                edges_zi = np.linspace(0.0, 1.0, nbins + 1)

            centers_zi = 0.5 * (edges_zi[:-1] + edges_zi[1:])
            widths_zi = edges_zi[1:] - edges_zi[:-1]

            if zeta_i_init.size > 0:
                counts_zi_init, _ = np.histogram(zeta_i_init, bins=edges_zi)
                N_zi_init = counts_zi_init.sum()
                if N_zi_init > 0:
                    probs_zi_init = counts_zi_init / N_zi_init
                else:
                    probs_zi_init = np.zeros_like(counts_zi_init, dtype=float)
            else:
                probs_zi_init = np.zeros(nbins, dtype=float)

            if zeta_i_cur.size > 0:
                counts_zi_cur, _ = np.histogram(zeta_i_cur, bins=edges_zi)
                N_zi_cur = counts_zi_cur.sum()
                if N_zi_cur > 0:
                    probs_zi_cur = counts_zi_cur / N_zi_cur
                else:
                    probs_zi_cur = np.zeros_like(counts_zi_cur, dtype=float)
            else:
                probs_zi_cur = np.zeros(nbins, dtype=float)

            self.ax_zeta_per.bar(
                centers_zi,
                probs_zi_init,
                width=widths_zi,
                align='center',
                alpha=0.3,
                color='red',
                edgecolor='none',
                label='initial',
                zorder=0.5
            )
            self.ax_zeta_per.bar(
                centers_zi,
                probs_zi_cur,
                width=widths_zi,
                align='center',
                alpha=0.8,
                color='C0',
                edgecolor='black',
                label='current',
                zorder=1.0
            )
            norm_zi = float(probs_zi_cur.sum())
            self.ax_zeta_per.set_xlabel(r'$\zeta_i = \ln(\Omega / J_i)$')
            self.ax_zeta_per.set_ylabel('probability')
            self.ax_zeta_per.set_title('Per-bond ζ distribution')
            self.ax_zeta_per.text(
                0.98, 0.95,
                f"norm = {norm_zi:.3f}",
                transform=self.ax_zeta_per.transAxes,
                ha='right',
                va='top',
                fontsize=9
            )
            self.ax_zeta_per.legend(loc='best', fontsize=8)

            # All-bonds ζ: current vs initial
            zeta_all_init = np.array([])
            zeta_all_cur = np.array([])

            if J_pos_init.size > 0 and Omega_init is not None and Omega_init > 0:
                zeta_all_init = np.log(Omega_init / J_pos_init)

            if J_pos_cur.size > 0 and Omega_cur is not None and Omega_cur > 0:
                zeta_all_cur = np.log(Omega_cur / J_pos_cur)

            if zeta_all_init.size > 0:
                data_za = zeta_all_init
            else:
                data_za = zeta_all_cur

            if data_za.size > 0:
                zmin_a = float(data_za.min())
                zmax_a = float(data_za.max())
                if zmin_a == zmax_a:
                    zmin_a -= 0.5
                    zmax_a += 0.5
                edges_za = np.linspace(zmin_a, zmax_a, nbins + 1)
            else:
                edges_za = np.linspace(0.0, 1.0, nbins + 1)

            centers_za = 0.5 * (edges_za[:-1] + edges_za[1:])
            widths_za = edges_za[1:] - edges_za[:-1]

            if zeta_all_init.size > 0:
                counts_za_init, _ = np.histogram(zeta_all_init, bins=edges_za)
                N_za_init = counts_za_init.sum()
                if N_za_init > 0:
                    probs_za_init = counts_za_init / N_za_init
                else:
                    probs_za_init = np.zeros_like(counts_za_init, dtype=float)
            else:
                probs_za_init = np.zeros(nbins, dtype=float)

            if zeta_all_cur.size > 0:
                counts_za_cur, _ = np.histogram(zeta_all_cur, bins=edges_za)
                N_za_cur = counts_za_cur.sum()
                if N_za_cur > 0:
                    probs_za_cur = counts_za_cur / N_za_cur
                else:
                    probs_za_cur = np.zeros_like(counts_za_cur, dtype=float)
            else:
                probs_za_cur = np.zeros(nbins, dtype=float)

            self.ax_zeta_all.bar(
                centers_za,
                probs_za_init,
                width=widths_za,
                align='center',
                alpha=0.3,
                color='red',
                edgecolor='none',
                label='initial',
                zorder=0.5
            )
            self.ax_zeta_all.bar(
                centers_za,
                probs_za_cur,
                width=widths_za,
                align='center',
                alpha=0.8,
                color='C0',
                edgecolor='black',
                label='current',
                zorder=1.0
            )
            norm_za = float(probs_za_cur.sum())
            self.ax_zeta_all.set_xlabel(r'$\zeta = \ln(\Omega / J)$ (all bonds)')
            self.ax_zeta_all.set_ylabel('probability')
            self.ax_zeta_all.set_title('All-bonds ζ distribution')
            self.ax_zeta_all.text(
                0.98, 0.95,
                f"norm = {norm_za:.3f}",
                transform=self.ax_zeta_all.transAxes,
                ha='right',
                va='top',
                fontsize=9
            )
            self.ax_zeta_all.legend(loc='best', fontsize=8)

            # Adjust layout
            self.fig_hist.subplots_adjust(left=0.08, right=0.98,
                                          bottom=0.08, top=0.92,
                                          wspace=0.3, hspace=0.35)
            self.canvas_hist.draw_idle()

        # ----- Update RG flow plot -----
        self.update_rg_plot(Omega_cur)

        # ----- 3D view (only if enabled) -----
        if not self.show_3d.get():
            return  # 3D is turned off

        # Ensure 3D window exists
        if self.top3d is None:
            self.create_3d_window()

        if self.ax3 is None:
            return

        # ------------- 3D-style view -------------
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

    def update_rg_plot(self, Omega):
        """Update RG flow data and redraw the RG plot."""
        # Update data arrays if we have a valid Omega and current J
        if self.J_current is not None and Omega is not None:
            if len(self.rg_steps) == 0 or self.rg_steps[-1] != self.step_count:
                self.rg_steps.append(self.step_count)
                self.rg_omegas.append(Omega)
            else:
                self.rg_omegas[-1] = Omega

        # Draw
        self.ax_rg.clear()
        self.ax_rg.set_xlabel("SDRG step")
        self.ax_rg.set_ylabel(r'$\Omega = \max J$')
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

                if (s, j) in decimated_set:
                    base_color = (1.0, 0.2, 0.2)
                else:
                    base_color = (0.2, 0.4, 1.0)

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

        # ====== Axis labels with matched perpendicular distance d ======
        d = 0.6 * margin_y

        # X label: below the middle of the bottom edge (between O and S)
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

        # Y label: perpendicular offset from the middle of the right edge (S–SB)
        right_mid = 0.5 * (S + SB)

        b_hat = b_dir / np.linalg.norm(b_dir)
        n_candidate = np.array([b_hat[1], -b_hat[0]])
        if n_candidate[0] < 0:
            n_candidate = -n_candidate
        n_hat = n_candidate

        y_label_pos = right_mid + d * n_hat
        y_label_x, y_label_y = y_label_pos

        label_angle = np.degrees(np.arctan2(b_dir[1], b_dir[0]))  # parallel to skewed axis

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
                          label='Decimated bonds')

        self.ax3.legend(handles=[blue_patch, red_patch],
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
