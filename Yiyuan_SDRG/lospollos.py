#!/usr/bin/env python3
"""
SDRG ground-state viewer (batch mode only).

- Sample an ensemble of random AFM chains (Heisenberg-like couplings).
- Run SDRG down to the last bond for each sample:
    - At each step, freeze the strongest bond into a singlet.
    - Keep track of all singlets (pairs of original site indices).
    - Track the last decimated (effective) bond energy as the "first excitation gap".
- Main window:
    - Top: ensemble singlet-based "correlation profile"
      C(r) = -3/4 * (number of singlets at distance r) / N, where N is total #singlets.
    - Bottom: histogram of the first excitation gaps across the ensemble.
- Separate window:
    - Visualizes singlet patterns for a selectable sample:
      spins on a circle, singlets as straight chords.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------- Sampling distributions ---------------------- #

def sample_fermi_dirac(shape, rng: np.random.Generator):
    """
    Very simple rejection sampler for a toy 'Fermi-Dirac-like'
    positive distribution, just to have another profile.
    (Not a physically exact FD function; used only as a random profile.)
    """
    size = int(np.prod(shape))
    out = np.empty(size, dtype=float)
    i = 0
    while i < size:
        # Exponential proposal
        x = rng.exponential(scale=1.0)
        u = rng.random()
        # Toy "Fermi-like" acceptance probability
        p_accept = np.exp(x) / (np.exp(x) + 1.0)
        if u < p_accept:
            out[i] = x
            i += 1
    return out.reshape(shape)


def sample_couplings(L: int, M: int, dist_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an M×L array of couplings J_{sample, bond}.

    We interpret J_row[i] as the coupling between site i and i+1 (mod L),
    so this is a periodic chain.
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
        # Fallback
        J = rng.random((M, L))
    return J


# ---------------------- GUI class ---------------------- #

class SDRGGroundStateGUI:
    def __init__(self, master):
        self.master = master
        master.title("SDRG ground-state viewer — singlet profile + gap distribution")

        # Ensemble data
        self.J_init = None              # (M, L) array of initial couplings
        self.singlets_per_sample = None # list of length M, each a list of (i,j) singlet pairs
        self.C_avg = None               # C_avg[r] = -3/4 * (#singlets at distance r)/N
        self.gaps = None                # array of length M: last decimated (effective) bond energy
        self.L = None
        self.M = None

        # Progress / status
        self.progressbar = None

        # Ground-state viewer (separate window)
        self.show_gs = tk.BooleanVar(value=False)
        self.gs_top = None
        self.fig_gs = None
        self.ax_gs = None
        self.canvas_gs = None
        self.gs_slider = None
        self.gs_sample_label = None
        self.gs_sample_index = 0

        # Layout: main frame
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- Controls on the left ---- #

        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(control_frame, text="Chain length L:").grid(row=0, column=0, sticky="w")
        self.entry_L = ttk.Entry(control_frame, width=8)
        self.entry_L.insert(0, "32")
        self.entry_L.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(control_frame, text="# samples M:").grid(row=0, column=2, sticky="w")
        self.entry_M = ttk.Entry(control_frame, width=8)
        self.entry_M.insert(0, "64")
        self.entry_M.grid(row=0, column=3, padx=4, pady=2)

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

        ttk.Label(control_frame, text="# bins (gap hist):").grid(row=2, column=0, sticky="w")
        self.entry_bins = ttk.Entry(control_frame, width=6)
        self.entry_bins.insert(0, "20")
        self.entry_bins.grid(row=2, column=1, padx=4, pady=2)

        # We keep an entry for distance but only for potential extension; unused in hist now.
        ttk.Label(control_frame, text="(distance entry unused)").grid(row=2, column=2, columnspan=2, sticky="w")

        # Toggle for ground-state viewer window
        self.check_gs = ttk.Checkbutton(
            control_frame,
            text="Show ground-state viewer",
            variable=self.show_gs,
            command=self.on_toggle_gs
        )
        self.check_gs.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 2))

        # Buttons
        self.button_sample = ttk.Button(control_frame, text="Sample ensemble", command=self.on_sample)
        self.button_sample.grid(row=4, column=0, padx=2, pady=4, sticky="ew")

        self.button_run = ttk.Button(control_frame, text="Run SDRG to GS", command=self.on_run_sdrg)
        self.button_run.grid(row=4, column=1, padx=2, pady=4, sticky="ew")

        # Status + progress bar
        self.status_label = ttk.Label(self.left_frame, text="Status: ready")
        self.status_label.pack(side=tk.TOP, anchor="w", pady=(4, 2))

        self.progressbar = ttk.Progressbar(self.left_frame, orient="horizontal", length=260, mode="determinate")
        self.progressbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 5))

        # ---- Correlation / gap plots on the right ---- #

        self.fig_corr = plt.Figure(figsize=(8, 6))
        gs = self.fig_corr.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
        self.ax_corr_avg = self.fig_corr.add_subplot(gs[0, 0])
        self.ax_corr_hist = self.fig_corr.add_subplot(gs[1, 0])

        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, master=self.right_frame)
        self.canvas_corr_widget = self.canvas_corr.get_tk_widget()
        self.canvas_corr_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.update_corr_plots()

        # Dummy entry_dist just to keep compatibility with earlier layout (not used)
        self.entry_dist = ttk.Entry(control_frame, width=6)
        self.entry_dist.insert(0, "1")
        self.entry_dist.grid_remove()  # hide it

    # ---------------------- Sampling + SDRG ---------------------- #

    def on_sample(self):
        """Sample a new ensemble of random couplings."""
        try:
            L = int(self.entry_L.get())
            M = int(self.entry_M.get())
            seed = int(self.entry_seed.get())
            if L <= 1 or M <= 0 or L % 2 != 0:
                # We want even L so that L/2 singlets per chain is natural.
                raise ValueError
        except ValueError:
            self.status_label.config(text="Status: invalid L or M (L must be even, L>1, M>0)")
            return

        self.L = L
        self.M = M

        dist_name = self.combo_dist.get()
        rng = np.random.default_rng(seed)
        self.J_init = sample_couplings(L, M, dist_name, rng)

        # Reset SDRG-derived data
        self.singlets_per_sample = None
        self.C_avg = None
        self.gaps = None

        self.progressbar['mode'] = 'determinate'
        self.progressbar['maximum'] = 1
        self.progressbar['value'] = 0

        self.status_label.config(text="Status: ensemble sampled (no SDRG yet)")
        self.update_corr_plots()
        self.update_gs_viewer()

    def on_run_sdrg(self):
        """Run SDRG down to the last bond for all samples (batch mode)."""
        if self.J_init is None:
            self.status_label.config(text="Status: sample ensemble first")
            return
        L = self.L
        M = self.M
        if L is None or M is None:
            self.status_label.config(text="Status: internal error (L,M)")
            return

        # Each chain of length L (even) eventually yields L/2 singlets.
        # Total decimations ~ M * (L/2).
        est_max_decim = M * (L // 2)
        if est_max_decim <= 0:
            est_max_decim = 1
        self.progressbar['mode'] = 'determinate'
        self.progressbar['maximum'] = est_max_decim
        self.progressbar['value'] = 0

        self.status_label.config(text="Status: running SDRG...")
        self.master.update_idletasks()

        singlets_all = []
        gaps_all = []
        decim_done = 0

        for s in range(M):
            J_row = self.J_init[s, :]
            singlets_s, gap_s = self.run_sdrg_single_chain(J_row, L)
            singlets_all.append(singlets_s)
            gaps_all.append(gap_s)
            decim_done += len(singlets_s)
            if decim_done > est_max_decim:
                decim_done = est_max_decim
            self.progressbar['value'] = decim_done
            self.master.update_idletasks()

        # Snap bar to full at the end
        if decim_done < est_max_decim:
            self.progressbar['value'] = est_max_decim

        self.singlets_per_sample = singlets_all
        self.gaps = np.array(gaps_all, dtype=float)
        self.status_label.config(text="Status: SDRG complete (ground-state singlets + gaps stored)")

        # Compute ensemble-averaged correlation profile from singlet distances
        self.compute_correlations()
        self.update_corr_plots()
        self.update_gs_viewer()

    def run_sdrg_single_chain(self, J_row, L):
        """
        Run SDRG for one chain (one sample) until no bonds remain.

        Representation:
        - Nodes: original sites {0,...,L-1}, we never relabel them.
        - Edges: dict mapping frozenset({i,j}) -> J_ij > 0.
        - Neighbors: adjacency sets.

        At each step:
        - Find edge with max J_ij.
        - Record singlet (i,j).
        - Find the remaining neighbors k of i and l of j (if any).
        - Generate effective coupling J_eff = J_ik J_jl / (2 J_ij) between k and l (if both exist).
        - Remove i and j and all incident edges.

        Returns:
            singlets: list of (i,j) singlet pairs
            final_gap: the last decimated J_max (first excitation gap scale)
        """
        edges = {}
        neighbors = {i: set() for i in range(L)}

        # Build initial periodic chain edges from J_row
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
        final_gap = 0.0  # default in case edges is empty

        while edges:
            # Pick the strongest bond
            key_max, J_max = max(edges.items(), key=lambda kv: kv[1])
            i, j = tuple(key_max)
            singlets.append((i, j))
            final_gap = float(J_max)  # update the last decimated energy scale

            # Neighbors excluding each other
            nei_i = set(neighbors.get(i, set()))
            if j in nei_i:
                nei_i.remove(j)
            nei_j = set(neighbors.get(j, set()))
            if i in nei_j:
                nei_j.remove(i)

            # Generate effective bond between k and l if both have external neighbors
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

            # Remove i and j, along with all incident edges
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

        return singlets, final_gap

    # ---------------------- Correlation computation ---------------------- #

    def compute_correlations(self):
        """
        From the singlet structure across the ensemble, compute:

        C_avg[r] = -3/4 * (number of singlet pairs at distance r) / N,

        where distance r is the minimal distance on the ring, and
        N is the total number of singlet pairs across all samples.
        """
        if self.singlets_per_sample is None or self.L is None or self.M is None:
            self.C_avg = None
            return

        L = self.L
        d_max = L // 2
        if d_max < 1:
            self.C_avg = None
            return

        counts = np.zeros(d_max + 1, dtype=int)
        total_pairs = 0

        for singlets in self.singlets_per_sample:
            total_pairs += len(singlets)
            for (i, j) in singlets:
                dx = abs(i - j)
                d = min(dx, L - dx)
                if 1 <= d <= d_max:
                    counts[d] += 1

        C_avg = np.zeros(d_max + 1, dtype=float)
        if total_pairs > 0:
            C_avg = -0.75 * counts / float(total_pairs)

        self.C_avg = C_avg

    # ---------------------- Correlation / gap plots ---------------------- #

    def get_hist_bins(self):
        try:
            nb = int(self.entry_bins.get())
            if nb < 1:
                nb = 1
        except ValueError:
            nb = 20
        return nb

    def update_corr_plots(self):
        """Refresh both plots in the main window."""
        self.ax_corr_avg.clear()
        self.ax_corr_hist.clear()

        if self.C_avg is None or self.L is None or self.gaps is None:
            # No data yet
            self.ax_corr_avg.set_title("Singlet-based correlation profile (no data)")
            self.ax_corr_avg.set_xlabel("|i - j| (minimal distance on ring)")
            self.ax_corr_avg.set_ylabel("C(r) = -3/4 · (singlets at distance r) / N")

            self.ax_corr_hist.set_title("Gap distribution (no data)")
            self.ax_corr_hist.set_xlabel("gap (last J_eff)")
            self.ax_corr_hist.set_ylabel("count")

            self.fig_corr.tight_layout()
            self.canvas_corr.draw_idle()
            return

        L = self.L
        d_max = L // 2
        distances = np.arange(1, d_max + 1)
        y = self.C_avg[1:d_max + 1]

        # Top plot: C(r) vs distance
        self.ax_corr_avg.plot(distances, y, marker='o', linestyle='-')
        self.ax_corr_avg.set_xlabel("|i - j| along chain (minimal on ring)")
        self.ax_corr_avg.set_ylabel("C(r) = -3/4 · (singlets at distance r) / N")
        self.ax_corr_avg.set_title("Ensemble singlet-based 'correlation' profile")

        # Bottom plot: histogram of first excitation gaps
        gaps = self.gaps
        gaps_pos = gaps[gaps > 0]
        nb = self.get_hist_bins()

        if gaps_pos.size == 0:
            self.ax_corr_hist.set_title("Gap distribution (no positive gaps)")
            self.ax_corr_hist.set_xlabel("gap (last J_eff)")
            self.ax_corr_hist.set_ylabel("count")
        else:
            g_min = float(gaps_pos.min())
            g_max = float(gaps_pos.max())
            if g_max == g_min:
                g_min -= 0.5 * abs(g_min) if g_min != 0 else -0.5
                g_max += 0.5 * abs(g_max) if g_max != 0 else 0.5
            self.ax_corr_hist.hist(
                gaps_pos,
                bins=nb,
                range=(g_min, g_max),
                edgecolor='black',
                alpha=0.7
            )
            self.ax_corr_hist.set_xlabel("First excitation gap (last J_eff)")
            self.ax_corr_hist.set_ylabel("count")
            self.ax_corr_hist.set_title("Distribution of first excitation gaps across ensemble")

            # Annotate mean gap
            mean_gap = float(np.mean(gaps_pos))
            self.ax_corr_hist.text(
                0.98, 0.95,
                f"mean gap ≈ {mean_gap:.3g}",
                transform=self.ax_corr_hist.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )

        self.fig_corr.tight_layout()
        self.canvas_corr.draw_idle()

    # ---------------------- Ground-state viewer window ---------------------- #

    def on_toggle_gs(self):
        """Open/close the ground-state viewer window."""
        if self.show_gs.get():
            if self.gs_top is None:
                self.create_gs_window()
            self.update_gs_viewer()
        else:
            if self.gs_top is not None:
                self.on_close_gs()

    def create_gs_window(self):
        """Create the separate window that shows singlet patterns per sample."""
        if self.gs_top is not None:
            return
        self.gs_top = tk.Toplevel(self.master)
        self.gs_top.title("SDRG ground-state singlet patterns (per sample)")

        fig_frame = ttk.Frame(self.gs_top)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_gs = plt.Figure(figsize=(6, 6))
        self.ax_gs = self.fig_gs.add_subplot(111)
        self.ax_gs.set_aspect('equal', adjustable='box')
        self.ax_gs.axis('off')

        self.canvas_gs = FigureCanvasTkAgg(self.fig_gs, master=fig_frame)
        self.canvas_gs_widget = self.canvas_gs.get_tk_widget()
        self.canvas_gs_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Slider for sample index
        slider_frame = ttk.Frame(self.gs_top)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        ttk.Label(slider_frame, text="Sample index:").pack(side=tk.LEFT)
        max_index = max(0, (self.M or 1) - 1)
        self.gs_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=max_index,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            command=self.on_gs_slider_change
        )
        self.gs_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.gs_sample_label = ttk.Label(slider_frame, text="(no data)")
        self.gs_sample_label.pack(side=tk.LEFT, padx=5)

        self.gs_top.protocol("WM_DELETE_WINDOW", self.on_close_gs)

    def on_close_gs(self):
        """Handle closing the ground-state viewer window."""
        try:
            if self.gs_top is not None:
                self.gs_top.destroy()
        except Exception:
            pass
        self.gs_top = None
        self.fig_gs = None
        self.ax_gs = None
        self.canvas_gs = None
        self.gs_slider = None
        self.gs_sample_label = None
        self.show_gs.set(False)

    def on_gs_slider_change(self, value):
        """Callback when the sample index slider changes."""
        try:
            idx = int(float(value))
        except ValueError:
            idx = 0
        if self.M is not None:
            idx = max(0, min(idx, self.M - 1))
        else:
            idx = 0
        self.gs_sample_index = idx
        self.update_gs_viewer()

    def update_gs_viewer(self):
        """Redraw the singlet pattern for the currently selected sample."""
        if not self.show_gs.get() or self.gs_top is None or self.ax_gs is None:
            return

        self.ax_gs.clear()
        self.ax_gs.set_aspect('equal', adjustable='box')
        self.ax_gs.axis('off')

        if self.L is None:
            self.fig_gs.tight_layout()
            self.canvas_gs.draw_idle()
            return

        L = self.L
        theta = np.linspace(0, 2 * np.pi, L, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        # Draw sites as dots
        self.ax_gs.scatter(x, y, s=20, c='k')

        # Label site indices around the circle
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
            s = max(0, min(self.gs_sample_index, self.M - 1))
            pairs = self.singlets_per_sample[s]
            for (i, j) in pairs:
                xi, yi = x[i], y[i]
                xj, yj = x[j], y[j]
                # Straight chord between the two sites
                self.ax_gs.plot(
                    [xi, xj],
                    [yi, yj],
                    color='C0',
                    linewidth=1.2,
                    alpha=0.8
                )
            self.ax_gs.set_title(f"Sample {s}: singlets (lines) on ring of sites")

        self.ax_gs.set_xlim(-1.3, 1.3)
        self.ax_gs.set_ylim(-1.3, 1.3)

        if self.gs_sample_label is not None:
            if self.singlets_per_sample is None or self.M is None:
                self.gs_sample_label.config(text="(no data)")
            else:
                self.gs_sample_label.config(text=f"{self.gs_sample_index} / {self.M - 1}")

        self.fig_gs.tight_layout()
        self.canvas_gs.draw_idle()


# ---------------------- Entry point ---------------------- #

def main():
    root = tk.Tk()
    app = SDRGGroundStateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
