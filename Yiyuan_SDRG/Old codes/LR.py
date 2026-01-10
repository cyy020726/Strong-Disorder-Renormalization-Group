"""
long_range_section6_gui_mastereq_shells_v4.py

Fix vs v2/v3:
-------------
If you plot a density P(J) on a log-x axis, very large values are not "wrong" by
themselves, but they are typically not the quantity you want to compare visually.

For a log-scale x-axis, the natural object is the density in the log variable:
    v = log10(J),     P_v(v) = dP/dv = (ln 10) * J * P(J).

Similarly for r:
    u = log10(r),     P_u(u) = (ln 10) * r * P(r).

This file keeps *all* SDRG / master-equation evolution in true r and J variables.
We only do the above Jacobian conversion at the moment of plotting.

So:
- Numerics histogram is built in r and J (density wrt r and J).
- Theory curve is built in r and J.
- Fixed point is built in r and mapped to J.
- Plot shows the converted densities wrt log10(r) and log10(J) to avoid blow-ups
  and compare apples-to-apples on log x-axes.

Meaning of GUI parameters:
--------------------------
- dOmega: energy shell width in Omega. Shell n corresponds to Omega in:
          [Omega_low, Omega_high] with Omega_low = Omega_high - dOmega.
          Larger dOmega = fewer shells (faster, coarser). Smaller dOmega = more
          shells (slower, finer).
- N_theory: number of sites in the "theory" large ring used to represent the
            master-equation evolution as a Monte Carlo process. Larger N_theory
            = smoother orange curve, but slower.

"""

from __future__ import annotations

import math
import time
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Small "safe parse" helpers for GUI strings
# =============================================================================

def safe_int(s: str, default: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return int(default)

def safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return float(default)


# =============================================================================
# Distributions: sampling + analytic PDFs
# =============================================================================

DIST_NAMES = [
    "Regular lattice (constant)",
    "Uniform(0,1)",
    "Exponential(λ=1)",
    "Abs Gaussian N(0,1)",
    "Log-normal(μ=0,σ=1)",
    "Gamma(k=2,θ=1)",
    "Fermi-Dirac(μ=0,T=1)",
]

def sample_fermi_dirac_positive(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample X >= 0 from pdf  p(x) = 1/(ln2) * 1/(exp(x)+1).

    Rejection sampling:
      envelope g(x)=exp(-x)  (Exp(1))
      accept prob = exp(x)/(exp(x)+1)
    """
    out = np.empty(size, dtype=float)
    i = 0
    while i < size:
        x = rng.exponential(scale=1.0)
        u = rng.random()
        if u < (math.exp(x) / (math.exp(x) + 1.0)):
            out[i] = x
            i += 1
    return out

def sample_gaps(num_gaps: int,
                dist_name: str,
                rng: np.random.Generator,
                scale: float,
                min_gap: float) -> np.ndarray:
    """
    Sample i.i.d. positive gaps with a hard-core shift:
        gap = min_gap + scale * X,    X >= 0 from a base distribution.
    """
    num_gaps = int(num_gaps)
    if num_gaps <= 0:
        return np.zeros((0,), dtype=float)

    scale = max(float(scale), 0.0)
    min_gap = max(float(min_gap), 0.0)

    if dist_name == "Regular lattice (constant)":
        X = np.ones(num_gaps, dtype=float)

    elif dist_name == "Uniform(0,1)":
        X = rng.random(num_gaps)

    elif dist_name == "Exponential(λ=1)":
        X = rng.exponential(scale=1.0, size=num_gaps)

    elif dist_name == "Abs Gaussian N(0,1)":
        X = np.abs(rng.normal(loc=0.0, scale=1.0, size=num_gaps))

    elif dist_name == "Log-normal(μ=0,σ=1)":
        X = rng.lognormal(mean=0.0, sigma=1.0, size=num_gaps)

    elif dist_name == "Gamma(k=2,θ=1)":
        X = rng.gamma(shape=2.0, scale=1.0, size=num_gaps)

    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        X = sample_fermi_dirac_positive(num_gaps, rng)

    else:
        X = rng.random(num_gaps)

    gaps = min_gap + scale * np.asarray(X, dtype=float)
    return gaps


def analytic_gap_pdf(dist_name: str, r: np.ndarray, scale: float, min_gap: float) -> np.ndarray:
    """
    Analytic PDF for gaps r, consistent with:
        r = min_gap + scale * X,  with X >= 0.

    Returns p(r) (density w.r.t. r).
    """
    r = np.asarray(r, dtype=float)
    scale = max(float(scale), 1e-300)
    min_gap = max(float(min_gap), 0.0)

    x = (r - min_gap) / scale
    out = np.zeros_like(r)

    mask = x >= 0
    xm = x[mask]

    if dist_name == "Regular lattice (constant)":
        # approximate delta by narrow Gaussian
        r0 = min_gap + scale * 1.0
        sigma = max(0.02 * r0, 1e-6)
        out = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(-0.5 * ((r - r0) / sigma) ** 2)
        return out

    if dist_name == "Uniform(0,1)":
        out[mask] = (1.0 / scale) * ((xm >= 0) & (xm <= 1)).astype(float)

    elif dist_name == "Exponential(λ=1)":
        out[mask] = (1.0 / scale) * np.exp(-xm)

    elif dist_name == "Abs Gaussian N(0,1)":
        out[mask] = (1.0 / scale) * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * xm * xm)

    elif dist_name == "Log-normal(μ=0,σ=1)":
        m2 = xm > 0
        out_mask = np.zeros_like(xm)
        xx = xm[m2]
        out_mask[m2] = (1.0 / (xx * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (np.log(xx) ** 2))
        out[mask] = (1.0 / scale) * out_mask

    elif dist_name == "Gamma(k=2,θ=1)":
        out[mask] = (1.0 / scale) * (xm * np.exp(-xm))

    elif dist_name == "Fermi-Dirac(μ=0,T=1)":
        out[mask] = (1.0 / (scale * math.log(2.0))) * (1.0 / (np.exp(xm) + 1.0))

    else:
        out[mask] = (1.0 / scale) * ((xm >= 0) & (xm <= 1)).astype(float)

    return out


def gaussian_smooth_1d(h: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    Simple Gaussian smoothing on a 1D histogram array (convolution).
    """
    h = np.asarray(h, dtype=float)
    if sigma_bins <= 0:
        return h
    radius = int(max(1, round(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= max(np.sum(k), 1e-300)
    return np.convolve(h, k, mode="same")


def normalize_density_on_bins(p: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Normalize a bin-wise density p so that sum_k p_k * Δx_k = 1.
    This is the correct normalization for arrays that represent a histogram density
    on (possibly log-spaced) bins.
    """
    p = np.asarray(p, dtype=float)
    widths = np.diff(np.asarray(bin_edges, dtype=float))
    if p.size != widths.size:
        s = float(np.sum(p))
        return p / max(s, 1e-300)
    s = float(np.sum(p * widths))
    return p / max(s, 1e-300)


# =============================================================================
# SDRG update rules: distance renormalization
# =============================================================================

def gamma_renormalize(gamma: float) -> float:
    """
    gamma -> (gamma^2 (1+gamma))/2
    """
    g = float(gamma)
    return 0.5 * g * g * (1.0 + g)


def renormalized_distance_nn(RL: float, rho: float, RR: float, alpha: float, gamma: float) -> float:
    """
    Nearest-neighbor effective distance:
        r_tilde = (1+gamma)^{1/alpha} * (RL * RR / rho)
    """
    RL = max(float(RL), 1e-300)
    RR = max(float(RR), 1e-300)
    rho = max(float(rho), 1e-300)
    alpha = float(alpha)
    gamma = float(gamma)
    return ((1.0 + gamma) ** (1.0 / alpha)) * (RL * RR / rho)


def renormalized_distance_alltoall(RL: float, rho: float, RR: float, alpha: float, gamma: float) -> float:
    """
    Section-6 / Eq. (35)-(36) style f(RL, RR, rho).

    r_bare = RL + rho + RR

    term_geom = ((r_bare*rho)/(RL*RR))^{-alpha} = (RL*RR/(r_bare*rho))^{alpha}

    bracket factors:
      fL = 1 - (1/(1+rho/RL))^alpha
      fR = 1 - (1/(1+rho/RR))^alpha

    anisotropy prefactor:
      for gamma=1 this becomes 1/2 as in your Eq.(36).
    """
    RL = max(float(RL), 1e-300)
    RR = max(float(RR), 1e-300)
    rho = max(float(rho), 1e-300)
    alpha = float(alpha)
    gamma = float(gamma)

    r_bare = RL + rho + RR
    term_geom = (RL * RR / (r_bare * rho)) ** alpha

    fL = 1.0 - (1.0 / (1.0 + rho / RL)) ** alpha
    fR = 1.0 - (1.0 / (1.0 + rho / RR)) ** alpha

    A = (1.0 / (1.0 + gamma)) * term_geom * fL * fR
    if A < 1e-18:
        return r_bare
    return r_bare * (1.0 + A) ** (-1.0 / alpha)


# =============================================================================
# A fast ring SDRG implementation (heap + linked list)
# =============================================================================

@dataclass
class RingSDRG:
    values: np.ndarray
    left: np.ndarray
    right: np.ndarray
    active: np.ndarray
    version: np.ndarray
    heap: List[Tuple[float, int, int]]
    n_active: int
    alpha: float
    gamma: float
    mode: str  # "Nearest-neighbor" or "All-to-all"

    @staticmethod
    def from_iid_gaps(n_sites: int,
                     dist_name: str,
                     rng: np.random.Generator,
                     scale: float,
                     min_gap: float,
                     alpha: float,
                     gamma: float,
                     mode: str) -> "RingSDRG":
        n_sites = int(n_sites)
        gaps = sample_gaps(n_sites, dist_name, rng, scale, min_gap)
        values = np.array(gaps, dtype=float)

        left = np.arange(n_sites, dtype=int) - 1
        right = np.arange(n_sites, dtype=int) + 1
        left[0] = n_sites - 1
        right[-1] = 0

        active = np.ones(n_sites, dtype=bool)
        version = np.zeros(n_sites, dtype=int)

        heap: List[Tuple[float, int, int]] = []
        for i in range(n_sites):
            heap.append((float(values[i]), int(i), int(version[i])))
        heapq.heapify(heap)

        return RingSDRG(
            values=values,
            left=left,
            right=right,
            active=active,
            version=version,
            heap=heap,
            n_active=n_sites,
            alpha=float(alpha),
            gamma=float(gamma),
            mode=str(mode),
        )

    def all_active_gaps(self) -> np.ndarray:
        return self.values[self.active].copy()

    def current_min_gap(self) -> Optional[Tuple[float, int]]:
        while self.heap:
            v, idx, ver = self.heap[0]
            if (not self.active[idx]) or (ver != self.version[idx]):
                heapq.heappop(self.heap)
                continue
            return float(v), int(idx)
        return None

    def current_Omega(self) -> float:
        mg = self.current_min_gap()
        if mg is None:
            return 0.0
        rho, _ = mg
        rho = max(float(rho), 1e-300)
        return rho ** (-self.alpha)

    def step_one_decimation(self) -> bool:
        if self.n_active < 3:
            return False

        mg = self.current_min_gap()
        if mg is None:
            return False
        rho, rho_id = mg

        RL_id = int(self.left[rho_id])
        RR_id = int(self.right[rho_id])

        if (not self.active[RL_id]) or (not self.active[RR_id]):
            return False

        RL = float(self.values[RL_id])
        RR = float(self.values[RR_id])
        rho = float(rho)

        if self.mode == "Nearest-neighbor":
            new_gap = renormalized_distance_nn(RL, rho, RR, self.alpha, self.gamma)
        else:
            new_gap = renormalized_distance_alltoall(RL, rho, RR, self.alpha, self.gamma)

        # SDRG monotonicity guard: do not generate a gap smaller than the decimated one.
        eps = 1e-12
        if not np.isfinite(new_gap):
            new_gap = RL + rho + RR
        new_gap = max(float(new_gap), float(rho) * (1.0 + eps))

        L_of_RL = int(self.left[RL_id])
        R_of_RR = int(self.right[RR_id])

        self.values[RL_id] = float(new_gap)
        self.version[RL_id] += 1

        self.left[RL_id] = L_of_RL
        self.right[RL_id] = R_of_RR
        self.right[L_of_RL] = RL_id
        self.left[R_of_RR] = RL_id

        self.active[rho_id] = False
        self.active[RR_id] = False
        self.version[rho_id] += 1
        self.version[RR_id] += 1
        self.n_active -= 2

        heapq.heappush(self.heap, (float(new_gap), int(RL_id), int(self.version[RL_id])))

        self.gamma = gamma_renormalize(self.gamma)
        return True

    def decimate_until_Omega_below(self, Omega_low: float, max_steps: int = 10_000_000) -> int:
        Omega_low = float(max(Omega_low, 0.0))
        n = 0
        while n < max_steps and self.n_active >= 3:
            Om = self.current_Omega()
            if Om < Omega_low or Om <= 0:
                break
            if not self.step_one_decimation():
                break
            n += 1
        return n


# =============================================================================
# Fixed-point shape (Section 6)
# =============================================================================

def fixed_point_pdf_r(r: np.ndarray, rho: float) -> np.ndarray:
    """
    Section-6 fixed point in r: P(r) ∝ r^{-3/2} θ(r-rho).
    We keep shape and then normalize numerically on the plotting range.
    """
    r = np.asarray(r, dtype=float)
    rho = float(max(rho, 1e-300))
    out = np.zeros_like(r)
    mask = r >= rho
    out[mask] = r[mask] ** (-1.5)
    return out


# =============================================================================
# GUI
# =============================================================================

class Section6LongRangeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Section-6 Long-range SDRG validation (numerics vs master-equation flow)")

        # user parameters
        self.var_L = tk.StringVar(value="64")
        self.var_M = tk.StringVar(value="200")
        self.var_alpha = tk.StringVar(value="2.0")
        self.var_gamma = tk.StringVar(value="1.0")
        self.var_mode = tk.StringVar(value="All-to-all")
        self.var_dist = tk.StringVar(value="Abs Gaussian N(0,1)")
        self.var_scale = tk.StringVar(value="1.0")
        self.var_min_gap = tk.StringVar(value="0.5")
        self.var_seed = tk.StringVar(value="0")

        self.var_bins = tk.StringVar(value="60")

        self.var_dOmega = tk.StringVar(value="0.5")
        self.var_shells = tk.StringVar(value="80")

        self.var_N_theory = tk.StringVar(value="40000")
        self.var_smooth_sigma = tk.StringVar(value="1.2")

        # internal state
        self.snapshots: List[Dict] = []
        self.current_index = 0
        self.is_playing = False
        self.play_ms = 200

        self.r_bins: Optional[np.ndarray] = None
        self.J_bins: Optional[np.ndarray] = None
        self.r_cent: Optional[np.ndarray] = None
        self.J_cent: Optional[np.ndarray] = None

        self.fig = plt.Figure(figsize=(9.0, 7.0))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.32)
        self.ax_r = self.fig.add_subplot(gs[0, 0])
        self.ax_J = self.fig.add_subplot(gs[1, 0])

        self._build_layout()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        frm = ttk.LabelFrame(left, text="Parameters")
        frm.pack(fill=tk.X)

        def row(rr, label, var, width=10):
            ttk.Label(frm, text=label).grid(row=rr, column=0, sticky="w", pady=2)
            ent = ttk.Entry(frm, textvariable=var, width=width)
            ent.grid(row=rr, column=1, sticky="w", pady=2)
            return ent

        rr = 0
        row(rr, "L (sites):", self.var_L); rr += 1
        row(rr, "M (samples):", self.var_M); rr += 1
        row(rr, "alpha:", self.var_alpha); rr += 1
        row(rr, "gamma:", self.var_gamma); rr += 1

        ttk.Label(frm, text="Connectivity:").grid(row=rr, column=0, sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=self.var_mode, values=["Nearest-neighbor", "All-to-all"],
                     state="readonly", width=16).grid(row=rr, column=1, sticky="w", pady=2)
        rr += 1

        ttk.Label(frm, text="Initial gap dist:").grid(row=rr, column=0, sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=self.var_dist, values=DIST_NAMES,
                     state="readonly", width=20).grid(row=rr, column=1, sticky="w", pady=2)
        rr += 1

        row(rr, "scale:", self.var_scale); rr += 1
        row(rr, "min_gap:", self.var_min_gap); rr += 1
        row(rr, "seed:", self.var_seed); rr += 1
        row(rr, "# bins:", self.var_bins); rr += 1

        ttk.Separator(frm, orient="horizontal").grid(row=rr, column=0, columnspan=2, sticky="ew", pady=6)
        rr += 1

        row(rr, "dOmega:", self.var_dOmega); rr += 1
        row(rr, "# shells:", self.var_shells); rr += 1

        ttk.Separator(frm, orient="horizontal").grid(row=rr, column=0, columnspan=2, sticky="ew", pady=6)
        rr += 1

        row(rr, "N_theory:", self.var_N_theory, width=12); rr += 1
        row(rr, "smooth sigma:", self.var_smooth_sigma); rr += 1

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=(8, 2))

        self.btn_compute = ttk.Button(btns, text="Compute", command=self.on_compute)
        self.btn_compute.grid(row=0, column=0, padx=2, sticky="ew")
        self.btn_play = ttk.Button(btns, text="Play", command=self.on_play)
        self.btn_play.grid(row=0, column=1, padx=2, sticky="ew")
        self.btn_pause = ttk.Button(btns, text="Pause", command=self.on_pause)
        self.btn_pause.grid(row=0, column=2, padx=2, sticky="ew")
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop)
        self.btn_stop.grid(row=0, column=3, padx=2, sticky="ew")

        for c in range(4):
            btns.columnconfigure(c, weight=1)

        self.slider = tk.Scale(left, from_=0, to=0, orient=tk.HORIZONTAL,
                               showvalue=True, resolution=1, command=self.on_slider)
        self.slider.pack(fill=tk.X, pady=(6, 2))
        self.slider.configure(state="disabled")

        self.lbl_info = ttk.Label(left, text="No data yet.")
        self.lbl_info.pack(fill=tk.X, pady=(4, 2))

        canvas = FigureCanvasTkAgg(self.fig, master=right)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

        self._plot_empty()

    def _plot_empty(self):
        self.ax_r.clear()
        self.ax_J.clear()

        self.ax_r.set_title("Distance distribution (density in log10 r)")
        self.ax_r.set_xlabel("r (log x-axis)")
        self.ax_r.set_ylabel("dP/d log10(r)")

        self.ax_J.set_title("Coupling distribution (density in log10 J)")
        self.ax_J.set_xlabel("J (log x-axis)")
        self.ax_J.set_ylabel("dP/d log10(J)")

        self.ax_r.set_xscale("log")
        self.ax_J.set_xscale("log")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_compute(self):
        L = safe_int(self.var_L.get(), 64)
        M = safe_int(self.var_M.get(), 200)
        alpha = safe_float(self.var_alpha.get(), 2.0)
        gamma0 = safe_float(self.var_gamma.get(), 1.0)
        mode = self.var_mode.get()
        dist_name = self.var_dist.get()
        scale = safe_float(self.var_scale.get(), 1.0)
        min_gap = safe_float(self.var_min_gap.get(), 0.5)
        seed = safe_int(self.var_seed.get(), 0)

        nbins = safe_int(self.var_bins.get(), 60)
        dOmega = safe_float(self.var_dOmega.get(), 0.5)
        n_shells = safe_int(self.var_shells.get(), 80)

        N_theory = safe_int(self.var_N_theory.get(), 40000)
        smooth_sigma = safe_float(self.var_smooth_sigma.get(), 1.2)

        if L < 6 or M <= 0 or alpha <= 0 or gamma0 <= 0 or min_gap <= 0:
            messagebox.showerror("Input error", "Please use L>=6, M>0, alpha>0, gamma>0, min_gap>0.")
            return
        if nbins < 10:
            messagebox.showerror("Input error", "Please use #bins >= 10.")
            return
        if dOmega <= 0 or n_shells <= 0:
            messagebox.showerror("Input error", "Please use dOmega>0 and #shells>0.")
            return
        if N_theory < 2000:
            messagebox.showerror("Input error", "Please use N_theory >= 2000.")
            return

        self._set_buttons_state("disabled")
        self.root.update_idletasks()

        t0 = time.time()
        rng = np.random.default_rng(seed)

        # Numerical samples
        samples: List[RingSDRG] = []
        for _ in range(M):
            sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            samples.append(RingSDRG.from_iid_gaps(
                n_sites=L,
                dist_name=dist_name,
                rng=sub_rng,
                scale=scale,
                min_gap=min_gap,
                alpha=alpha,
                gamma=gamma0,
                mode=mode
            ))

        Omega0 = max(s.current_Omega() for s in samples)
        if Omega0 <= 0:
            messagebox.showerror("Compute error", "Omega0 computed as <= 0.")
            self._set_buttons_state("normal")
            return

        # Theory ring (large)
        theory_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        theory = RingSDRG.from_iid_gaps(
            n_sites=N_theory,
            dist_name=dist_name,
            rng=theory_rng,
            scale=scale,
            min_gap=min_gap,
            alpha=alpha,
            gamma=gamma0,
            mode=mode
        )

        # Bins in r and J (used for actual densities P(r), P(J))
        total_lengths = [float(np.sum(s.all_active_gaps())) for s in samples]
        r_min = max(min_gap * 0.8, 1e-12)
        r_max = max(max(total_lengths) * 1.2, r_min * 10.0)

        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
        r_cent = np.sqrt(r_bins[:-1] * r_bins[1:])
        self.r_bins = r_bins
        self.r_cent = r_cent

        J_min = r_max ** (-alpha)
        J_max = r_min ** (-alpha)
        J_bins = np.logspace(np.log10(J_min), np.log10(J_max), nbins + 1)
        J_cent = np.sqrt(J_bins[:-1] * J_bins[1:])
        self.J_bins = J_bins
        self.J_cent = J_cent

        LN10 = math.log(10.0)

        def pooled_pdf(samples_now: List[RingSDRG]) -> Tuple[np.ndarray, np.ndarray]:
            gaps = np.concatenate([s.all_active_gaps() for s in samples_now])
            gaps = gaps[gaps > 0]
            if gaps.size == 0:
                return np.zeros_like(r_cent), np.zeros_like(J_cent)

            hr, _ = np.histogram(gaps, bins=r_bins, density=True)

            J = gaps ** (-alpha)
            J = J[np.isfinite(J) & (J > 0)]
            hJ, _ = np.histogram(J, bins=J_bins, density=True)
            return hr, hJ

        def theory_pdf(step: int) -> Tuple[np.ndarray, np.ndarray]:
            if step == 0:
                pr = analytic_gap_pdf(dist_name, r_cent, scale, min_gap)
                pr = normalize_density_on_bins(pr, r_bins)

                r_of_J = J_cent ** (-1.0 / alpha)
                drdJ = (1.0 / alpha) * (J_cent ** (-1.0 / alpha - 1.0))
                pJ = analytic_gap_pdf(dist_name, r_of_J, scale, min_gap) * drdJ
                pJ = normalize_density_on_bins(pJ, J_bins)
                return pr, pJ

            gaps = theory.all_active_gaps()
            gaps = gaps[gaps > 0]
            if gaps.size == 0:
                return np.zeros_like(r_cent), np.zeros_like(J_cent)

            pr, _ = np.histogram(gaps, bins=r_bins, density=True)
            J = gaps ** (-alpha)
            J = J[np.isfinite(J) & (J > 0)]
            pJ, _ = np.histogram(J, bins=J_bins, density=True)

            pr = gaussian_smooth_1d(pr, smooth_sigma)
            pJ = gaussian_smooth_1d(pJ, smooth_sigma)

            pr = normalize_density_on_bins(pr, r_bins)
            pJ = normalize_density_on_bins(pJ, J_bins)
            return pr, pJ

        def fixed_point_curves(rho: float) -> Tuple[np.ndarray, np.ndarray]:
            pr = fixed_point_pdf_r(r_cent, rho=rho)
            pr = normalize_density_on_bins(pr, r_bins)

            r_of_J = J_cent ** (-1.0 / alpha)
            drdJ = (1.0 / alpha) * (J_cent ** (-1.0 / alpha - 1.0))
            pJ = fixed_point_pdf_r(r_of_J, rho=rho) * drdJ
            pJ = normalize_density_on_bins(pJ, J_bins)
            return pr, pJ

        snapshots: List[Dict] = []

        hr0, hJ0 = pooled_pdf(samples)
        tr0, tJ0 = theory_pdf(step=0)
        rho0 = Omega0 ** (-1.0 / alpha)
        fp_r0, fp_J0 = fixed_point_curves(rho=rho0)

        snapshots.append({
            "shell": 0,
            "Omega_high": Omega0,
            "Omega_low": max(0.0, Omega0 - dOmega),
            "rho": rho0,
            "hist_r": hr0,
            "hist_J": hJ0,
            "theory_r": tr0,
            "theory_J": tJ0,
            "fp_r": fp_r0,
            "fp_J": fp_J0,
        })

        for n in range(1, n_shells + 1):
            Omega_high = Omega0 - (n - 1) * dOmega
            Omega_low = max(0.0, Omega_high - dOmega)
            if Omega_high <= 0:
                break

            for s in samples:
                s.decimate_until_Omega_below(Omega_low)
            theory.decimate_until_Omega_below(Omega_low)

            hr, hJ = pooled_pdf(samples)
            tr, tJ = theory_pdf(step=n)

            rho = (Omega_low if Omega_low > 0 else max(Omega_high, 1e-300)) ** (-1.0 / alpha)
            fp_r, fp_J = fixed_point_curves(rho=rho)

            snapshots.append({
                "shell": n,
                "Omega_high": Omega_high,
                "Omega_low": Omega_low,
                "rho": rho,
                "hist_r": hr,
                "hist_J": hJ,
                "theory_r": tr,
                "theory_J": tJ,
                "fp_r": fp_r,
                "fp_J": fp_J,
            })

            if all(s.n_active < 3 for s in samples):
                break
            if n % 5 == 0:
                self.root.update_idletasks()

        self.snapshots = snapshots
        self.current_index = 0

        self.slider.configure(state="normal", from_=0, to=max(0, len(snapshots) - 1))
        self.slider.set(0)
        self._render_snapshot(0)

        dt = time.time() - t0
        self.lbl_info.config(
            text=f"Computed {len(snapshots)} shells in {dt:.2f}s. "
                 f"(N_theory={N_theory}, dOmega={dOmega})"
        )

        self._set_buttons_state("normal")

    def _render_snapshot(self, idx: int):
        if not self.snapshots or self.r_bins is None or self.J_bins is None or self.r_cent is None or self.J_cent is None:
            self._plot_empty()
            return

        LN10 = math.log(10.0)

        idx = int(max(0, min(idx, len(self.snapshots) - 1)))
        snap = self.snapshots[idx]

        r_bins = self.r_bins
        J_bins = self.J_bins
        r_cent = self.r_cent
        J_cent = self.J_cent

        r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])
        J_mid = np.sqrt(J_bins[:-1] * J_bins[1:])

        # Convert densities (wrt r/J) into densities wrt log10(r)/log10(J) for plotting:
        #   dP/d log10(r) = (ln 10) * r * P(r)
        #   dP/d log10(J) = (ln 10) * J * P(J)
        hist_r_plot = LN10 * r_mid * snap["hist_r"]
        theory_r_plot = LN10 * r_cent * snap["theory_r"]
        fp_r_plot = LN10 * r_cent * snap["fp_r"]

        hist_J_plot = LN10 * J_mid * snap["hist_J"]
        theory_J_plot = LN10 * J_cent * snap["theory_J"]
        fp_J_plot = LN10 * J_cent * snap["fp_J"]

        self.ax_r.clear()
        self.ax_J.clear()

        # r panel
        self.ax_r.bar(
            r_bins[:-1],
            hist_r_plot,
            width=np.diff(r_bins),
            align="edge",
            color="C0",
            alpha=0.35,
            edgecolor="C0",
            linewidth=0.4,
            label="Numerics (hist)",
            zorder=1,
        )
        self.ax_r.plot(r_cent, theory_r_plot, color="C1", linewidth=2.0, label="Master eq (MC / large ring)")
        self.ax_r.plot(r_cent, fp_r_plot, color="C2", linestyle="--", linewidth=2.0, label="Fixed point")

        self.ax_r.set_xscale("log")
        self.ax_r.set_title("Distance distribution (density in log10 r)")
        self.ax_r.set_xlabel("r (log x-axis)")
        self.ax_r.set_ylabel("dP/d log10(r)")
        self.ax_r.legend(loc="best", fontsize=9)

        # J panel
        self.ax_J.bar(
            J_bins[:-1],
            hist_J_plot,
            width=np.diff(J_bins),
            align="edge",
            color="C0",
            alpha=0.35,
            edgecolor="C0",
            linewidth=0.4,
            label="Numerics (hist)",
            zorder=1,
        )
        self.ax_J.plot(J_cent, theory_J_plot, color="C1", linewidth=2.0, label="Master eq (MC / large ring)")
        self.ax_J.plot(J_cent, fp_J_plot, color="C2", linestyle="--", linewidth=2.0, label="Fixed point")

        self.ax_J.set_xscale("log")
        self.ax_J.set_title("Coupling distribution (density in log10 J)")
        self.ax_J.set_xlabel("J (log x-axis)")
        self.ax_J.set_ylabel("dP/d log10(J)")
        self.ax_J.legend(loc="best", fontsize=9)

        # annotate RG scale
        self.ax_r.text(
            0.98, 0.96,
            f"shell={snap['shell']}\nΩ∈[{snap['Omega_low']:.3g},{snap['Omega_high']:.3g}]\nρ≈{snap['rho']:.3g}",
            transform=self.ax_r.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_slider(self, value):
        if not self.snapshots:
            return
        idx = safe_int(value, 0)
        self.current_index = idx
        self._render_snapshot(idx)

    def on_play(self):
        if not self.snapshots:
            messagebox.showinfo("Play", "Compute first.")
            return
        if self.is_playing:
            return
        self.is_playing = True
        self._play_tick()

    def _play_tick(self):
        if not self.is_playing:
            return
        nxt = self.current_index + 1
        if nxt >= len(self.snapshots):
            self.is_playing = False
            return
        self.slider.set(nxt)
        self.current_index = nxt
        self.root.after(self.play_ms, self._play_tick)

    def on_pause(self):
        self.is_playing = False

    def on_stop(self):
        self.is_playing = False
        if self.snapshots:
            self.slider.set(0)
            self.current_index = 0

    def _set_buttons_state(self, state: str):
        for b in [self.btn_compute, self.btn_play, self.btn_pause, self.btn_stop]:
            try:
                b.configure(state=state)
            except Exception:
                pass
        if state == "disabled":
            self.slider.configure(state="disabled")


def main():
    root = tk.Tk()
    app = Section6LongRangeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
