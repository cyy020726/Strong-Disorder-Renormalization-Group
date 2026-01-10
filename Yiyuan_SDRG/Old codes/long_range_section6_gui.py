"""
long_range_section6_gui.py

A heavily-annotated (and intentionally very explicit) GUI to validate the
*long-range Heisenberg chain* SDRG statistics against the **master-equation
distribution flow** discussed in Section 6 of the user's uploaded paper (SDRG.pdf)
and in the Kettemann long-range SDRG paper.

What this script does (high level)
----------------------------------
1) NUMERICS (blue histogram)
   We generate many disorder realizations ("samples") of a 1D chain on a ring.
   Each sample is specified by its *nearest-neighbor spacings* (gaps) r_i > 0.

   The long-range couplings are assumed monotonic with distance:
       J(r) = r^{-alpha}    (we set the microscopic prefactor J0 = 1)

   SDRG then repeatedly decimates the *strongest bond* (largest J), which is
   equivalent to the *smallest distance* among the current nearest-neighbor gaps.

   When the strongest bond (i,j) is removed as a singlet, the two neighboring
   distances R_L and R_R are merged into a new effective distance \tilde r
   between the outer sites (l,m). The key physical content of Kettemann's long-range
   SDRG is that this merged distance is *renormalized* from the naive sum
       r_bare = R_L + rho + R_R
   to a smaller effective distance \tilde r = f(R_L, R_R, rho) that depends on
   the anisotropy gamma and the power-law exponent alpha.

2) THEORY: "master equation evolution" (orange curve)
   The paper's master equation is a continuum description for how the *probability
   distribution of distances* evolves under SDRG, in the thermodynamic limit.

   Numerically, the simplest robust way to obtain the master-equation evolution is:
   run SDRG on a **very large ring** (N_theory sites) using the same decimation rule.
   In the large-N limit, this large-ring evolution approximates the master-equation flow.

   Important: the *initial condition* for the theory evolution is determined by
   the user's selected *analytic* input distribution (Gaussian, exponential, etc.),
   NOT by the observed histogram of the numerical SDRG samples.

3) THEORY: asymptotic fixed point (green dashed)
   In the approximation of Section 6 where distance renormalization is neglected
   inside the master equation, one obtains the strong-disorder fixed-point distance PDF:
       P_fp(r; rho) = rho^{1/2} / (2 r^{3/2}) * Theta(r - rho)

   Because we plot **log r** and **log J**, we convert this to PDFs in log variables:
       y = log r  =>  p_fp(y) = r P_fp(r) = (1/2) exp(-(y - log rho)/2),  y >= log rho
       z = log J  =>  p_fp(z) = J P_fp(J) = (1/(2 alpha)) Omega^{-1/(2 alpha)} exp(z/(2 alpha)),
                      z <= log Omega
   where Omega is the current SDRG energy scale and rho = Omega^{-1/alpha}.

Plots
-----
We produce two plots (stacked):
  (A) distribution of log r  (nearest-neighbor distances)
  (B) distribution of log J  (couplings J = r^{-alpha})

Colors requested by user:
  - Histogram (numerics): BLUE
  - Master-equation evolution curve: ORANGE
  - Fixed point curve: GREEN and dashed

Notes
-----
- This is a self-contained file (no external project dependencies).
- It uses Tkinter + Matplotlib.
- Computation is kept simple but reasonably efficient by using a heap-based SDRG
  implementation on a ring (linked list of gaps).

If you want this to *exactly* match a legacy GUI layout you already have, the
easiest path is to transplant the THEORY parts (orange/green overlays) into your
existing code. This file is designed to be readable and modifiable first.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
# 1) Small utilities
# ============================================================

def safe_int(x: str, default: int) -> int:
    """Parse int from a Tk string; fall back to default."""
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: str, default: float) -> float:
    """Parse float from a Tk string; fall back to default."""
    try:
        return float(x)
    except Exception:
        return default


def gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    Simple Gaussian smoothing in *bin-index space*.
    This is used only to make the orange "theory" curve visually smooth.

    We intentionally avoid SciPy to keep the file dependency-free.
    """
    if sigma_bins <= 0:
        return y
    # Build a symmetric kernel of ~ 6 sigma
    radius = int(max(3, math.ceil(3.0 * sigma_bins)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-0.5 * (xs / sigma_bins) ** 2)
    kern /= kern.sum()
    return np.convolve(y, kern, mode="same")


# ============================================================
# 2) Input distributions (sampling + analytic PDFs)
# ============================================================

DIST_NAMES = [
    "Constant",
    "Uniform[0,1]",
    "Exponential(mean=1)",
    "AbsGaussian(N(0,1))",
    "LogNormal(mu=0,sigma=1)",
    "Gamma(k=2,theta=1)",
    "Fermi-Dirac(f(x)∝1/(e^x+1))",
]


def sample_base_X(dist_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample the base nonnegative random variable X for the chosen distribution.
    We always produce X >= 0.

    The physical gap is then:  r = min_gap + scale * X
    """
    if dist_name == "Constant":
        return np.ones(n, dtype=float)
    if dist_name == "Uniform[0,1]":
        return rng.random(n)
    if dist_name == "Exponential(mean=1)":
        return rng.exponential(scale=1.0, size=n)
    if dist_name == "AbsGaussian(N(0,1))":
        return np.abs(rng.normal(loc=0.0, scale=1.0, size=n))
    if dist_name == "LogNormal(mu=0,sigma=1)":
        return rng.lognormal(mean=0.0, sigma=1.0, size=n)
    if dist_name == "Gamma(k=2,theta=1)":
        # shape=k=2, scale=theta=1
        return rng.gamma(shape=2.0, scale=1.0, size=n)
    if dist_name == "Fermi-Dirac(f(x)∝1/(e^x+1))":
        # We can sample by rejection from Exp(1) envelope (simple and robust).
        # Target density: f(x) = 1/(ln 2) * 1/(e^x+1), x>=0
        # Envelope: g(x)=e^{-x}, accept with prob proportional to f/g.
        out = np.empty(n, dtype=float)
        k = 0
        # f/g = e^x/(e^x+1) = 1/(1+e^{-x}) in (1/2, 1). max=1
        while k < n:
            x = rng.exponential(scale=1.0)
            u = rng.random()
            if u < (1.0 / (1.0 + math.exp(-x))):
                out[k] = x
                k += 1
        return out
    raise ValueError(f"Unknown dist_name: {dist_name}")


def base_pdf_X(dist_name: str, x: np.ndarray) -> np.ndarray:
    """
    Analytic PDF f_X(x) for the base variable X (defined on x>=0, except Uniform on [0,1]).
    Returned as numpy array with same shape as x.

    This is used to build a smooth initial theoretical curve.
    """
    x = np.asarray(x, dtype=float)
    fx = np.zeros_like(x)

    if dist_name == "Constant":
        # delta distribution; handled separately when plotting
        return fx

    if dist_name == "Uniform[0,1]":
        fx[(x >= 0.0) & (x <= 1.0)] = 1.0
        return fx

    if dist_name == "Exponential(mean=1)":
        m = x >= 0.0
        fx[m] = np.exp(-x[m])
        return fx

    if dist_name == "AbsGaussian(N(0,1))":
        m = x >= 0.0
        fx[m] = math.sqrt(2.0 / math.pi) * np.exp(-0.5 * x[m] ** 2)
        return fx

    if dist_name == "LogNormal(mu=0,sigma=1)":
        # f(x)=1/(x sqrt(2π)) exp(-(ln x)^2/2) for x>0
        m = x > 0.0
        fx[m] = (1.0 / (x[m] * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (np.log(x[m]) ** 2))
        return fx

    if dist_name == "Gamma(k=2,theta=1)":
        # f(x) = x e^{-x} for x>=0
        m = x >= 0.0
        fx[m] = x[m] * np.exp(-x[m])
        return fx

    if dist_name == "Fermi-Dirac(f(x)∝1/(e^x+1))":
        # normalized: ∫0∞ dx 1/(e^x+1) = ln 2
        m = x >= 0.0
        fx[m] = 1.0 / (math.log(2.0) * (np.exp(x[m]) + 1.0))
        return fx

    raise ValueError(f"Unknown dist_name: {dist_name}")


def gap_pdf_r(dist_name: str, r: np.ndarray, scale: float, min_gap: float) -> np.ndarray:
    """
    Analytic PDF of the physical gap r = min_gap + scale * X.

    f_r(r) = (1/scale) f_X((r-min_gap)/scale)
    """
    r = np.asarray(r, dtype=float)
    fr = np.zeros_like(r)
    if dist_name == "Constant":
        return fr
    x = (r - min_gap) / max(scale, 1e-300)
    fr = base_pdf_X(dist_name, x) / max(scale, 1e-300)
    # enforce support r>=min_gap
    fr[r < min_gap] = 0.0
    return fr


# ============================================================
# 3) SDRG decimation on a ring of gaps (heap + linked list)
# ============================================================

def gamma_renormalize(gamma: float) -> float:
    """
    Standard SDRG anisotropy flow for the XXZ chain (used in user's legacy code):
        gamma -> 0.5 * gamma^2 * (1 + gamma)
    """
    return 0.5 * gamma * gamma * (1.0 + gamma)


def renormalized_distance_nn(RL: float, rho: float, RR: float, alpha: float, gamma: float) -> float:
    """
    Nearest-neighbor SDRG effective coupling (mapped into an effective "distance").

    Legacy mapping used in earlier code:
        \tilde r = (1+gamma)^{1/alpha} * (RL * RR / rho)

    This comes from:
        J_eff = (1/(1+gamma)) * (J_L * J_R) / J_rho
    with J ~ r^{-alpha}, then solve for r.

    Note: This *is not* the long-range master-equation of Section 6; it is included only
    as an option for comparison.
    """
    return (max(1.0 + gamma, 1e-300) ** (1.0 / max(alpha, 1e-300))) * (RL * RR / max(rho, 1e-300))


def renormalized_distance_alltoall(RL: float, rho: float, RR: float, alpha: float, gamma: float) -> float:
    r"""
    Kettemann long-range geometric SDRG distance renormalization.

    We decimate the bond (i,j) of length rho. The two neighboring gaps are:
        R_L = distance(l,i)
        R_R = distance(j,m)

    The *bare* merged distance between l and m is:
        r_bare = R_L + rho + R_R.

    The *renormalized* distance is:
        \tilde r = r_bare * [ 1 + A ]^{-1/alpha}

    with
        A = (1/(1+gamma)) * ( (r_bare * rho)/(R_L * R_R) )^{-alpha}
            * [1 - (R_L/(R_L+rho))^{alpha}] * [1 - (R_R/(R_R+rho))^{alpha}]

    This matches the structure shown in SDRG.pdf (Section 6) and Kettemann's paper.
    """
    eps = 1e-300
    RL = max(RL, eps)
    RR = max(RR, eps)
    rho = max(rho, eps)
    r_bare = RL + rho + RR

    term_geom = (RL * RR / (r_bare * rho)) ** alpha  # NOTE: matches Eq. (35)/(36): ((r_bare*rho)/(RL*RR))^{-alpha}
    term_L = 1.0 - (RL / (RL + rho)) ** alpha
    term_R = 1.0 - (RR / (RR + rho)) ** alpha

    A = (1.0 / max(1.0 + gamma, eps)) * term_geom * term_L * term_R
    # if A is tiny, avoid numerical issues:
    if A < 1e-18:
        return r_bare
    return r_bare * (1.0 + A) ** (-1.0 / alpha)


@dataclass
class RingSDRG:
    """
    A single SDRG instance on a periodic chain (ring) represented purely by nearest-neighbor gaps.

    We do not store spin operators etc. We only need distances r between neighboring surviving spins,
    because couplings are monotonic in distance: J(r) = r^{-alpha}. The strongest coupling is the
    smallest gap.

    Internal representation:
      - We have N "gap nodes" arranged in a ring.
      - Each active node corresponds to one current nearest-neighbor gap.
      - We maintain:
          value[i]  = current gap length at node i
          left[i], right[i] = neighbor node indices in the ring of *active* nodes
          active[i] = whether node i is still present
      - A heap stores (gap_value, node_id, version_id) so we can pop the current minimum gap quickly.
    """

    values: np.ndarray
    left: np.ndarray
    right: np.ndarray
    active: np.ndarray
    version: np.ndarray
    heap: List[Tuple[float, int, int]]
    n_active: int
    alpha: float
    gamma: float
    mode: str  # "All-to-all" or "Nearest-neighbor"

    @classmethod
    def from_iid_gaps(
        cls,
        n_sites: int,
        dist_name: str,
        scale: float,
        min_gap: float,
        alpha: float,
        gamma: float,
        mode: str,
        rng: np.random.Generator,
    ) -> "RingSDRG":
        # Sample gaps (length = n_sites) because ring has same number of gaps as sites.
        X = sample_base_X(dist_name, n_sites, rng)
        gaps = min_gap + scale * X
        gaps = np.asarray(gaps, dtype=float)

        left = np.arange(n_sites, dtype=int) - 1
        left[0] = n_sites - 1
        right = np.arange(n_sites, dtype=int) + 1
        right[-1] = 0

        active = np.ones(n_sites, dtype=bool)
        version = np.zeros(n_sites, dtype=int)
        heap = [(float(gaps[i]), int(i), 0) for i in range(n_sites)]
        import heapq
        heapq.heapify(heap)

        return cls(
            values=gaps,
            left=left,
            right=right,
            active=active,
            version=version,
            heap=heap,
            n_active=n_sites,
            alpha=float(alpha),
            gamma=float(gamma),
            mode=mode,
        )

    def current_min_gap(self) -> Optional[Tuple[float, int]]:
        """Return (min_gap_value, node_id) for the current active configuration."""
        import heapq
        while self.heap:
            v, i, ver = self.heap[0]
            if (not self.active[i]) or (ver != self.version[i]):
                heapq.heappop(self.heap)
                continue
            return float(v), int(i)
        return None

    def current_Omega(self) -> float:
        """Current SDRG energy scale Omega = max bond strength = min_gap^{-alpha}."""
        mg = self.current_min_gap()
        if mg is None:
            return 0.0
        rho, _ = mg
        return rho ** (-self.alpha)

    def step_one_decimation(self) -> bool:
        """
        Perform ONE SDRG decimation (remove the strongest bond).

        Returns:
          True  if a decimation was performed
          False if the chain is too small to decimate further
        """
        if self.n_active < 3:
            return False  # need at least three gaps to have left/right neighbors

        mg = self.current_min_gap()
        if mg is None:
            return False
        rho, rho_id = mg

        # Identify neighboring gaps (left and right in the ring of gaps)
        RL_id = int(self.left[rho_id])
        RR_id = int(self.right[rho_id])

        if (not self.active[RL_id]) or (not self.active[RR_id]):
            # This should not happen if the linked list is consistent,
            # but we guard against corruption.
            return False

        RL = float(self.values[RL_id])
        RR = float(self.values[RR_id])
        rho = float(rho)

        # Compute the new effective distance
        if self.mode == "Nearest-neighbor":
            new_gap = renormalized_distance_nn(RL, rho, RR, self.alpha, self.gamma)
        else:
            new_gap = renormalized_distance_alltoall(RL, rho, RR, self.alpha, self.gamma)

        # Merge (RL, rho, RR) -> new_gap stored at RL_id.
        # Remove rho_id and RR_id from the active ring.
        L_of_RL = int(self.left[RL_id])
        R_of_RR = int(self.right[RR_id])

        # Update RL_id node to hold the merged gap and connect across.
        self.values[RL_id] = new_gap
        self.version[RL_id] += 1

        # Patch the ring links:
        self.left[RL_id] = L_of_RL
        self.right[RL_id] = R_of_RR
        self.right[L_of_RL] = RL_id
        self.left[R_of_RR] = RL_id

        # Deactivate the removed nodes:
        self.active[rho_id] = False
        self.active[RR_id] = False

        # Mark them as outdated in heap by bumping version (optional).
        self.version[rho_id] += 1
        self.version[RR_id] += 1

        self.n_active -= 2

        # Push updated RL_id into heap
        import heapq
        heapq.heappush(self.heap, (float(new_gap), int(RL_id), int(self.version[RL_id])))

        # Renormalize anisotropy
        self.gamma = gamma_renormalize(self.gamma)

        return True

    def all_active_gaps(self) -> np.ndarray:
        """Return an array of all currently active gaps (distances)."""
        return self.values[self.active].copy()


# ============================================================
# 4) Fixed point curves (green dashed)
# ============================================================

def fixed_point_logr_pdf(y: np.ndarray, rho: float) -> np.ndarray:
    """
    Fixed point PDF in y = log r (derived from P_fp(r;rho) = rho^{1/2} / (2 r^{3/2}) for r>=rho).

    p_fp(y) = r P_fp(r) = (1/2) exp(-(y - log rho)/2), for y >= log rho
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    y0 = math.log(max(rho, 1e-300))
    m = y >= y0
    out[m] = 0.5 * np.exp(-0.5 * (y[m] - y0))
    return out


def fixed_point_logJ_pdf(z: np.ndarray, Omega: float, alpha: float) -> np.ndarray:
    """
    Fixed point PDF in z = log J.

    Start from:
      P_fp(J;Omega) = (1/(2 alpha Omega)) * (Omega/J)^{1 - 1/(2 alpha)},  0 < J <= Omega.

    Convert to z:
      p_fp(z) = J P_fp(J) = (1/(2 alpha)) * Omega^{-1/(2 alpha)} * exp(z/(2 alpha)),
                for z <= log Omega.
    """
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    if Omega <= 0:
        return out
    zmax = math.log(Omega)
    m = z <= zmax
    out[m] = (1.0 / (2.0 * alpha)) * (Omega ** (-1.0 / (2.0 * alpha))) * np.exp(z[m] / (2.0 * alpha))
    return out


# ============================================================
# 5) Analytic initial curves (for step 0)
# ============================================================

def analytic_initial_logr_pdf(
    dist_name: str, y: np.ndarray, scale: float, min_gap: float
) -> np.ndarray:
    """
    Initial PDF in y=log r for the chosen analytic distribution.
    """
    y = np.asarray(y, dtype=float)
    r = np.exp(y)
    fr = gap_pdf_r(dist_name, r, scale, min_gap)
    # transform to log variable: p_y(y)=r f_r(r)
    return r * fr


def analytic_initial_logJ_pdf(
    dist_name: str, z: np.ndarray, scale: float, min_gap: float, alpha: float
) -> np.ndarray:
    """
    Initial PDF in z=log J for the chosen analytic distribution, where J = r^{-alpha}.
    """
    z = np.asarray(z, dtype=float)
    J = np.exp(z)
    # r = J^{-1/alpha}
    r = J ** (-1.0 / alpha)
    fr = gap_pdf_r(dist_name, r, scale, min_gap)
    # P_J(J) = f_r(r(J)) * |dr/dJ|, dr/dJ = -(1/alpha) J^{-1/alpha -1}
    PJ = fr * (1.0 / alpha) * (J ** (-1.0 / alpha - 1.0))
    # p_z(z) = J * P_J(J)
    return J * PJ


# ============================================================
# 6) Snapshot container
# ============================================================

@dataclass
class Snapshot:
    """
    One stored state at a given global decimation step index.
    """
    step: int
    Omega_rep: float                 # representative Omega (median over numeric samples)
    logr_values: np.ndarray          # pooled log r from numeric samples
    logJ_values: np.ndarray          # pooled log J from numeric samples
    logr_theory_curve: Tuple[np.ndarray, np.ndarray]  # (bin_centers, density) for orange curve
    logJ_theory_curve: Tuple[np.ndarray, np.ndarray]  # (bin_centers, density) for orange curve


# ============================================================
# 7) The GUI
# ============================================================

class LongRangeSection6GUI:
    """
    Tkinter GUI that runs:
      - Many small SDRG samples (blue histogram)
      - One large SDRG "theory" run (orange curve)
      - Fixed point overlay (green dashed)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Long-range SDRG validation (Section 6)")

        # -------------------------
        # Parameters (Tk variables)
        # -------------------------
        self.var_L = tk.StringVar(value="64")          # sites per sample
        self.var_M = tk.StringVar(value="200")         # number of samples
        self.var_alpha = tk.StringVar(value="1.2")     # power-law exponent
        self.var_gamma = tk.StringVar(value="1.0")     # XXZ anisotropy parameter
        self.var_dist = tk.StringVar(value=DIST_NAMES[3])  # AbsGaussian
        self.var_scale = tk.StringVar(value="1.0")
        self.var_min_gap = tk.StringVar(value="0.5")
        self.var_seed = tk.StringVar(value="0")
        self.var_mode = tk.StringVar(value="All-to-all")   # All-to-all (Kettemann)
        self.var_bins = tk.StringVar(value="60")

        # Theory knobs
        self.var_N_theory = tk.StringVar(value="40000")    # large ring size
        self.var_smooth_sigma = tk.StringVar(value="1.2")  # smoothing in bins

        # How many global decimation steps to compute/store
        self.var_steps = tk.StringVar(value="80")

        # -------------------------
        # Internal state
        # -------------------------
        self.snapshots: List[Snapshot] = []
        self.current_index = 0
        self.is_playing = False
        self.play_ms = 200  # time between frames in Play mode

        # -------------------------
        # Build layout
        # -------------------------
        self._build_layout()

    # -------------------------
    # Layout
    # -------------------------
    def _build_layout(self):
        # Main container
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # Left: plots, Right: controls
        left = ttk.Frame(outer)
        right = ttk.Frame(outer)

        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y")

        # --- Matplotlib figure (2 rows)
        self.fig = Figure(figsize=(8.0, 6.0), dpi=100)
        self.ax_r = self.fig.add_subplot(2, 1, 1)
        self.ax_J = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Controls
        lab = ttk.LabelFrame(right, text="Parameters", padding=10)
        lab.pack(fill="x")

        def row(label: str, var: tk.StringVar):
            fr = ttk.Frame(lab)
            fr.pack(fill="x", pady=2)
            ttk.Label(fr, text=label, width=18).pack(side="left")
            ttk.Entry(fr, textvariable=var, width=14).pack(side="right")

        row("L (sites)", self.var_L)
        row("M (samples)", self.var_M)
        row("alpha", self.var_alpha)
        row("gamma (init)", self.var_gamma)

        frd = ttk.Frame(lab)
        frd.pack(fill="x", pady=2)
        ttk.Label(frd, text="Distribution", width=18).pack(side="left")
        ttk.OptionMenu(frd, self.var_dist, self.var_dist.get(), *DIST_NAMES).pack(side="right", fill="x", expand=True)

        row("scale", self.var_scale)
        row("min_gap", self.var_min_gap)
        row("seed", self.var_seed)

        frm = ttk.Frame(lab)
        frm.pack(fill="x", pady=2)
        ttk.Label(frm, text="Connectivity", width=18).pack(side="left")
        ttk.OptionMenu(frm, self.var_mode, self.var_mode.get(), "All-to-all", "Nearest-neighbor").pack(side="right", fill="x", expand=True)

        row("# bins", self.var_bins)

        lab2 = ttk.LabelFrame(right, text="Theory (orange)", padding=10)
        lab2.pack(fill="x", pady=(10, 0))
        row2 = lambda l, v: row(l, v)  # reuse
        row2("N_theory", self.var_N_theory)
        row2("smooth_sigma", self.var_smooth_sigma)

        lab3 = ttk.LabelFrame(right, text="Run", padding=10)
        lab3.pack(fill="x", pady=(10, 0))
        row3 = lambda l, v: row(l, v)
        row3("snapshots (steps)", self.var_steps)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(10, 0))

        self.btn_compute = ttk.Button(btns, text="Compute", command=self.on_compute)
        self.btn_play = ttk.Button(btns, text="Play", command=self.on_play)
        self.btn_pause = ttk.Button(btns, text="Pause", command=self.on_pause)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop)

        self.btn_compute.pack(fill="x", pady=2)
        self.btn_play.pack(fill="x", pady=2)
        self.btn_pause.pack(fill="x", pady=2)
        self.btn_stop.pack(fill="x", pady=2)

        # Slider for snapshots
        self.slider = ttk.Scale(
            right, from_=0.0, to=0.0, orient="horizontal", command=self.on_slider
        )
        self.slider.pack(fill="x", pady=(10, 0))
        self.lbl_step = ttk.Label(right, text="Step: -")
        self.lbl_step.pack(fill="x")

        # Initial empty plot
        self._plot_empty()

    # -------------------------
    # Event handlers
    # -------------------------
    def on_compute(self):
        """
        Run the full computation and store snapshots.

        This is the "main button" for validation.
        """
        try:
            L = safe_int(self.var_L.get(), 64)
            M = safe_int(self.var_M.get(), 200)
            alpha = safe_float(self.var_alpha.get(), 1.2)
            gamma0 = safe_float(self.var_gamma.get(), 1.0)
            dist_name = self.var_dist.get()
            scale = safe_float(self.var_scale.get(), 1.0)
            min_gap = safe_float(self.var_min_gap.get(), 0.5)
            seed = safe_int(self.var_seed.get(), 0)
            mode = self.var_mode.get()
            n_bins = max(10, safe_int(self.var_bins.get(), 60))
            N_theory = max(2000, safe_int(self.var_N_theory.get(), 40000))
            smooth_sigma = max(0.0, safe_float(self.var_smooth_sigma.get(), 1.2))
            n_steps = max(1, safe_int(self.var_steps.get(), 80))
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        if L < 6:
            messagebox.showerror("Input error", "Please use L >= 6.")
            return
        if alpha <= 0:
            messagebox.showerror("Input error", "Please use alpha > 0.")
            return
        if scale < 0 or min_gap <= 0:
            messagebox.showerror("Input error", "Please use scale >= 0 and min_gap > 0.")
            return

        # Disable buttons during compute
        self._set_buttons_state("disabled")
        self.root.update_idletasks()

        t0 = time.time()

        # -------------------------
        # Build M numerical samples
        # -------------------------
        rng = np.random.default_rng(seed)
        samples: List[RingSDRG] = []
        for m in range(M):
            # independent seeds for each sample
            sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            samples.append(RingSDRG.from_iid_gaps(
                n_sites=L,
                dist_name=dist_name,
                scale=scale,
                min_gap=min_gap,
                alpha=alpha,
                gamma=gamma0,
                mode=mode,
                rng=sub_rng
            ))

        # -------------------------
        # Build one large "theory" run
        # -------------------------
        rng_th = np.random.default_rng(seed + 1234567)
        theory = RingSDRG.from_iid_gaps(
            n_sites=N_theory,
            dist_name=dist_name,
            scale=scale,
            min_gap=min_gap,
            alpha=alpha,
            gamma=gamma0,
            mode=mode,
            rng=rng_th
        )

        # -------------------------
        # Decide binning in log variables
        # -------------------------
        # We will pick fixed log-bins from initial data ranges.
        # This makes "same scale" comparisons easy: both hist and theory curve are evaluated on same bins.
        init_gaps = np.concatenate([s.all_active_gaps() for s in samples])
        init_y = np.log(init_gaps)

        init_J = init_gaps ** (-alpha)
        init_z = np.log(init_J)

        # Expand ranges a bit for safety
        y_min, y_max = float(np.min(init_y)), float(np.max(init_y))
        z_min, z_max = float(np.min(init_z)), float(np.max(init_z))
        y_pad = 0.15 * (y_max - y_min + 1e-12)
        z_pad = 0.15 * (z_max - z_min + 1e-12)
        y_bins = np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)
        z_bins = np.linspace(z_min - z_pad, z_max + z_pad, n_bins + 1)
        y_cent = 0.5 * (y_bins[:-1] + y_bins[1:])
        z_cent = 0.5 * (z_bins[:-1] + z_bins[1:])

        self.snapshots = []

        # -------------------------
        # Helper: compute pooled hist data from current samples
        # -------------------------
        def pooled_logs_from_samples() -> Tuple[np.ndarray, np.ndarray, float]:
            gaps_all = []
            Omegas = []
            for s in samples:
                mg = s.current_min_gap()
                if mg is not None:
                    Omegas.append(s.current_Omega())
                gaps_all.append(s.all_active_gaps())
            gaps = np.concatenate(gaps_all) if gaps_all else np.array([], dtype=float)
            gaps = gaps[gaps > 0]
            y = np.log(gaps)

            J = gaps ** (-alpha)
            z = np.log(J)

            Omega_rep = float(np.median(Omegas)) if Omegas else 0.0
            return y, z, Omega_rep

        # -------------------------
        # Helper: build orange theory curve
        # -------------------------
        def build_theory_curves(step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Returns:
              (y_cent, p_theory_y, z_cent, p_theory_z)

            At step=0 we use analytic PDFs (smooth).
            At step>0 we use the large-ring (theory) SDRG state, converted to a smooth line.
            """
            if step == 0:
                p_y = analytic_initial_logr_pdf(dist_name, y_cent, scale, min_gap)
                p_z = analytic_initial_logJ_pdf(dist_name, z_cent, scale, min_gap, alpha)
                # Normalize numerically over the displayed bins (purely for plotting robustness)
                p_y = p_y / max(np.trapz(p_y, y_cent), 1e-300)
                p_z = p_z / max(np.trapz(p_z, z_cent), 1e-300)
                return y_cent, p_y, z_cent, p_z

            # For step>0, use the *theory ring*'s current gaps:
            g = theory.all_active_gaps()
            g = g[g > 0]
            y = np.log(g)
            J = g ** (-alpha)
            z = np.log(J)

            # Histogram in log variables with density=True => this is pdf in y/z.
            hy, _ = np.histogram(y, bins=y_bins, density=True)
            hz, _ = np.histogram(z, bins=z_bins, density=True)

            # Smooth for a clean orange curve
            hy = gaussian_smooth_1d(hy, smooth_sigma)
            hz = gaussian_smooth_1d(hz, smooth_sigma)

            # Renormalize after smoothing (optional but keeps comparisons clean)
            hy = hy / max(np.trapz(hy, y_cent), 1e-300)
            hz = hz / max(np.trapz(hz, z_cent), 1e-300)

            return y_cent, hy, z_cent, hz

        # -------------------------
        # Main evolution loop over snapshots
        # -------------------------
        for step in range(n_steps):
            # 1) Snapshot data from samples at current step
            y_num, z_num, Omega_rep = pooled_logs_from_samples()

            # 2) Build theory curves for current step
            y_line_x, y_line, z_line_x, z_line = build_theory_curves(step)

            self.snapshots.append(Snapshot(
                step=step,
                Omega_rep=Omega_rep,
                logr_values=y_num,
                logJ_values=z_num,
                logr_theory_curve=(y_line_x, y_line),
                logJ_theory_curve=(z_line_x, z_line),
            ))

            # 3) Advance one SDRG decimation per sample (if possible)
            for s in samples:
                s.step_one_decimation()

            # 4) Advance theory ring by one decimation as well
            theory.step_one_decimation()

            # Keep UI responsive in long runs
            if step % 10 == 0:
                self.root.update_idletasks()

        # Update slider
        self.current_index = 0
        self.slider.configure(from_=0.0, to=max(0, len(self.snapshots) - 1))
        self.slider.set(0.0)
        self._render_snapshot(0)

        dt = time.time() - t0
        self.lbl_step.config(text=f"Computed {len(self.snapshots)} snapshots in {dt:.2f}s")

        self._set_buttons_state("normal")

    def on_play(self):
        """Start auto-advancing the snapshot slider."""
        if not self.snapshots:
            return
        self.is_playing = True
        self._play_tick()

    def on_pause(self):
        """Pause auto-play."""
        self.is_playing = False

    def on_stop(self):
        """Stop auto-play and return to the first snapshot."""
        self.is_playing = False
        if self.snapshots:
            self.current_index = 0
            self.slider.set(0.0)
            self._render_snapshot(0)

    def on_slider(self, val: str):
        """Slider callback (val is a string)."""
        if not self.snapshots:
            return
        idx = int(round(float(val)))
        idx = max(0, min(idx, len(self.snapshots) - 1))
        self.current_index = idx
        self._render_snapshot(idx)

    def _play_tick(self):
        if not self.is_playing or not self.snapshots:
            return
        idx = self.current_index + 1
        if idx >= len(self.snapshots):
            idx = 0
        self.current_index = idx
        self.slider.set(float(idx))
        self._render_snapshot(idx)
        self.root.after(self.play_ms, self._play_tick)

    # -------------------------
    # Plotting helpers
    # -------------------------
    def _plot_empty(self):
        self.ax_r.clear()
        self.ax_J.clear()
        self.ax_r.set_title("log r distribution")
        self.ax_J.set_title("log J distribution")
        self.ax_r.set_xlabel("log r")
        self.ax_J.set_xlabel("log J")
        self.ax_r.set_ylabel("density")
        self.ax_J.set_ylabel("density")
        self.fig.tight_layout()
        self.canvas.draw()

    def _render_snapshot(self, idx: int):
        snap = self.snapshots[idx]
        self.lbl_step.config(text=f"Step {snap.step}   (Omega_med ≈ {snap.Omega_rep:.4g})")

        # Parse params needed for fixed point
        alpha = safe_float(self.var_alpha.get(), 1.2)
        mode = self.var_mode.get()

        # Representative Omega and corresponding rho
        Omega = max(snap.Omega_rep, 1e-300)
        rho = Omega ** (-1.0 / alpha)

        # --- Panel A: log r
        self.ax_r.clear()
        self.ax_r.set_title("Nearest-neighbor distances: log r")
        self.ax_r.set_xlabel("log r")
        self.ax_r.set_ylabel("density")

        # Blue histogram (numerics)
        self.ax_r.hist(
            snap.logr_values,
            bins=int(max(10, safe_int(self.var_bins.get(), 60))),
            density=True,
            color="tab:blue",
            alpha=0.55,
            label="Numerics (hist)"
        )

        # Orange theory curve
        x_y, p_y = snap.logr_theory_curve
        self.ax_r.plot(x_y, p_y, color="tab:orange", linewidth=2.0, label="Master equation (large-ring)")

        # Green dashed fixed point (only meaningful for All-to-all long-range)
        if mode == "All-to-all":
            p_fp = fixed_point_logr_pdf(x_y, rho=rho)
            self.ax_r.plot(x_y, p_fp, color="tab:green", linestyle="--", linewidth=2.0, label="Fixed point")

        self.ax_r.legend(loc="best")

        # --- Panel B: log J
        self.ax_J.clear()
        self.ax_J.set_title("Couplings: log J (with J=r^{-alpha})")
        self.ax_J.set_xlabel("log J")
        self.ax_J.set_ylabel("density")

        self.ax_J.hist(
            snap.logJ_values,
            bins=int(max(10, safe_int(self.var_bins.get(), 60))),
            density=True,
            color="tab:blue",
            alpha=0.55,
            label="Numerics (hist)"
        )

        x_z, p_z = snap.logJ_theory_curve
        self.ax_J.plot(x_z, p_z, color="tab:orange", linewidth=2.0, label="Master equation (large-ring)")

        if mode == "All-to-all":
            pJ_fp = fixed_point_logJ_pdf(x_z, Omega=Omega, alpha=alpha)
            self.ax_J.plot(x_z, pJ_fp, color="tab:green", linestyle="--", linewidth=2.0, label="Fixed point")

        self.ax_J.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------
    # UI state helper
    # -------------------------
    def _set_buttons_state(self, state: str):
        for b in [self.btn_compute, self.btn_play, self.btn_pause, self.btn_stop]:
            try:
                b.configure(state=state)
            except Exception:
                pass


def main():
    root = tk.Tk()
    app = LongRangeSection6GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
