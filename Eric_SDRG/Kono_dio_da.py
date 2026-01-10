import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1) Sampling nearest-neighbor distances (positive)
# =============================================================================

def sample_nn_distances(L: int, M: int, dist_name: str, rng: np.random.Generator,
                        *, shift_positive: float = 1e-6) -> np.ndarray:
    """
    Sample an ensemble of M chains, each with L nearest-neighbor distances (periodic ring).
    Returns array shape (M, L), with strictly positive distances.
    """
    if dist_name == "Uniform(0,1)":
        d = rng.random((M, L))
    elif dist_name == "Exponential(λ=1)":
        d = rng.exponential(scale=1.0, size=(M, L))
    elif dist_name == "Abs Gaussian N(0,1)":
        d = np.abs(rng.normal(loc=0.0, scale=1.0, size=(M, L)))
    elif dist_name == "Log-normal(μ=0,σ=1)":
        d = rng.lognormal(mean=0.0, sigma=1.0, size=(M, L))
    else:
        # fallback
        d = rng.random((M, L))

    # ensure strictly positive (avoid exact zeros)
    d = d + shift_positive
    return d


# =============================================================================
# 2) Kettemann f-function for the next-nearest-neighbor effective distance
#    (as written in your SDRG(1).pdf Eq. (38))
# =============================================================================


def _stable_one_minus_pow_inv1p(x: float, alpha: float) -> float:
    """
    Computes: 1 - (1/(1+x))^alpha  for x>=0 in a numerically stable way.
    """
    # (1/(1+x))^alpha = exp(-alpha * log(1+x))
    t = -alpha * np.log1p(x)
    # 1 - exp(t) = -expm1(t)  (stable when t ~ 0)
    return float(-np.expm1(t))

def f_kettemann(RL: float, RR: float, rho: float, alpha: float, gamma: float,
                eps: float = 1e-12) -> float:
    """
    Numerically stable Kettemann f(RL,RR,rho).
    eps clamps distances away from 0 to prevent division-by-zero/underflow.
    """
    RL = max(float(RL), eps)
    RR = max(float(RR), eps)
    rho = max(float(rho), eps)

    r = RL + RR + rho

    x = (r * rho) / (RL * RR)
    x = max(x, eps)
    x_malpha = x ** (-alpha)

    # Stable versions of:
    # aL = 1 - (1/(1+rho/RL))^alpha,  aR similar
    aL = _stable_one_minus_pow_inv1p(rho / RL, alpha)
    aR = _stable_one_minus_pow_inv1p(rho / RR, alpha)

    # Clamp away from 0 (still keeps physics: aL,aR in (0,1))
    aL = max(aL, eps)
    aR = max(aR, eps)

    prod_term = (aL * aR) ** (-1.0 / alpha)
    inside = 1.0 + 0.5 * x_malpha * prod_term

    d_new = r * (inside ** (-1.0 / alpha))

    return max(float(d_new), eps)


# =============================================================================
# 3) One RG run on one chain of NN distances
# =============================================================================

@dataclass
class RGSnapshot:
    # store data at some RG "time"
    step: int
    rho: float                 # current minimal distance (strongest bond distance)
    Omega: float               # current RG energy scale Omega = J0 * rho^{-alpha}
    zetas: np.ndarray          # zeta values for current NN couplings: zeta = ln(Omega/J_i)


def rg_single_chain_distances(
    d_init: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    mode: str,                        # "bare" or "kettemann"
    J0: float = 1.0,
    snapshot_steps: Optional[List[int]] = None,
) -> Tuple[float, List[RGSnapshot]]:
    """
    Perform the distance-based SDRG flow on a single periodic chain.

    Representation:
      d[k] = distance between site k and site k+1 (mod Nsites), length = Nbonds = Nsites

    Decimation:
      pick bond b with smallest distance rho = d[b] (largest coupling),
      define RL = d[b-1], RR = d[b+1] (mod),
      remove rho bond and its adjacent bond RR (two spins are removed => bond list shrinks by 2),
      replace RL by d_new where
         - bare: d_new = RL + rho + RR
         - kettemann: d_new = f_kettemann(RL, RR, rho)

    Returns:
      gap_proxy: smallest Omega encountered (last scale) as a proxy
      snapshots: list of RGSnapshot at requested steps
    """
    if mode not in ("bare", "kettemann"):
        raise ValueError("mode must be 'bare' or 'kettemann'")

    d = [float(x) for x in d_init]
    snapshots: List[RGSnapshot] = []
    snapshot_set = set(snapshot_steps or [])

    min_Omega = np.inf
    step = 0

    while len(d) >= 3:
        n = len(d)
        arr = np.maximum(np.array(d, dtype=float), 1e-12)

        # strongest bond = smallest distance
        b = int(arr.argmin())

        # rotate so that b -> 0 (decimated bond at index 0)
        # after rotation:
        #   rho = d[0]
        #   RR  = d[1]
        #   RL  = d[-1]
        d = list(np.roll(arr, -b))

        rho = float(d[0])
        RR  = float(d[1])
        RL  = float(d[-1])

        Omega = J0 * rho ** (-alpha)
        J_nn  = J0 * (np.array(d, dtype=float) ** (-alpha))
        zeta  = np.log(Omega / J_nn)

        if step in snapshot_set:
            snapshots.append(RGSnapshot(step=step, rho=rho, Omega=Omega, zetas=zeta.copy()))

        # compute new effective distance between the two outer neighbors
        if mode == "bare":
            d_new = RL + rho + RR
            d_new = max(d_new, 1e-12)
        else:
            d_new = f_kettemann(RL, RR, rho, alpha=alpha, gamma=gamma, eps=1e-12)

        # remove bonds 0 (rho) and 1 (RR), replace RL (which is last) by d_new
        # New ring bonds in order:
        #   [d_new] + old d[2:-1]
        d = [d_new] + d[2:-1]

        step += 1

    

    gap_proxy = float(min_Omega) if np.isfinite(min_Omega) else 0.0
    return gap_proxy, snapshots


# =============================================================================
# 4) Ensemble driver: run both modes, collect histograms vs fixed point
# =============================================================================

def fixed_point_P_zeta(zeta: np.ndarray, Gamma: float) -> np.ndarray:
    """
    Strong-disorder fixed point (finite-width) exponential in zeta:
      P(zeta) = (1/Gamma) exp(-zeta/Gamma)
    In Kettemann's long-range discussion the width tends to Gamma -> 2 alpha. :contentReference[oaicite:5]{index=5}
    """
    return (1.0 / Gamma) * np.exp(-zeta / Gamma)


def run_ensemble_compare(
    *,
    L: int,
    M: int,
    dist_name: str,
    alpha: float,
    gamma: float,
    seed: int = 0,
    snapshot_fracs: List[float] = (0.0, 0.25, 0.5, 0.75),
    nbins: int = 60,
):
    """
    Run an ensemble of distance chains, for both:
      - bare update
      - kettemann update
    and compare zeta histograms at selected RG "times".

    snapshot_fracs are fractions of the maximum number of steps ~ (L-3)/2.
    """
    rng = np.random.default_rng(seed)
    d_ens = sample_nn_distances(L=L, M=M, dist_name=dist_name, rng=rng)

    max_steps = (L - 3) // 2  # rough number of decimations until ~3 bonds remain
    snap_steps = sorted({int(frac * max_steps) for frac in snapshot_fracs})

    modes = ["bare", "kettemann"]
    all_snaps: Dict[str, Dict[int, List[np.ndarray]]] = {m: {s: [] for s in snap_steps} for m in modes}
    gaps: Dict[str, List[float]] = {m: [] for m in modes}

    for s in range(M):
        d0 = d_ens[s]

        for mode in modes:
            gap_proxy, snaps = rg_single_chain_distances(
                d0, alpha=alpha, gamma=gamma, mode=mode, snapshot_steps=snap_steps
            )
            gaps[mode].append(gap_proxy)
            for snap in snaps:
                all_snaps[mode][snap.step].append(snap.zetas)

    # ---- Plot: zeta distributions vs exponential fixed point with Gamma=2alpha ----
    Gamma_fp = 2.0 * alpha  # from Kettemann: finite width tends to 2α in the LR case. :contentReference[oaicite:6]{index=6}

    fig, axes = plt.subplots(len(snap_steps), 1, figsize=(7.0, 2.6 * len(snap_steps)), sharex=True)
    if len(snap_steps) == 1:
        axes = [axes]

    # define common binning based on pooled data
    pooled = []
    for mode in modes:
        for st in snap_steps:
            for z in all_snaps[mode][st]:
                pooled.append(z)
    if len(pooled) == 0:
        raise RuntimeError("No snapshots collected; increase L/M or adjust snapshot_fracs.")
    pooled = np.concatenate(pooled)
    zmax = np.quantile(pooled, 0.995)
    bins = np.linspace(0.0, max(1e-9, zmax), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for ax, st in zip(axes, snap_steps):
        for mode in modes:
            Z = np.concatenate(all_snaps[mode][st]) if len(all_snaps[mode][st]) else np.array([])
            if Z.size == 0:
                continue
            counts, _ = np.histogram(Z, bins=bins, density=True)
            ax.plot(centers, counts, label=f"{mode} (step {st})")

        # fixed point curve
        ax.plot(centers, fixed_point_P_zeta(centers, Gamma=Gamma_fp), linestyle="--", label=r"FP: $(1/\Gamma)e^{-\zeta/\Gamma}$")
        ax.set_ylabel("density")
        ax.set_title(f"ζ distribution at RG step {st} (L={L}, M={M}, dist={dist_name}, α={alpha}, γ={gamma})")
        ax.legend()

    axes[-1].set_xlabel(r"$\zeta=\ln(\Omega/J)$  (using NN couplings $J\propto r^{-\alpha}$)")
    plt.tight_layout()
    plt.show()

    # ---- Quick sanity stats on "gap proxy" ----
    gaps_bare = np.array(gaps["bare"])
    gaps_ket = np.array(gaps["kettemann"])
    print("Gap proxy (min Ω encountered) summary:")
    print(f"  bare     : median={np.median(gaps_bare):.4g}, mean={np.mean(gaps_bare):.4g}")
    print(f"  kettemann: median={np.median(gaps_ket):.4g}, mean={np.mean(gaps_ket):.4g}")


# =============================================================================
# 5) Example run
# =============================================================================
if __name__ == "__main__":
    run_ensemble_compare(
        L=128,
        M=200,
        dist_name="Abs Gaussian N(0,1)",
        alpha=2.0,
        gamma=1.0,
        seed=3,
        snapshot_fracs=[0.0, 0.2, 0.5, 0.8],
        nbins=70,
    )
