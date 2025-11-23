import numpy as np
import matplotlib.pyplot as plt

# ---------- Disorder sampling (same as in burgerking.py) ----------

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
    else:
        J = rng.random((M, L))
    return J


# ---------- Single-chain SDRG with singlet tracking ----------

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
        Approximate finite-size excitation gap (smallest RG scale).
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

        # neighbor indices as in Ma–Dasgupta rule
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


# ---------- Ensemble driver ----------

def sdrg_ensemble(L, M, dist_name="Uniform(0,1)", seed=0, store_all_events=True):
    """
    Run SDRG on an ensemble of random chains and collect statistics.

    Parameters
    ----------
    L : int
        System size (number of sites).
    M : int
        Number of disorder realizations.
    dist_name : str
        Name of initial J distribution (match GUI to get same couplings).
    seed : int
        RNG seed.
    store_all_events : bool
        If True, store all singlet events (more memory).

    Returns
    -------
    gaps : np.ndarray, shape (M,)
        Approximate finite-size gaps for each sample (Δ_L ~ min Ω).
    all_lengths : np.ndarray or None
        Singlet lengths ℓ for all events in all samples (if store_all_events).
    all_omegas : np.ndarray or None
        Corresponding RG scales Ω for those events (if store_all_events).
    L0 : int
        System size (echo).
    """
    rng = np.random.default_rng(seed)
    J_ensemble = sample_couplings(L, M, dist_name, rng)

    gaps = np.empty(M, dtype=float)
    all_lengths = []
    all_omegas = []

    for s in range(M):
        J_init = J_ensemble[s, :]
        gap, events, L0 = sdrg_single_chain(J_init, store_all_events=store_all_events)
        gaps[s] = gap

        if store_all_events:
            for Omega, i_site, j_site in events:
                d = abs(j_site - i_site)
                d = min(d, L0 - d)  # periodic distance
                all_lengths.append(d)
                all_omegas.append(Omega)

    if store_all_events:
        return gaps, np.array(all_lengths, dtype=int), np.array(all_omegas, dtype=float), L0
    else:
        return gaps, None, None, L0


# ---------- Derived observables ----------

def avg_correlations_from_singlets(all_lengths, L0, M):
    """
    Compute disorder-averaged equal-time correlations from singlet-length distribution.

    SDRG approximation (random-singlet phase):
       <S_i · S_{i+r}>_avg ≈ -(3/4) P_singlet(r),

    where P_singlet(r) = probability that a pair at distance r is a singlet.

    Parameters
    ----------
    all_lengths : 1D array of ints
        Singlet distances ℓ from all samples.
    L0 : int
        Chain length.
    M : int
        Number of samples.

    Returns
    -------
    r_vals : np.ndarray
        Distances r = 0..L0//2.
    C_r : np.ndarray
        Average correlations <S_i·S_{i+r}>.
    """
    max_r = L0 // 2
    counts = np.zeros(max_r + 1, dtype=int)

    for d in all_lengths:
        if d <= max_r:
            counts[d] += 1

    # Each sample has L0 pairs (i, i+r) per r on a ring -> total M*L0
    total_pairs = M * L0
    P_r = counts / total_pairs
    C_r = -0.75 * P_r

    r_vals = np.arange(max_r + 1)
    return r_vals, C_r


def length_vs_energy_stats(all_lengths, all_omegas):
    """
    Convenience: return arrays for plotting or fitting ℓ vs Ω.

    Returns
    -------
    ℓ : np.ndarray
    Ω : np.ndarray
    """
    return np.asarray(all_lengths), np.asarray(all_omegas)

def plot_gap_distribution(gaps, bins=50, title="Gap Distribution"):
    plt.figure(figsize=(5,4))
    plt.hist(-np.log(gaps), bins=bins, density=True, alpha=0.7)
    plt.xlabel("Gap Δ")
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


# ---------- Minimal sanity check ----------

if __name__ == "__main__":
    L = 100
    M = 500
    gaps, all_lengths, all_omegas, L0 = sdrg_ensemble(
        L, M, dist_name="Uniform(0,1)", seed=123, store_all_events=True
    )
    print("Gaps:", gaps)
    print("Sample singlet lengths:", all_lengths[:10])
    print("Sample Omegas:", all_omegas[:10])
    plot_gap_distribution(gaps)