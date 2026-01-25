import numpy as np
import matplotlib.pyplot as plt


def sample_initial_distances(N_sites, kind="uniform", rng=None, **kwargs):
    """
    Return a 1D array 'd' of length N_sites - 1 with positive neighbor distances.

    kind:
      - "uniform": U[a, b]
      - "exponential": Exp(scale)
      - "lognormal": lognormal(mean, sigma)
    Extra parameters via kwargs:
      - uniform: a, b
      - exponential: scale
      - lognormal: mean, sigma
    """
    if rng is None:
        rng = np.random.default_rng()

    N_bonds = N_sites - 1

    if kind == "uniform":
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 2.0)
        d = rng.uniform(a, b, size=N_bonds)

    elif kind == "exponential":
        scale = kwargs.get("scale", 1.0)
        d = rng.exponential(scale=scale, size=N_bonds)

    elif kind == "lognormal":
        mean = kwargs.get("mean", 0.0)
        sigma = kwargs.get("sigma", 0.5)
        d = rng.lognormal(mean=mean, sigma=sigma, size=N_bonds)

    else:
        raise ValueError(f"Unknown distance distribution kind={kind}")

    return d


def sdrg_step_distances(d, N_sites):
    """
    One SDRG step in the pure distance picture.

    d: 1D numpy array of distances between neighboring active sites (length L = N_sites - 1)
    N_sites: current number of active sites.

    Returns:
      d_new: updated distances (numpy array)
      N_sites_new: updated site count
      d_dec: decimated (shortest) distance
    """
    L = len(d)
    if L == 0 or N_sites <= 1:
        return d, N_sites, None

    # find index of smallest distance
    k = int(np.argmin(d))
    d_dec = d[k]

    # edge cases vs interior
    if L == 1:
        # only one bond left; decimate it and remove the two sites
        d_new = np.array([], dtype=float)
        N_sites_new = N_sites - 2
        return d_new, N_sites_new, d_dec

    if k == 0:
        # decimate the first bond; remove first two segments: d[0], d[1]
        if L >= 2:
            d_new = d[2:].copy()
        else:
            d_new = np.array([], dtype=float)
        N_sites_new = N_sites - 2

    elif k == L - 1:
        # decimate the last bond; remove last two segments: d[L-2], d[L-1]
        if L >= 2:
            d_new = d[:-2].copy()
        else:
            d_new = np.array([], dtype=float)
        N_sites_new = N_sites - 2

    else:
        # interior bond: combine [k-1, k, k+1] into one
        d_new_segment = d[k-1] + d[k] + d[k+1]
        d_new = np.concatenate([d[:k-1], np.array([d_new_segment]), d[k+2:]])
        N_sites_new = N_sites - 2

    return d_new, N_sites_new, d_dec    


def run_distance_sdrg(N_sites, kind="uniform", rng=None, **dist_kwargs):
    """
    Run the pure distance SDRG until <= 1 site remains.

    Returns:
      decimated_d: array of decimated distances in order of RG steps
      u_values:    array of fractions of sites remaining at each step
    """
    if rng is None:
        rng = np.random.default_rng()

    # initial distances
    d = sample_initial_distances(N_sites, kind=kind, rng=rng, **dist_kwargs)
    N0 = N_sites
    N_curr = N_sites

    decimated_d = []
    u_values = []

    while N_curr > 1 and len(d) > 0:
        d, N_curr, d_dec = sdrg_step_distances(d, N_curr)
        if d_dec is None:
            break
        decimated_d.append(d_dec)
        u_values.append(N_curr / N0)

    return np.array(decimated_d), np.array(u_values)



def collect_sdrg_samples(
    N_sites=64, n_realizations=200, kind="uniform", rng=None, **dist_kwargs
):
    """
    Run distance SDRG for many realizations, collect all (u, d_dec) pairs.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_u = []
    all_d = []

    for _ in range(n_realizations):
        dec_d, u_vals = run_distance_sdrg(
            N_sites=N_sites, kind=kind, rng=rng, **dist_kwargs
        )
        all_u.append(u_vals)
        all_d.append(dec_d)

    # concatenate into single arrays
    if len(all_u) == 0:
        return np.array([]), np.array([])

    u_cat = np.concatenate(all_u)
    d_cat = np.concatenate(all_d)
    return u_cat, d_cat



def bin_by_u(u, d, u_bins):
    """
    u: array of fractions of sites remaining
    d: array of decimated distances (same length as u)
    u_bins: list/array of thresholds, e.g. [0.75, 0.5, 0.25, 0.125]

    Returns:
      d_per_bin: list of arrays of distances for each u-bin
      labels: string labels for each bin
    """
    edges = np.concatenate(([1.0], u_bins))  # e.g. [1.0, 0.75, 0.5, 0.25, 0.125]
    d_per_bin = []
    labels = []

    for k in range(len(edges) - 1):
        u_high = edges[k]
        u_low = edges[k + 1]
        mask = (u <= u_high) & (u > u_low)
        d_per_bin.append(d[mask])
        labels.append(f"{u_low:.3f} < u ≤ {u_high:.3f}")
    return d_per_bin, labels



def plot_distance_pdf_flow(d_per_bin, labels, nbins=30):
    """
    Plot log–log histograms of P(d) for different RG stages.
    """
    for d_vals, lab in zip(d_per_bin, labels):
        if len(d_vals) == 0:
            continue
        hist, edges = np.histogram(d_vals, bins=nbins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.loglog(centers, hist, marker='o', linestyle='-', label=lab)

    plt.xlabel("decimated distance d")
    plt.ylabel("P(d)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N_sites = 256
    n_realizations = 200

    # collect decimated distances for uniform initial distances
    u_cat, d_cat = collect_sdrg_samples(
        N_sites=N_sites,
        n_realizations=n_realizations,
        kind="uniform",
        rng=rng,
        a=1.0,
        b=2.0,
    )

    # define RG stages (fractions of sites remaining)
    u_bins = np.array([0.75, 0.5, 0.25, 0.125])

    d_per_bin, labels = bin_by_u(u_cat, d_cat, u_bins)

    # plot the evolving PDFs
    plot_distance_pdf_flow(d_per_bin, labels)