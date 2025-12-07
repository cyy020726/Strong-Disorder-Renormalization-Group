import numpy as np
import matplotlib.pyplot as plt



def generate_positions(N, L=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.sort(rng.uniform(0.0, L, size=N))
    x = x/x[0]
    return x


def generate_scaled_positions(N, L=1.0, rng=None):
    """
    Returns sorted positions x with min nearest-neighbor spacing rescaled to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.sort(rng.uniform(0.0, L, size=N))
    diffs = np.diff(x)
    dmin = np.min(diffs)
    x_scaled = x / dmin  # now min spacing = 1
    return x_scaled


def build_couplings_from_positions(x, alpha, J0=1.0):
    N = len(x)
    J = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i+1, N):
            r = abs(x[i] - x[j])
            # avoid infinity if two positions accidentally coincide
            if r == 0.0: # <------------this is shit coding floating point operations will cause this to almost never be zero
                continue
            J_ij = J0 / (r ** alpha)
            J[i, j] = J_ij
            J[j, i] = J_ij
    return J

def plot_chain_positions(x):
    N = len(x)
    y = np.zeros_like(x)

    plt.figure(figsize=(8, 1.5))
    plt.scatter(x, y, s=40)
    for i, xi in enumerate(x):
        plt.text(xi, 0.05, str(i), ha='center', va='bottom', fontsize=8)

    plt.yticks([])
    plt.xlabel("position x")
    plt.title("Random-distance 1D chain (min spacing = 1)")
    plt.tight_layout()
    plt.show()

positions = generate_scaled_positions(10)
print(positions)
plot_chain_positions(positions)
