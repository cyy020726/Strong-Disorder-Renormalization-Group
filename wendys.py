import numpy as np
import matplotlib.pyplot as plt



def generate_positions(N, L=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.sort(rng.uniform(0.0, L, size=N))
    x = x/x[0]
    return x

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

print(generate_positions(10))