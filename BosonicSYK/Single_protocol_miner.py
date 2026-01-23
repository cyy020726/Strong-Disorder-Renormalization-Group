#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chain_protocol_miner_periodic_even_sparse_v3.py

SDRG-on-SD (two-site SD decimation protocol) data miner for a *periodic* 1D chain
(even L only), with *sparse* Green's-function restoration consistent with the
appendix paragraph you pasted.

Key design choices (matching your protocol):
  1) Forward pass decimates the strongest bond B=(i,i+1) on a periodic active ring.
     At each decimation, we solve the *local two-site* SD fixed point exactly for B,
     with unknown on-site self-energies on the two decimated sites.

  2) We update only:
       - induced bond between the boundary sites A=(i-1,i+2) (frequency-dependent), and
       - shifts of boundary free kernels (frequency-dependent).

  3) We store into a buffer ONLY the local data needed for later restoration:
       - K_BB^{-1}(i\nu_m) (the 2n x 2n inverse kernel on the decimated block), and
       - K_AB, K_BA, represented here by the boundary couplings (mu_L, mu_R).

     We deliberately do NOT store K_AA or K_AA^eff at decimation time.

  4) After the forward pass reaches two remaining active sites, we solve that final
     two-site SD problem and obtain final {kappa_x^{(\infty)}, Sigma_x^{(\infty)}}.

  5) Restoration pass: for each buffered decimation record, we reconstruct K_AA from
     the final on-site data only, build the Schur complement K_AA^eff, invert it to get
     G_AA^eff, and then restore *only* G_AB, G_BB (and optionally G_BA) using the block
     inversion identities. We do NOT store any G_AA blocks to avoid double counting.

Output is an .npz file containing a sparse list of available Green's functions:
  - pairs_i, pairs_j: integer arrays of equal length P
  - G_pairs_iw: complex array with shape (M, P, nrep, nrep)

where G_pairs_iw[:,p,:,:] is G_{pairs_i[p], pairs_j[p]}(i\nu_m) as an nrep x nrep
replica matrix at each Matsubara frequency.

The accompanying plotting GUI (analysis script) can read this format.
"""

from __future__ import annotations

import json
import math
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import numpy as np


# ----------------------------
# Numerics helpers
# ----------------------------

def matsubara_bosonic(beta: float, M: int) -> np.ndarray:
    """Return the first M bosonic Matsubara frequencies: nu_m = 2*pi*m/beta, m=0..M-1."""
    m = np.arange(M, dtype=float)
    return (2.0 * np.pi / beta) * m


def _fft_forward_time_to_nu(x_tau: np.ndarray, beta: float) -> np.ndarray:
    """FFT convention: x(i nu_m) = \int_0^beta d tau e^{i nu_m tau} x(tau).

    With a uniform grid tau_j = j*beta/M and the discrete FFT, this is implemented as:
        x_nu[m] = (beta/M) * sum_j exp(+i 2pi m j/M) x_tau[j]
               = (beta/M) * FFT(x_tau)[m]

    We use numpy.fft.fft which has + sign in the exponent.
    """
    M = x_tau.shape[0]
    return (beta / M) * np.fft.fft(x_tau, axis=0)


def _fft_inverse_nu_to_time(x_nu: np.ndarray, beta: float) -> np.ndarray:
    """Inverse of _fft_forward_time_to_nu, returning x_tau on tau-grid.

        x_tau[j] = (1/beta) * sum_m exp(-i 2pi m j/M) x_nu[m]
                 = (1/beta) * IFFT(x_nu)[j] * M

    numpy.fft.ifft implements (1/M) sum_m exp(+i 2pi m j/M) x_nu[m].
    So we need exp(-i ...) which is complex conjugate; however if x_nu corresponds
    to the forward transform above, then the correct inverse is:
        x_tau = (1/beta) * np.fft.ifft(x_nu) * M
    because our forward used +i exponent.
    """
    M = x_nu.shape[0]
    return (M / beta) * np.fft.ifft(x_nu, axis=0)


def symmetrize_last2(A: np.ndarray) -> np.ndarray:
    """Hermitian-symmetrize the last two axes: (A + A^T)/2.  Works for real/complex."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def inf_norm_matrix(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat, ord=np.inf))


def safe_inv(mat: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Invert a square matrix with an optional ridge term."""
    if ridge > 0:
        mat = mat + ridge * np.eye(mat.shape[0], dtype=mat.dtype)
    return np.linalg.inv(mat)


# ----------------------------
# Two-site SD solver (general replica matrices)
# ----------------------------

@dataclass
class TwoSiteParams:
    beta: float
    M: int
    q: int
    J: float
    m2: float
    nrep: int
    mixing: float = 0.35
    tol: float = 1e-10
    max_iter: int = 250
    ridge: float = 1e-12
    enforce_symmetry: bool = True


def solve_two_site_sd(
    kappa1_nu: np.ndarray,   # (M,n,n)
    kappa2_nu: np.ndarray,   # (M,n,n)
    Omega_nu: np.ndarray,    # (M,n,n)
    p: TwoSiteParams,
    Sigma1_init_nu: np.ndarray | None = None,
    Sigma2_init_nu: np.ndarray | None = None,
    progress_cb=None,
):
    """Solve the two-site SD fixed point for sites (1,2).

    Kernel at each Matsubara frequency:
        K_BB = [[kappa1 - Sigma1,  -Omega],
                [-Omega,          kappa2 - Sigma2]]

    Closure (time domain) on *site diagonal* blocks:
        Sigma_x^{ab}(tau) = J^2 [G_xx^{ab}(tau)]^{q-1}

    Returns:
        Sigma1_nu, Sigma2_nu, GBB_nu (M,2n,2n)
    """

    M = p.M
    n = p.nrep

    if Sigma1_init_nu is None:
        Sigma1_nu = np.zeros((M, n, n), dtype=np.complex128)
    else:
        Sigma1_nu = Sigma1_init_nu.astype(np.complex128, copy=True)

    if Sigma2_init_nu is None:
        Sigma2_nu = np.zeros((M, n, n), dtype=np.complex128)
    else:
        Sigma2_nu = Sigma2_init_nu.astype(np.complex128, copy=True)

    # Pre-allocations
    GBB_nu = np.zeros((M, 2 * n, 2 * n), dtype=np.complex128)

    # Initial compute
    def compute_GBB(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
        out = np.zeros_like(GBB_nu)
        for m in range(M):
            A11 = kappa1_nu[m] - S1[m]
            A22 = kappa2_nu[m] - S2[m]
            K = np.zeros((2 * n, 2 * n), dtype=np.complex128)
            K[:n, :n] = A11
            K[n:, n:] = A22
            K[:n, n:] = -Omega_nu[m]
            K[n:, :n] = -Omega_nu[m]
            if p.enforce_symmetry:
                K = symmetrize_last2(K)
            out[m] = safe_inv(K, ridge=p.ridge)
            if p.enforce_symmetry:
                out[m] = symmetrize_last2(out[m])
        return out

    # Iteration
    for it in range(p.max_iter):
        GBB_nu = compute_GBB(Sigma1_nu, Sigma2_nu)

        # Extract site-diagonal blocks
        G11_nu = GBB_nu[:, :n, :n]
        G22_nu = GBB_nu[:, n:, n:]

        # To time domain
        G11_tau = _fft_inverse_nu_to_time(G11_nu, p.beta)
        G22_tau = _fft_inverse_nu_to_time(G22_nu, p.beta)

        # Closure
        # elementwise power (q-1), allowing complex (take principal branch)
        Sigma1_tau_new = (p.J ** 2) * (G11_tau ** (p.q - 1))
        Sigma2_tau_new = (p.J ** 2) * (G22_tau ** (p.q - 1))

        # Back to Matsubara
        Sigma1_nu_new = _fft_forward_time_to_nu(Sigma1_tau_new, p.beta)
        Sigma2_nu_new = _fft_forward_time_to_nu(Sigma2_tau_new, p.beta)

        if p.enforce_symmetry:
            Sigma1_nu_new = symmetrize_last2(Sigma1_nu_new)
            Sigma2_nu_new = symmetrize_last2(Sigma2_nu_new)

        # Mix
        a = p.mixing
        Sigma1_next = (1 - a) * Sigma1_nu + a * Sigma1_nu_new
        Sigma2_next = (1 - a) * Sigma2_nu + a * Sigma2_nu_new

        # Residual for stopping
        resid = max(
            float(np.max(np.abs(Sigma1_next - Sigma1_nu))),
            float(np.max(np.abs(Sigma2_next - Sigma2_nu))),
        )

        Sigma1_nu, Sigma2_nu = Sigma1_next, Sigma2_next

        if progress_cb is not None:
            progress_cb(it + 1, p.max_iter, resid)

        if resid < p.tol:
            break

    # Final G
    GBB_nu = compute_GBB(Sigma1_nu, Sigma2_nu)
    return Sigma1_nu, Sigma2_nu, GBB_nu


# ----------------------------
# Protocol: forward decimation + sparse restoration
# ----------------------------


def _make_base_kappa(beta: float, M: int, nrep: int, m2: float) -> np.ndarray:
    nu = matsubara_bosonic(beta, M)
    base = np.zeros((M, nrep, nrep), dtype=np.complex128)
    for m in range(M):
        base[m] = (nu[m] ** 2 + m2) * np.eye(nrep, dtype=np.complex128)
    return base


def _seed_sigma_from_tau_constants(beta: float, M: int, nrep: int, sig_d_tau: float, sig_od_tau: float) -> np.ndarray:
    """Build a replica-symmetric Sigma(τ) constant in τ, then FFT to Matsubara."""
    Sig_tau = np.zeros((M, nrep, nrep), dtype=np.complex128)
    mat = np.full((nrep, nrep), sig_od_tau, dtype=np.complex128)
    np.fill_diagonal(mat, sig_d_tau)
    for j in range(M):
        Sig_tau[j] = mat
    Sig_nu = _fft_forward_time_to_nu(Sig_tau, beta)
    Sig_nu = symmetrize_last2(Sig_nu)
    return Sig_nu


def _bond_strength(mu_nu: np.ndarray) -> float:
    # mu_nu shape (M,n,n)
    return float(sum(inf_norm_matrix(mu_nu[m]) for m in range(mu_nu.shape[0])))


def run_protocol_periodic_even_sparse(
    L: int,
    mu_init: list[np.ndarray],
    p: TwoSiteParams,
    sig_d_tau_init: float,
    sig_od_tau_init: float,
    progress_cb=None,
):
    """Run the periodic-even protocol and return sparse Green's functions.

    Parameters
    ----------
    L : even integer >= 4
    mu_init : list of length L, each element mu_x(i nu) as (M,n,n)
              representing the directed bond from site x to site x+1 (mod L).
              Initial mu_x must be uniform replica diagonal as requested, but
              the protocol will generate full replica matrices after decimation.

    Returns dict with keys suitable for np.savez.
    """

    if L < 4 or (L % 2) != 0:
        raise ValueError("This miner is restricted to even L >= 4.")

    M = p.M
    n = p.nrep

    # Base free kernel and cumulative shifts
    base_kappa = _make_base_kappa(p.beta, M, n, p.m2)
    delta_kappa = np.zeros((M, L, n, n), dtype=np.complex128)

    # Final on-site self-energies (filled when a site is decimated / terminal)
    Sigma_final = np.zeros((M, L, n, n), dtype=np.complex128)
    Sigma_known = np.zeros((L,), dtype=bool)

    # Seed for two-site solver
    Sigma_seed_nu = _seed_sigma_from_tau_constants(p.beta, M, n, sig_d_tau_init, sig_od_tau_init)

    # Active periodic ring representation
    sites: list[int] = list(range(L))
    bonds: list[np.ndarray] = [mu.copy() for mu in mu_init]

    # Buffer records
    # Each record stores site ids (i,j,L,R) and the local data for restoration.
    buffer = []

    # Progress bookkeeping
    n_steps_forward = (L // 2) - 1  # number of decimations until 2 sites remain

    def wrap_progress(stage: str, step: int, stepmax: int, it: int, itmax: int, resid: float):
        if progress_cb is not None:
            progress_cb(stage, step, stepmax, it, itmax, resid)

    # Forward decimation
    for step in range(n_steps_forward):
        N = len(sites)
        assert N == len(bonds)

        # Choose strongest bond index
        strengths = [_bond_strength(bonds[k]) for k in range(N)]
        k_star = int(np.argmax(strengths))

        # Rotate so that strongest bond is between sites[0] and sites[1]
        if k_star != 0:
            sites = sites[k_star:] + sites[:k_star]
            bonds = bonds[k_star:] + bonds[:k_star]

        i = sites[0]
        j = sites[1]
        Lsite = sites[-1]
        Rsite = sites[2]

        Omega = bonds[0]
        mu_R = bonds[1]
        mu_L = bonds[-1]

        # Local two-site solve on B=(i,j)
        kappa_i = base_kappa + delta_kappa[:, i]
        kappa_j = base_kappa + delta_kappa[:, j]

        def local_sd_progress(it: int, itmax: int, resid: float):
            wrap_progress("local_sd", step + 1, n_steps_forward, it, itmax, resid)

        Sig_i, Sig_j, GBB = solve_two_site_sd(
            kappa_i, kappa_j, Omega, p,
            Sigma1_init_nu=Sigma_seed_nu,
            Sigma2_init_nu=Sigma_seed_nu,
            progress_cb=local_sd_progress,
        )

        Sigma_final[:, i] = Sig_i
        Sigma_final[:, j] = Sig_j
        Sigma_known[i] = True
        Sigma_known[j] = True

        # Extract D1, F, D2
        D1 = GBB[:, :n, :n]
        F = GBB[:, :n, n:]
        D2 = GBB[:, n:, n:]

        # On-site shifts on boundary sites
        for m in range(M):
            delta_kappa[m, Lsite] -= mu_L[m] @ D1[m] @ mu_L[m]
            delta_kappa[m, Rsite] -= mu_R[m] @ D2[m] @ mu_R[m]
            if p.enforce_symmetry:
                delta_kappa[m, Lsite] = symmetrize_last2(delta_kappa[m, Lsite])
                delta_kappa[m, Rsite] = symmetrize_last2(delta_kappa[m, Rsite])

        # Induced bond (symmetrized for potential non-commutation)
        mu_eff = np.zeros_like(Omega)
        for m in range(M):
            mu_eff[m] = -0.5 * (mu_L[m] @ F[m] @ mu_R[m] + mu_R[m] @ F[m] @ mu_L[m])
            if p.enforce_symmetry:
                mu_eff[m] = symmetrize_last2(mu_eff[m])

        # Store buffer record: we store the *local* inverse block and the boundary couplings
        buffer.append({
            "i": i, "j": j, "L": Lsite, "R": Rsite,
            "KBB_inv": GBB,
            "mu_L": mu_L,
            "mu_R": mu_R,
        })

        # Remove decimated sites and incident bonds, add induced bond
        # (After rotation, this is a simple slice operation.)
        sites = sites[2:]
        bonds = bonds[2:-1] + [mu_eff]

        wrap_progress("forward", step + 1, n_steps_forward, 0, 1, 0.0)

    # Terminal two-site solve on remaining sites
    assert len(sites) == 2 and len(bonds) == 2
    a_site, b_site = sites[0], sites[1]
    # Effective hopping between the two remaining sites includes both directed bonds
    Omega_term = np.zeros_like(bonds[0])
    for m in range(M):
        Omega_term[m] = bonds[0][m] + bonds[1][m].T
        if p.enforce_symmetry:
            Omega_term[m] = symmetrize_last2(Omega_term[m])

    kappa_a = base_kappa + delta_kappa[:, a_site]
    kappa_b = base_kappa + delta_kappa[:, b_site]

    def term_sd_progress(it: int, itmax: int, resid: float):
        wrap_progress("terminal_sd", n_steps_forward + 1, n_steps_forward + 1, it, itmax, resid)

    Sig_a, Sig_b, GBB_term = solve_two_site_sd(
        kappa_a, kappa_b, Omega_term, p,
        Sigma1_init_nu=Sigma_seed_nu,
        Sigma2_init_nu=Sigma_seed_nu,
        progress_cb=term_sd_progress,
    )

    Sigma_final[:, a_site] = Sig_a
    Sigma_final[:, b_site] = Sig_b
    Sigma_known[a_site] = True
    Sigma_known[b_site] = True

    if not bool(np.all(Sigma_known)):
        missing = np.where(~Sigma_known)[0].tolist()
        raise RuntimeError(f"Internal error: some sites never received a Sigma: {missing}")

    # Final kappa array
    kappa_final = np.zeros((M, L, n, n), dtype=np.complex128)
    for x in range(L):
        kappa_final[:, x] = base_kappa + delta_kappa[:, x]

    # ----------------------------
    # Restoration (sparse)
    # ----------------------------

    pair_map: dict[tuple[int, int], np.ndarray] = {}

    def add_pair(i_: int, j_: int, G_nu: np.ndarray):
        # G_nu shape (M,n,n)
        key = (int(i_), int(j_))
        if key in pair_map:
            return
        pair_map[key] = G_nu.astype(np.complex128, copy=True)

    # Add terminal block pairs (available)
    D1t = GBB_term[:, :n, :n]
    Ft = GBB_term[:, :n, n:]
    D2t = GBB_term[:, n:, n:]
    add_pair(a_site, a_site, D1t)
    add_pair(a_site, b_site, Ft)
    add_pair(b_site, a_site, np.swapaxes(Ft, -1, -2))
    add_pair(b_site, b_site, D2t)

    # Restore buffered records
    nbuf = len(buffer)
    for r, rec in enumerate(buffer, start=1):
        i = rec["i"]
        j = rec["j"]
        Ls = rec["L"]
        Rs = rec["R"]
        KBB_inv = rec["KBB_inv"]
        mu_L = rec["mu_L"]
        mu_R = rec["mu_R"]

        # Build K_AA from final on-site data (diagonal in site-block sense)
        KAA = np.zeros((M, 2 * n, 2 * n), dtype=np.complex128)
        for m in range(M):
            KLL = kappa_final[m, Ls] - Sigma_final[m, Ls]
            KRR = kappa_final[m, Rs] - Sigma_final[m, Rs]
            KAA[m, :n, :n] = KLL
            KAA[m, n:, n:] = KRR

        # K_AB and K_BA from stored boundary couplings
        KAB = np.zeros_like(KAA)
        KBA = np.zeros_like(KAA)
        for m in range(M):
            KAB[m, :n, :n] = -mu_L[m]
            KAB[m, n:, n:] = -mu_R[m]
            KBA[m, :n, :n] = -mu_L[m]
            KBA[m, n:, n:] = -mu_R[m]

        # Schur complement and inverses per frequency
        GAA_eff = np.zeros_like(KAA)
        GAB = np.zeros_like(KAA)
        GBA = np.zeros_like(KAA)
        GBB_rest = np.zeros((M, 2 * n, 2 * n), dtype=np.complex128)

        for m in range(M):
            KAA_eff = KAA[m] - KAB[m] @ KBB_inv[m] @ KBA[m]
            if p.enforce_symmetry:
                KAA_eff = symmetrize_last2(KAA_eff)
            GAA = safe_inv(KAA_eff, ridge=p.ridge)
            if p.enforce_symmetry:
                GAA = symmetrize_last2(GAA)
            GAA_eff[m] = GAA

            GAB[m] = -GAA @ KAB[m] @ KBB_inv[m]
            GBA[m] = -KBB_inv[m] @ KBA[m] @ GAA
            GBB_rest[m] = KBB_inv[m] + KBB_inv[m] @ KBA[m] @ GAA @ KAB[m] @ KBB_inv[m]

            if p.enforce_symmetry:
                GBB_rest[m] = symmetrize_last2(GBB_rest[m])

        # Store only BB and AB/BA blocks as available Green's functions
        add_pair(i, i, GBB_rest[:, :n, :n])
        add_pair(i, j, GBB_rest[:, :n, n:])
        add_pair(j, i, GBB_rest[:, n:, :n])
        add_pair(j, j, GBB_rest[:, n:, n:])

        add_pair(Ls, i, GAB[:, :n, :n])
        add_pair(Ls, j, GAB[:, :n, n:])
        add_pair(Rs, i, GAB[:, n:, :n])
        add_pair(Rs, j, GAB[:, n:, n:])

        add_pair(i, Ls, GBA[:, :n, :n])
        add_pair(j, Ls, GBA[:, n:, :n])
        add_pair(i, Rs, GBA[:, :n, n:])
        add_pair(j, Rs, GBA[:, n:, n:])

        wrap_progress("restore", r, nbuf, 0, 1, 0.0)

    # Pack pairs into arrays
    keys = sorted(pair_map.keys())
    P = len(keys)
    pairs_i = np.array([k[0] for k in keys], dtype=np.int32)
    pairs_j = np.array([k[1] for k in keys], dtype=np.int32)

    G_pairs_iw = np.zeros((M, P, n, n), dtype=np.complex128)
    for idx, k in enumerate(keys):
        G_pairs_iw[:, idx] = pair_map[k]

    # Metadata
    nu = matsubara_bosonic(p.beta, p.M)
    buffer_meta = np.array([[rec["i"], rec["j"], rec["L"], rec["R"]] for rec in buffer], dtype=np.int32)

    payload = {
        "beta": float(p.beta),
        "M": int(p.M),
        "q": int(p.q),
        "J": float(p.J),
        "m2": float(p.m2),
        "nrep": int(p.nrep),
        "L": int(L),
        "nu": nu,
        "pairs_i": pairs_i,
        "pairs_j": pairs_j,
        "G_pairs_iw": G_pairs_iw,
        "kappa_final": kappa_final,
        "Sigma_final": Sigma_final,
        "terminal_pair": np.array([a_site, b_site], dtype=np.int32),
        "buffer_meta": buffer_meta,
        "params_json": json.dumps({
            "note": "Sparse restoration: only BB and AB/BA blocks per decimation record, no AA blocks stored.",
            "init_sigma_tau": {"sig_d": sig_d_tau_init, "sig_od": sig_od_tau_init},
        }, indent=2),
    }

    return payload


# ----------------------------
# Bond initialization: uniform replica diagonal
# ----------------------------


def _scalar_to_mu_matrix(M: int, nrep: int, scalar: float) -> np.ndarray:
    out = np.zeros((M, nrep, nrep), dtype=np.complex128)
    mat = scalar * np.eye(nrep, dtype=np.complex128)
    for m in range(M):
        out[m] = mat
    return out


def sample_bond_scalars(
    L: int,
    mode: str,
    seed: int | None,
    # manual
    mu_uniform: float,
    mu_weak: float,
    mu_strong: float,
    strong_pos: int,
    mu_list_str: str,
    # random
    dist: str,
    abs_value: bool,
    gauss_mean: float,
    gauss_std: float,
    uni_low: float,
    uni_high: float,
    fd_loc: float,
    fd_scale: float,
    exp_scale: float,
    gamma_shape: float,
    gamma_scale: float,
) -> np.ndarray:
    """Return an array of length L of scalar bond amplitudes."""
    rng = np.random.default_rng(None if seed in (None, "") else int(seed))

    mode = mode.strip().lower()
    dist = dist.strip().lower()

    if mode.startswith("manual"):
        if "uniform" in mode:
            vals = np.full(L, float(mu_uniform), dtype=float)
        elif "one-strong" in mode or "one strong" in mode:
            vals = np.full(L, float(mu_weak), dtype=float)
            pos = int(strong_pos) % L
            vals[pos] = float(mu_strong)
        elif "list" in mode:
            parts = [p.strip() for p in mu_list_str.split(",") if p.strip()]
            if len(parts) != L:
                raise ValueError(f"Manual list must have exactly L={L} entries.")
            vals = np.array([float(x) for x in parts], dtype=float)
        else:
            raise ValueError(f"Unknown manual mode: {mode}")

    elif mode.startswith("random"):
        if dist in ("gaussian", "normal"):
            vals = rng.normal(loc=float(gauss_mean), scale=float(gauss_std), size=L)
        elif dist in ("uniform",):
            vals = rng.uniform(low=float(uni_low), high=float(uni_high), size=L)
        elif dist in ("fermi-dirac", "fermi", "fd", "logistic"):
            # Interpret "Fermi-Dirac" as a logistic distribution.
            # This is the natural distribution with pdf proportional to f(1-f).
            vals = rng.logistic(loc=float(fd_loc), scale=float(fd_scale), size=L)
        elif dist in ("exponential", "exp"):
            vals = rng.exponential(scale=float(exp_scale), size=L)
        elif dist in ("gamma",):
            vals = rng.gamma(shape=float(gamma_shape), scale=float(gamma_scale), size=L)
        else:
            raise ValueError(f"Unknown random distribution: {dist}")

        if abs_value:
            vals = np.abs(vals)

    else:
        raise ValueError(f"Unknown bond initialization mode: {mode}")

    return vals


# ----------------------------
# Tkinter GUI
# ----------------------------

class MinerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDRG-on-SD Miner (Periodic Even L, Sparse Restore)")
        self.geometry("1180x740")

        self._build_vars()
        self._build_ui()

        self.payload = None
        self.fig = None

    def _build_vars(self):
        # Model / numerics
        self.var_L = tk.IntVar(value=12)
        self.var_nrep = tk.IntVar(value=4)
        self.var_beta = tk.DoubleVar(value=80.0)
        self.var_M = tk.IntVar(value=256)
        self.var_q = tk.IntVar(value=4)
        self.var_J = tk.DoubleVar(value=1.0)
        self.var_m2 = tk.DoubleVar(value=1.0)

        self.var_mixing = tk.DoubleVar(value=0.35)
        self.var_tol = tk.DoubleVar(value=1e-10)
        self.var_maxiter = tk.IntVar(value=250)
        self.var_ridge = tk.DoubleVar(value=1e-12)

        self.var_sig_d_tau = tk.DoubleVar(value=0.5)
        self.var_sig_od_tau = tk.DoubleVar(value=0.2)

        # Bond init
        self.var_mode = tk.StringVar(value="Manual: uniform")
        self.var_seed = tk.StringVar(value="")
        self.var_mu_uniform = tk.DoubleVar(value=0.3)
        self.var_mu_weak = tk.DoubleVar(value=0.1)
        self.var_mu_strong = tk.DoubleVar(value=1.0)
        self.var_strong_pos = tk.IntVar(value=0)
        self.var_mu_list = tk.StringVar(value="")

        self.var_dist = tk.StringVar(value="Gaussian")
        self.var_abs = tk.BooleanVar(value=True)
        self.var_gauss_mean = tk.DoubleVar(value=0.0)
        self.var_gauss_std = tk.DoubleVar(value=1.0)
        self.var_uni_low = tk.DoubleVar(value=0.0)
        self.var_uni_high = tk.DoubleVar(value=1.0)
        self.var_fd_loc = tk.DoubleVar(value=0.0)
        self.var_fd_scale = tk.DoubleVar(value=1.0)
        self.var_exp_scale = tk.DoubleVar(value=1.0)
        self.var_gamma_shape = tk.DoubleVar(value=2.0)
        self.var_gamma_scale = tk.DoubleVar(value=1.0)

        self.var_outfile = tk.StringVar(value="")

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Left: parameters
        nb = ttk.Notebook(left)
        nb.pack(fill="y", expand=False)

        tab_model = ttk.Frame(nb)
        tab_bonds = ttk.Frame(nb)
        tab_num = ttk.Frame(nb)
        nb.add(tab_model, text="Model")
        nb.add(tab_bonds, text="Bonds")
        nb.add(tab_num, text="Solver")

        def row(parent, r, label, widget):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="ew", padx=4, pady=3)
            parent.grid_columnconfigure(1, weight=1)

        # Model tab
        r = 0
        row(tab_model, r, "Chain length L (even)", ttk.Entry(tab_model, textvariable=self.var_L, width=10)); r += 1
        row(tab_model, r, "Replicas n", ttk.Entry(tab_model, textvariable=self.var_nrep, width=10)); r += 1
        row(tab_model, r, "β", ttk.Entry(tab_model, textvariable=self.var_beta, width=10)); r += 1
        row(tab_model, r, "M Matsubara", ttk.Entry(tab_model, textvariable=self.var_M, width=10)); r += 1
        row(tab_model, r, "q (even)", ttk.Entry(tab_model, textvariable=self.var_q, width=10)); r += 1
        row(tab_model, r, "J", ttk.Entry(tab_model, textvariable=self.var_J, width=10)); r += 1
        row(tab_model, r, "m²", ttk.Entry(tab_model, textvariable=self.var_m2, width=10)); r += 1
        ttk.Separator(tab_model).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        row(tab_model, r, "Σ_d(τ) init", ttk.Entry(tab_model, textvariable=self.var_sig_d_tau, width=10)); r += 1
        row(tab_model, r, "Σ_od(τ) init", ttk.Entry(tab_model, textvariable=self.var_sig_od_tau, width=10)); r += 1

        # Bonds tab
        r = 0
        mode_cb = ttk.Combobox(tab_bonds, textvariable=self.var_mode, state="readonly",
                               values=[
                                   "Manual: uniform",
                                   "Manual: one-strong",
                                   "Manual: list",
                                   "Random: draw",
                               ])
        row(tab_bonds, r, "Init mode", mode_cb); r += 1
        row(tab_bonds, r, "Seed (random)", ttk.Entry(tab_bonds, textvariable=self.var_seed, width=12)); r += 1
        ttk.Separator(tab_bonds).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1

        row(tab_bonds, r, "μ_uniform", ttk.Entry(tab_bonds, textvariable=self.var_mu_uniform, width=10)); r += 1
        row(tab_bonds, r, "μ_weak", ttk.Entry(tab_bonds, textvariable=self.var_mu_weak, width=10)); r += 1
        row(tab_bonds, r, "μ_strong", ttk.Entry(tab_bonds, textvariable=self.var_mu_strong, width=10)); r += 1
        row(tab_bonds, r, "strong pos (0..L-1)", ttk.Entry(tab_bonds, textvariable=self.var_strong_pos, width=10)); r += 1
        row(tab_bonds, r, "μ list (comma, len L)", ttk.Entry(tab_bonds, textvariable=self.var_mu_list, width=22)); r += 1

        ttk.Separator(tab_bonds).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1

        dist_cb = ttk.Combobox(tab_bonds, textvariable=self.var_dist, state="readonly",
                               values=["Gaussian", "Uniform", "Fermi-Dirac", "Exponential", "Gamma"])
        row(tab_bonds, r, "Random dist", dist_cb); r += 1
        row(tab_bonds, r, "Abs(value)", ttk.Checkbutton(tab_bonds, variable=self.var_abs)); r += 1
        row(tab_bonds, r, "Gauss mean", ttk.Entry(tab_bonds, textvariable=self.var_gauss_mean, width=10)); r += 1
        row(tab_bonds, r, "Gauss std", ttk.Entry(tab_bonds, textvariable=self.var_gauss_std, width=10)); r += 1
        row(tab_bonds, r, "Unif low", ttk.Entry(tab_bonds, textvariable=self.var_uni_low, width=10)); r += 1
        row(tab_bonds, r, "Unif high", ttk.Entry(tab_bonds, textvariable=self.var_uni_high, width=10)); r += 1
        row(tab_bonds, r, "FD loc", ttk.Entry(tab_bonds, textvariable=self.var_fd_loc, width=10)); r += 1
        row(tab_bonds, r, "FD scale", ttk.Entry(tab_bonds, textvariable=self.var_fd_scale, width=10)); r += 1
        row(tab_bonds, r, "Exp scale", ttk.Entry(tab_bonds, textvariable=self.var_exp_scale, width=10)); r += 1
        row(tab_bonds, r, "Gamma shape", ttk.Entry(tab_bonds, textvariable=self.var_gamma_shape, width=10)); r += 1
        row(tab_bonds, r, "Gamma scale", ttk.Entry(tab_bonds, textvariable=self.var_gamma_scale, width=10)); r += 1

        # Solver tab
        r = 0
        row(tab_num, r, "Mixing α", ttk.Entry(tab_num, textvariable=self.var_mixing, width=10)); r += 1
        row(tab_num, r, "tol", ttk.Entry(tab_num, textvariable=self.var_tol, width=10)); r += 1
        row(tab_num, r, "max iters", ttk.Entry(tab_num, textvariable=self.var_maxiter, width=10)); r += 1
        row(tab_num, r, "ridge", ttk.Entry(tab_num, textvariable=self.var_ridge, width=10)); r += 1

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Run miner", command=self.on_run).pack(fill="x", pady=2)
        ttk.Button(btns, text="Save output (.npz)", command=self.on_save, state="disabled").pack(fill="x", pady=2)
        self.btn_save = btns.winfo_children()[-1]

        # Progress bars
        prog = ttk.LabelFrame(left, text="Progress")
        prog.pack(fill="x", pady=10)

        self.lbl_stage = ttk.Label(prog, text="Stage: -")
        self.lbl_stage.pack(anchor="w", padx=6, pady=2)

        self.pb_step = ttk.Progressbar(prog, orient="horizontal", length=260, mode="determinate")
        self.pb_step.pack(fill="x", padx=6, pady=4)
        self.lbl_step = ttk.Label(prog, text="Step: -")
        self.lbl_step.pack(anchor="w", padx=6, pady=2)

        self.pb_iter = ttk.Progressbar(prog, orient="horizontal", length=260, mode="determinate")
        self.pb_iter.pack(fill="x", padx=6, pady=4)
        self.lbl_iter = ttk.Label(prog, text="Iter: -")
        self.lbl_iter.pack(anchor="w", padx=6, pady=2)

        # Right: output summary
        out = ttk.LabelFrame(right, text="Output")
        out.pack(fill="both", expand=True)

        self.txt = tk.Text(out, height=20, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)

    def _progress_cb(self, stage, step, stepmax, it, itmax, resid):
        self.lbl_stage.config(text=f"Stage: {stage}")

        self.pb_step["maximum"] = max(1, stepmax)
        self.pb_step["value"] = min(step, stepmax)
        self.lbl_step.config(text=f"Step: {step}/{stepmax}")

        self.pb_iter["maximum"] = max(1, itmax)
        self.pb_iter["value"] = min(it, itmax)
        if stage in ("local_sd", "terminal_sd"):
            self.lbl_iter.config(text=f"Iter: {it}/{itmax}   resid={resid:.3e}")
        else:
            self.lbl_iter.config(text=f"Iter: -")

        self.update_idletasks()

    def _append(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def on_run(self):
        try:
            L = int(self.var_L.get())
            if L < 4 or (L % 2) != 0:
                raise ValueError("L must be even and >= 4.")

            nrep = int(self.var_nrep.get())
            beta = float(self.var_beta.get())
            M = int(self.var_M.get())
            q = int(self.var_q.get())
            if q % 2 != 0:
                raise ValueError("q must be even.")
            J = float(self.var_J.get())
            m2 = float(self.var_m2.get())

            p = TwoSiteParams(
                beta=beta,
                M=M,
                q=q,
                J=J,
                m2=m2,
                nrep=nrep,
                mixing=float(self.var_mixing.get()),
                tol=float(self.var_tol.get()),
                max_iter=int(self.var_maxiter.get()),
                ridge=float(self.var_ridge.get()),
                enforce_symmetry=True,
            )

            sig_d_tau = float(self.var_sig_d_tau.get())
            sig_od_tau = float(self.var_sig_od_tau.get())

            # Bond scalars
            mode = self.var_mode.get()
            dist = self.var_dist.get()
            seed = self.var_seed.get()
            seed_val = None if seed.strip() == "" else int(seed)

            scalars = sample_bond_scalars(
                L=L,
                mode=mode,
                seed=seed_val,
                mu_uniform=float(self.var_mu_uniform.get()),
                mu_weak=float(self.var_mu_weak.get()),
                mu_strong=float(self.var_mu_strong.get()),
                strong_pos=int(self.var_strong_pos.get()),
                mu_list_str=self.var_mu_list.get(),
                dist=dist,
                abs_value=bool(self.var_abs.get()),
                gauss_mean=float(self.var_gauss_mean.get()),
                gauss_std=float(self.var_gauss_std.get()),
                uni_low=float(self.var_uni_low.get()),
                uni_high=float(self.var_uni_high.get()),
                fd_loc=float(self.var_fd_loc.get()),
                fd_scale=float(self.var_fd_scale.get()),
                exp_scale=float(self.var_exp_scale.get()),
                gamma_shape=float(self.var_gamma_shape.get()),
                gamma_scale=float(self.var_gamma_scale.get()),
            )

            # Initial mu matrices: uniform replica diagonal, frequency-independent scalar
            mu_init = [_scalar_to_mu_matrix(M, nrep, float(s)) for s in scalars]

            self.txt.delete("1.0", "end")
            self._append("Running protocol...\n")

            payload = run_protocol_periodic_even_sparse(
                L=L,
                mu_init=mu_init,
                p=p,
                sig_d_tau_init=sig_d_tau,
                sig_od_tau_init=sig_od_tau,
                progress_cb=self._progress_cb,
            )

            self.payload = payload

            P = int(payload["pairs_i"].shape[0])
            self._append("Done.")
            self._append(f"Available Green's functions: {P} directed pairs")
            self._append(f"Terminal pair: {payload['terminal_pair'].tolist()}")
            self._append(f"Buffer records: {payload['buffer_meta'].shape[0]}")
            self._append("\nFirst 20 available pairs (i->j):")
            for k in range(min(20, P)):
                self._append(f"  {int(payload['pairs_i'][k])} -> {int(payload['pairs_j'][k])}")

            self.btn_save.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_save(self):
        if self.payload is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save miner output",
            defaultextension=".npz",
            filetypes=[("NumPy zip", "*.npz")],
        )
        if not path:
            return

        try:
            np.savez_compressed(path, **self.payload)
            messagebox.showinfo("Saved", f"Saved output to\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    app = MinerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
