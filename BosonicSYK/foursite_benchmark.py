#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""four_site_sd_benchmark_gui.py

Benchmark GUI: Exact 4-site SD solver vs. SDRG-style "two-site decimation" protocol.

What this program does
----------------------
You choose model/numerical parameters and three hoppings on a 4-site open chain:

    1 --(mu12)-- 2 --(mu23)-- 3 --(mu34)-- 4

The code provides two solvers:

  (A) Exact 4-site SD:
      Solve the full 4-site SD fixed point by iterating
          G(iν) = K(iν)^{-1},   Σ(τ) = J^2 [G(τ)]^{q-1}
      where Σ is site-diagonal (but a full replica matrix), and K includes all hoppings.

  (B) Protocol (your SDRG-on-SD local decimation idea):
      1) Solve the 2-site SD problem on the strong bond (2,3) exactly.
      2) Use the Schur-complement update (with K_BB^{-1} from step 1) to induce
         diagonal shifts on sites 1 and 4 and an effective dynamic bond between 1 and 4.
      3) Solve the reduced 2-site SD problem on (1,4) exactly (with the induced bond).
      4) Restore an approximate full 4-site Green's function by block-inversion identities.

No external databases are used.

Replica matrices
----------------
We keep replica matrices fully general during the solve (nrep×nrep matrices).
The SD closure is applied entrywise in τ-space:
    Σ^{ab}(τ) = J^2 * (G^{ab}(τ))^{q-1}

This is the most direct formulation, avoiding projector decompositions.

Practical stability
-------------------
Bosonic zero modes can be numerically delicate. Two stabilizers are exposed:

  - nu0_reg: adds a small +nu0_reg to the diagonal of each on-site kernel at ν_0.
  - ridge:   adds a small +ridge * I to the full matrix before inversion.

Plotting
--------
You can plot any G_{ij}^{ab} in frequency or imaginary-time domain, and overlay
Exact vs Protocol (or plot their difference). A list of available site-pairs is shown.

"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Helpers: grids and FFT conventions (match the user's previous code)
# =============================================================================

def matsubara_nu(beta: float, M: int) -> np.ndarray:
    """Bosonic Matsubara grid used in the user's code: ν_n = 2π n / β, n=0..M-1."""
    n = np.arange(int(M), dtype=np.float64)
    return (2.0 * np.pi / float(beta)) * n


def tau_grid(beta: float, M: int) -> np.ndarray:
    """Uniform τ grid: τ_k = k β / M."""
    M = int(M)
    return (float(beta) / M) * np.arange(M, dtype=np.float64)


def nu_to_tau(beta: float, F_nu: np.ndarray) -> np.ndarray:
    """Convert an array over ν to τ using the convention:

        F(τ) = (1/β) * FFT[F(iν)]

    Works for arrays whose first axis is the Matsubara index.
    """
    return (1.0 / float(beta)) * np.fft.fft(F_nu, axis=0)


def tau_to_nu(beta: float, F_tau: np.ndarray) -> np.ndarray:
    """Inverse of nu_to_tau:

        F(iν) = β * IFFT[F(τ)]
    """
    return float(beta) * np.fft.ifft(F_tau, axis=0)


def symmetrize_replica_matrix(arr: np.ndarray) -> np.ndarray:
    """Symmetrize replica matrices: X <- (X + X^T)/2.

    Parameters
    ----------
    arr : ndarray
        Either shape (..., n, n) or (n, n).
    """
    if arr.ndim == 2:
        return 0.5 * (arr + arr.T)
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def rs_project_matrix(mat: np.ndarray) -> np.ndarray:
    """Project a replica matrix to the RS submanifold by averaging:
    - all diagonal entries are replaced by their mean
    - all off-diagonal entries are replaced by their mean

    This is optional and only intended to suppress numerical drift.
    """
    n = mat.shape[-1]
    diag = np.diagonal(mat, axis1=-2, axis2=-1)
    diag_mean = np.mean(diag, axis=-1, keepdims=True)  # (..., 1)
    # mask off-diagonal
    off = mat.copy()
    idx = np.arange(n)
    off[..., idx, idx] = np.nan
    off_mean = np.nanmean(off, axis=(-2, -1), keepdims=True)  # (..., 1, 1)
    out = np.empty_like(mat)
    out[...] = off_mean
    out[..., idx, idx] = diag_mean
    return out


def ensure_complex(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)


# =============================================================================
# Block-matrix multiplication helpers for 2x2 blocks with replica matrices
# =============================================================================

def block2_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two 2x2 block matrices with replica blocks.

    Both A and B have shape (M, 2, 2, n, n) or (2, 2, n, n) (broadcastable over M).
    Output shape follows broadcasting.
    """
    # Promote to have Matsubara axis if needed
    if A.ndim == 4:
        A = A[None, ...]
    if B.ndim == 4:
        B = B[None, ...]

    M = max(A.shape[0], B.shape[0])
    if A.shape[0] == 1 and M > 1:
        A = np.repeat(A, M, axis=0)
    if B.shape[0] == 1 and M > 1:
        B = np.repeat(B, M, axis=0)

    out = np.zeros_like(A)
    # out_ij = sum_k A_ik @ B_kj
    for i in range(2):
        for j in range(2):
            s = 0.0
            for k in range(2):
                s = s + A[:, i, k] @ B[:, k, j]
            out[:, i, j] = s
    return out


# =============================================================================
# Two-site SD solver with general replica matrices and (possibly ν-dependent) hopping
# =============================================================================

@dataclass
class SolverParams:
    beta: float
    M: int
    nrep: int
    m2: float
    J: float
    q_body: int
    max_iter: int
    mix: float
    tol: float
    ridge: float
    nu0_reg: float
    symmetrize_each_iter: bool
    rs_project_each_iter: bool
    # Initial condition seed (constant in τ, implemented as ν0-only coefficient)
    sigma_d_init_tau: float
    sigma_od_init_tau: float


def _seed_sigma_nu_from_rs_tau_constant(
    *,
    beta: float,
    M: int,
    nrep: int,
    sigma_d_tau: float,
    sigma_od_tau: float,
) -> np.ndarray:
    """Construct an RS seed for Σ(iν) corresponding to constants in τ.

    We seed only the ν0 component, because a constant-in-τ value has Fourier
    coefficients Σ(iν0)=β Σ(τ) and Σ(iν_{n!=0})=0.

    Returns
    -------
    Sigma_nu : ndarray, shape (M,nrep,nrep)
    """
    M = int(M)
    n = int(nrep)
    Sigma_nu = np.zeros((M, n, n), dtype=np.complex128)
    if n <= 0:
        return Sigma_nu
    # ν0 coefficient
    Sigma0 = np.full((n, n), float(beta) * float(sigma_od_tau), dtype=np.complex128)
    np.fill_diagonal(Sigma0, float(beta) * float(sigma_d_tau))
    Sigma_nu[0] = Sigma0
    return Sigma_nu


def solve_sd_two_site(
    *,
    kappa0_1: np.ndarray,  # (M,n,n)
    kappa0_2: np.ndarray,  # (M,n,n)
    Omega: np.ndarray,     # (M,n,n)
    params: SolverParams,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the two-site SD equations self-consistently.

    Returns
    -------
    Sigma1_nu, Sigma2_nu : ndarray
        Shapes (M,n,n)
    GBB_nu : ndarray
        Shape (M,2,2,n,n) containing K_BB^{-1} blocks.
    """
    beta = float(params.beta)
    M = int(params.M)
    n = int(params.nrep)
    nu = matsubara_nu(beta, M)

    kappa0_1 = ensure_complex(kappa0_1)
    kappa0_2 = ensure_complex(kappa0_2)
    Omega = ensure_complex(Omega)

    # Unknowns to iterate
    # Seed with an RS constant-in-τ initial condition (ν0-only coefficient).
    Sigma_seed = _seed_sigma_nu_from_rs_tau_constant(
        beta=beta,
        M=M,
        nrep=n,
        sigma_d_tau=float(params.sigma_d_init_tau),
        sigma_od_tau=float(params.sigma_od_init_tau),
    )
    Sigma1_nu = Sigma_seed.copy()
    Sigma2_nu = Sigma_seed.copy()

    Irep = np.eye(n, dtype=np.complex128)

    for it in range(1, params.max_iter + 1):
        # Dyson: build & invert for each ν
        GBB_nu = np.zeros((M, 2, 2, n, n), dtype=np.complex128)

        # A little progress granularity inside each iteration
        for k in range(M):
            K11 = kappa0_1[k] - Sigma1_nu[k]
            K22 = kappa0_2[k] - Sigma2_nu[k]

            # Stabilize ν0
            if k == 0 and params.nu0_reg != 0.0:
                K11 = K11 + params.nu0_reg * Irep
                K22 = K22 + params.nu0_reg * Irep

            # Build full 2n x 2n matrix
            top = np.concatenate([K11, -Omega[k]], axis=1)
            bot = np.concatenate([-Omega[k].T, K22], axis=1)
            K = np.concatenate([top, bot], axis=0)

            if params.ridge != 0.0:
                K = K + params.ridge * np.eye(2 * n, dtype=np.complex128)

            try:
                G = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    "Two-site inversion failed (singular matrix). "
                    "Increase ridge / nu0_reg, or change parameters."
                )

            G11 = G[0:n, 0:n]
            G12 = G[0:n, n:2 * n]
            G21 = G[n:2 * n, 0:n]
            G22b = G[n:2 * n, n:2 * n]

            GBB_nu[k, 0, 0] = G11
            GBB_nu[k, 0, 1] = G12
            GBB_nu[k, 1, 0] = G21
            GBB_nu[k, 1, 1] = G22b

            if progress_cb and (k % max(1, M // 12) == 0):
                frac = (it - 1 + (k + 1) / M) / max(1, params.max_iter)
                progress_cb(frac, f"Two-site SD: iter {it}/{params.max_iter} (ν {k+1}/{M})")

        # SD closure: Σ(τ) = J^2 [G(τ)]^{q-1} entrywise for on-site blocks
        G11_tau = nu_to_tau(beta, GBB_nu[:, 0, 0])
        G22_tau = nu_to_tau(beta, GBB_nu[:, 1, 1])

        # Use real part in closure (consistent with user's previous implementation)
        G11_tau_r = np.real(G11_tau)
        G22_tau_r = np.real(G22_tau)

        power = int(params.q_body) - 1
        Sigma1_tau_new = (float(params.J) ** 2) * np.power(G11_tau_r, power)
        Sigma2_tau_new = (float(params.J) ** 2) * np.power(G22_tau_r, power)

        Sigma1_nu_new = tau_to_nu(beta, Sigma1_tau_new)
        Sigma2_nu_new = tau_to_nu(beta, Sigma2_tau_new)

        if params.symmetrize_each_iter:
            Sigma1_nu_new = symmetrize_replica_matrix(Sigma1_nu_new)
            Sigma2_nu_new = symmetrize_replica_matrix(Sigma2_nu_new)

        if params.rs_project_each_iter:
            Sigma1_nu_new = rs_project_matrix(Sigma1_nu_new)
            Sigma2_nu_new = rs_project_matrix(Sigma2_nu_new)

        # Mix
        Sigma1_next = (1.0 - params.mix) * Sigma1_nu + params.mix * Sigma1_nu_new
        Sigma2_next = (1.0 - params.mix) * Sigma2_nu + params.mix * Sigma2_nu_new

        # Convergence
        diff = max(
            float(np.max(np.abs(Sigma1_next - Sigma1_nu))),
            float(np.max(np.abs(Sigma2_next - Sigma2_nu))),
        )

        Sigma1_nu, Sigma2_nu = Sigma1_next, Sigma2_next

        if progress_cb and (it == 1 or it % 10 == 0):
            frac = it / max(1, params.max_iter)
            progress_cb(frac, f"Two-site SD: iter {it}/{params.max_iter}  diff={diff:.3e}")

        if diff < params.tol:
            break

    # Final Dyson with converged Σ
    # (We re-compute GBB_nu one last time so output is consistent.)
    GBB_nu_final = np.zeros((M, 2, 2, n, n), dtype=np.complex128)
    for k in range(M):
        K11 = kappa0_1[k] - Sigma1_nu[k]
        K22 = kappa0_2[k] - Sigma2_nu[k]
        if k == 0 and params.nu0_reg != 0.0:
            K11 = K11 + params.nu0_reg * Irep
            K22 = K22 + params.nu0_reg * Irep
        top = np.concatenate([K11, -Omega[k]], axis=1)
        bot = np.concatenate([-Omega[k].T, K22], axis=1)
        K = np.concatenate([top, bot], axis=0)
        if params.ridge != 0.0:
            K = K + params.ridge * np.eye(2 * n, dtype=np.complex128)
        G = np.linalg.inv(K)
        GBB_nu_final[k, 0, 0] = G[0:n, 0:n]
        GBB_nu_final[k, 0, 1] = G[0:n, n:2 * n]
        GBB_nu_final[k, 1, 0] = G[n:2 * n, 0:n]
        GBB_nu_final[k, 1, 1] = G[n:2 * n, n:2 * n]

    return Sigma1_nu, Sigma2_nu, GBB_nu_final


# =============================================================================
# Exact 4-site SD solver
# =============================================================================

def solve_sd_four_site_exact(
    *,
    mu12: float,
    mu23: float,
    mu34: float,
    params: SolverParams,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, np.ndarray]:
    """Solve 4-site SD fixed point exactly (site-space 4 blocks, full inversion each ν)."""
    beta = float(params.beta)
    M = int(params.M)
    n = int(params.nrep)
    nu = matsubara_nu(beta, M)
    Irep = np.eye(n, dtype=np.complex128)

    # Bare on-site kernel κ0_x(iν) = (ν^2+m^2)I  (no RG shifts in the exact solver)
    nu_sq = (nu ** 2 + float(params.m2)).astype(np.complex128)
    kappa0 = np.zeros((M, 4, n, n), dtype=np.complex128)
    for k in range(M):
        for x in range(4):
            kappa0[k, x] = nu_sq[k] * Irep
    # Stabilize ν0
    if params.nu0_reg != 0.0:
        kappa0[0, :, :, :] += params.nu0_reg * Irep

    # Unknowns: Σ_x(iν)
    # Seed with an RS constant-in-τ initial condition (ν0-only coefficient).
    Sigma_seed = _seed_sigma_nu_from_rs_tau_constant(
        beta=beta,
        M=M,
        nrep=n,
        sigma_d_tau=float(params.sigma_d_init_tau),
        sigma_od_tau=float(params.sigma_od_init_tau),
    )
    Sigma = np.zeros((M, 4, n, n), dtype=np.complex128)
    for x in range(4):
        Sigma[:, x] = Sigma_seed

    # Hoppings are replica-diagonal in this benchmark
    mu12M = float(mu12) * Irep
    mu23M = float(mu23) * Irep
    mu34M = float(mu34) * Irep

    for it in range(1, params.max_iter + 1):
        # Dyson step: invert 4n x 4n matrix at each ν
        G_nu = np.zeros((M, 4, 4, n, n), dtype=np.complex128)

        for k in range(M):
            # Build 4x4 block kernel in site space
            blocks = [[None for _ in range(4)] for _ in range(4)]
            for x in range(4):
                blocks[x][x] = kappa0[k, x] - Sigma[k, x]

            # Off-diagonal blocks (open chain)
            blocks[0][1] = -mu12M
            blocks[1][0] = -mu12M.T
            blocks[1][2] = -mu23M
            blocks[2][1] = -mu23M.T
            blocks[2][3] = -mu34M
            blocks[3][2] = -mu34M.T
            for i in range(4):
                for j in range(4):
                    if blocks[i][j] is None:
                        blocks[i][j] = np.zeros((n, n), dtype=np.complex128)

            # Assemble
            row0 = np.concatenate(blocks[0], axis=1)
            row1 = np.concatenate(blocks[1], axis=1)
            row2 = np.concatenate(blocks[2], axis=1)
            row3 = np.concatenate(blocks[3], axis=1)
            K = np.concatenate([row0, row1, row2, row3], axis=0)
            if params.ridge != 0.0:
                K = K + params.ridge * np.eye(4 * n, dtype=np.complex128)

            try:
                G = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    "Exact 4-site inversion failed (singular matrix). "
                    "Increase ridge / nu0_reg, or change parameters."
                )

            # Extract 4x4 blocks
            for i in range(4):
                for j in range(4):
                    G_nu[k, i, j] = G[i * n:(i + 1) * n, j * n:(j + 1) * n]

            if progress_cb and (k % max(1, M // 10) == 0):
                frac = (it - 1 + (k + 1) / M) / max(1, params.max_iter)
                progress_cb(frac, f"Exact 4-site: iter {it}/{params.max_iter} (ν {k+1}/{M})")

        # SD closure on on-site blocks only
        Sigma_new = np.zeros_like(Sigma)
        power = int(params.q_body) - 1
        for x in range(4):
            Gxx_tau = nu_to_tau(beta, G_nu[:, x, x])
            Gxx_tau_r = np.real(Gxx_tau)
            Sigma_x_tau = (float(params.J) ** 2) * np.power(Gxx_tau_r, power)
            Sigma_new[:, x] = tau_to_nu(beta, Sigma_x_tau)

        if params.symmetrize_each_iter:
            Sigma_new = symmetrize_replica_matrix(Sigma_new)
        if params.rs_project_each_iter:
            Sigma_new = rs_project_matrix(Sigma_new)

        # Mix and check convergence
        Sigma_next = (1.0 - params.mix) * Sigma + params.mix * Sigma_new
        diff = float(np.max(np.abs(Sigma_next - Sigma)))
        Sigma = Sigma_next

        if progress_cb and (it == 1 or it % 10 == 0):
            frac = it / max(1, params.max_iter)
            progress_cb(frac, f"Exact 4-site: iter {it}/{params.max_iter}  diff={diff:.3e}")

        if diff < params.tol:
            break

    # Final Dyson for consistent output
    # (Repeat the inversion once with the final Σ.)
    G_nu_final = np.zeros((M, 4, 4, n, n), dtype=np.complex128)
    for k in range(M):
        blocks = [[None for _ in range(4)] for _ in range(4)]
        for x in range(4):
            blocks[x][x] = kappa0[k, x] - Sigma[k, x]
        blocks[0][1] = -mu12M
        blocks[1][0] = -mu12M.T
        blocks[1][2] = -mu23M
        blocks[2][1] = -mu23M.T
        blocks[2][3] = -mu34M
        blocks[3][2] = -mu34M.T
        for i in range(4):
            for j in range(4):
                if blocks[i][j] is None:
                    blocks[i][j] = np.zeros((n, n), dtype=np.complex128)
        row0 = np.concatenate(blocks[0], axis=1)
        row1 = np.concatenate(blocks[1], axis=1)
        row2 = np.concatenate(blocks[2], axis=1)
        row3 = np.concatenate(blocks[3], axis=1)
        K = np.concatenate([row0, row1, row2, row3], axis=0)
        if params.ridge != 0.0:
            K = K + params.ridge * np.eye(4 * n, dtype=np.complex128)
        G = np.linalg.inv(K)
        for i in range(4):
            for j in range(4):
                G_nu_final[k, i, j] = G[i * n:(i + 1) * n, j * n:(j + 1) * n]

    return {
        "nu": nu,
        "tau": tau_grid(beta, M),
        "Sigma_nu": Sigma,
        "G_nu": G_nu_final,
        "G_tau": nu_to_tau(beta, G_nu_final),
    }


# =============================================================================
# 4-site protocol solver (decimate 2-3, then solve 1-4)
# =============================================================================

def solve_sd_four_site_protocol(
    *,
    mu12: float,
    mu23: float,
    mu34: float,
    params: SolverParams,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, np.ndarray]:
    """Your SDRG-style protocol on 4 sites: local two-site solves + Schur updates + restore."""
    beta = float(params.beta)
    M = int(params.M)
    n = int(params.nrep)
    nu = matsubara_nu(beta, M)
    Irep = np.eye(n, dtype=np.complex128)

    # Quadratic on-site kernels κ0_x(iν) = (ν^2+m^2)I + δκ_x(iν)
    nu_sq = (nu ** 2 + float(params.m2)).astype(np.complex128)
    delta = np.zeros((4, M, n, n), dtype=np.complex128)  # δκ

    def build_kappa0(x: int) -> np.ndarray:
        """Current on-site kernel κ_x(iν) = (ν^2+m^2)I + δκ_x(iν).

        The ν0 stabilizer (nu0_reg) is applied in the Dyson inversions to avoid
        double counting across different solver components.
        """
        out = np.zeros((M, n, n), dtype=np.complex128)
        for k in range(M):
            out[k] = nu_sq[k] * Irep + delta[x, k]
        return out

    # ---------------------------------------------------------------------
    # Step 1: solve two-site SD on B={2,3} (indices 1 and 2 in 0-based)
    # ---------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.0, "Protocol: Step 1/4 - solve dimer (2,3)")

    kappa0_2 = build_kappa0(1)  # site 2
    kappa0_3 = build_kappa0(2)  # site 3

    Omega23 = np.zeros((M, n, n), dtype=np.complex128)
    for k in range(M):
        Omega23[k] = float(mu23) * Irep

    def cb1(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(0.25 * frac, msg)

    Sigma2, Sigma3, GBB = solve_sd_two_site(
        kappa0_1=kappa0_2,
        kappa0_2=kappa0_3,
        Omega=Omega23,
        params=params,
        progress_cb=cb1,
    )

    D1 = GBB[:, 0, 0]  # (M,n,n) corresponds to site 2 on-site block
    F = GBB[:, 0, 1]   # (M,n,n) corresponds to (2,3)
    D2 = GBB[:, 1, 1]  # (M,n,n) corresponds to site 3 on-site block

    # ---------------------------------------------------------------------
    # Step 2: Schur updates to boundary sites 1 and 4
    # ---------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.25, "Protocol: Step 2/4 - Schur updates (shifts + induced bond)")

    # Diagonal shifts (Schur complement):
    #   κ_1 <- κ_1 - μ12 · D1 · μ12,    κ_4 <- κ_4 - μ34 · D2 · μ34
    # Here μ12,μ34 are replica-diagonal in the benchmark GUI (scalars × I), but we
    # implement the matrix product form to match the Appendix convention.
    mu12M = float(mu12) * Irep
    mu34M = float(mu34) * Irep
    for k in range(M):
        delta[0, k] = delta[0, k] - (mu12M @ D1[k] @ mu12M)
        delta[3, k] = delta[3, k] - (mu34M @ D2[k] @ mu34M)

    # Induced (dynamic) bond between 1 and 4:
    #   μ14(iν) = 1/2( μ12 · F(iν) · μ34 + μ34 · F(iν) · μ12 )
    Omega14 = np.zeros((M, n, n), dtype=np.complex128)
    for k in range(M):
        Omega14[k] = 0.5 * (mu12M @ F[k] @ mu34M + mu34M @ F[k] @ mu12M)
    # Enforce symmetric kernel convention (K12 = K21^T)
    Omega14 = symmetrize_replica_matrix(Omega14)

    # ---------------------------------------------------------------------
    # Step 3: solve reduced two-site SD on A={1,4}
    # ---------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.35, "Protocol: Step 3/4 - solve effective dimer (1,4)")

    kappa0_1 = build_kappa0(0)  # site 1 (dressed)
    kappa0_4 = build_kappa0(3)  # site 4 (dressed)

    def cb2(frac: float, msg: str) -> None:
        if progress_cb:
            # map to [0.35, 0.85]
            progress_cb(0.35 + 0.50 * frac, msg)

    Sigma1, Sigma4, GAA = solve_sd_two_site(
        kappa0_1=kappa0_1,
        kappa0_2=kappa0_4,
        Omega=Omega14,
        params=params,
        progress_cb=cb2,
    )

    # ---------------------------------------------------------------------
    # Step 4: restore an approximate full 4-site Green's function
    # Partition: A={1,4}, B={2,3}
    # ---------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.86, "Protocol: Step 4/4 - restore full 4-site G via block inversion")

    # Build constant K_AB blocks
    KAB = np.zeros((2, 2, n, n), dtype=np.complex128)
    KAB[0, 0] = -float(mu12) * Irep  # 1 couples to 2
    KAB[1, 1] = -float(mu34) * Irep  # 4 couples to 3
    KBA = np.zeros((2, 2, n, n), dtype=np.complex128)
    KBA[0, 0] = KAB[0, 0].T
    KBA[1, 1] = KAB[1, 1].T

    # In this protocol, G_AA is the inverse of the Schur-complement effective kernel
    G_AA = GAA
    K_BB_inv = GBB

    # G_AB = - G_AA K_AB K_BB^{-1}
    tmp = block2_mul(KAB, K_BB_inv)
    G_AB = -block2_mul(G_AA, tmp)
    # G_BA = - K_BB^{-1} K_BA G_AA
    tmp2 = block2_mul(KBA, G_AA)
    G_BA = -block2_mul(K_BB_inv, tmp2)
    # G_BB = K_BB^{-1} + K_BB^{-1} K_BA G_AA K_AB K_BB^{-1}
    tmp3 = block2_mul(KBA, G_AA)
    tmp3 = block2_mul(tmp3, KAB)
    tmp3 = block2_mul(tmp3, K_BB_inv)
    corr = block2_mul(K_BB_inv, tmp3)
    G_BB = K_BB_inv + corr

    # Assemble full G_{ij} (sites 1..4) from A/B blocks
    # A indices: [1,4] -> site indices [0,3]
    # B indices: [2,3] -> site indices [1,2]
    G_full = np.zeros((M, 4, 4, n, n), dtype=np.complex128)

    # Fill A-A
    G_full[:, 0, 0] = G_AA[:, 0, 0]
    G_full[:, 0, 3] = G_AA[:, 0, 1]
    G_full[:, 3, 0] = G_AA[:, 1, 0]
    G_full[:, 3, 3] = G_AA[:, 1, 1]

    # Fill B-B
    G_full[:, 1, 1] = G_BB[:, 0, 0]
    G_full[:, 1, 2] = G_BB[:, 0, 1]
    G_full[:, 2, 1] = G_BB[:, 1, 0]
    G_full[:, 2, 2] = G_BB[:, 1, 1]

    # Fill A-B and B-A
    # A-B: rows [1,4] cols [2,3] => (0,1) and (3,2)
    G_full[:, 0, 1] = G_AB[:, 0, 0]  # 1-2
    G_full[:, 0, 2] = G_AB[:, 0, 1]  # 1-3
    G_full[:, 3, 1] = G_AB[:, 1, 0]  # 4-2
    G_full[:, 3, 2] = G_AB[:, 1, 1]  # 4-3

    # B-A
    G_full[:, 1, 0] = G_BA[:, 0, 0]  # 2-1
    G_full[:, 2, 0] = G_BA[:, 1, 0]  # 3-1
    G_full[:, 1, 3] = G_BA[:, 0, 1]  # 2-4
    G_full[:, 2, 3] = G_BA[:, 1, 1]  # 3-4

    if progress_cb:
        progress_cb(1.0, "Protocol: done")

    # Package
    Sigma_all = np.zeros((M, 4, n, n), dtype=np.complex128)
    Sigma_all[:, 0] = Sigma1
    Sigma_all[:, 1] = Sigma2
    Sigma_all[:, 2] = Sigma3
    Sigma_all[:, 3] = Sigma4

    return {
        "nu": nu,
        "tau": tau_grid(beta, M),
        "Sigma_nu": Sigma_all,
        "G_nu": G_full,
        "G_tau": nu_to_tau(beta, G_full),
        "protocol_details": {
            "GBB": GBB,
            "GAA": GAA,
            "Omega14": Omega14,
            "delta_kappa": delta,
        },
    }


# =============================================================================
# GUI
# =============================================================================

class BenchmarkGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("4-site SD Benchmark: Exact vs Protocol")
        self.geometry("1400x860")

        # Results
        self.res_exact: Optional[Dict[str, np.ndarray]] = None
        self.res_proto: Optional[Dict[str, np.ndarray]] = None

        # Progress state
        self.progress_exact = tk.DoubleVar(value=0.0)
        self.progress_proto = tk.DoubleVar(value=0.0)
        self.status_exact = tk.StringVar(value="Exact solver: idle")
        self.status_proto = tk.StringVar(value="Protocol solver: idle")

        # Inputs (defaults chosen to be reasonably stable)
        self.beta_var = tk.DoubleVar(value=40.0)
        self.M_var = tk.IntVar(value=128)
        self.nrep_var = tk.IntVar(value=3)
        self.m2_var = tk.DoubleVar(value=1.0)
        self.J_var = tk.DoubleVar(value=1.0)
        self.q_var = tk.IntVar(value=4)

        # RS seed for replica matrices (constant in τ; implemented via ν0-only coefficient)
        # If you want a purely replica-diagonal solution, set sigma_od=0.
        self.sigma_d_init_var = tk.DoubleVar(value=0.0)
        self.sigma_od_init_var = tk.DoubleVar(value=0.05)

        self.mu12_var = tk.DoubleVar(value=0.20)
        self.mu23_var = tk.DoubleVar(value=1.00)
        self.mu34_var = tk.DoubleVar(value=0.20)

        self.max_iter_var = tk.IntVar(value=3000)
        self.mix_var = tk.DoubleVar(value=0.20)
        self.tol_var = tk.DoubleVar(value=1e-8)
        self.ridge_var = tk.DoubleVar(value=1e-10)
        self.nu0_reg_var = tk.DoubleVar(value=1e-6)
        self.sym_each_iter_var = tk.BooleanVar(value=True)
        self.rs_project_var = tk.BooleanVar(value=False)

        # Plot controls
        self.site_i_var = tk.IntVar(value=1)
        self.site_j_var = tk.IntVar(value=1)
        self.rep_a_var = tk.IntVar(value=0)
        self.rep_b_var = tk.IntVar(value=0)

        self.domain_var = tk.StringVar(value="ν")  # ν or τ
        self.component_var = tk.StringVar(value="Re")  # Re/Im/Abs
        self.which_var = tk.StringVar(value="Overlay")  # Exact/Protocol/Overlay/Diff

        # Style controls
        self.fig_w_var = tk.DoubleVar(value=7.6)
        self.fig_h_var = tk.DoubleVar(value=4.6)
        self.title_fs_var = tk.IntVar(value=12)
        self.label_fs_var = tk.IntVar(value=11)
        self.legend_fs_var = tk.IntVar(value=10)
        self.tick_fs_var = tk.IntVar(value=10)

        self._build_ui()

    # -------------------------- UI layout --------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, width=430)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left: inputs, run buttons, progress, plot controls
        self._build_inputs(left)
        self._build_progress(left)
        self._build_plot_controls(left)
        self._build_available_lists(left)

        # Right: plot
        self._build_plot(right)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, var: tk.Variable, width: int = 10) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT)

    def _build_inputs(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Model & Numerical Parameters")
        box.pack(fill=tk.X, padx=6, pady=6)

        self._add_labeled_entry(box, "β", self.beta_var)
        self._add_labeled_entry(box, "M (freq/time grid)", self.M_var)
        self._add_labeled_entry(box, "nrep", self.nrep_var)
        self._add_labeled_entry(box, "m²", self.m2_var)
        self._add_labeled_entry(box, "J", self.J_var)
        self._add_labeled_entry(box, "q (body)", self.q_var)

        self._add_labeled_entry(box, "Init Σ_d(τ) (RS seed)", self.sigma_d_init_var)
        self._add_labeled_entry(box, "Init Σ_od(τ) (RS seed)", self.sigma_od_init_var)

        sep = ttk.Separator(box, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=4, pady=6)

        self._add_labeled_entry(box, "μ12 (weak)", self.mu12_var)
        self._add_labeled_entry(box, "μ23 (strong)", self.mu23_var)
        self._add_labeled_entry(box, "μ34 (weak)", self.mu34_var)

        sep2 = ttk.Separator(box, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=4, pady=6)

        self._add_labeled_entry(box, "max_iter", self.max_iter_var)
        self._add_labeled_entry(box, "mix (0..1)", self.mix_var)
        self._add_labeled_entry(box, "tol", self.tol_var, width=14)
        self._add_labeled_entry(box, "ridge", self.ridge_var, width=14)
        self._add_labeled_entry(box, "nu0_reg", self.nu0_reg_var, width=14)

        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(row, text="Symmetrize Σ each iter", variable=self.sym_each_iter_var).pack(anchor=tk.W)
        ttk.Checkbutton(row, text="RS-project Σ each iter", variable=self.rs_project_var).pack(anchor=tk.W)

        btns = ttk.Frame(box)
        btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Solve Exact", command=self._run_exact_thread).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Solve Protocol", command=self._run_proto_thread).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Solve Both", command=self._run_both_thread).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Clear Results", command=self._clear_results).pack(side=tk.RIGHT, padx=3)

    def _build_progress(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Progress")
        box.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(box, textvariable=self.status_exact, wraplength=390).pack(anchor=tk.W, padx=6, pady=2)
        pb1 = ttk.Progressbar(box, variable=self.progress_exact, maximum=1.0)
        pb1.pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(box, textvariable=self.status_proto, wraplength=390).pack(anchor=tk.W, padx=6, pady=2)
        pb2 = ttk.Progressbar(box, variable=self.progress_proto, maximum=1.0)
        pb2.pack(fill=tk.X, padx=6, pady=3)

    def _build_plot_controls(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Plot Controls")
        box.pack(fill=tk.X, padx=6, pady=6)

        # Site / replica selectors
        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="site i", width=8).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=4, textvariable=self.site_i_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="site j", width=8).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=4, textvariable=self.site_j_var, width=5).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(box)
        row2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row2, text="rep a", width=8).pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=0, to=50, textvariable=self.rep_a_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="rep b", width=8).pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=0, to=50, textvariable=self.rep_b_var, width=5).pack(side=tk.LEFT, padx=2)

        # Domain/component/which
        row3 = ttk.Frame(box)
        row3.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row3, text="Domain", width=10).pack(side=tk.LEFT)
        ttk.Combobox(row3, textvariable=self.domain_var, values=["ν", "τ"], width=8, state="readonly").pack(side=tk.LEFT)
        ttk.Label(row3, text="Component", width=10).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(row3, textvariable=self.component_var, values=["Re", "Im", "Abs"], width=8, state="readonly").pack(side=tk.LEFT)

        row4 = ttk.Frame(box)
        row4.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row4, text="Display", width=10).pack(side=tk.LEFT)
        ttk.Combobox(
            row4,
            textvariable=self.which_var,
            values=["Exact", "Protocol", "Overlay", "Diff (P-E)"],
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Button(row4, text="Plot", command=self._plot).pack(side=tk.RIGHT, padx=3)
        ttk.Button(row4, text="Save PDF", command=self._save_pdf).pack(side=tk.RIGHT, padx=3)

        # Style controls
        style = ttk.LabelFrame(box, text="Style")
        style.pack(fill=tk.X, padx=0, pady=6)
        self._add_labeled_entry(style, "Fig width (in)", self.fig_w_var)
        self._add_labeled_entry(style, "Fig height (in)", self.fig_h_var)
        self._add_labeled_entry(style, "Title fontsize", self.title_fs_var)
        self._add_labeled_entry(style, "Label fontsize", self.label_fs_var)
        self._add_labeled_entry(style, "Legend fontsize", self.legend_fs_var)
        self._add_labeled_entry(style, "Tick fontsize", self.tick_fs_var)

    def _build_available_lists(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Available Green's functions")
        box.pack(fill=tk.BOTH, expand=False, padx=6, pady=6)

        ttk.Label(box, text="Exact solver provides all 16 site pairs (i,j).\nProtocol provides:").pack(anchor=tk.W, padx=6, pady=2)
        self.proto_list = tk.Listbox(box, height=8)
        self.proto_list.pack(fill=tk.X, padx=6, pady=4)
        self._refresh_available_lists()

    def _build_plot(self, parent: ttk.Frame) -> None:
        # A dedicated host so we can fully replace the plot (no overlay artifacts)
        self.plot_host = ttk.Frame(parent)
        self.plot_host.pack(fill=tk.BOTH, expand=True)
        self.canvas = None
        self.fig = None
        self.ax = None
        self._plot_placeholder = ttk.Label(self.plot_host, text="No data yet.")
        self._plot_placeholder.pack(fill=tk.BOTH, expand=True)

    def _recreate_plot_canvas(self, fig_w: float, fig_h: float) -> None:
        # Destroy existing canvas/placeholder and rebuild with a fresh Figure.
        if getattr(self, 'canvas', None) is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None
        if getattr(self, '_plot_placeholder', None) is not None:
            try:
                self._plot_placeholder.destroy()
            except Exception:
                pass
            self._plot_placeholder = None
        self.fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_host)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    # -------------------------- parameter parsing --------------------------
    def _get_params(self) -> Tuple[SolverParams, float, float, float]:
        try:
            beta = float(self.beta_var.get())
            M = int(self.M_var.get())
            nrep = int(self.nrep_var.get())
            m2 = float(self.m2_var.get())
            J = float(self.J_var.get())
            q = int(self.q_var.get())
            sigma_d_init = float(self.sigma_d_init_var.get())
            sigma_od_init = float(self.sigma_od_init_var.get())
            mu12 = float(self.mu12_var.get())
            mu23 = float(self.mu23_var.get())
            mu34 = float(self.mu34_var.get())
            max_iter = int(self.max_iter_var.get())
            mix = float(self.mix_var.get())
            tol = float(self.tol_var.get())
            ridge = float(self.ridge_var.get())
            nu0_reg = float(self.nu0_reg_var.get())
            sym = bool(self.sym_each_iter_var.get())
            rsproj = bool(self.rs_project_var.get())
        except Exception as e:
            raise ValueError(f"Could not parse inputs: {e}")

        if M <= 8 or (M & (M - 1)) != 0:
            # Not strictly required, but FFT sizes that are powers of two are fastest.
            # We don't hard-fail; we just warn.
            pass
        if nrep <= 0:
            raise ValueError("nrep must be >= 1")
        if q < 3:
            raise ValueError("q must be >= 3")
        if not (0.0 < mix <= 1.0):
            raise ValueError("mix must be in (0,1]")
        if max_iter <= 0:
            raise ValueError("max_iter must be >= 1")

        sp = SolverParams(
            beta=beta,
            M=M,
            nrep=nrep,
            m2=m2,
            J=J,
            q_body=q,
            max_iter=max_iter,
            mix=mix,
            tol=tol,
            ridge=ridge,
            nu0_reg=nu0_reg,
            symmetrize_each_iter=sym,
            rs_project_each_iter=rsproj,
            sigma_d_init_tau=sigma_d_init,
            sigma_od_init_tau=sigma_od_init,
        )
        return sp, mu12, mu23, mu34

    # -------------------------- threading helpers --------------------------
    def _set_exact_progress(self, frac: float, msg: str) -> None:
        self.progress_exact.set(float(np.clip(frac, 0.0, 1.0)))
        self.status_exact.set(msg)
        self.update_idletasks()

    def _set_proto_progress(self, frac: float, msg: str) -> None:
        self.progress_proto.set(float(np.clip(frac, 0.0, 1.0)))
        self.status_proto.set(msg)
        self.update_idletasks()

    def _run_exact_thread(self) -> None:
        self._disable_buttons(True)
        self._set_exact_progress(0.0, "Exact solver: starting")
        t = threading.Thread(target=self._run_exact, daemon=True)
        t.start()

    def _run_proto_thread(self) -> None:
        self._disable_buttons(True)
        self._set_proto_progress(0.0, "Protocol solver: starting")
        t = threading.Thread(target=self._run_proto, daemon=True)
        t.start()

    def _run_both_thread(self) -> None:
        self._disable_buttons(True)
        self._set_exact_progress(0.0, "Exact solver: starting")
        self._set_proto_progress(0.0, "Protocol solver: queued")
        t = threading.Thread(target=self._run_both, daemon=True)
        t.start()

    def _disable_buttons(self, is_busy: bool) -> None:
        # Simple approach: disable all Buttons under the first parameter box.
        # Tkinter doesn't provide a global registry; we keep it minimal.
        state = "disabled" if is_busy else "normal"
        for child in self.winfo_children():
            for sub in child.winfo_children():
                for w in sub.winfo_children():
                    if isinstance(w, ttk.Button):
                        w.configure(state=state)

    def _run_exact(self) -> None:
        try:
            sp, mu12, mu23, mu34 = self._get_params()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Input error", str(e)))
            self.after(0, lambda: self._disable_buttons(False))
            return

        def cb(frac: float, msg: str) -> None:
            self.after(0, lambda: self._set_exact_progress(frac, msg))

        try:
            res = solve_sd_four_site_exact(mu12=mu12, mu23=mu23, mu34=mu34, params=sp, progress_cb=cb)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Exact solver error", str(e)))
            self.after(0, lambda: self._set_exact_progress(0.0, "Exact solver: error"))
            self.after(0, lambda: self._disable_buttons(False))
            return

        self.res_exact = res
        self.after(0, lambda: self._set_exact_progress(1.0, "Exact solver: done"))
        self.after(0, self._refresh_available_lists)
        self.after(0, lambda: self._disable_buttons(False))

    def _run_proto(self) -> None:
        try:
            sp, mu12, mu23, mu34 = self._get_params()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Input error", str(e)))
            self.after(0, lambda: self._disable_buttons(False))
            return

        def cb(frac: float, msg: str) -> None:
            self.after(0, lambda: self._set_proto_progress(frac, msg))

        try:
            res = solve_sd_four_site_protocol(mu12=mu12, mu23=mu23, mu34=mu34, params=sp, progress_cb=cb)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Protocol solver error", str(e)))
            self.after(0, lambda: self._set_proto_progress(0.0, "Protocol solver: error"))
            self.after(0, lambda: self._disable_buttons(False))
            return

        self.res_proto = res
        self.after(0, lambda: self._set_proto_progress(1.0, "Protocol solver: done"))
        self.after(0, self._refresh_available_lists)
        self.after(0, lambda: self._disable_buttons(False))

    def _run_both(self) -> None:
        # Run exact then protocol
        self._run_exact()
        self._run_proto()

    def _clear_results(self) -> None:
        self.res_exact = None
        self.res_proto = None
        self.progress_exact.set(0.0)
        self.progress_proto.set(0.0)
        self.status_exact.set("Exact solver: idle")
        self.status_proto.set("Protocol solver: idle")
        self._refresh_available_lists()
        self.ax.clear()
        self.ax.set_title("No data yet")
        self.canvas.draw()

    # -------------------------- availability lists --------------------------
    def _refresh_available_lists(self) -> None:
        self.proto_list.delete(0, tk.END)
        # For the 4-site protocol, all (i,j) are available after restoration.
        # We show them explicitly.
        pairs = [(i, j) for i in range(1, 5) for j in range(1, 5)]
        for (i, j) in pairs:
            self.proto_list.insert(tk.END, f"G[{i},{j}]  (available)")

    # -------------------------- plotting --------------------------
    def _extract_series(self, res: Dict[str, np.ndarray], domain: str, i: int, j: int, a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) for chosen domain and indices."""
        i0 = i - 1
        j0 = j - 1
        a0 = int(a)
        b0 = int(b)
        nrep = res["G_nu"].shape[-1]
        if a0 < 0 or b0 < 0 or a0 >= nrep or b0 >= nrep:
            raise ValueError(f"Replica indices out of range: a,b must be in [0,{nrep-1}]")

        if domain == "ν":
            x = res["nu"]
            y = res["G_nu"][:, i0, j0, a0, b0]
        else:
            x = res["tau"]
            y = res["G_tau"][:, i0, j0, a0, b0]
        return x, y

    def _plot(self) -> None:
        if self.res_exact is None and self.res_proto is None:
            messagebox.showinfo("No data", "Run at least one solver first.")
            return

        try:
            i = int(self.site_i_var.get())
            j = int(self.site_j_var.get())
            a = int(self.rep_a_var.get())
            b = int(self.rep_b_var.get())
        except Exception:
            messagebox.showerror("Input error", "Invalid site/replica indices")
            return

        domain = self.domain_var.get()
        comp = self.component_var.get()
        which = self.which_var.get()

        try:
            fig_w = float(self.fig_w_var.get())
            fig_h = float(self.fig_h_var.get())
        except Exception:
            fig_w, fig_h = 7.6, 4.6
        self._recreate_plot_canvas(fig_w, fig_h)
        self.ax.clear()

        def comp_fn(z: np.ndarray) -> np.ndarray:
            if comp == "Re":
                return np.real(z)
            if comp == "Im":
                return np.imag(z)
            return np.abs(z)

        title = f"G[{i},{j}]^{{a={a},b={b}}}  domain={domain}"

        # Extract and plot
        if which in ("Exact", "Overlay", "Diff (P-E)"):
            if self.res_exact is None:
                messagebox.showerror("Missing", "Exact results not available. Run 'Solve Exact' first.")
                return
        if which in ("Protocol", "Overlay", "Diff (P-E)"):
            if self.res_proto is None:
                messagebox.showerror("Missing", "Protocol results not available. Run 'Solve Protocol' first.")
                return

        if which == "Exact":
            x, y = self._extract_series(self.res_exact, domain, i, j, a, b)
            self.ax.plot(x, comp_fn(y), label="Exact")
        elif which == "Protocol":
            x, y = self._extract_series(self.res_proto, domain, i, j, a, b)
            self.ax.plot(x, comp_fn(y), label="Protocol")
        elif which == "Overlay":
            x1, y1 = self._extract_series(self.res_exact, domain, i, j, a, b)
            x2, y2 = self._extract_series(self.res_proto, domain, i, j, a, b)
            self.ax.plot(x1, comp_fn(y1), label="Exact")
            self.ax.plot(x2, comp_fn(y2), label="Protocol", linestyle="--")
            # Show a simple scalar error summary
            err = float(np.max(np.abs(y2 - y1)))
            title += f"   max|Δ|={err:.3e}"
        else:  # Diff (P-E)
            x1, y1 = self._extract_series(self.res_exact, domain, i, j, a, b)
            x2, y2 = self._extract_series(self.res_proto, domain, i, j, a, b)
            self.ax.plot(x1, comp_fn(y2 - y1), label="Protocol - Exact")

        self.ax.set_title(title, fontsize=int(self.title_fs_var.get()))
        self.ax.set_xlabel(r"$\nu_k$" if domain == "ν" else r"$\tau$", fontsize=int(self.label_fs_var.get()))
        self.ax.set_ylabel(
            rf"$\mathrm{{{comp}}}\,G_{{{i}{j}}}^{{{a}{b}}}$" if comp in ("Re","Im")
            else rf"$|G_{{{i},{j}}}^{{{a},{b}}}|$",
            fontsize=int(self.label_fs_var.get()),
        )
        self.ax.tick_params(labelsize=int(self.tick_fs_var.get()))
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(fontsize=int(self.legend_fs_var.get()))
        self.canvas.draw()

    def _save_pdf(self) -> None:
        if getattr(self, 'fig', None) is None:
            messagebox.showinfo("No plot", "Nothing to save yet — click Plot first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save plot as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, format="pdf", bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return
        messagebox.showinfo("Saved", f"Saved plot to:\n{path}")


def main() -> None:
    app = BenchmarkGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
