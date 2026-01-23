#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble miner GUI for the SDRG-on-SD *periodic even L* chain protocol with *sparse* restoration.

CHANGES (2026-01-13)
--------------------
1) The SDRG pipeline and GUI are unchanged.
2) Data storage is now *raw per-sample data* in Matsubara frequency:
      - For each sample s, we store every restored Green's function G_{xy}^{ab}(iν_m) available
        from the sparse restoration stage, as complex numbers on the Matsubara grid.
      - NO ensemble averaging / typical values are computed in the miner.

Because the number of restored directed pairs (x→y) varies per sample, the output is stored
in an appendable HDF5 database with a CSR-like layout:

  offsets[s]..offsets[s+1]-1  are the records belonging to sample s.

Datasets
--------
- beta: float64 attribute
- nu:   (M,) float64 dataset
- L, nrep, M: int attributes
- Nsamp_requested: int attribute
- Nsamp_done: int attribute
- sample_offsets: (Nsamp_done+1,) int64 dataset   (prefix sums, offsets[0]=0)
- pairs_i_all: (Nrec,) int32 dataset
- pairs_j_all: (Nrec,) int32 dataset
- G_iw_all:    (Nrec, M, nrep, nrep) complex128 dataset  (record-major)
- sample_seed: (Nsamp_done,) int64 dataset
- sample_mu_scalars: (Nsamp_done, L) float64 dataset
- params_json: UTF-8 JSON string dataset (run settings)

This format is designed so the analysis GUI can compute ensemble mean/typical *on demand*
for any selected (x,y,a,b) without re-running the miner.
"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

try:
    import h5py
except Exception as e:
    raise RuntimeError(
        "This version writes raw per-sample data to an HDF5 database and requires 'h5py'.\n"
        f"Import error: {e}"
    )

try:
    import Single_protocol_miner as base
except Exception as e:
    raise RuntimeError(
        "Cannot import the single-sample protocol engine 'Single_protocol_miner'.\n"
        "Place this ensemble miner in the same folder as 'Single_protocol_miner.py'.\n\n"
        f"Import error: {e}"
    )


class EnsembleMinerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDRG-on-SD Ensemble Miner (Periodic Even L, Sparse Restore) — RAW DB")
        self.geometry("1200x760")

        self._build_vars()
        self._build_ui()

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_flag = False

    # -----------------
    # UI state
    # -----------------

    def _build_vars(self):
        # Model / numerics
        self.var_L = tk.IntVar(value=10)
        self.var_beta = tk.DoubleVar(value=10.0)
        self.var_M = tk.IntVar(value=128)
        self.var_q = tk.IntVar(value=4)
        self.var_J = tk.DoubleVar(value=1.0)
        self.var_m2 = tk.DoubleVar(value=1.0)
        self.var_nrep = tk.IntVar(value=6)

        # Two-site SD solver controls
        self.var_mixing = tk.DoubleVar(value=0.35)
        self.var_tol = tk.DoubleVar(value=1e-10)
        self.var_max_iter = tk.IntVar(value=250)
        self.var_ridge = tk.DoubleVar(value=1e-12)
        self.var_enforce_sym = tk.BooleanVar(value=True)

        # Initial Sigma(τ) constants (replica-symmetric seed)
        self.var_sig_d_tau = tk.DoubleVar(value=0.0)
        self.var_sig_od_tau = tk.DoubleVar(value=0.0)

        # Ensemble
        self.var_Nsamp = tk.IntVar(value=50)
        self.var_seed = tk.StringVar(value="")

        # Distribution
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

        # Output
        self.var_outfile = tk.StringVar(value="")

        # Progress
        self.var_status = tk.StringVar(value="Idle.")
        self.var_prog_samples = tk.DoubleVar(value=0.0)
        self.var_prog_inner = tk.DoubleVar(value=0.0)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ----------------
        # Left: controls
        # ----------------

        # Run controls
        runbox = ttk.LabelFrame(left, text="Run")
        runbox.pack(fill=tk.X, pady=(0, 10))

        btnrow = ttk.Frame(runbox)
        btnrow.pack(fill=tk.X, padx=6, pady=6)
        self.btn_choose = ttk.Button(btnrow, text="Choose output .h5...", command=self._choose_outfile)
        self.btn_choose.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_run = ttk.Button(btnrow, text="Run ensemble", command=self._run)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btnrow2 = ttk.Frame(runbox)
        btnrow2.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.btn_stop = ttk.Button(btnrow2, text="Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(runbox, textvariable=self.var_outfile, wraplength=520).pack(fill=tk.X, padx=6, pady=(0, 6))

        # Status
        statbox = ttk.LabelFrame(left, text="Progress")
        statbox.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(statbox, textvariable=self.var_status, wraplength=520, justify="left").pack(fill=tk.X, padx=6, pady=(6, 6))

        ttk.Label(statbox, text="Samples").pack(anchor="w", padx=6)
        self.pb_samples = ttk.Progressbar(statbox, variable=self.var_prog_samples, maximum=100.0)
        self.pb_samples.pack(fill=tk.X, padx=6, pady=(0, 6))

        ttk.Label(statbox, text="Within-sample (stage)").pack(anchor="w", padx=6)
        self.pb_inner = ttk.Progressbar(statbox, variable=self.var_prog_inner, maximum=100.0)
        self.pb_inner.pack(fill=tk.X, padx=6, pady=(0, 6))

        # Model parameters
        model = ttk.LabelFrame(left, text="Model / numerics")
        model.pack(fill=tk.X, pady=(0, 10))
        model.columnconfigure(0, weight=0)
        model.columnconfigure(1, weight=1)

        row = 0
        for lab, var in [
            ("L (even)", self.var_L),
            ("beta", self.var_beta),
            ("M (Matsubara modes)", self.var_M),
            ("q", self.var_q),
            ("J", self.var_J),
            ("m^2", self.var_m2),
            ("nrep", self.var_nrep),
            ("Nsamples", self.var_Nsamp),
            ("seed (optional)", self.var_seed),
        ]:
            ttk.Label(model, text=lab).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(model, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
            row += 1

        # Solver controls
        solver = ttk.LabelFrame(left, text="Two-site SD solver")
        solver.pack(fill=tk.X, pady=(0, 10))
        solver.columnconfigure(0, weight=0)
        solver.columnconfigure(1, weight=1)

        row = 0
        for lab, var in [
            ("mixing alpha", self.var_mixing),
            ("tol", self.var_tol),
            ("max_iter", self.var_max_iter),
            ("ridge", self.var_ridge),
            ("seed Sigma_d(τ)", self.var_sig_d_tau),
            ("seed Sigma_od(τ)", self.var_sig_od_tau),
        ]:
            ttk.Label(solver, text=lab).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(solver, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
            row += 1

        ttk.Checkbutton(
            solver, text="enforce symmetry (Hermitian symmetrize)", variable=self.var_enforce_sym
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(3, 6))

        # Distribution controls on the right
        distbox = ttk.LabelFrame(right, text="Bond distribution (draw per sample)")
        distbox.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(distbox)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Distribution").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        cb = ttk.Combobox(
            top,
            textvariable=self.var_dist,
            state="readonly",
            values=["Gaussian", "Uniform", "Fermi-Dirac", "Exponential", "Gamma"],
            width=18,
        )
        cb.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Checkbutton(top, text="abs(value)", variable=self.var_abs).grid(row=0, column=2, sticky="w", padx=8, pady=4)

        # parameter grid
        grid = ttk.Frame(distbox)
        grid.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        grid.columnconfigure(0, weight=0)
        grid.columnconfigure(1, weight=1)

        params = [
            ("Gaussian mean", self.var_gauss_mean),
            ("Gaussian std", self.var_gauss_std),
            ("Uniform low", self.var_uni_low),
            ("Uniform high", self.var_uni_high),
            ("Fermi-Dirac loc", self.var_fd_loc),
            ("Fermi-Dirac scale", self.var_fd_scale),
            ("Exponential scale", self.var_exp_scale),
            ("Gamma shape", self.var_gamma_shape),
            ("Gamma scale", self.var_gamma_scale),
        ]

        for r, (lab, var) in enumerate(params):
            ttk.Label(grid, text=lab).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(grid, textvariable=var).grid(row=r, column=1, sticky="ew", padx=4, pady=3)

        help_txt = (
            "Notes:\n"
            "- For 'Fermi-Dirac' we use a logistic distribution.\n"
            "- 'abs(value)' is applied after drawing.\n"
            "- Each sample draws L independent bond scalars and initializes μ_x(iν) = μ_x I_{rep}.\n"
            "- Output is a RAW HDF5 database (no averaging in miner)."
        )
        ttk.Label(distbox, text=help_txt, justify="left", wraplength=600).pack(fill=tk.X, padx=8, pady=(0, 8))

    # -----------------
    # Actions
    # -----------------

    def _choose_outfile(self):
        path = filedialog.asksaveasfilename(
            title="Save RAW ensemble database",
            defaultextension=".h5",
            filetypes=[("HDF5 database", "*.h5 *.hdf5"), ("All files", "*")],
        )
        if path:
            self.var_outfile.set(path)

    def _stop(self):
        self._stop_flag = True
        self.var_status.set("Stop requested. Waiting for current step to finish...")
        self.btn_stop.configure(state=tk.DISABLED)

    def _validate(self):
        L = int(self.var_L.get())
        if L < 4 or (L % 2) != 0:
            raise ValueError("L must be an even integer >= 4.")
        M = int(self.var_M.get())
        if M < 8:
            raise ValueError("M must be >= 8.")
        q = int(self.var_q.get())
        if q % 2 != 0 or q < 2:
            raise ValueError("q must be even and >= 2.")
        Ns = int(self.var_Nsamp.get())
        if Ns <= 0:
            raise ValueError("Nsamples must be positive.")
        nrep = int(self.var_nrep.get())
        if nrep <= 0:
            raise ValueError("nrep must be positive.")

    def _run(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            messagebox.showwarning("Busy", "A run is already in progress.")
            return

        try:
            self._validate()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        outfp = self.var_outfile.get().strip()
        if not outfp:
            messagebox.showerror("Missing output", "Choose an output .h5 file first.")
            return

        # Create/overwrite safeguard
        if os.path.exists(outfp):
            if not messagebox.askyesno("Overwrite?", f"File exists:\n{outfp}\n\nOverwrite?"):
                return

        self._stop_flag = False
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_choose.configure(state=tk.DISABLED)

        self.var_prog_samples.set(0.0)
        self.var_prog_inner.set(0.0)
        self.var_status.set("Starting...")

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        h5f = None
        ns_done = 0
        try:
            outfp = self.var_outfile.get().strip()
            L = int(self.var_L.get())
            beta = float(self.var_beta.get())
            M = int(self.var_M.get())
            q = int(self.var_q.get())
            J = float(self.var_J.get())
            m2 = float(self.var_m2.get())
            nrep = int(self.var_nrep.get())
            Ns = int(self.var_Nsamp.get())

            seed_str = self.var_seed.get().strip()
            base_seed = None if seed_str == "" else int(seed_str)

            p = base.TwoSiteParams(
                beta=beta,
                M=M,
                q=q,
                J=J,
                m2=m2,
                nrep=nrep,
                mixing=float(self.var_mixing.get()),
                tol=float(self.var_tol.get()),
                max_iter=int(self.var_max_iter.get()),
                ridge=float(self.var_ridge.get()),
                enforce_symmetry=bool(self.var_enforce_sym.get()),
            )

            sig_d_tau = float(self.var_sig_d_tau.get())
            sig_od_tau = float(self.var_sig_od_tau.get())

            dist = self.var_dist.get().strip().lower()
            abs_value = bool(self.var_abs.get())

            # Fix a RNG for the ensemble; each sample uses an independent stream
            rng = np.random.default_rng(base_seed)

            def set_status(msg: str):
                self.after(0, lambda: self.var_status.set(msg))

            def set_sample_prog(frac01: float):
                self.after(0, lambda: self.var_prog_samples.set(100.0 * frac01))

            def set_inner_prog(val: float):
                self.after(0, lambda: self.var_prog_inner.set(float(val)))

            # Precompute nu once
            nu = base.matsubara_bosonic(beta, M)

            # --- Create the HDF5 database ---
            set_status("Creating HDF5 database...")
            h5f = h5py.File(outfp, "w")
            h5f.attrs["schema"] = "raw_pairs_v1"
            h5f.attrs["beta"] = float(beta)
            h5f.attrs["L"] = int(L)
            h5f.attrs["nrep"] = int(nrep)
            h5f.attrs["M"] = int(M)
            h5f.attrs["Nsamp_requested"] = int(Ns)

            h5f.create_dataset("nu", data=nu.astype(np.float64))

            # Offsets: fixed-size; we will fill as we go (offsets[0]=0)
            d_offsets = h5f.create_dataset("sample_offsets", shape=(Ns + 1,), dtype=np.int64)
            d_offsets[...] = 0

            d_seed = h5f.create_dataset("sample_seed", shape=(Ns,), dtype=np.int64)
            d_mu = h5f.create_dataset("sample_mu_scalars", shape=(Ns, L), dtype=np.float64)

            # Appendable record datasets
            chunk_pairs = 256  # number of (x->y) records per chunk in pair arrays
            d_pi = h5f.create_dataset(
                "pairs_i_all",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(chunk_pairs,),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            d_pj = h5f.create_dataset(
                "pairs_j_all",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(chunk_pairs,),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            d_G = h5f.create_dataset(
                "G_iw_all",
                shape=(0, M, nrep, nrep),
                maxshape=(None, M, nrep, nrep),
                dtype=np.complex128,
                chunks=(1, M, nrep, nrep),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            params = {
                "engine": "periodic_even_sparse_protocol",
                "storage": {"schema": "raw_pairs_v1", "note": "raw per-sample restored G(iw) stored; no averaging in miner"},
                "L": L,
                "beta": beta,
                "M": M,
                "q": q,
                "J": J,
                "m2": m2,
                "nrep": nrep,
                "Nsamp_requested": Ns,
                "seed": base_seed,
                "distribution": {
                    "name": self.var_dist.get(),
                    "abs": abs_value,
                    "gaussian": {"mean": float(self.var_gauss_mean.get()), "std": float(self.var_gauss_std.get())},
                    "uniform": {"low": float(self.var_uni_low.get()), "high": float(self.var_uni_high.get())},
                    "fermi_dirac_logistic": {"loc": float(self.var_fd_loc.get()), "scale": float(self.var_fd_scale.get())},
                    "exponential": {"scale": float(self.var_exp_scale.get())},
                    "gamma": {"shape": float(self.var_gamma_shape.get()), "scale": float(self.var_gamma_scale.get())},
                },
                "two_site_solver": {
                    "mixing": float(self.var_mixing.get()),
                    "tol": float(self.var_tol.get()),
                    "max_iter": int(self.var_max_iter.get()),
                    "ridge": float(self.var_ridge.get()),
                    "enforce_symmetry": bool(self.var_enforce_sym.get()),
                    "init_sigma_tau": {"sig_d": sig_d_tau, "sig_od": sig_od_tau},
                },
                "note": (
                    "Sparse restoration per appendix: not all G_{xy} exist in every sample. "
                    "Database stores only those restored by the protocol."
                ),
            }
            dt_str = h5py.string_dtype(encoding="utf-8")
            h5f.create_dataset("params_json", data=json.dumps(params, indent=2), dtype=dt_str)

            # --- Main loop ---
            set_status("Running ensemble (raw storage)...")
            total_records = 0
            ns_done = 0

            for s in range(Ns):
                if self._stop_flag:
                    raise RuntimeError("Stopped by user.")

                # Map dist name to base's accepted names
                if dist in ("gaussian", "normal"):
                    dist_name = "Gaussian"
                elif dist in ("uniform",):
                    dist_name = "Uniform"
                elif dist in ("fermi-dirac", "fermi", "fd", "logistic"):
                    dist_name = "Fermi-Dirac"
                elif dist in ("exponential", "exp"):
                    dist_name = "Exponential"
                elif dist in ("gamma",):
                    dist_name = "Gamma"
                else:
                    dist_name = self.var_dist.get().strip()

                sample_seed = int(rng.integers(0, 2**31 - 1))
                d_seed[s] = sample_seed

                mu_scalars = base.sample_bond_scalars(
                    L=L,
                    mode="Random",
                    seed=sample_seed,
                    mu_uniform=0.0,
                    mu_weak=0.0,
                    mu_strong=0.0,
                    strong_pos=0,
                    mu_list_str="",
                    dist=dist_name,
                    abs_value=abs_value,
                    gauss_mean=float(self.var_gauss_mean.get()),
                    gauss_std=float(self.var_gauss_std.get()),
                    uni_low=float(self.var_uni_low.get()),
                    uni_high=float(self.var_uni_high.get()),
                    fd_loc=float(self.var_fd_loc.get()),
                    fd_scale=float(self.var_fd_scale.get()),
                    exp_scale=float(self.var_exp_scale.get()),
                    gamma_shape=float(self.var_gamma_shape.get()),
                    gamma_scale=float(self.var_gamma_scale.get()),
                ).astype(np.float64)
                d_mu[s, :] = mu_scalars

                mu_init = [base._scalar_to_mu_matrix(M, nrep, float(mu_scalars[x])) for x in range(L)]

                def per_sample_cb(stage: str, step: int, stepmax: int, it: int, itmax: int, resid: float):
                    # inner progress: 0..60 forward/terminal, 60..100 restore
                    if stage in ("local_sd", "forward"):
                        frac = (step / max(1, stepmax))
                        set_inner_prog(60.0 * frac)
                    elif stage in ("terminal_sd",):
                        set_inner_prog(60.0)
                    elif stage in ("restore",):
                        frac = (step / max(1, stepmax))
                        set_inner_prog(60.0 + 40.0 * frac)
                    else:
                        set_inner_prog(0.0)

                    if stage in ("local_sd", "terminal_sd"):
                        msg = f"Sample {s+1}/{Ns}: {stage} iter {it}/{itmax}, resid={resid:.3e}"
                    else:
                        msg = f"Sample {s+1}/{Ns}: {stage} {step}/{stepmax}"
                    set_status(msg)

                payload = base.run_protocol_periodic_even_sparse(
                    L=L,
                    mu_init=mu_init,
                    p=p,
                    sig_d_tau_init=sig_d_tau,
                    sig_od_tau_init=sig_od_tau,
                    progress_cb=per_sample_cb,
                )

                pairs_i = payload["pairs_i"].astype(np.int32)
                pairs_j = payload["pairs_j"].astype(np.int32)
                G_pairs_iw = payload["G_pairs_iw"].astype(np.complex128)  # (M,P,n,n)

                P = int(pairs_i.shape[0])

                # Append new records
                if P > 0:
                    new_total = total_records + P
                    d_pi.resize((new_total,))
                    d_pj.resize((new_total,))
                    d_G.resize((new_total, M, nrep, nrep))

                    d_pi[total_records:new_total] = pairs_i
                    d_pj[total_records:new_total] = pairs_j
                    # Store record-major (P,M,n,n) for sequential reads by record
                    d_G[total_records:new_total, :, :, :] = np.transpose(G_pairs_iw, (1, 0, 2, 3))

                    total_records = new_total

                d_offsets[s + 1] = total_records
                ns_done = s + 1

                set_sample_prog(ns_done / Ns)

            # Finalize
            h5f.attrs["Nsamp_done"] = int(ns_done)

            set_inner_prog(100.0)
            set_status(f"Done. Saved RAW database to: {outfp}")

            self.after(0, lambda: messagebox.showinfo("Saved", f"Saved RAW ensemble database to\n{outfp}"))

        except Exception as e:
            msg = str(e)
            if msg.strip().lower().startswith("stopped by user"):
                self.after(0, lambda: messagebox.showinfo("Stopped", msg))
            else:
                self.after(0, lambda: messagebox.showerror("Error", msg))
        finally:
            try:
                if h5f is not None:
                    # Ensure partial runs still record how many samples were completed.
                    if "Nsamp_done" not in h5f.attrs:
                        h5f.attrs["Nsamp_done"] = int(ns_done)
                    h5f.flush()
                    h5f.close()
            except Exception:
                pass
            self.after(0, self._finish_ui)

    def _finish_ui(self):
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_choose.configure(state=tk.NORMAL)


def main():
    app = EnsembleMinerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
