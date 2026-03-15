"""
IMEX Runge-Kutta time integrator for the Boltzmann equation.

Handles splitting into:
  - Explicit part: transport via the SLDG operator
  - Implicit part: BGK-penalised collision operator

Supports arbitrary GSA (globally stiffly accurate) IMEX-RK Butcher tables.
"""

import numpy as np
from util.helper_functions import compute_moments_and_maxwellian


def imex_rk_step(f_n, dt, epsilon, table, sldg, Q_solver, xgrid, v, v1, v2):
    """Advance the distribution function by one IMEX-RK time step.

    Parameters
    ----------
    f_n     : (Nx, Nv, Nv) current distribution
    dt      : float, time step
    epsilon : float, Knudsen number
    table   : IMEXRKTable with stages, c_ex, a_ex, c_im, a_im
    sldg    : SLDGIntegrator (transport operator)
    Q_solver: BoltzmannSolver (collision operator)
    xgrid   : XGrid
    v, v1, v2 : velocity arrays

    Returns
    -------
    f_{n+1} : (Nx, Nv, Nv) updated distribution (last stage for GSA schemes)
    """
    c_ex, a_ex = table.c_ex, table.a_ex
    c_im, a_im = table.c_im, table.a_im
    stages = table.stages

    A_tilde = a_im[1:, 1:]

    f_stg, M_stg, beta_stg, Q_stg = [], [], [], []

    for k in range(stages):
        Sf_k = sldg.integrate(f_n, v, c_ex[k] * dt)

        if k == 0:
            M_k, beta_k, *_ = compute_moments_and_maxwellian(Sf_k, v, v1, v2)
            akk = a_im[k][k]
            f_k = (beta_k * akk * dt * M_k + epsilon * Sf_k) / (epsilon + beta_k * akk * dt)
            M_k, beta_k, *_ = compute_moments_and_maxwellian(f_k, v, v1, v2)
            Q_k = np.zeros_like(f_k)
            for ix in range(xgrid.num_nodes):
                Q_k[ix] = Q_solver.spectral_2v(f_k[ix], f_k[ix])
        else:
            # Macro prediction via Shu-Osher form for stages k >= 2
            if k == 1:
                M_k, beta_k, *_ = compute_moments_and_maxwellian(Sf_k, v, v1, v2)
            else:
                SF = np.array([
                    sldg.integrate(f_stg[i], v, (c_ex[k] - c_ex[i]) * dt)
                    for i in range(k)])
                Ainv = np.linalg.inv(A_tilde[:k-1, :k-1])
                w = A_tilde[k-1, :k-1] @ Ainv
                macro = (1 - w @ np.ones(k-1)) * Sf_k + np.tensordot(w, SF[1:k], axes=([0], [0]))
                M_k, beta_k, *_ = compute_moments_and_maxwellian(macro, v, v1, v2)

            EX = np.zeros_like(f_n)
            IM = np.zeros_like(f_n)
            for j in range(k):
                pen = (Q_stg[j] - beta_stg[j] * (M_stg[j] - f_stg[j])) / epsilon
                EX += a_ex[k][j] * dt * sldg.integrate(pen, v, (c_ex[k] - c_ex[j]) * dt)
                IM += a_im[k][j] * dt * sldg.integrate(
                    beta_stg[j] * (M_stg[j] - f_stg[j]) / epsilon,
                    v, (c_im[k] - c_im[j]) * dt)

            akk = a_im[k][k]
            f_k = (beta_k * akk * dt * M_k + epsilon * (Sf_k + EX + IM)) / (epsilon + beta_k * akk * dt)
            M_k, beta_k, *_ = compute_moments_and_maxwellian(f_k, v, v1, v2)

            Q_k = np.zeros_like(f_k)
            for ix in range(xgrid.num_nodes):
                Q_k[ix] = Q_solver.spectral_2v(f_k[ix], f_k[ix])

        f_stg.append(f_k)
        M_stg.append(M_k)
        beta_stg.append(beta_k)
        Q_stg.append(Q_k)

    return f_stg[-1]
