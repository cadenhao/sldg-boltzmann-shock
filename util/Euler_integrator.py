"""
Euler-limit time integrator (transport + Maxwellian projection).

Derived from the IMEX integrator by dropping the collision operator and
projecting onto the local Maxwellian at each stage, which reduces the
Boltzmann equation to the compressible Euler equations.
"""

import numpy as np
from util.helper_functions import compute_moments_and_maxwellian, maxwellian


def euler_rk_step(rho_n, u1_n, u2_n, theta_n, dt, table,
                  sldg, xgrid, v, v1, v2):
    """Advance moments by one Euler-limit RK step.

    Works with macroscopic quantities and converts to/from distribution
    functions internally.

    Returns
    -------
    (rho, u1, u2, theta) at the next time level.
    """
    f_n = maxwellian(rho_n, u1_n, u2_n, theta_n, v1, v2)

    c_ex = table.c_ex
    A_tilde = table.a_im[1:, 1:]
    stages = table.stages

    f_stg = []

    for k in range(stages):
        Sf_k = sldg.integrate(f_n, v, c_ex[k] * dt)

        if k <= 1:
            M_k, *_ = compute_moments_and_maxwellian(Sf_k, v, v1, v2)
        else:
            SF = np.array([
                sldg.integrate(f_stg[i], v, (c_ex[k] - c_ex[i]) * dt)
                for i in range(k)])
            Ainv = np.linalg.inv(A_tilde[:k-1, :k-1])
            w = A_tilde[k-1, :k-1] @ Ainv
            macro = ((1 - w @ np.ones(k-1)) * Sf_k +
                     np.tensordot(w, SF[1:k], axes=([0], [0])))
            M_k, *_ = compute_moments_and_maxwellian(macro, v, v1, v2)

        f_stg.append(M_k)

    _, _, rho, u1, u2, theta = compute_moments_and_maxwellian(
        f_stg[-1], v, v1, v2)
    return rho, u1, u2, theta
