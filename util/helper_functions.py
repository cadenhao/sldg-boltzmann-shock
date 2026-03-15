"""
Helper functions for the SLDG-IMEX Boltzmann solver.

Provides Gauss-Legendre quadrature, Lagrange interpolation,
the 2-D Maxwellian distribution, and moment computation.
"""

import numpy as np
from scipy.integrate import trapezoid


def legendre_quadrature(n, a, b):
    """Gauss-Legendre points and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    w = 0.5 * (b - a) * w
    return x, w


def lagrange_at(pts, x):
    """Evaluate Lagrange basis polynomials defined by *pts* at query points *x*.

    Returns an array of shape (len(pts), len(x)) where entry [i, j] is l_i(x_j).
    """
    pts = np.asarray(pts, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(pts)
    L = np.ones((n, len(x)), dtype=float)
    for i in range(n):
        for j in range(n):
            if j != i:
                L[i] *= (x - pts[j]) / (pts[i] - pts[j])
    return L


def maxwellian(rho, u1, u2, theta, v1, v2):
    """Evaluate the 2-D Maxwellian distribution M[rho, u, theta](v).

    Parameters
    ----------
    rho, u1, u2, theta : (Nx,) arrays — macroscopic quantities
    v1, v2             : (Nv, Nv) arrays — velocity mesh

    Returns
    -------
    M : (Nx, Nv, Nv) array
    """
    v1 = v1[np.newaxis, :, :]
    v2 = v2[np.newaxis, :, :]
    rho = rho[:, np.newaxis, np.newaxis]
    u1 = u1[:, np.newaxis, np.newaxis]
    u2 = u2[:, np.newaxis, np.newaxis]
    theta = theta[:, np.newaxis, np.newaxis]

    return rho / (2 * np.pi * theta) * np.exp(
        -((u1 - v1)**2 + (u2 - v2)**2) / (2 * theta))


# Keep old name as alias for backward compatibility
Maxwellian = maxwellian


def compute_moments_and_maxwellian(f, v, v1, v2):
    """Compute macroscopic moments and the local Maxwellian from f.

    Parameters
    ----------
    f        : (Nx, Nv, Nv) distribution function
    v        : (Nv,) velocity nodes
    v1, v2   : (Nv, Nv) velocity meshgrid

    Returns
    -------
    M      : (Nx, Nv, Nv) Maxwellian
    beta   : (Nx, 1, 1) penalisation weight (= rho)
    rho    : (Nx,) density
    u1, u2 : (Nx,) bulk velocities
    theta  : (Nx,) temperature
    """
    rho = trapezoid(trapezoid(f, v, axis=2), v, axis=1)
    u1 = (trapezoid(trapezoid(v1[np.newaxis] * f, v, axis=2), v, axis=1)
           / rho)
    u2 = (trapezoid(trapezoid(v2[np.newaxis] * f, v, axis=2), v, axis=1)
           / rho)
    theta = 0.5 * trapezoid(
        trapezoid(
            ((u1[:, np.newaxis, np.newaxis] - v1[np.newaxis])**2 +
             (u2[:, np.newaxis, np.newaxis] - v2[np.newaxis])**2) * f,
            v, axis=2),
        v, axis=1) / rho

    M = maxwellian(rho, u1, u2, theta, v1, v2)
    beta = rho[:, np.newaxis, np.newaxis]
    return M, beta, rho, u1, u2, theta
