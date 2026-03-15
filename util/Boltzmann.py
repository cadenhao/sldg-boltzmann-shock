"""
Spectral solver for the 2-D Boltzmann collision operator (Maxwell molecules).

Implements the Fourier-spectral method of Mouhot & Pareschi (2006) for
computing Q(f, g) on a uniform velocity grid.
"""

import numpy as np


class BoltzmannSolver:
    """Spectral Boltzmann collision operator on a 2-D velocity grid."""

    def __init__(self, vgrid):
        self.vgrid = vgrid

    def spectral_2v(self, f, g):
        """Evaluate the collision integral Q(f, g) on the 2-D velocity grid.

        Parameters
        ----------
        f, g : (Nv, Nv) arrays in velocity space

        Returns
        -------
        Q : (Nv, Nv) collision integral
        """
        l1, l2 = self.vgrid.l1, self.vgrid.l2
        Mv, Rv = self.vgrid.Mv, self.vgrid.Rv
        B = 1 / (2 * np.pi)

        theta = np.reshape(
            np.linspace(0.5 * np.pi / Mv, np.pi - 0.5 * np.pi / Mv, Mv),
            (1, 1, Mv))
        w_theta = np.pi / Mv

        # Kernel modes
        proj = l1[:, :, np.newaxis] * np.cos(theta) + l2[:, :, np.newaxis] * np.sin(theta)
        alpha = 2 * Rv * self._sinc(Rv * proj)
        alpha_ = 2 * Rv * self._sinc(
            Rv * np.sqrt(l1[:, :, np.newaxis]**2 + l2[:, :, np.newaxis]**2 - proj**2))

        f3 = f[:, :, np.newaxis]
        g3 = g[:, :, np.newaxis]
        fft_f = np.fft.fft2(f3, axes=(0, 1))
        fft_g = np.fft.fft2(g3, axes=(0, 1))

        Q = w_theta * 2 * B * np.real(np.sum(
            np.fft.ifft2(alpha * fft_f, axes=(0, 1)) *
            np.fft.ifft2(alpha_ * fft_g, axes=(0, 1)) -
            f3 * np.fft.ifft2(fft_g * alpha * alpha_, axes=(0, 1)),
            axis=2))
        return Q

    # Keep old API name as alias
    Spectral_2V = spectral_2v

    @staticmethod
    def _sinc(x):
        """Un-normalized sinc: sin(x)/x."""
        return np.sinc(x / np.pi)


# Backward-compatible alias
Boltzmann_solver = BoltzmannSolver
