"""
Spatial and velocity grid classes for the SLDG-IMEX solver.
"""

import numpy as np
from util.helper_functions import legendre_quadrature


class XGrid:
    """One-dimensional spatial grid built from Gauss-Legendre nodes on each element."""

    def __init__(self, config):
        self.domain_left = config['domain_left']
        self.domain_right = config['domain_right']
        self.num_elements = config['num_elements']
        self.num_legendre = config['num_legendre']
        self.dx = (self.domain_right - self.domain_left) / self.num_elements
        self.num_nodes = self.num_elements * self.num_legendre
        self.gl_pts, self.gl_wts = legendre_quadrature(self.num_legendre, 0, 1)
        self.x = np.array([
            pt for i in range(self.num_elements)
            for pt in (self.gl_pts + i) * self.dx
        ])


class VGrid:
    """Two-dimensional velocity grid with Fourier frequency arrays.

    Parameters from *config*:
        Nv — number of grid points per direction
        Lv — half-width of the velocity domain
    """

    def __init__(self, config):
        self.Nv = config['Nv']
        self.Lv = config['Lv']

        # Spectral support parameters (Mouhot-Pareschi)
        self.S = self.Lv * (2 / (1 + 3 * np.sqrt(2)))
        self.Rv = 2 * self.S
        self.Mv = 4

        # Uniform velocity nodes
        self.dv = 2 * self.Lv / self.Nv
        self.v = self.dv * np.arange(-self.Nv / 2, self.Nv / 2)
        self.v1, self.v2 = np.meshgrid(self.v, self.v, indexing='ij')

        # Fourier frequencies
        self.k = (np.pi / self.Lv *
                  np.concatenate([np.arange(0, self.Nv / 2),
                                  np.arange(-self.Nv / 2, 0)]))
        self.l1, self.l2 = np.meshgrid(self.k, self.k, indexing='ij')
