"""
Semi-Lagrangian Discontinuous Galerkin (SLDG) transport integrator.

Computes S(dt) f — the exact transport operator for v . nabla_x f —
using a DG framework with Lagrange interpolation on Gauss-Legendre nodes,
combined with an optional bound-preserving (BP/LMPP) limiter.
"""

import numpy as np
from util.helper_functions import lagrange_at


class SLDGIntegrator:
    """SLDG transport operator with configurable boundary and limiter."""

    def __init__(self, xgrid, boundary_type='periodic',
                 limiter='None', BL=None, BR=None):
        self.xgrid = xgrid
        self.boundary_type = boundary_type
        self.limiter = limiter
        self.BL = BL
        self.BR = BR

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def integrate(self, f, v, dt):
        """Apply the SLDG transport operator: Sf = S(dt) f.

        Parameters
        ----------
        f  : (Nx, Nv, Nv) distribution function on nodal DG grid
        v  : (Nv,) velocity nodes
        dt : float, time step (may include an RK-coefficient factor)

        Returns
        -------
        Sf : (Nx, Nv, Nv) transported distribution
        """
        xg = self.xgrid
        gl_pts = xg.gl_pts
        num_el = xg.num_elements
        n_leg = xg.num_legendre
        dx = xg.dx
        Nv = len(v)

        Sf = np.zeros_like(f)

        for ie in range(num_el):
            e_left = ie * dx
            for iv in range(Nv):
                shift = v[iv] * dt
                A, B = self._updating_matrices(shift, dx, n_leg)

                ups_left = e_left - shift
                lid, rid = self._search_segments(ups_left)
                lid_bc = self._apply_bc(lid)
                rid_bc = self._apply_bc(rid)

                f_left = self._fetch_element(f, lid_bc, iv, n_leg)
                f_right = self._fetch_element(f, rid_bc, iv, n_leg)

                sl = slice(ie * n_leg, (ie + 1) * n_leg)
                if self.limiter == 'BP':
                    Sf[sl, iv, :] = self._bp_limit(
                        f_left, f_right, A, B,
                        e_left, dx, gl_pts, lid_bc, rid_bc)
                else:
                    Sf[sl, iv, :] = (
                        np.tensordot(A, f_left, axes=([1], [0])) +
                        np.tensordot(B, f_right, axes=([1], [0])))

        return Sf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_element(self, f, elem_id, iv, n_leg):
        """Retrieve nodal values for one element, respecting inflow BCs."""
        n_el = self.xgrid.num_elements
        if self.boundary_type == 'inflow':
            if elem_id < 1e-9:
                return self.BL[:, iv, :]
            if elem_id - n_el + 1 > 1e-9:
                return self.BR[:, iv, :]
        s = slice(elem_id * n_leg, (elem_id + 1) * n_leg)
        return f[s, iv, :]

    def _bp_limit(self, f_left, f_right, A, B,
                  e_left, dx, gl_pts, lid, rid):
        """Bound-preserving (LMPP) limiter applied after prediction."""
        f_pred = (np.tensordot(A, f_left, axes=([1], [0])) +
                  np.tensordot(B, f_right, axes=([1], [0])))

        pts_here = gl_pts * dx + e_left
        x_fine = np.linspace(e_left, e_left + dx, 6)
        gl_l = gl_pts * dx + lid * dx
        gl_r = gl_pts * dx + rid * dx
        fine_l = np.linspace(lid * dx, (lid + 1) * dx, 6)
        fine_r = np.linspace(rid * dx, (rid + 1) * dx, 6)

        fp_eval = np.tensordot(lagrange_at(pts_here, x_fine), f_pred,
                               axes=([0], [0]))
        fl_eval = np.tensordot(lagrange_at(gl_l, fine_l), f_left,
                               axes=([0], [0]))
        fr_eval = np.tensordot(lagrange_at(gl_r, fine_r), f_right,
                               axes=([0], [0]))

        mean = fp_eval.mean(axis=0)
        M_new = fp_eval.max(axis=0)
        m_new = fp_eval.min(axis=0)
        M_old = np.maximum(fl_eval.max(axis=0), fr_eval.max(axis=0))
        m_old = np.minimum(fl_eval.min(axis=0), fr_eval.min(axis=0))

        EPS = 1e-9
        th_max = np.where(M_new - M_old > EPS,
                          (M_old - mean) / (M_new - mean + EPS), 1.0)
        th_min = np.where(m_new - m_old < -EPS,
                          (m_old - mean) / (m_new - mean + EPS), 1.0)
        lam = np.minimum(np.minimum(th_max, th_min), 1.0)
        lam = lam.reshape(1, 1, -1)

        return (f_pred - mean) * lam + mean

    def _updating_matrices(self, shift, dx, size):
        """Build the mass-transfer matrices A, B for a given shift."""
        gl_pts, gl_wts = self.xgrid.gl_pts, self.xgrid.gl_wts

        alpha = 1 - (shift % dx) / dx
        if abs(alpha - 1) < 1e-9:
            alpha = 0

        w_pj = gl_wts[:, np.newaxis, np.newaxis]
        w_q = gl_wts[np.newaxis, np.newaxis, :]

        l_A = lagrange_at(gl_pts, alpha + gl_pts * (1 - alpha))[np.newaxis]
        lj_A = lagrange_at(gl_pts, gl_pts * (1 - alpha))[:, np.newaxis]
        A = np.sum((1 - alpha) / w_pj * w_q * l_A * lj_A, axis=2)

        l_B = lagrange_at(gl_pts, alpha * gl_pts)[np.newaxis]
        lj_B = lagrange_at(gl_pts, alpha * (gl_pts - 1) + 1)[:, np.newaxis]
        B = np.sum(alpha / w_pj * w_q * l_B * lj_B, axis=2)

        return A, B

    def _search_segments(self, ups_left):
        """Find the upstream element indices for the back-traced interval."""
        dx = self.xgrid.dx
        raw = (ups_left - self.xgrid.domain_left) / dx
        if abs(raw - round(raw)) < 1e-9:
            left_id = int(round(raw))
        else:
            left_id = int(np.floor(raw))
        return left_id, left_id + 1

    def _apply_bc(self, idx):
        """Map an element index through the chosen boundary condition."""
        n = self.xgrid.num_elements
        if self.boundary_type == 'periodic':
            return idx % n
        if self.boundary_type == 'neumann':
            if idx < 1e-9:
                return -idx
            if idx - n + 1 > 1e-9:
                return 2 * (n - 1) - idx
            return idx
        return idx  # inflow: leave as-is (handled in _fetch_element)
