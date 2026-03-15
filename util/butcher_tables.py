"""
Butcher tables for IMEX Runge-Kutta methods.

Provides the explicit and implicit coefficient arrays for:
  - plain   : Forward-Backward Euler (1st order, Type CK)
  - DP2A242 : 2nd-order diagonally-implicit (Type A)
  - ARS443  : 3rd-order Ascher-Ruuth-Spiteri 4-stage (Type CK)
"""

import numpy as np


class ButcherTable:
    """Base class for Butcher tables."""

    def __init__(self, name, stages):
        self.name = name
        self.stages = stages
        self.type = 'Base'


class IMEXRKTable(ButcherTable):
    """IMEX Runge-Kutta Butcher table (explicit + implicit tableaux)."""

    def __init__(self, name, stages, c_ex, a_ex, b_ex, c_im, a_im, b_im):
        super().__init__(name, stages)
        self.c_ex = c_ex
        self.a_ex = a_ex
        self.b_ex = b_ex
        self.c_im = c_im
        self.a_im = a_im
        self.b_im = b_im
        self.type = 'IMEX-RK'


# ── Forward-Backward Euler (Type CK) ─────────────────────────────────

_plain = IMEXRKTable(
    name='plain', stages=2,
    c_ex=np.array([0, 1]),
    a_ex=np.array([[0, 0],
                   [1, 0]]),
    b_ex=np.array([1, 0]),
    c_im=np.array([0, 1]),
    a_im=np.array([[0, 0],
                   [0, 1]]),
    b_im=np.array([0, 1]),
)

# ── ARS443 (Type CK) ─────────────────────────────────────────────────

_ars443 = IMEXRKTable(
    name='ARS443', stages=5,
    c_ex=np.array([0, 1/2, 2/3, 1/2, 1]),
    a_ex=np.array([
        [0,    0,    0,    0,    0],
        [1/2,  0,    0,    0,    0],
        [11/18, 1/18, 0,   0,    0],
        [5/6, -5/6,  1/2,  0,    0],
        [1/4,  7/4,  3/4, -7/4,  0],
    ]),
    b_ex=np.array([1/4, 7/4, 3/4, -7/4, 0]),
    c_im=np.array([0, 1/2, 2/3, 1/2, 1]),
    a_im=np.array([
        [0,    0,    0,    0,    0],
        [0,    1/2,  0,    0,    0],
        [0,    1/6,  1/2,  0,    0],
        [0,   -1/2,  1/2,  1/2,  0],
        [0,    3/2, -3/2,  1/2,  1/2],
    ]),
    b_im=np.array([0, 3/2, -3/2, 1/2, 1/2]),
)

# ── DP2A242 (Type A) ─────────────────────────────────────────────────

_gamma = 2
_dp2a242 = IMEXRKTable(
    name='DP2A242', stages=4,
    c_ex=np.array([0, 0, 1, 1]),
    a_ex=np.array([
        [0,   0,   0,   0],
        [0,   0,   0,   0],
        [0,   1,   0,   0],
        [0,   1/2, 1/2, 0],
    ]),
    b_ex=np.array([0, 1/2, 1/2, 0]),
    c_im=np.array([_gamma, 0, 1, 1]),
    a_im=np.array([
        [_gamma,   0,          0,              0],
        [-_gamma,  _gamma,     0,              0],
        [0,        1-_gamma,   _gamma,         0],
        [0,        1/2,        1/2-_gamma,     _gamma],
    ]),
    b_im=np.array([0, 1/2, 1/2-_gamma, _gamma]),
)

# ── Registry ──────────────────────────────────────────────────────────

TABLES = {
    'plain':   _plain,
    'ARS443':  _ars443,
    'DP2A242': _dp2a242,
}


def get_butcher_table(name):
    """Look up a Butcher table by name.

    Raises ValueError if the name is not recognised.
    """
    table = TABLES.get(name)
    if table is None:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(TABLES)}")
    return table
