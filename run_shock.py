"""
Sod Shock Tube Simulation via SLDG-IMEX Methods

Solves the Boltzmann equation for the Sod shock tube problem using
Semi-Lagrangian Discontinuous Galerkin (SLDG) spatial discretization
with IMEX Runge-Kutta time integration and bound-preserving (BP) limiters.

Supports multiple IMEX-RK schemes:
  - plain   : Forward-Backward Euler (1st order)
  - DP2A242 : Diagonally-implicit 2nd-order scheme (Type A)
  - ARS443  : Ascher-Ruuth-Spiteri 4-stage 3rd-order scheme (Type CK)

Usage:
    python run_shock.py --method ARS443
    python run_shock.py --method DP2A242
    python run_shock.py --method plain
    python run_shock.py --method all        # run all three methods
    python run_shock.py --method ARS443 --no-parallel
"""

import argparse
import time
from functools import partial
import multiprocessing as mp

import numpy as np

from util.discretization import XGrid, VGrid
from util.Boltzmann import BoltzmannSolver
from util.SLDG_integrator import SLDGIntegrator
from util.helper_functions import maxwellian, compute_moments_and_maxwellian
from util.IMEX_integrator import imex_rk_step
from util.butcher_tables import get_butcher_table

AVAILABLE_METHODS = ['plain', 'DP2A242', 'ARS443']

BASE_CONFIG = {
    'domain_left': 0.0,
    'domain_right': 1.0,
    'Nv': 32,
    'Lv': 7,
}

PARAM_GRID = [
    (num_legendre, CFL, num_elements, epsilon)
    for num_legendre in [2]
    for CFL in [0.5, 2]
    for num_elements in [80]
    for epsilon in [1e-2, 1e-8]
]

TIME_SLICES = [0.1, 0.2]
LIMITER = 'BP'


def run_single_case(num_legendre, CFL, num_elements, epsilon,
                    rk_method, time_slices):
    """Run one simulation case for a given (method, parameter) combination."""
    config = {
        **BASE_CONFIG,
        'rk_method': rk_method,
        'num_legendre': num_legendre,
        'CFL': CFL,
        'num_elements': num_elements,
        'epsilon': epsilon,
    }

    table = get_butcher_table(rk_method)
    if table.type != 'IMEX-RK':
        raise ValueError(f"Expected IMEX-RK table, got {table.type}")

    xgrid = XGrid(config)
    vgrid = VGrid(config)
    Nx = xgrid.num_nodes
    v, v1, v2 = vgrid.v, vgrid.v1, vgrid.v2

    dt_base = CFL * xgrid.dx / vgrid.Lv

    print(f"\n[{rk_method}] L={num_legendre}, CFL={CFL}, "
          f"E={num_elements}, eps={epsilon}")

    # Sod shock tube initial conditions
    rho = np.concatenate([np.ones(Nx // 2), 0.125 * np.ones(Nx // 2)])
    u1 = np.zeros(Nx)
    u2 = np.zeros(Nx)
    theta = np.concatenate([np.ones(Nx // 2), 0.25 * np.ones(Nx // 2)])
    f = maxwellian(rho, u1, u2, theta, v1, v2)

    # Inflow boundary Maxwellians
    n_leg = num_legendre
    ML = maxwellian(np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), np.ones(n_leg), v1, v2)
    MR = maxwellian(0.125 * np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), 0.25 * np.ones(n_leg), v1, v2)

    sldg = SLDGIntegrator(xgrid, 'neumann', LIMITER, ML, MR)
    Q_solver = BoltzmannSolver(vgrid)

    # Time-stepping loop
    wall_start = time.time()
    t = 0.0
    t_final = max(time_slices)
    snapshots = {}
    next_idx = 0

    try:
        while t < t_final:
            dt = dt_base
            if next_idx < len(time_slices):
                gap = time_slices[next_idx] - t
                if gap < dt:
                    dt = gap
            dt = min(dt, t_final - t)
            if np.isclose(dt, 0.0):
                break

            f = imex_rk_step(f, dt, epsilon, table, sldg, Q_solver,
                             xgrid, v, v1, v2)
            t += dt

            if next_idx < len(time_slices) and np.isclose(t, time_slices[next_idx]):
                _, _, rho_s, u1_s, u2_s, th_s = compute_moments_and_maxwellian(
                    f, v, v1, v2)
                snapshots[time_slices[next_idx]] = {
                    'rho': rho_s.copy(), 'u1': u1_s.copy(),
                    'u2': u2_s.copy(), 'theta': th_s.copy(),
                }
                print(f"  snapshot at t={t:.4f}")
                next_idx += 1

            progress = t / t_final * 100
            elapsed = time.time() - wall_start
            print(f"\r  {progress:5.1f}% | t={t:.4f}/{t_final} | "
                  f"{elapsed:.0f}s", end="", flush=True)

        wall_total = time.time() - wall_start
        print(f"\n  done in {wall_total:.1f}s")

        return {
            'state': 'completed',
            'num_legendre': num_legendre, 'CFL': CFL,
            'num_elements': num_elements, 'epsilon': epsilon,
            'final_time': t, 'computation_time': wall_total,
            'time_step_size': dt_base,
            'snapshots': snapshots, 'x_grid': xgrid.x,
            'config_params': config,
        }

    except Exception as exc:
        wall_total = time.time() - wall_start
        print(f"\n  FAILED at t={t:.4f}: {exc}")
        return {
            'state': 'error',
            'num_legendre': num_legendre, 'CFL': CFL,
            'num_elements': num_elements, 'epsilon': epsilon,
            'error': str(exc), 'final_time': t,
            'computation_time': wall_total, 'config_params': config,
        }


def run_method(rk_method, parallel=True, cpus=4):
    """Run all parameter combinations for one IMEX-RK method."""
    output_file = f'data/results_shock_{rk_method.lower()}_BP.npz'
    results = []

    def _save(result):
        results.append(result)
        np.savez_compressed(output_file, results=results)

    worker = partial(run_single_case, rk_method=rk_method,
                     time_slices=TIME_SLICES)

    if parallel:
        with mp.Pool(processes=cpus) as pool:
            for params in PARAM_GRID:
                pool.apply_async(worker, args=params,
                                 callback=_save)
            pool.close()
            pool.join()
    else:
        for params in PARAM_GRID:
            _save(worker(*params))

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="SLDG-IMEX Sod shock tube solver")
    parser.add_argument(
        '--method', type=str, default='ARS443',
        choices=AVAILABLE_METHODS + ['all'],
        help='IMEX-RK method to use (default: ARS443)')
    parser.add_argument(
        '--no-parallel', action='store_true',
        help='Disable multiprocessing')
    parser.add_argument(
        '--cpus', type=int, default=4,
        help='Number of worker processes (default: 4)')
    args = parser.parse_args()

    methods = AVAILABLE_METHODS if args.method == 'all' else [args.method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Running {method}")
        print(f"{'='*60}")
        run_method(method, parallel=not args.no_parallel, cpus=args.cpus)


if __name__ == '__main__':
    main()
