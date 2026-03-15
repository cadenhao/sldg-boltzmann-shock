"""
Euler Reference Solution Generator

Generates reference solutions for the Sod shock tube problem in the
hydrodynamic limit (epsilon -> 0).  Uses transport + Maxwellian projection
(no collision operator), which reduces the Boltzmann equation to the
compressible Euler equations.

Output: data/ref_1e-8.npz

Usage:
    python euler_ref.py
    python euler_ref.py --no-parallel
"""

import argparse
import time
from functools import partial
import multiprocessing as mp

import numpy as np

from util.discretization import XGrid, VGrid
from util.SLDG_integrator import SLDGIntegrator
from util.helper_functions import maxwellian
from util.Euler_integrator import euler_rk_step
from util.butcher_tables import get_butcher_table

OUTPUT_FILE = 'data/ref_1e-8.npz'
TIME_SLICES = [0.1, 0.2]
LIMITER = 'BP'

BASE_CONFIG = {
    'rk_method': 'ARS443',
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
    for epsilon in [1e-8]
]


def run_euler_case(num_legendre, CFL, num_elements, epsilon, time_slices):
    """Run one Euler reference case (transport + Maxwellian projection)."""
    config = {
        **BASE_CONFIG,
        'num_legendre': num_legendre,
        'CFL': CFL,
        'num_elements': num_elements,
        'epsilon': epsilon,
    }

    table = get_butcher_table(config['rk_method'])
    if table.type != 'IMEX-RK':
        raise ValueError(f"Expected IMEX-RK table, got {table.type}")

    xgrid = XGrid(config)
    vgrid = VGrid(config)
    Nx = xgrid.num_nodes
    v, v1, v2 = vgrid.v, vgrid.v1, vgrid.v2
    dt_base = CFL * xgrid.dx / vgrid.Lv

    print(f"\n[Euler ref] L={num_legendre}, CFL={CFL}, "
          f"E={num_elements}, eps={epsilon}")

    # Sod shock tube initial conditions (moments only)
    rho = np.concatenate([np.ones(Nx // 2), 0.125 * np.ones(Nx // 2)])
    u1 = np.zeros(Nx)
    u2 = np.zeros(Nx)
    theta = np.concatenate([np.ones(Nx // 2), 0.25 * np.ones(Nx // 2)])

    n_leg = num_legendre
    ML = maxwellian(np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), np.ones(n_leg), v1, v2)
    MR = maxwellian(0.125 * np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), 0.25 * np.ones(n_leg), v1, v2)

    sldg = SLDGIntegrator(xgrid, 'neumann', LIMITER, ML, MR)

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

            rho, u1, u2, theta = euler_rk_step(
                rho, u1, u2, theta, dt, table, sldg, xgrid, v, v1, v2)
            t += dt

            if next_idx < len(time_slices) and np.isclose(t, time_slices[next_idx]):
                snapshots[time_slices[next_idx]] = {
                    'rho': rho.copy(), 'u1': u1.copy(),
                    'u2': u2.copy(), 'theta': theta.copy(),
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Euler reference solutions")
    parser.add_argument('--no-parallel', action='store_true')
    parser.add_argument('--cpus', type=int, default=4)
    args = parser.parse_args()

    results = []

    def _save(result):
        results.append(result)
        np.savez_compressed(OUTPUT_FILE, results=results)

    worker = partial(run_euler_case, time_slices=TIME_SLICES)

    if not args.no_parallel:
        with mp.Pool(processes=args.cpus) as pool:
            for params in PARAM_GRID:
                pool.apply_async(worker, args=params, callback=_save)
            pool.close()
            pool.join()
    else:
        for params in PARAM_GRID:
            _save(worker(*params))

    print(f"\nEuler reference saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
