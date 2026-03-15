"""
Real-Time Shock Tube Visualization

Runs the SLDG-IMEX solver for the Sod shock tube problem and plots
density, velocity, and temperature as the simulation evolves.

Usage:
    python visualize.py
    python visualize.py --epsilon 1e-2 --CFL 2 --method DP2A242
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from util.discretization import XGrid, VGrid
from util.Boltzmann import BoltzmannSolver
from util.SLDG_integrator import SLDGIntegrator
from util.helper_functions import maxwellian, compute_moments_and_maxwellian
from util.IMEX_integrator import imex_rk_step
from util.butcher_tables import get_butcher_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time shock tube visualization")
    parser.add_argument('--method', type=str, default='ARS443',
                        choices=['ARS443', 'DP2A242', 'plain'])
    parser.add_argument('--CFL', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--elements', type=int, default=80)
    parser.add_argument('--tfinal', type=float, default=0.2)
    parser.add_argument('--plot-every', type=int, default=1,
                        help="Update plot every N time steps")
    return parser.parse_args()


def build_config(args):
    return {
        'rk_method': args.method,
        'domain_left': 0.0,
        'domain_right': 1.0,
        'num_legendre': 2,
        'CFL': args.CFL,
        'num_elements': args.elements,
        'epsilon': args.epsilon,
        'Nv': 32,
        'Lv': 7,
        'final_time': args.tfinal,
    }


def setup_simulation(config):
    """Initialize grids, solvers, and Sod shock tube initial conditions."""
    table = get_butcher_table(config['rk_method'])
    if table.type != 'IMEX-RK':
        raise ValueError(f"Expected IMEX-RK table, got {table.type}")

    xgrid = XGrid(config)
    vgrid = VGrid(config)
    Nx = xgrid.num_nodes
    v, v1, v2 = vgrid.v, vgrid.v1, vgrid.v2
    dt = config['CFL'] * xgrid.dx / vgrid.Lv

    # Sod shock tube initial conditions
    rho = np.concatenate([np.ones(Nx // 2), 0.125 * np.ones(Nx // 2)])
    u1 = np.zeros(Nx)
    u2 = np.zeros(Nx)
    theta = np.concatenate([np.ones(Nx // 2), 0.25 * np.ones(Nx // 2)])
    f = maxwellian(rho, u1, u2, theta, v1, v2)

    n_leg = config['num_legendre']
    ML = maxwellian(np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), np.ones(n_leg), v1, v2)
    MR = maxwellian(0.125 * np.ones(n_leg), np.zeros(n_leg),
                    np.zeros(n_leg), 0.25 * np.ones(n_leg), v1, v2)

    sldg = SLDGIntegrator(xgrid, 'neumann', 'BP', ML, MR)
    Q_solver = BoltzmannSolver(vgrid)

    print(f"Grid: {Nx} nodes, {config['Nv']}x{config['Nv']} velocity points")
    print(f"dt = {dt:.6f}, T_final = {config['final_time']}, "
          f"~{int(config['final_time'] / dt)} steps")

    return {
        'f': f, 'dt': dt, 'table': table, 'sldg': sldg,
        'Q_solver': Q_solver, 'xgrid': xgrid,
        'v': v, 'v1': v1, 'v2': v2, 'x': xgrid.x,
    }


def create_figure(config):
    """Set up the three-panel figure for density, velocity, temperature."""
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Shock Tube  (eps={config["epsilon"]}, '
                 f'CFL={config["CFL"]}, {config["rk_method"]})')

    labels = [('Density', r'$\rho$', 'b'),
              ('Velocity', r'$u_1$', 'r'),
              ('Temperature', r'$\theta$', 'm')]
    lines = {}
    for ax, (title, ylabel, color) in zip(axes, labels):
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], f'{color}-', linewidth=2)
        lines[ylabel] = line

    plt.tight_layout()
    return fig, axes, lines


def update_figure(x, rho, u1, theta, t, step, axes, lines):
    """Redraw the three panels with the latest data."""
    data_map = [(r'$\rho$', rho), (r'$u_1$', u1), (r'$\theta$', theta)]
    for ax, (key, y) in zip(axes, data_map):
        lines[key].set_data(x, y)
        ax.set_xlim(x.min(), x.max())
        margin = max(0.1 * (y.max() - y.min()), 0.05)
        ax.set_ylim(y.min() - margin, y.max() + margin)

    axes[0].figure.suptitle(
        f'Shock Tube  t={t:.4f}  step={step}')
    plt.draw()
    plt.pause(0.01)


def run(config, plot_every):
    """Main simulation loop with live plotting."""
    sim = setup_simulation(config)
    fig, axes, lines = create_figure(config)

    f = sim['f']
    dt, t_final = sim['dt'], config['final_time']
    t, step = 0.0, 0
    wall0 = time.time()

    try:
        while t < t_final:
            f = imex_rk_step(f, dt, config['epsilon'], sim['table'],
                             sim['sldg'], sim['Q_solver'],
                             sim['xgrid'], sim['v'], sim['v1'], sim['v2'])
            t += dt
            step += 1

            if step % plot_every == 0:
                _, _, rho, u1, _, theta = compute_moments_and_maxwellian(
                    f, sim['v'], sim['v1'], sim['v2'])
                update_figure(sim['x'], rho, u1, theta, t, step, axes, lines)
                pct = t / t_final * 100
                print(f"\r{pct:5.1f}%  t={t:.4f}  step={step}  "
                      f"wall={time.time()-wall0:.1f}s",
                      end="", flush=True)

        # Final frame
        _, _, rho, u1, _, theta = compute_moments_and_maxwellian(
            f, sim['v'], sim['v1'], sim['v2'])
        update_figure(sim['x'], rho, u1, theta, t, step, axes, lines)

        total = time.time() - wall0
        print(f"\n\nDone: {step} steps in {total:.1f}s "
              f"({total/step:.4f}s/step)")
        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nInterrupted.")
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    config = build_config(args)
    print("="*50)
    print("Real-Time Shock Tube Visualization")
    print("="*50)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50)
    run(config, args.plot_every)
