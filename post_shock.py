"""
Post-Processing and Plotting for Shock Tube Results

Loads simulation results from multiple IMEX-RK methods and plots them
against reference solutions (MATLAB finite-difference or Euler limit).

Generates comparison figures (EPS + PNG) in the plots/ directory.

Usage:
    python post_shock.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHODS = {
    'plain_BP': {
        'data_file': 'data/results_shock_plain_BP.npz',
        'display_name': 'FBEuler',
        'color': 'red',
        'dash_pattern': [4, 2],
    },
    'dp2a242_BP': {
        'data_file': 'data/results_shock_dp2a242_BP.npz',
        'display_name': 'DP2A242',
        'color': 'green',
        'dash_pattern': [2, 2],
    },
    'ars443_BP': {
        'data_file': 'data/results_shock_ars443_BP.npz',
        'display_name': 'ARS443',
        'color': 'blue',
        'dash_pattern': [1, 1],
    },
}

REF_DIR = 'data'
OUTPUT_DIR = 'plots'
OUTPUT_PREFIX = 'shock_comparison'

PROCESS_EPSILONS = [1e-2, 1e-8]

TITLE_SIZE = 19
LABEL_SIZE = 15
LEGEND_SIZE = 12
TICK_SIZE = 15

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_results(filepath):
    """Load completed simulation cases from an .npz file."""
    if not os.path.exists(filepath):
        print(f"  {filepath} not found — skipping")
        return []
    data = np.load(filepath, allow_pickle=True)
    return [c for c in data['results'] if c['state'] == 'completed']


def load_reference(epsilon, ref_dir=REF_DIR):
    """Load the appropriate reference solution for a given epsilon.

    For epsilon ~ 1e-8 the reference comes from the Python Euler solver
    (ref_1e-8.npz).  For other values of epsilon a MATLAB finite-difference
    reference (.mat) is used.
    """
    if np.isclose(epsilon, 1e-8):
        path = os.path.join(ref_dir, 'ref_1e-8.npz')
        raw = np.load(path, allow_pickle=True)
        cases = [c for c in raw['results'] if c['state'] == 'completed']
        if not cases:
            raise ValueError(f"No completed case in {path}")
        case = cases[0]
        ref = {'x_ref': case['x_grid'], 'is_euler': True, 'snapshots': {}}
        for t_val, snap in case['snapshots'].items():
            ref['snapshots'][t_val] = {
                'rho_ref': snap['rho'],
                'u1_ref': snap['u1'],
                'theta_ref': snap['theta'],
            }
    else:
        path = os.path.join(ref_dir, f'ref_{epsilon:.6e}.mat')
        mat = io.loadmat(path)
        ref = {'x_ref': mat['X'].flatten(), 'is_euler': False, 'snapshots': {}}
        if 'rho_all' in mat and 'times' in mat:
            times = mat['times'].flatten()
            for i, t_val in enumerate(times):
                ref['snapshots'][t_val] = {
                    'rho_ref': mat['rho_all'][:, i],
                    'u1_ref': mat['u1_all'][:, i],
                    'theta_ref': mat['theta_all'][:, i],
                }
        else:
            ref['snapshots'] = None
            ref['rho_ref'] = mat['rho'].flatten()
            ref['u1_ref'] = mat['u1'].flatten()
            ref['theta_ref'] = mat['theta'].flatten()

    return ref


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(matching, ref, params, method_data, save_dir):
    """Generate density / velocity / temperature comparison plots."""
    ref_label = 'Euler(ref)' if ref['is_euler'] else 'AP-FD(ref)'
    all_times = sorted({t for case in matching.values()
                        for t in case['snapshots']})

    for t_val in all_times:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        quantities = ['rho', 'u1', 'theta']
        titles = [r'$\rho$', r'$u_1$', r'$\theta$']

        for ax, q, title in zip(axes, quantities, titles):
            # Reference curve
            if ref.get('snapshots') and t_val in ref['snapshots']:
                ax.plot(ref['x_ref'], ref['snapshots'][t_val][f'{q}_ref'],
                        '-', lw=2, label=ref_label, color='black')
            elif ref.get('snapshots') is None:
                ax.plot(ref['x_ref'], ref[f'{q}_ref'],
                        '-', lw=2, label=ref_label, color='black')
            else:
                continue

            # Method curves
            for key, case in matching.items():
                if t_val not in case['snapshots']:
                    continue
                snap = case['snapshots'][t_val]
                vals = snap[q]
                if np.isnan(vals).any():
                    print(f"  NaN in {key}/{q} at t={t_val} — skipping")
                    continue
                cfg = method_data[key]['config']
                ax.plot(case['x_grid'], vals,
                        color=cfg['color'],
                        dashes=cfg['dash_pattern'],
                        lw=2, label=cfg['display_name'])

            ax.set_xlabel('$x$', fontsize=LABEL_SIZE)
            ax.set_title(title, fontsize=TITLE_SIZE)
            ax.legend(fontsize=LEGEND_SIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=TICK_SIZE)

        plt.tight_layout()

        stem = (f"{OUTPUT_PREFIX}_L{params['num_legendre']}_"
                f"CFL{params['CFL']}_E{params['num_elements']}_"
                f"eps{params['epsilon']:.0e}_t{t_val}")
        for fmt in ('eps', 'png'):
            plt.savefig(f"{save_dir}/{stem}.{fmt}",
                        dpi=300, bbox_inches='tight', format=fmt)
        plt.close()
        print(f"  plot: {stem}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all method data
    method_data = {}
    for key, cfg in METHODS.items():
        cases = load_results(cfg['data_file'])
        if cases:
            method_data[key] = {'cases': cases, 'config': cfg}
            print(f"  {cfg['display_name']}: {len(cases)} cases")
    if not method_data:
        print("No data found. Run the simulations first.")
        return

    # Collect unique parameter combos
    combos = set()
    for data in method_data.values():
        for c in data['cases']:
            if PROCESS_EPSILONS is None or c['epsilon'] in PROCESS_EPSILONS:
                combos.add((c['epsilon'], c['CFL'],
                            c['num_elements'], c['num_legendre']))
    combos = sorted(combos)

    # Load reference data
    refs = {}
    for eps in {c[0] for c in combos}:
        try:
            refs[eps] = load_reference(eps)
        except Exception as exc:
            print(f"  ref for eps={eps} failed: {exc}")

    # Generate comparison plots
    for eps, CFL, num_el, n_leg in combos:
        if eps not in refs:
            continue
        params = {'epsilon': eps, 'CFL': CFL,
                  'num_elements': num_el, 'num_legendre': n_leg}
        matching = {}
        for key, data in method_data.items():
            for case in data['cases']:
                if (case['epsilon'] == eps and case['CFL'] == CFL and
                        case['num_elements'] == num_el and
                        case['num_legendre'] == n_leg):
                    matching[key] = case
                    break
        if len(matching) < 2:
            continue
        print(f"\neps={eps}, CFL={CFL}, E={num_el}: "
              f"{list(matching.keys())}")
        plot_comparison(matching, refs[eps], params, method_data, OUTPUT_DIR)


if __name__ == '__main__':
    main()
