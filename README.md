# SLDG-IMEX Boltzmann Solver for the Sod Shock Tube

A Semi-Lagrangian Discontinuous Galerkin (SLDG) solver for the Boltzmann equation with BGK-penalised IMEX Runge-Kutta time integration and bound-preserving limiters.  The solver is validated on the classical **Sod shock tube** problem across kinetic and hydrodynamic regimes.

| Kinetic regime (ε = 10⁻²) | Hydrodynamic limit (ε = 10⁻⁸) |
|---|---|
| ![kinetic](plots/shock_comparison_L2_CFL2_E80_eps1e-02_t0.2.png) | ![hydro](plots/shock_comparison_L2_CFL2_E80_eps1e-08_t0.2.png) |

## Background

The Boltzmann equation describes the evolution of a particle distribution function f(t, x, v) through transport and binary collisions:

$$\partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon}\, Q(f, f)$$

where ε is the **Knudsen number**.  As ε → 0 the solution converges to a local Maxwellian and the macroscopic density, velocity, and temperature satisfy the **compressible Euler equations**.

This project implements:
- **Spatial discretisation** — SLDG (Semi-Lagrangian Discontinuous Galerkin) with Gauss-Legendre nodal basis and characteristic back-tracing, plus a local maximum-principle-preserving (LMPP) bound-preserving limiter.
- **Time integration** — IMEX Runge-Kutta schemes with BGK penalisation, ensuring uniform stability across all ε.  Three schemes are provided: Forward-Backward Euler (1st order), DP2A242 (2nd order), and ARS443 (3rd order).
- **Collision operator** — Fourier-spectral method for 2-D Maxwell molecules (Mouhot & Pareschi, 2006).
- **Reference solvers** — A MATLAB finite-difference IMEX solver (`FD_solver.m`) and an Euler-limit solver, used for cross-validation.

## Project Structure

```
├── run_shock.py            # Main simulation driver (supports all IMEX-RK methods)
├── euler_ref.py            # Euler-limit reference solution generator
├── post_shock.py           # Post-processing: load results and generate comparison plots
├── visualize.py            # Real-time visualization of density, velocity, temperature
│
├── util/
│   ├── discretization.py   # Spatial (XGrid) and velocity (VGrid) grid classes
│   ├── helper_functions.py # Maxwellian, moments, Lagrange interpolation, quadrature
│   ├── Boltzmann.py        # Spectral Boltzmann collision operator
│   ├── SLDG_integrator.py  # SLDG transport operator with BP limiter
│   ├── IMEX_integrator.py  # IMEX-RK time stepper for the full Boltzmann equation
│   ├── Euler_integrator.py # Euler-limit time stepper (transport + projection)
│   └── butcher_tables.py   # IMEX-RK Butcher tableaux (plain, DP2A242, ARS443)
│
├── FD_solver.m             # MATLAB: finite-difference IMEX reference solver
├── Euler_solver.m          # MATLAB: Euler-limit reference solver
├── collisionB.m            # MATLAB: spectral collision operator
├── upwind.m                # MATLAB: 2nd-order upwind transport with minmod limiter
├── Maxwellian.m            # MATLAB: Maxwellian distribution function
│
├── data/                   # Simulation results and reference data (.npz, .mat)
├── plots/                  # Generated comparison figures (.png, .eps)
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simulation (e.g. with ARS443)
python run_shock.py --method ARS443

# Run all three methods
python run_shock.py --method all

# Generate Euler reference solution
python euler_ref.py

# Generate comparison plots
python post_shock.py

# Real-time visualization
python visualize.py --method ARS443 --epsilon 1e-6 --CFL 0.5
```

### Command-Line Options

**`run_shock.py`**
| Flag | Description | Default |
|---|---|---|
| `--method` | `plain`, `DP2A242`, `ARS443`, or `all` | `ARS443` |
| `--no-parallel` | Disable multiprocessing | off |
| `--cpus` | Number of worker processes | 4 |

**`visualize.py`**
| Flag | Description | Default |
|---|---|---|
| `--method` | IMEX-RK scheme | `ARS443` |
| `--CFL` | CFL number | 0.5 |
| `--epsilon` | Knudsen number | 1e-6 |
| `--elements` | Number of spatial elements | 80 |
| `--tfinal` | Final simulation time | 0.2 |
| `--plot-every` | Update plot every N steps | 1 |

## Numerical Methods

### SLDG Transport

The spatial domain is divided into elements, each carrying Gauss-Legendre nodal values.  Transport is handled exactly by tracing characteristics backward in time and reconstructing via Lagrange interpolation — this removes the CFL restriction of standard explicit DG methods, allowing CFL > 1.

### Bound-Preserving Limiter

A local maximum-principle-preserving (LMPP) limiter is applied after each transport step.  It scales the deviation from the cell mean so that point values remain within the bounds of the upstream data, preventing spurious oscillations near shocks.

### IMEX-RK Time Integration

The collision operator is split into a stiff BGK-penalisation term (treated implicitly) and a remainder (treated explicitly), following the micro-macro decomposition framework.  This yields a scheme that is:
- **Asymptotic-preserving**: correctly captures the Euler limit as ε → 0 without resolving ε-scale time steps.
- **Uniformly stable**: the implicit treatment of the stiff relaxation allows large time steps across all regimes.

### Spectral Collision Operator

The Boltzmann collision integral for 2-D Maxwell molecules is computed via the Fourier-spectral method of Mouhot & Pareschi, discretising the angular variable on S¹ and evaluating convolutions through FFT.

## References

- C. Mouhot, L. Pareschi, "Fast algorithms for computing the Boltzmann collision operator," *Mathematics of Computation*, 75(256):1833–1852, 2006.
- U. Ascher, S. Ruuth, R. Spiteri, "Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations," *Applied Numerical Mathematics*, 25(2–3):151–167, 1997.
- J. Qiu, C.-W. Shu, "Positivity preserving semi-Lagrangian discontinuous Galerkin formulation," *Journal of Computational Physics*, 230(23):8386–8409, 2011.
