# Identification-Robust Inference for Complier Cumulative-Incidence Effects under Competing Risks

This repository contains the Python code and simulation results for:

**Naresh Garg (2026), “Identification-Robust Inference for Complier Cumulative-Incidence Effects under Competing Risks.”**

## Overview

The paper studies inference for the complier cumulative-incidence effect

$$
\psi_j(\tau)=\frac{F_{1j}(\tau)-F_{0j}(\tau)}{p_1-p_0}
=\frac{N_j(\tau)}{\kappa},
$$

where $F_{zj}(\tau)$ is the cause-$j$ cumulative incidence under instrument assignment $Z=z$, and $\kappa=p_1-p_0$ is the compliance contrast.

When $\kappa$ is close to zero, the usual Wald estimator $\widehat{\psi}_j(\tau)=\widehat{N}_j(\tau)/\widehat{\kappa}$ can be unstable.

## Main contribution

The paper develops an identification-robust confidence set by inverting the studentized score for

$$
N_j(\tau)-b\kappa=0.
$$

The method combines classical Fieller–Anderson–Rubin inversion with Aalen–Johansen influence functions for right-censored competing-risk data. It remains valid under weak compliance and can return bounded, unbounded, disconnected, or empty confidence sets.

## Repository contents

* `simulate_weak_iv_cif.py`: core data-generating process and estimation functions.
* `simulate_weak_iv_cif_extended.py`: main simulation driver.
* `make_simulation_figures.py`: reproduces the simulation figures.
* `extended_results_final/extended_simulation_results.csv`: aggregate simulation results.
* `extended_results_final/extended_run_metadata.json`: simulation settings and software information.

All data used in the study are simulated; no confidential or external datasets are required.

## Requirements

Python $3.11$ or later with:

```bash
pip install numpy pandas matplotlib
```

## Quick test

```bash
python simulate_weak_iv_cif_extended.py \
  --output smoke_results \
  --replications 2 \
  --primary-replications 2 \
  --quadrature-nodes 20 \
  --seed 20260819 \
  --workers 1
```

## Full simulation

```bash
python simulate_weak_iv_cif_extended.py \
  --output extended_results_final \
  --replications 2000 \
  --primary-replications 5000 \
  --quadrature-nodes 60 \
  --seed 20260819 \
  --workers 4
```

The full simulation generates aggregate and replicate-level results. Runtime depends on the computer and number of workers.

## Figures

After completing the full simulation, run:

```bash
python make_simulation_figures.py
```

The figures are saved in the `figures/` directory.

## Reproducibility

The reported simulations use random seed `20260819`. Exact package versions and simulation settings are recorded in:

```text
extended_results_final/extended_run_metadata.json
```

## Citation

```bibtex
@article{garg2026identification,
  author  = {Garg, Naresh},
  title   = {Identification-Robust Inference for Complier
             Cumulative-Incidence Effects under Competing Risks},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

Add a `LICENSE` file before public release. The MIT License is appropriate if unrestricted reuse of the code is intended.
