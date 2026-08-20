# Identification-Robust Inference for Complier Cumulative-Incidence Effects under Competing Risks

This repository contains the simulation and figure-generation code for the manuscript:

> **Identification-Robust Inference for Complier Cumulative-Incidence Effects under Competing Risks**
> Naresh Garg, 2026.

## Overview

Instrumental-variable analyses with competing-risk outcomes often identify a cause-specific complier cumulative-incidence effect through a Wald-type ratio,

[
\psi_j(\tau)
============

\frac{F_{1j}(\tau)-F_{0j}(\tau)}
{P(A=1\mid Z=1)-P(A=1\mid Z=0)}.
]

Here, the numerator is the instrument contrast in cause-specific cumulative incidence at horizon (\tau), and the denominator is the treatment-uptake or compliance contrast. When compliance is weak, direct division by the estimated first stage can produce an unstable ratio estimator and poorly interpretable Wald intervals.

The proposed method avoids dividing by the estimated compliance contrast. Instead, it constructs a confidence set by inverting the studentized undivided moment

[
N_j(\tau)-b\kappa=0,
]

over the feasible effect range (b\in[-1,1]).

## Main contribution

The general Fieller/Anderson–Rubin test-inversion principle is classical. The contribution of this paper is its rigorous specialization to a cause-specific complier cumulative-incidence effect under right-censored competing risks.

Specifically, the paper:

* combines arm-specific Aalen–Johansen estimators with an undivided LATE moment;
* derives the joint influence representation of the cumulative-incidence and treatment-uptake contrasts;
* incorporates the exact finite-jump correction required for tied discrete event times;
* retains the covariance between the reduced-form and first-stage estimators;
* solves the resulting quadratic score inequality exactly, allowing ordinary, disconnected, full and empty confidence sets;
* establishes identification-robust coverage under local-to-zero compliance; and
* evaluates the method under fixed-strength, local-to-zero, severe-censoring, tied-time and assumption-violation scenarios.

The method reports weak identification through a wide or full feasible confidence set rather than concealing it through an unstable ratio approximation.

## Scope and assumptions

The current implementation considers:

* one randomized binary instrument (Z);
* one binary treatment (A);
* one event cause (j);
* one fixed analysis horizon (\tau);
* consistency and no interference;
* instrument randomization;
* the exclusion restriction;
* monotonicity;
* independent censoring conditional on the instrument;
* a fixed finite event-time grid; and
* no baseline-covariate adjustment.

The informative-censoring and defier simulations are deliberately outside the assumptions of the theorem. They are included as diagnostic stress tests and should not be interpreted as evidence supporting theoretical coverage.

## Repository structure

```text
simulate_weak_iv_cif.py
    Core simulation and inference functions, including:
    - data generation;
    - Aalen–Johansen estimation;
    - finite-grid influence functions;
    - tied-time finite-jump correction;
    - product Gauss–Hermite quadrature truth;
    - exact quadratic score-set inversion; and
    - ratio-Wald inference.

simulate_weak_iv_cif_extended.py
    Main simulation driver covering all 44 settings used in the
    extended simulation study.

make_simulation_figures.py
    Reconstructs the three manuscript figures from the aggregate
    and replicate-level outputs.

extended_results_final/
    extended_simulation_results.csv
        Final setting-level simulation summaries.

    extended_run_metadata.json
        Seed, software versions, arguments, quadrature diagnostics,
        replication counts and runtime metadata.

figures/
    Generated PDF and PNG figures.
```

The large replicate-level file

```text
extended_results_final/extended_replicate_results.csv
```

is generated automatically when the extended simulation is run. It is required for reconstructing the figures.

## Software requirements

The code requires Python 3.11 or later and the following packages:

* NumPy;
* pandas;
* matplotlib.

The final simulation was run using:

```text
Python 3.12.13
NumPy 2.3.5
pandas 2.2.3
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib
```

If a `requirements.txt` file is provided, use:

```bash
python -m pip install -r requirements.txt
```

## Quick execution test

The following reduced run checks that the simulation code executes correctly:

```bash
python simulate_weak_iv_cif_extended.py \
  --output smoke_results \
  --replications 2 \
  --primary-replications 2 \
  --quadrature-nodes 20 \
  --seed 20260819 \
  --workers 1
```

This quick run is only a software smoke test. Two replications per setting are insufficient for estimating coverage, bias or interval-length distributions.

## Reproducing the final simulations

Run:

```bash
python simulate_weak_iv_cif_extended.py \
  --output extended_results_final \
  --replications 2000 \
  --primary-replications 5000 \
  --quadrature-nodes 60 \
  --seed 20260819 \
  --workers 4
```

The final design contains 44 settings and 172,000 generated datasets:

* 12 fixed-strength settings;
* 16 local-to-zero settings;
* 4 severe independent-censoring settings;
* 8 paired tied-time influence-function ablation settings; and
* 4 assumption-violation stress-test settings.

The primary fixed-strength and local-to-zero settings use 5,000 replications per cell. The supplementary settings use 2,000 replications per cell.

## Simulation design

The treatment probabilities are

[
q_0(X,U)
========

\operatorname{expit}(-1+0.45X+0.55U),
]

and

[
q_1(X,U)
========

q_0(X,U)+\delta{1-q_0(X,U)}.
]

Because

[
q_1-q_0=\delta(1-q_0),
]

the distribution of ((X,U)) among compliers does not depend on (\delta). Product Gauss–Hermite quadrature gives the common true complier effect

[
\psi=-0.09649172685174347,
]

with compliance multiplier

[
E(1-q_0)=0.7114055309464351.
]

The local-to-zero settings are constructed so that

[
\sqrt n,\kappa_n=c,
\qquad
c\in{0.5,1,2,4},
]

for

[
n\in{500,1000,2500,5000}.
]

## Generated outputs

The extended simulation writes:

```text
extended_results_final/extended_simulation_results.csv
extended_results_final/extended_replicate_results.csv
extended_results_final/extended_run_metadata.json
```

### Aggregate output

`extended_simulation_results.csv` contains:

* score-set coverage;
* ratio-Wald coverage;
* Monte Carlo standard errors;
* mean and median interval lengths;
* upper interval-length quantiles;
* full-set frequency;
* disconnected-set frequency;
* empty-set frequency;
* mean and standard deviation of the estimated first stage;
* frequency of near-zero estimated first stages;
* ratio bias and RMSE;
* realized censoring proportion; and
* cause-1 event rate.

### Replicate-level output

`extended_replicate_results.csv` contains:

* reduced-form estimate (\widehat N);
* first-stage estimate (\widehat\kappa);
* ratio estimate and Wald endpoints;
* exact score-set components;
* quadratic coefficients;
* variance and covariance terms;
* score statistic at the truth;
* confidence-set topology;
* realized censoring; and
* event-rate diagnostics.

### Metadata

`extended_run_metadata.json` records:

* random seed;
* command-line arguments;
* Python, NumPy and pandas versions;
* quadrature nodes and stability check;
* true effect;
* compliance multiplier;
* number of settings;
* total number of replications; and
* wall-clock runtime.

## Reproducing the figures

After running the complete extended simulation, execute:

```bash
python make_simulation_figures.py
```

This produces PDF and PNG versions of:

```text
figures/local_coverage_informativeness.pdf
figures/local_coverage_informativeness.png

figures/denominator_instability.pdf
figures/denominator_instability.png

figures/score_set_topology.pdf
figures/score_set_topology.png
```

The figures display:

1. score-set coverage and full-set frequency under local-to-zero compliance;
2. the heavy-tailed length distribution of ratio-Wald intervals; and
3. the frequencies of ordinary, disconnected, full and empty score sets.

## Interpretation of the simulation output

Coverage should always be interpreted together with Monte Carlo uncertainty and interval informativeness.

A ratio-Wald interval may attain nominal or conservative coverage by becoming extremely long when the estimated first stage approaches zero. Therefore, coverage alone is not sufficient for comparing the procedures.

The score-inversion method is designed to represent loss of identification explicitly. Under weak compliance, a full feasible set is not a computational failure; it indicates that the data contain insufficient information to distinguish among candidate effects.

Rows with

```text
theorem_valid = false
```

correspond to assumption-violation diagnostics and are outside the theoretical coverage guarantee.

## Reproducibility notes

* The main random seed is `20260819`.
* Product Gauss–Hermite quadrature uses 60 nodes.
* Events win censoring ties.
* The cause-specific cumulative incidence is evaluated at horizon (\tau=8).
* The feasible effect range is ([-1,1]).
* The main confidence level is 95%.
* The finite-jump correction is used in all theorem-valid settings.
* The tied-time ablation uses paired random streams.
* No machine-learning nuisance model is fitted.
* No training, validation or tuning data are used.
* Parallel execution changes runtime but not the assigned replicate-level random streams.

## Citation

Until a journal DOI or preprint identifier becomes available, cite the manuscript and repository as:

```bibtex
@unpublished{garg2026identification,
  author = {Garg, Naresh},
  title  = {Identification-Robust Inference for Complier
            Cumulative-Incidence Effects under Competing Risks},
  year   = {2026},
  note   = {Manuscript and reproducibility repository},
  url    = {https://github.com/nareshng/Identification-Robust-Inference-for-Complier-Cumulative-Incidence-Effects-under-Competing-Risks}
}
```

## Limitations

The current method does not cover:

* nonrandomized instruments;
* multivalued or continuous instruments;
* multiple treatments;
* baseline-covariate adjustment;
* censoring dependent on treatment or baseline covariates;
* simultaneous inference over multiple causes or horizons; or
* machine-learning nuisance estimation.

These extensions require additional identification conditions, adjusted observed-data influence functions and new theoretical justification.
