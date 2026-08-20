# Candidate 1: reproducibility and result alignment

## Bottom line

The attached `simulation_results(1).csv` supports the paper's qualitative
conclusion, but it is not the numerical source for the revised manuscript.
It contains 1,000 replications per cell and uses three simulation-noisy
truth values. In this DGP, however,
`q1 - q0 = delta * (1 - q0)`, so the complier distribution and target effect
do not depend on `delta`. Product Gauss--Hermite quadrature gives the common
truth

```
psi = -0.09649172685174347
E(1-q0) = 0.7114055309464351.
```

The final paper therefore uses a corrected run with 5,000 replications in
each primary fixed-strength and local-to-zero cell, and 2,000 in each
supplementary cell.

## Alignment with the attached aggregate

- Attached score coverage: 0.943--0.968.
- Final fixed-strength score coverage: 0.9446--0.9534.
- Eleven of 12 coverage differences are within two combined Monte Carlo
  standard errors. The exception is `delta=0.35`, `n=1000`, no censoring:
  0.968 attached versus 0.9446 final.
- Mean score-set length differs by at most 0.0216, full-set frequency by at
  most 0.0136, and mean first stage by at most 0.0034.
- Weak-IV Wald mean length and ratio RMSE do not align numerically because
  rare near-zero denominators dominate these heavy-tailed means. The paper
  reports medians and tail quantiles instead.

The row-by-row comparison is in `results_alignment.csv`.

## Expanded design

The final run contains 44 settings and 172,000 generated datasets:

- 12 fixed-strength cells;
- 16 local-to-zero cells with
  `delta_n = c / (0.7114055309464351 * sqrt(n))` for
  `c in {0.5, 1, 2, 4}` and `n in {500, 1000, 2500, 5000}`;
- four severe independent-censoring cells;
- eight paired finite-jump/tie-correction ablation cells; and
- four labelled assumption-violation diagnostics.

Replicate-level output retains exact score-set components, quadratic
coefficients, variance terms, Wald endpoints, the reduced form, and the
estimated first stage.

## Reproduce

```bash
python simulate_weak_iv_cif_extended.py \
  --output extended_results_final \
  --replications 2000 \
  --primary-replications 5000 \
  --workers 4

python make_simulation_figures.py
python compare_attached_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  Candidate1_Weak_IV_Competing_Risks.tex
```

## Main files

- `Candidate1_Weak_IV_Competing_Risks.tex` and `.pdf`: revised paper.
- `simulate_weak_iv_cif.py`: core estimator, exact score geometry, and
  deterministic quadrature truth.
- `simulate_weak_iv_cif_extended.py`: all 44 simulation settings.
- `make_simulation_figures.py`: the three vector result figures.
- `extended_results_final/extended_simulation_results.csv`: aggregate output.
- `extended_results_final/extended_replicate_results.csv`: replicate output.
- `extended_results_final/extended_run_metadata.json`: complete run metadata.
