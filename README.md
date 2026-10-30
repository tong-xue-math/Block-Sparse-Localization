# Robust Anchor-Based Localization under Spatially Correlated Outliers

MATLAB implementation of robust target localization from anchor-distance measurements in the presence of structured, cluster-dependent anchor corruptions.

## Overview

The code generates clustered targets and near/far anchors, perturbs squared-distance measurements with dense noise and sparse gross errors, and compares three convex recovery approaches. Estimated corruption patterns are used to cluster targets, identify unreliable anchors, and recover target coordinates by least squares using the remaining measurements.

After centering at a reference anchor, the measurement model is

$$
\widetilde M = XQ + E + S,
$$

where $X$ contains centered anchor coordinates, $Q$ contains centered target coordinates, $E$ denotes dense measurement noise, and $S$ denotes gross corruption. For the null-space approaches, $R X = 0$, so the recovery constraint is imposed on $R(\widetilde M-S)$.

## Methods

The `method` argument of `recover_S.m` selects one of three approaches:

| Method in code | Label in plots | Recovery approach |
| --- | --- | --- |
| `joint` | RSCB | Per-target joint coordinate and sparse-error recovery with an exact linear equality constraint |
| `l1_nuc` | RSCLNuc | Entrywise $\ell_1$ plus nuclear-norm regularization under a null-space residual constraint |
| `l1` | RSCL | Entrywise $\ell_1$ regularization under a null-space residual constraint |

For `l1_nuc` and `l1`, the first `n_near` rows of the recovered corruption matrix are constrained to zero. The `joint` implementation does not impose that constraint and uses an exact equality rather than a noise-tolerant residual bound.

## Requirements

- MATLAB (developed with MATLAB R2025a)
- [CVX](https://cvxr.com/cvx/) (used for convex optimization; CVX 2.2 was used in development)
- MATLAB Statistics and Machine Learning Toolbox (for `kmeans`)

Install and configure CVX (including running `cvx_setup`) before executing the experiments.

## Files

| File | Purpose |
| --- | --- |
| `Block_sparsity_main.m` | Runs the three methods across repeated synthetic trials, reports summary statistics, and saves results. |
| `generate_data.m` | Generates target clusters, anchors, noisy/corrupted squared distances, and centered linearized measurements. |
| `recover_S.m` | Solves the convex corruption-recovery problems. |
| `clustering_recover_Q.m` | Builds a support-overlap graph, performs spectral clustering, identifies corrupted anchors, and estimates target locations by least squares. |
| `plot_curves_vs_k.m` | Sweeps the number of corrupted anchors per cluster and plots support score and median relative localization error. |
| `plot_heatmap_support.m` | Currently contains a repeated comparison-experiment script; despite its filename, the supplied version does not generate a heatmap. |

**Filename note:** Some uploaded files had a `(1)` suffix. Rename these to their MATLAB function names (`generate_data.m`, `recover_S.m`, and `clustering_recover_Q.m`) before running the scripts. Rename `plot_heatmap_support(1).m` to `plot_heatmap_support.m` if including it in the repository.

## Quick start

Place the `.m` files in the same directory, open MATLAB in that directory, and run:

```matlab
Block_sparsity_main
```

To generate performance curves over the sparsity level $k$, run:

```matlab
plot_curves_vs_k
```

Both scripts invoke `generate_data`, `recover_S`, and `clustering_recover_Q` automatically.

## Default experimental setup

The main comparison script uses:

| Parameter | Value | Description |
| --- | --- | --- |
| `d` | 2 | Spatial dimension |
| `n` | 99 | Number of target nodes |
| `n_c` | 3 | Number of target clusters |
| `sigma` | 8 | Target-cluster spread |
| `n_near` | 2 | Number of near anchors |
| `n_far` | 9 | Number of far anchors |
| `epsilon` | 0.01 | Multiplicative noise scale for squared distances |
| `k` | 5 | Corrupted far anchors selected per cluster |
| `corrupt_level` | 0.4 | Gross-error magnitude relative to squared distance |
| `lambda` | 5 | Nuclear-norm regularization weight |
| `c_tau` | 1.2 | Residual-tolerance scaling factor |
| `num_trials` | 50 | Number of repeated trials |

The curve script instead uses `k_list = 2:7` and `num_trials = 10`. Trials use deterministic seeds (`seed = t`). Parameters can be edited at the top of each script.

## Evaluation and outputs

The pipeline evaluates recovered corrupted-anchor sets against the synthetic ground truth and reports:

- **Support score:** fraction of correctly matched corrupted anchors across clusters, after matching recovered and true clusters.
- **Relative Frobenius error:** relative Frobenius norm of the centered coordinate error.
- **Mean, median, and maximum node error:** statistics of per-node relative coordinate errors.
- **Solve rate:** fraction of trials with CVX status `Solved` or `Inaccurate/Solved` (all per-target problems must qualify for `joint`).

`Block_sparsity_main.m` prints means and standard deviations and saves `compare_recovery_results.mat` in the current MATLAB working directory. The file includes trial-level outputs, generated data, summary metrics, and solver statuses.

`plot_curves_vs_k.m` saves `curves_vs_k_results.mat`, `curves.pdf`, and `curves.png` in the current MATLAB working directory. The PDF uses vector graphics; the PNG is exported at 300 dpi.

## Implementation notes

- The clustering stage retains the top-`k` entries in each recovered corruption column, constructs a Jaccard support-overlap graph, applies normalized spectral clustering, and selects frequently occurring anchors within each recovered cluster.
- Localization is performed by least squares after excluding the recovered corrupted anchors and the reference-anchor row.
- The synthetic data generator assumes the number of targets is divisible by the number of clusters and is written for two-dimensional coordinates.
- The current support score uses known corrupted-anchor sets for evaluation; it is not an unsupervised estimate of recovery quality.
- This repository contains experimental research code. The supplied files have not been independently executed or benchmarked as part of preparing this README.

## Citation

If you use this code in academic work, please cite the associated paper once its bibliographic details are available.
