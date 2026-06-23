# Neural Hedging Minimal Task

This repository contains the code and outputs for a Master of Financial
Engineering research project on neural-network hedging architectures for
European call options.

The project starts from a controlled Black--Scholes benchmark, where analytic
pricing and hedging references are available, and then extends the selected
shared Markov MLP architecture to parameter robustness, parameter-conditioned
hedging, transaction costs, supporting tail-risk objectives where retained, and
Heston stock-and-option hedging.

## Cleaned submission branch

The original working files are kept on `main`. This branch reorganises the code
and outputs for submission review.

Original notebooks and exploratory outputs are retained for traceability, but
the recommended entry point is the single consolidated notebook in `notebooks/`.

## Submission-facing notebook

The recommended notebook for reproducing and inspecting the report results is:

`notebooks/01_reproduce_report_results.ipynb`

It consolidates the experiments still present in the current report. Removed
extensions and old exploratory experiments are preserved only in the archive for
traceability.

The previous two-notebook cleanup structure has been superseded. The old
`01_main_report_experiments.ipynb` and
`02_appendix_and_supporting_experiments.ipynb` files are preserved in
`notebooks/archive/old_cleaning_attempts/`.

## Experiments

1. Black--Scholes final benchmark
2. Architecture selection
3. Parameter robustness under retraining
4. Single-scenario generalization failure
5. Parameter-conditioned hedger
6. Main Heston stock-and-option extension
7. Data-generation comparison
8. Transaction-cost extension
9. CVaR-loss extension
10. Observable-volatility Heston robustness check

## Repository guide

| Path | Contents |
|---|---|
| `notebooks/01_reproduce_report_results.ipynb` | Clean end-to-end report-results notebook |
| `notebooks/archive/original_notebooks/` | Preserved copies of the original development notebooks |
| `notebooks/archive/old_cleaning_attempts/` | Superseded two-notebook cleanup attempt |
| `docs/repo_cleanup_audit.md` | Initial repository audit before cleanup |
| `docs/results_manifest.md` | Mapping from outputs to report use |
| `docs/repo_cleanup_feedback.md` | Cleanup notes, residual risks, and suggestions |
| `results/report/` | Report-ready CSV outputs, grouped by experiment |
| `figures/report/` | Report-ready figures, grouped by experiment |
| `archive/` | Exploratory, superseded, or ambiguous material |

## Reproducibility notes

- The main entry point is `notebooks/01_reproduce_report_results.ipynb`.
- By default, the notebook loads and displays saved report-ready outputs and runs quick validation checks.
- Set `RUN_MODE = "full_generation"` near the top of the notebook to generate the clean submission outputs through callable `src` functions.
- Some expensive neural-network results are regenerated as explicitly labelled reported-run tables rather than retrained in the lightweight Colab pass.
- In Google Colab, set `SAVE_TO_DRIVE = True` to mirror outputs to `MyDrive/MFE_neural_hedging_report_outputs/`.
- The notebook writes progress to `logs/run_progress.csv` and creates `report_outputs_latest.zip` at the end of a run.
- Seeds are set in notebooks where possible.
- Heston full-information and observable-volatility outputs are separated by filename or folder.
- Carr--Madan validation code and exported CSVs support the COS-pricer validation discussed in the report.
- The Heston full-information result should be generated with `USE_OBSERVABLE_VOL = False`.
- The observable-volatility robustness check should be generated with `USE_OBSERVABLE_VOL = True`.

## Setup

Install the Python dependencies into a fresh environment:

```bash
pip install -r requirements.txt
```

TensorFlow is used by the Black--Scholes neural hedging notebooks. PyTorch is
used by the Heston and transaction-cost notebooks.

## Original minimal-task files

The initial repository included `deep_hedging.py` and `deep_hedging.ipynb`, which
implement the first Black--Scholes minimal task. They are preserved because they
are useful for tracing the project history, but the cleaned branch should be
reviewed through the submission-facing notebooks first.
