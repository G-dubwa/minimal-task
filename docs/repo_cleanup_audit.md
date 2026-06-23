# Repository Cleanup Audit

## Initial branch and commit

- Initial branch: `main`
- Initial commit: `b44e10f52b42136a194cdeb1135eaf82ea36badb`
- Cleanup branch created: `cleaned-submission-version`
- Initial status: clean working tree before cleanup

## Current folder structure

Initial top-level repository contents were flat: the repository root contained the development notebooks, `deep_hedging.py`, `README.md`, and `.gitignore`. No `docs/`, `notebooks/`, `results/`, `figures/`, or `archive/` folders existed before this cleanup branch.

Initial directories found by `find . -maxdepth 2 -type d | sort`:

```text
.
./.git
./.git/hooks
./.git/info
./.git/logs
./.git/objects
./.git/refs
```

## Notebooks found

```text
./deep_hedging.ipynb
./final_benchmark_comparison.ipynb
./heston_stock_option_neural_hedging_COS_obsvol_multiseed (1).ipynb
./heston_stock_option_neural_hedging_COS_obsvol_multiseed_corrected.ipynb
./heston_stock_option_neural_hedging_second_review_revised.ipynb
./ideal_minimal_task_hyperparameter_tuning_with_architectures.ipynb
./parameter_robustness_study.ipynb
./transaction_cost_neural_hedging_extension (1).ipynb
```

## Result CSVs found

No committed `.csv` files were found in the initial repository checkout.

## Figure and output files found

No committed `.png`, `.pdf`, or `.zip` files were found in the initial repository checkout.

## Large files found

No files larger than 20 MB were found.

## Obvious duplicates

- `heston_stock_option_neural_hedging_COS_obsvol_multiseed (1).ipynb` and `heston_stock_option_neural_hedging_COS_obsvol_multiseed_corrected.ipynb` are closely related Heston COS notebooks. The corrected notebook is the safer entry point because it includes the reproducible Carr--Madan validation section and correction note.
- `heston_stock_option_neural_hedging_second_review_revised.ipynb` appears to be an earlier full-information Heston revision with related outputs and diagnostics.
- `deep_hedging.py` and `deep_hedging.ipynb` cover the original minimal Black--Scholes task in script and notebook forms.

## Files that appear report-ready

- `final_benchmark_comparison.ipynb`: consolidated Black--Scholes benchmark, robustness, single-scenario generalization, parameter-conditioned hedger, and data-generation sections.
- `parameter_robustness_study.ipynb`: focused robustness retraining study.
- `heston_stock_option_neural_hedging_COS_obsvol_multiseed_corrected.ipynb`: corrected Heston COS stock-and-option experiment with Carr--Madan validation code.
- `transaction_cost_neural_hedging_extension (1).ipynb`: transaction-cost supporting experiment.

## Files that appear exploratory or ambiguous

- `ideal_minimal_task_hyperparameter_tuning_with_architectures.ipynb`: useful for architecture selection, but contains a placeholder robustness function and looks like a development/tuning notebook.
- `heston_stock_option_neural_hedging_COS_obsvol_multiseed (1).ipynb`: superseded by the corrected Heston notebook.
- `heston_stock_option_neural_hedging_second_review_revised.ipynb`: useful background for full-information Heston results, but superseded for validation by the corrected COS notebook.
- Root-level generated outputs are not present in the checkout, so there is no direct file evidence to classify final CSV/figure outputs beyond notebook save paths.

## Immediate risks

- The original `.gitignore` ignored `*.csv`, `*.png`, and `*.pdf`, which would hide report outputs needed for submission traceability.
- Heston output names in the existing notebooks include several unsuffixed filenames. The corrected notebook adds mode-awareness, but reviewers should still run full-information and observable-volatility modes into separated folders or copy outputs into the new `results/main/heston_fullinfo/` and `results/appendix/heston_obsvol/` folders.
- Carr--Madan validation code is present in the corrected Heston notebook, but validation CSV files are not committed in the initial checkout because no generated outputs are present.
- Some notebooks contain long training runs. The cleanup should not rerun them automatically unless the group explicitly wants a fresh computational run.
