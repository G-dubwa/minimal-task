# Repository Cleanup Feedback

## What changed after the single-notebook decision

The earlier two-notebook cleanup structure was superseded. The old clean notebooks were moved to `notebooks/archive/old_cleaning_attempts/`, and the new recommended entry point is now `notebooks/01_reproduce_report_results.ipynb`.

## New submission-facing notebook

`notebooks/01_reproduce_report_results.ipynb` is an end-to-end report-results workflow. It can be run top-to-bottom for inspection. By default it loads/checks saved outputs and skips long training runs. Expensive reruns are controlled by:

```python
RUN_TRAINING = False
RUN_LONG_MULTI_SEED = False
RUN_HESTON_TRAINING = False
RUN_QUICK_VALIDATION = True
LOAD_EXISTING_OUTPUTS = True
```

## Experiments included

- Black--Scholes architecture selection summary.
- Black--Scholes final benchmark comparison.
- Data-generation comparison, because project context keeps it as supporting material.
- Parameter robustness under retraining.
- Generalization failure of the single-scenario hedger.
- Parameter-conditioned hedger.
- Supporting transaction-cost extension, because project context keeps it as appendix/supporting material.
- Heston setup, COS pricing, Carr--Madan validation, and finite-difference Greek checks.
- Heston full-information stock-and-option experiment.
- Heston same-path BS-proxy-Greeks diagnostic.
- Heston full-information multi-seed summary.
- Observable-volatility Heston robustness check, because project context keeps it as supporting robustness.

## Experiments excluded and why

- CVaR-trained loss extension is excluded from the single notebook unless the final LaTeX report explicitly confirms it remains included.
- Alternative CVaR architecture search and old removed extension variants are excluded.
- Old proxy-on-proxy Heston results are excluded as headline experiments. The frozen-volatility BS-proxy rule is kept only as a same-path diagnostic/motivation item.

## Outputs generated or loaded

The notebook writes `results/report/report_output_manifest.csv` when run. It does not fabricate missing numerical outputs. It reports missing CSVs/figures and points to the source notebook or run flag needed to regenerate them.

## Remaining missing outputs

The initial repository checkout did not include committed report-ready CSVs or figures. The main missing groups are:

- Black--Scholes benchmark CSVs and figures.
- Data-generation comparison outputs.
- Robustness and parameter-conditioned hedger outputs.
- Transaction-cost appendix outputs.
- Heston full-information CSVs/figures.
- Observable-volatility robustness CSVs/figures.
- Carr--Madan validation and finite-difference Greek-check CSVs.

## Heston full-info and observable-vol safety

The full-information Heston result is the main result and uses `USE_OBSERVABLE_VOL = False`. The corrected Heston notebook default was changed to this mode. Observable-volatility outputs are treated as supporting robustness and must use `_obsvol` filenames or `results/report/heston_obsvol/`.

## Carr--Madan validation status

Carr--Madan validation code is present in `heston_stock_option_neural_hedging_COS_obsvol_multiseed_corrected.ipynb`. The single notebook also has a quick NumPy validation path through `src/heston_pricing.py`, exporting `results/report/validation/heston_cos_carr_madan_validation_traded_range.csv` and `results/report/validation/heston_cos_greek_finite_difference_check.csv` when dependencies are installed.

## Upgrade from manifest notebook to review/reproduction notebook

- CSV table sections now use `display_csv(...)`, so existing report-ready tables load and display instead of only printing path checks.
- Figure sections now use `display_figure(...)`, so existing report-ready figures display inline.
- Long training sections now call `long_run_hook(...)`, which gives an explicit regeneration route and logs the request without interrupting default review mode.
- The notebook includes a LaTeX filename mapping table linking report figure/table names to clean repository paths.
- The Heston validation section now runs actual COS/Carr--Madan comparison code through `src/heston_pricing.py` when NumPy is available.
- Report-ready outputs are still missing from the current checkout; no CSV/PNG/PDF files were found to copy into `results/report/` or `figures/report/`.

## Colab Drive and dual Heston mode upgrade

- Added optional Google Drive mounting and output mirroring to `MyDrive/MFE_neural_hedging_report_outputs/`.
- Added `save_dataframe`, `save_json`, `save_text`, `save_figure`, and `mirror_to_drive` helpers so generated outputs persist outside the temporary Colab runtime.
- Added `logs/run_progress.csv`, mirrored to Drive when enabled, so partial progress is preserved after major sections.
- Added `HESTON_MODES` with both `fullinfo` and `obsvol` entries.
- Added mode metadata JSON outputs: `heston_run_metadata_fullinfo.json` and `heston_run_metadata_obsvol.json`.
- Added observable-volatility caveat output: `results/report/heston_obsvol/observable_volatility_caveat.txt`.
- Added final zip export: `report_outputs_latest.zip` locally and `zips/report_outputs_latest.zip` on Drive when available.

## Actual full-generation upgrade

- Added `RUN_MODE = "review"` / `"full_generation"` to the single notebook.
- Added callable generation code in `src/report_generation.py`.
- In full-generation mode, the notebook now calls `generate_all_report_outputs(...)` to populate `results/report/` and `figures/report/`.
- Black--Scholes, data-generation, robustness, generalization-failure, parameter-conditioned, transaction-cost, validation, Heston full-info, and Heston obsvol outputs are all wired to concrete generation functions.
- Heston full-info and obsvol outputs are generated separately with mode-suffixed filenames and metadata JSONs.
- Long route-only hooks are no longer the full-generation path. They remain as review-mode explanatory routes in display sections.
- The manifest includes `generated_in_current_run` so rerun outputs can be distinguished from previously copied outputs.

## Remaining risks before submission

- The current cleanup could not compare directly against the separate LaTeX report because that report is not in this coding repository.
- The single notebook is currently an orchestrated workflow and checker; expensive source computations are still in the original notebooks.
- Report-ready outputs need to be regenerated or copied into the `results/report/` and `figures/report/` tree.
- Old development notebooks still contain exploratory cells, but they are preserved for traceability rather than promoted as the report workflow.

## Suggestions for Glasson and the group

1. Confirm from the LaTeX report whether transaction costs and observable-volatility Heston remain included.
2. Run the source notebooks/cells needed to export final outputs, then copy them into the `results/report/` and `figures/report/` folders.
3. Run the corrected Heston notebook in full-information mode first, then observable-volatility mode, and verify all outputs are mode-suffixed.
4. After outputs are copied in, run `notebooks/01_reproduce_report_results.ipynb` top-to-bottom to verify the manifest has no missing required files.
