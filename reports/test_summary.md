# M10.1 integration test record

The five dashboard acceptance cases POST four raw measurements to the same HTTP handler used by the dashboard. Each checks the expected class, five model probabilities, the log-transformed fecal coliform feature, SHAP contributions and the decision-support response. They use representative holdout observations for Excellent, Good, Poor, Very Poor and Unsuitable with the frozen classifier. Passing these five cases verifies integration. Model accuracy requires evaluation across the holdout set.

## Verification on 11 November 2026

| Case | Result |
| --- | --- |
| TC-10.1-01, GitHub Actions on a pull request | [PR 4](https://github.com/ajstarks117/Water-Quality-Monitoring/pull/4) triggered the workflow. GitHub reported an account billing lock before runner startup. A successful hosted run is pending. |
| TC-10.1-02, all five predicted classes | Five dashboard API cases passed locally. |
| TC-10.1-03, full suite from repository root | 222 passed, zero failures, 1,359 warnings in 83.93 seconds (local Windows run). |
| Standard CI selection in a clean checkout with generated inputs | 199 passed, zero failures, 23 official-data cases deselected, 1,302 warnings in 63.96 seconds (local Windows run). |

GitHub branch protection on main requires the up-to-date **Dashboard pytest** check from GitHub Actions and applies to administrators. After the account lock clears, rerun the check on the current PR head before merging.

## Dataset coverage

Run `python -m pytest` from the prepared project root for complete coverage. For the standard suite, in a fresh checkout install requirements.txt, run `python -m tests.prepare_ci_data`, then run `python -m pytest -m "not official_cpcb"`. This generates 512 deterministic rows and split artifacts in the disposable checkout. The acceptance cases use their own measurements and the tracked frozen model.

| Tests carrying the official_cpcb marker | Inputs and reason |
| --- | --- |
| test_join_cpcb | The raw-source discovery, composite-key uniqueness and statewide join accounting cases check the six original files. Generated rows can test the join algorithm but cannot verify those files' keys or counts. The synthetic join cases also run in CI. |
| test_compute_wqi, test_data_dictionary_and_schema, test_load_data | These integration cases consume the official joined or labeled dataset to verify pipeline output, documented schema, class distribution and audit counts. A generated dataset cannot certify the published dataset description. |
| test_preprocessing, test_eda, test_engineering | These cases verify missing values, retained extremes, observed distribution or feature correlation in the labeled dataset. Those properties depend on the observations. |
| test_shap_explainer | The expected importance ranking depends on the official evaluation sample. The five per-prediction SHAP checks run in standard CI. |

The original dataset assertions remain in the local suite. No permission to redistribute the six specific CSVs was verified. No raw files or release assets were published. The portal directs users to the supplying agency for clarification. [NWDP disclaimer](https://nwdp.nwic.gov.in/footer/disclaimer)

## Integration fixes

The full run exposed a shape mismatch in the global SHAP bar plot. The plot now receives one samples-by-features matrix per class. Clean-checkout testing also exposed two fixture assumptions about timestamp naming and the first observation's missingness. The generator now provides the acquisition timestamp and a complete first observation.
