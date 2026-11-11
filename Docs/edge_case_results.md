# M10.2 dashboard edge-case results

On 11 November 2026, Playwright drove Microsoft Edge against the React/Vite dashboard and its existing Python HTTP API (`python -m app.api`). All five cases below used live endpoints. Successful submissions rendered local SHAP contributions from the frozen artifact. Inputs are listed in the order pH, DO (mg/L), BOD (mg/L), fecal coliform (MPN/100 mL).

| Case | Submitted measurements | Observed result |
| --- | --- | --- |
| TC-10.2-01 · Missing | 7, 7, 2, blank | **Pass.** Generate prediction displayed “Please provide all required parameters.” The browser sent zero prediction requests. Submitting the same invalid payload directly returned HTTP 422 with the same message. |
| TC-10.2-02 · Impossible | 20, 7, 2, 100 | **Pass.** The form displayed “Input value is outside the allowed data range. Please verify the measurement.” Zero browser prediction requests. The API independently returned 422. |
| TC-10.2-03 · Conflicting | 7, 7, 90, 100 | **Pass.** BOD alone exceeded its configured limit of 3 mg/L. Decision Support named BOD under Parameter interpretation and displayed High priority follow-up actions for it. |
| TC-10.2-04 · Borderline | 8.44, 8.7, 2.3, 800 | **Pass.** A complete observation from the local frozen holdout returned Poor with 100% model probability. The UI warned that changing only pH to 8.45 gives Very Poor. A second live submission confirmed that class. |
| TC-10.2-05 · Outside training support | 10, 7, 2, 100 | **Pass.** The API accepted pH 10 within the physical input range. The prediction view showed the pH-specific training-range warning (5.45–9.2) alongside the result. |

## Interpretation and limitations

All leaves of the frozen classifier have a maximum class proportion of 1.0. A real sample cannot exercise the existing probability-below-0.60 warning. The dashboard now also checks whether a small change in one measurement changes the predicted class. It checks both directions within the allowed bounds, using 0.01 pH units, 0.1 mg/L DO or BOD and 1 MPN/100 mL fecal coliform. These are screening increments. Laboratory precision varies, and stability under these probes cannot establish accuracy.

The range warning uses per-feature extrema from the raw training split before imputation. It can detect individual measurements beyond training support, but unusual combinations within all four ranges can pass. Warnings supplement the dashboard response and leave model probabilities, preprocessing, SHAP attributions and decision rules intact.

## Regression evidence

- Frontend production build passed. All 12 Playwright tests passed in 18.2 seconds, including the five live cases above. Run `npm test -- --workers=1` from app/frontend with the project Python environment active.
- Full unfiltered local pytest suite with the original datasets passed all 222 tests in 43.67 seconds with 1,359 warnings. The environment reports an existing scikit-learn artifact/runtime version mismatch. The frozen artifact was used as supplied.
- The existing required GitHub check runs the standard Python suite. Browser regression evidence above comes from the local Windows run.
