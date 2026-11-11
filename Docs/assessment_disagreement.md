# Krishna River assessment audit

## Finding

MPCB station 37 (Krishna River at Maighat, Sangli, Kolhapur region) lists the February 2026 monthly observation as pH 7.8, DO 6.3 mg/L, BOD 2.6 mg/L and fecal coliform 4 MPN/100 mL, with WQI 87.10. Its table also reports COD 16 mg/L and nitrate 0.92 mg/L. The project labeled dataset spans 1 January through 30 December 2021. No contemporaneous training observation establishes the condition of this station in February 2026. [MPCB station selector](https://www.mpcb.gov.in/water-quality/Kolhapur/22) · [Station 37 monthly table](https://mpcb.ecmpcb.in/envtdata/getWaterStation.php?cId=37&fdate=2026&tdate=2026)

| Assessment | What it measures | Result |
| --- | --- | --- |
| MPCB | Modified NSF index with piecewise quality subindices for DO in percent saturation, fecal coliform, pH and BOD. Their weights are 0.31, 0.28, 0.22 and 0.19 respectively. Higher is better. | 87.10 is in the 63–100 Good to Excellent band. The site also identifies Medium to Good, Bad and Bad to Very Bad bands. |
| Project label formula | Weighted arithmetic index over available target parameters. Higher means greater departure from ideal. | Replaying the four submitted measurements gives 80.09624, hence Very Poor. This replay is a diagnostic calculation independent of the model. |
| Frozen classifier | Learns the project's class from four raw inputs and derived log FC. | Very Poor, with model probability 1.0. |
| Configured rules | Checks each measured value against its configured reference. | All four measured values and the derived log FC pass. Optional measurements, including COD in the MPCB table, were outside the submitted assessment. |

The station table supplies DO concentration, while MPCB's published calculation uses saturation. Without the saturation input or the temperature and conversion conditions, this audit verifies the **reported** 87.10 rather than reproducing it independently. The two indices have different formulas, inputs, directions and class definitions. Neither their numbers nor their class labels can be substituted for one another. [MPCB annual report 2023–24, tables 5.25–5.27](https://www.mpcb.gov.in/sites/default/files/Establishment%20of%20MPCB/Seniority%20list/2014/AR_23-24_13062025.pdf)

## Target generation and model trace

1. In `src/data/compute_wqi.py`, the labeling pipeline scores up to 13 available parameters, normalizing inverse-standard weights over the active set. The executable minimum is three available parameters. The written core-parameter minimum in `Docs/wqi_standard.md` differs. Replaying the executable formula reproduced all 3,287 stored scores within floating-point tolerance.
2. For this four-input replay, the pH, DO, BOD and FC subindices are 53.333, 86.458, 86.667 and 1.600. Their unnormalized weights are 1/8.5, 1/5, 1/3 and 1/250. BOD and DO dominate. Both are within individual limits, yet both are far from the project ideals of zero BOD and 14.6 mg/L DO. Active weights normalize to produce 80.09624. Project upper boundaries are inclusive at 25 (Excellent), 50 (Good), 75 (Poor) and 100 (Very Poor), followed by Unsuitable.
3. The selected decision tree fits string targets. The explicit LabelEncoder in the training module belongs to the XGBoost candidate. The artifact's actual class order is Excellent, Good, Poor, Unsuitable, Very Poor. Its probability vector here is [0, 0, 0, 0, 1]. Phase 7 zips the vector with the classifier's classes and indexes SHAP by the predicted string. The HTTP response, nested decision output and React views use that label directly.
4. The frozen pipeline engineers log10(max(FC, 1)), then applies its fitted preprocessing and classifier. For FC 4 the derived value is 0.60205999. In model feature order the transformed query is [0.05239059, 0.02752271, -0.45351066, -0.06826815, -1.22858894]. Using saved split indices, the inference transform on original rows matched the saved training preprocessor applied to prepared features across all 2,629 training and 658 holdout rows (maximum absolute difference 8.9e-16). Saved target labels matched the original labeled rows on both splits.
5. For Very Poor, the SHAP base is 0.20000. Contributions are BOD +0.75070, pH +0.02794, DO +0.01439, log FC +0.00891 and raw FC −0.00194. The unrounded sum plus base equals 1.0 within numerical tolerance. These explain the classifier response, not a regulatory exceedance. A pure tree leaf assigns 100% to one class without establishing accuracy on this sample.

## Nearby labeled observations

The comparison uses complete original measurements (2,252 train and 558 holdout), aligned to saved split indices. Rank by Euclidean distance in training-standardized pH, DO, BOD and log FC, using the microbial measurement once. The closest observation in each split has a nonzero distance. The table contains aggregate and target evidence only.

| Split | Three closest distances | Stored project scores and actual classes in distance order |
| --- | --- | --- |
| Training | 0.106, 0.146, 0.155 | 90.595 Very Poor, 69.923 Poor, 66.530 Poor |
| Holdout | 0.076, 0.126, 0.229 | 90.593 Very Poor, 94.304 Very Poor, 74.587 Poor |

The nearest training and holdout observations both had BOD 3.2 mg/L. The nearby Poor examples had lower BOD (1.8–2.2 mg/L). These local examples support the plausibility of the project prediction under its own target. They do not establish external regulatory accuracy.

## Proposed follow-up requiring approval

Reconcile the written minimum-parameter policy with the executable target specification and distinguish project class names from regulatory designations in the standards documentation. If reproducing MPCB becomes a project requirement, first validate a separate versioned calculator using dated station records with known DO saturation, units and class definitions. Evaluate any proposed model against that explicitly chosen target on independent stations and dates, including calibration. This audit changes only dashboard interpretation and its tests.
