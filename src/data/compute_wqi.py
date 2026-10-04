"""Water Quality Index (WQI) Computation & Ground-Truth Class Labeling Engine.

This module implements the official Weighted Arithmetic Water Quality Index (WAWQI)
formulation as established in `docs/wqi_standard.md` (STD-WQI-CPCB-2026-V1), adhering
to Central Pollution Control Board (CPCB) and Bureau of Indian Standards (BIS IS 10500:2012)
specifications.

Mathematical Formulation:
    WQI = sum(w_i * q_i) / sum(w_i)
    where:
        q_i = |(V_i - V_0) / (S_i - V_0)| * 100
        w_i = K / S_i  (dynamically normalized over active/non-null parameters)

Data Leakage Prevention Guard:
    `wqi_score` is computed strictly to derive the ground-truth target labels
    (`wqi_class` and `Potability`). `wqi_score` MUST NEVER be used as a predictive
    feature in any machine learning training or test feature set.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compute_wqi")


# ---------------------------------------------------------------------------
# Standard Parameter Configurations (BIS IS 10500:2012 & CPCB Guidelines)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterStandard:
    """Standard regulatory threshold and ideal baseline for a single parameter."""
    code: str
    column_name: str
    standard_limit: float  # S_i
    ideal_value: float     # V_0
    unit: str
    description: str


# 13 Standard parameters mapped from docs/wqi_standard.md
WQI_PARAMETERS: Dict[str, ParameterStandard] = {
    "pH": ParameterStandard(
        code="pH",
        column_name="Potential of Hydrogen (pH)",
        standard_limit=8.5,
        ideal_value=7.0,
        unit="pH scale",
        description="Potential of Hydrogen",
    ),
    "DO": ParameterStandard(
        code="DO",
        column_name="Dissolved oxygen (mg/L)",
        standard_limit=5.0,
        ideal_value=14.6,
        unit="mg/L",
        description="Dissolved Oxygen (saturation at 0°C)",
    ),
    "BOD": ParameterStandard(
        code="BOD",
        column_name="Biochemical Oxygen Demand (mg/L)",
        standard_limit=3.0,
        ideal_value=0.0,
        unit="mg/L",
        description="Biochemical Oxygen Demand (3-day at 27°C)",
    ),
    "COD": ParameterStandard(
        code="COD",
        column_name="Chemical Oxygen Demand (mg/L)",
        standard_limit=10.0,
        ideal_value=0.0,
        unit="mg/L",
        description="Chemical Oxygen Demand",
    ),
    "EC": ParameterStandard(
        code="EC",
        column_name="Electric Conductivity (μS/cm)",
        standard_limit=750.0,
        ideal_value=0.0,
        unit="μS/cm",
        description="Electrical Conductivity",
    ),
    "TDS": ParameterStandard(
        code="TDS",
        column_name="Total Dissolved Solids (mg/L)",
        standard_limit=500.0,
        ideal_value=0.0,
        unit="mg/L",
        description="Total Dissolved Solids",
    ),
    "Turbidity": ParameterStandard(
        code="Turbidity",
        column_name="Turbidity (NTU)",
        standard_limit=5.0,
        ideal_value=0.0,
        unit="NTU",
        description="Turbidity",
    ),
    "TC": ParameterStandard(
        code="TC",
        column_name="Total Coliform (MPN/100mL)",
        standard_limit=500.0,
        ideal_value=0.0,
        unit="MPN/100mL",
        description="Total Coliform Bacteria",
    ),
    "FC": ParameterStandard(
        code="FC",
        column_name="Fecal Coliform (MPN/100mL)",
        standard_limit=250.0,
        ideal_value=0.0,
        unit="MPN/100mL",
        description="Fecal Coliform Bacteria",
    ),
    "Nitrate": ParameterStandard(
        code="Nitrate",
        column_name="Nitrate N (mgN/L)",
        standard_limit=45.0,
        ideal_value=0.0,
        unit="mgN/L",
        description="Nitrate Nitrogen",
    ),
    "Hardness": ParameterStandard(
        code="Hardness",
        column_name="Total Hardness (mgCaCO3/L)",
        standard_limit=200.0,
        ideal_value=0.0,
        unit="mgCaCO3/L",
        description="Total Hardness",
    ),
    "Chloride": ParameterStandard(
        code="Chloride",
        column_name="Chloride (mg/L)",
        standard_limit=250.0,
        ideal_value=0.0,
        unit="mg/L",
        description="Chloride",
    ),
    "Sulfate": ParameterStandard(
        code="Sulfate",
        column_name="Sulphate (mg/L)",
        standard_limit=200.0,
        ideal_value=0.0,
        unit="mg/L",
        description="Sulfate",
    ),
}

# Standard WQI Classification Boundaries (CPCB / Brown et al. 1970)
WQI_CLASS_BOUNDARIES = [
    (0.0, 25.0, "Excellent"),
    (25.0, 50.0, "Good"),
    (50.0, 75.0, "Poor"),
    (75.0, 100.0, "Very Poor"),
    (100.0, float("inf"), "Unsuitable"),
]


@dataclass
class WQIAuditReport:
    """Summary metrics of WQI calculation and class distribution."""
    total_input_records: int = 0
    valid_labeled_records: int = 0
    quarantined_records: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_percentages: Dict[str, float] = field(default_factory=dict)
    potable_count: int = 0
    non_potable_count: int = 0
    potable_percentage: float = 0.0
    mean_wqi: float = 0.0
    median_wqi: float = 0.0
    is_severely_imbalanced: bool = False
    dominant_class: str = ""
    dominant_percentage: float = 0.0


# ---------------------------------------------------------------------------
# Core Mathematical Sub-Index and WQI Calculation Functions
# ---------------------------------------------------------------------------
def compute_sub_index(
    param_name: str,
    value: float,
    config: Optional[Dict[str, ParameterStandard]] = None,
) -> float:
    """Calculate the sub-index quality rating (q_i) for a single parameter.

    Mathematical formulation from docs/wqi_standard.md (Section 2.2):
        q_i = |(V_i - V_0) / (S_i - V_0)| * 100

    Special boundary rules:
        - pH: V_0 = 7.0, S_i = 8.5
          If V_pH >= 7.0: q_pH = ((V_pH - 7.0) / (8.5 - 7.0)) * 100 = ((V_pH - 7.0) / 1.5) * 100
          If V_pH < 7.0:  q_pH = ((7.0 - V_pH) / (7.0 - 6.5)) * 100 = ((7.0 - V_pH) / 0.5) * 100
        - DO: V_0 = 14.6, S_i = 5.0
          q_DO = |(V_DO - 14.6) / (5.0 - 14.6)| * 100 = (|V_DO - 14.6| / 9.6) * 100
        - All other parameters: V_0 = 0.0
          q_i = |V_i / S_i| * 100

    Args:
        param_name: Parameter code (e.g. 'pH', 'DO', 'BOD', 'FC', etc.)
        value: Numeric measured concentration.
        config: Optional parameter configuration dictionary.

    Returns:
        float: Sub-index rating q_i (non-negative).
    """
    param_dict = config if config is not None else WQI_PARAMETERS
    if param_name not in param_dict:
        raise KeyError(f"Unknown WQI parameter code '{param_name}'. Valid: {list(param_dict.keys())}")

    std = param_dict[param_name]
    val = float(value)

    if param_name == "pH":
        if val >= 7.0:
            return ((val - 7.0) / (8.5 - 7.0)) * 100.0
        else:
            return ((7.0 - val) / (7.0 - 6.5)) * 100.0

    if param_name == "DO":
        return (abs(val - 14.6) / 9.6) * 100.0

    # General parameters with V_0 = 0.0
    return (abs(val - std.ideal_value) / (std.standard_limit - std.ideal_value)) * 100.0


def compute_wqi_score(
    row: Union[pd.Series, Dict[str, Any]],
    config: Optional[Dict[str, ParameterStandard]] = None,
    min_parameters: int = 3,
) -> float:
    """Compute the Weighted Arithmetic Water Quality Index (WAWQI) score for a single row.

    Implements dynamic weight redistribution across available (non-null) parameters:
        WQI = sum(w_i * q_i) / sum(w_i)
        where w_i = 1 / S_i for active parameters.

    Args:
        row: Series or dictionary mapping column names to parameter values.
        config: Optional parameter configuration dictionary.
        min_parameters: Minimum number of valid parameters required (default: 3).

    Returns:
        float: Numeric WQI score, or np.nan if fewer than min_parameters are present.
    """
    param_dict = config if config is not None else WQI_PARAMETERS

    active_q: List[float] = []
    active_inv_s: List[float] = []

    for param_name, std in param_dict.items():
        # Check both the standard column name and the short code as possible keys
        val = None
        if std.column_name in row and pd.notna(row[std.column_name]):
            val = row[std.column_name]
        elif param_name in row and pd.notna(row[param_name]):
            val = row[param_name]

        if val is None:
            continue

        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            continue

        if np.isnan(numeric_val):
            continue

        q_i = compute_sub_index(param_name, numeric_val, config=param_dict)
        inv_s_i = 1.0 / std.standard_limit

        active_q.append(q_i)
        active_inv_s.append(inv_s_i)

    # Minimum parameter constraint: reject sparse records
    if len(active_q) < min_parameters:
        return np.nan

    sum_w = sum(active_inv_s)
    sum_wq = sum(q * w for q, w in zip(active_q, active_inv_s))

    if sum_w <= 0.0:
        return np.nan

    return float(sum_wq / sum_w)


def assign_wqi_class(
    score: float,
    boundaries: Optional[List[Tuple[float, float, str]]] = None,
) -> Optional[str]:
    """Assign CPCB/Brown et al. categorical water quality class label from numeric WQI score.

    Class Boundaries (docs/wqi_standard.md Section 4.1):
        - 0 <= WQI <= 25:   'Excellent' (Class A)
        - 25 < WQI <= 50:   'Good' (Class B)
        - 50 < WQI <= 75:   'Poor' (Class C)
        - 75 < WQI <= 100:  'Very Poor' (Class D)
        - WQI > 100:        'Unsuitable' (Class E)

    Args:
        score: Numeric WQI score.
        boundaries: Optional list of (lower, upper, label) tuples.

    Returns:
        Optional[str]: Categorical class string, or None if score is NaN.
    """
    if pd.isna(score):
        return None

    score_val = float(score)
    bounds = boundaries if boundaries is not None else WQI_CLASS_BOUNDARIES

    for lower, upper, label in bounds:
        if lower == 0.0:
            if lower <= score_val <= upper:
                return label
        else:
            if lower < score_val <= upper:
                return label

    # Fallback for negative or exceptional scores
    if score_val < 0.0:
        return "Excellent"
    return "Unsuitable"


def assign_potability(score: float) -> Optional[int]:
    """Assign canonical binary Potability target from numeric WQI score.

    Potability Target Mapping (docs/wqi_standard.md Section 4.2):
        Potability = 1 if WQI <= 50.0 (Potable / Excellent or Good)
        Potability = 0 if WQI > 50.0  (Not Potable / Poor, Very Poor, Unsuitable)

    Args:
        score: Numeric WQI score.

    Returns:
        Optional[int]: 1 (potable), 0 (not potable), or None if score is NaN.
    """
    if pd.isna(score):
        return None
    return 1 if float(score) <= 50.0 else 0


# ---------------------------------------------------------------------------
# Full Dataset Pipeline Execution
# ---------------------------------------------------------------------------
def run_wqi_labeling_pipeline(
    input_joined_path: Path,
    output_labeled_path: Optional[Path] = None,
    output_quarantine_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    min_parameters: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, WQIAuditReport]:
    """Execute end-to-end WQI computation, class labeling, and audit reporting.

    1. Loads joined CPCB monitoring dataset.
    2. Computes numeric `wqi_score`, categorical `wqi_class`, and binary `Potability`.
    3. Segregates records meeting minimum parameter threshold from quarantined records.
    4. Writes labeled dataset to `cpcb_labeled.csv` and quarantine to `quarantine.csv`.
    5. Computes distribution metrics and writes `reports/wqi_class_distribution.csv`.
    6. Flags severe class imbalance (>90% in single class) if detected.

    Args:
        input_joined_path: Path to `data/interim/cpcb_joined.csv`.
        output_labeled_path: Path to save `data/interim/cpcb_labeled.csv`.
        output_quarantine_path: Path to save `data/interim/quarantine.csv`.
        output_report_path: Path to save `reports/wqi_class_distribution.csv`.
        min_parameters: Minimum number of required active parameters (default: 3).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, WQIAuditReport]:
            (df_labeled, df_quarantine, audit_report)
    """
    logger.info("Loading joined CPCB dataset from: %s", input_joined_path)
    df = pd.read_csv(input_joined_path, encoding="utf-8", encoding_errors="replace")

    total_records = len(df)
    logger.info("Total records in joined dataset: %d", total_records)

    # Compute WQI scores
    wqi_scores: List[float] = [
        compute_wqi_score(row, min_parameters=min_parameters) for _, row in df.iterrows()
    ]
    df["wqi_score"] = wqi_scores
    df["wqi_class"] = [assign_wqi_class(s) for s in wqi_scores]
    df["Potability"] = [assign_potability(s) for s in wqi_scores]

    # Split into valid labeled records and quarantined records
    valid_mask = df["wqi_score"].notna()
    df_labeled = df[valid_mask].copy()
    df_quarantine = df[~valid_mask].copy()

    # Reset indices
    df_labeled.reset_index(drop=True, inplace=True)
    df_quarantine.reset_index(drop=True, inplace=True)

    # Cast binary Potability to integer
    df_labeled["Potability"] = df_labeled["Potability"].astype(int)

    # Compute audit metrics
    report = WQIAuditReport(
        total_input_records=total_records,
        valid_labeled_records=len(df_labeled),
        quarantined_records=len(df_quarantine),
    )

    class_order = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"]
    counts = df_labeled["wqi_class"].value_counts().to_dict()
    for cls in class_order:
        report.class_counts[cls] = counts.get(cls, 0)
        report.class_percentages[cls] = (
            round((report.class_counts[cls] / len(df_labeled)) * 100, 2)
            if len(df_labeled) > 0 else 0.0
        )

    report.potable_count = int((df_labeled["Potability"] == 1).sum())
    report.non_potable_count = int((df_labeled["Potability"] == 0).sum())
    report.potable_percentage = (
        round((report.potable_count / len(df_labeled)) * 100, 2)
        if len(df_labeled) > 0 else 0.0
    )

    if len(df_labeled) > 0:
        report.mean_wqi = float(df_labeled["wqi_score"].mean())
        report.median_wqi = float(df_labeled["wqi_score"].median())

    # Check for severe class imbalance (>90% in single class)
    for cls, pct in report.class_percentages.items():
        if pct > 90.0:
            report.is_severely_imbalanced = True
            report.dominant_class = cls
            report.dominant_percentage = pct

    # Generate and save reports/wqi_class_distribution.csv
    report_rows = []
    for cls in class_order:
        report_rows.append({
            "wqi_class": cls,
            "cpcb_category": {
                "Excellent": "Class A",
                "Good": "Class B",
                "Poor": "Class C",
                "Very Poor": "Class D",
                "Unsuitable": "Class E",
            }.get(cls, ""),
            "wqi_range": {
                "Excellent": "0 - 25",
                "Good": "26 - 50",
                "Poor": "51 - 75",
                "Very Poor": "76 - 100",
                "Unsuitable": "> 100",
            }.get(cls, ""),
            "count": report.class_counts[cls],
            "percentage": report.class_percentages[cls],
            "binary_potability": 1 if cls in ["Excellent", "Good"] else 0,
        })

    df_distribution = pd.DataFrame(report_rows)

    if output_report_path is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        df_distribution.to_csv(output_report_path, index=False)
        logger.info("Saved WQI class distribution report to: %s", output_report_path)

    if output_labeled_path is not None:
        output_labeled_path.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_csv(output_labeled_path, index=False)
        logger.info("Saved labeled dataset (%d rows) to: %s", len(df_labeled), output_labeled_path)

    if output_quarantine_path is not None and len(df_quarantine) > 0:
        output_quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        df_quarantine.to_csv(output_quarantine_path, index=False)
        logger.info("Saved quarantined dataset (%d rows) to: %s", len(df_quarantine), output_quarantine_path)

    # Logging summary
    logger.info("================ WQI COMPUTATION & AUDIT SUMMARY ================")
    logger.info("  Total Joined Observations: %d", report.total_input_records)
    logger.info("  Valid Labeled Records: %d (%.1f%%)", report.valid_labeled_records, (report.valid_labeled_records / report.total_input_records) * 100)
    logger.info("  Quarantined Records (<%d params): %d (%.1f%%)", min_parameters, report.quarantined_records, (report.quarantined_records / report.total_input_records) * 100)
    logger.info("  Mean WQI: %.2f | Median WQI: %.2f", report.mean_wqi, report.median_wqi)
    logger.info("  Multi-Class Breakdown:")
    for cls in class_order:
        logger.info("    * %-12s: %5d rows (%5.2f%%)", cls, report.class_counts[cls], report.class_percentages[cls])
    logger.info("  Binary Target (Potability):")
    logger.info("    * Potable (1):     %5d rows (%5.2f%%)", report.potable_count, report.potable_percentage)
    logger.info("    * Non-Potable (0): %5d rows (%5.2f%%)", report.non_potable_count, 100.0 - report.potable_percentage)

    if report.is_severely_imbalanced:
        logger.warning(
            "SEVERE CLASS IMBALANCE DETECTED: Class '%s' constitutes %.2f%% (>90%%) of labeled dataset!",
            report.dominant_class, report.dominant_percentage
        )

    return df_labeled, df_quarantine, report


# ---------------------------------------------------------------------------
# Hand-Calculation Cross-Check Documentation
# ---------------------------------------------------------------------------
# Cross-Check Case 1: Pure / Pristine Water Baseline
#   Inputs: pH=7.0, DO=14.6 mg/L, BOD=0.0 mg/L
#   Sub-indices:
#     q_pH  = |7.0 - 7.0| / 1.5 * 100 = 0.0
#     q_DO  = |14.6 - 14.6| / 9.6 * 100 = 0.0
#     q_BOD = |0.0 - 0.0| / 3.0 * 100 = 0.0
#   Weights:
#     1/S_pH = 1/8.5 = 0.117647, 1/S_DO = 1/5 = 0.200000, 1/S_BOD = 1/3 = 0.333333
#   WQI = (0*0.117647 + 0*0.2 + 0*0.333333) / (0.117647 + 0.2 + 0.333333) = 0.0000
#   Expected Class: Excellent (Class A), Potability: 1
#
# Cross-Check Case 2: Exact BIS Permissible Limit Threshold Water
#   Inputs: pH=8.5, DO=5.0 mg/L, BOD=3.0 mg/L, FC=250.0 MPN/100mL
#   Sub-indices:
#     q_pH  = (8.5 - 7.0) / 1.5 * 100 = 100.0
#     q_DO  = |5.0 - 14.6| / 9.6 * 100 = 100.0
#     q_BOD = 3.0 / 3.0 * 100 = 100.0
#     q_FC  = 250.0 / 250.0 * 100 = 100.0
#   WQI = sum(100 * w_i) / sum(w_i) = 100.0000
#   Expected Class: Very Poor (Class D, upper boundary 100.0), Potability: 0
#
# Cross-Check Case 3: Representative Mixed Quality Surface Water Sample
#   Inputs: pH=7.6, DO=6.8 mg/L, BOD=2.1 mg/L, FC=120.0 MPN/100mL
#   Sub-indices:
#     q_pH  = (7.6 - 7.0) / 1.5 * 100 = 40.0
#     q_DO  = |6.8 - 14.6| / 9.6 * 100 = 81.25
#     q_BOD = (2.1 / 3.0) * 100 = 70.0
#     q_FC  = (120.0 / 250.0) * 100 = 48.0
#   Weights:
#     w_pH = 1/8.5 = 2/17 ~ 0.1176470588
#     w_DO = 1/5.0 = 0.2000000000
#     w_BOD = 1/3.0 = 0.3333333333
#     w_FC = 1/250.0 = 0.0040000000
#     sum(w) = 0.6549803921568628
#   Numerator:
#     40.0 * (1/8.5) + 81.25 * 0.2 + 70.0 * (1/3.0) + 48.0 * 0.004
#     = 4.70588235 + 16.25 + 23.33333333 + 0.192 = 44.481215686
#   WQI = 44.481215686 / 0.6549803921568628 = 67.91228595... ~ 67.91
#   Expected Class: Poor (Class C, 50 < WQI <= 75), Potability: 0


def main():
    """CLI entry point for Milestone 1.3 WQI computation."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    joined_csv_path = repo_root / "data" / "interim" / "cpcb_joined.csv"
    labeled_csv_path = repo_root / "data" / "interim" / "cpcb_labeled.csv"
    quarantine_csv_path = repo_root / "data" / "interim" / "quarantine.csv"
    report_csv_path = repo_root / "reports" / "wqi_class_distribution.csv"

    if not joined_csv_path.exists():
        raise FileNotFoundError(
            f"Pre-label joined dataset missing at {joined_csv_path}. "
            "Please run `python src/data/join_cpcb.py` first."
        )

    df_labeled, df_quarantine, report = run_wqi_labeling_pipeline(
        input_joined_path=joined_csv_path,
        output_labeled_path=labeled_csv_path,
        output_quarantine_path=quarantine_csv_path,
        output_report_path=report_csv_path,
        min_parameters=3,
    )

    print(
        f"\n[WQI Labeling Success] {len(df_labeled)} records labeled -> {labeled_csv_path}\n"
        f"  * Quarantined: {len(df_quarantine)} records -> {quarantine_csv_path}\n"
        f"  * Distribution report -> {report_csv_path}\n"
        f"  * Multi-class distribution: {report.class_counts}\n"
        f"  * Binary Potability: Potable={report.potable_count} ({report.potable_percentage}%), "
        f"Non-Potable={report.non_potable_count} ({100.0 - report.potable_percentage:.2f}%)\n"
    )
    return df_labeled, df_quarantine, report


if __name__ == "__main__":
    main()
