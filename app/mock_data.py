"""Illustrative dashboard fixtures. No model execution or observed station coordinates."""

# Display-only station observations for the overview and parameter analysis.
PARAMETERS = [
    ("Potential of Hydrogen (pH)", "pH", "pH", 7.2, "Within reference"),
    ("Dissolved oxygen (mg/L)", "DO", "mg/L", 4.8, "Below reference"),
    ("Biochemical Oxygen Demand (mg/L)", "BOD", "mg/L", 6.2, "Above reference"),
    ("Fecal Coliform (MPN/100mL)", "FC", "MPN/100mL", 320, "Above reference"),
]

# INTEGRATION POINT M9.4: use generate_decision_support for interpretations and actions.
ACTIONS = [
    {"priority": "High", "recommendation": "Repeat BOD and dissolved oxygen measurements at the sampling site."},
    {"priority": "High", "recommendation": "Collect a follow-up sample for microbial analysis."},
]

# INTEGRATION POINT M9.4: aggregate historical records by station, period and class.
DASHBOARD = {
    "mode": "Mock data", "state": "Maharashtra", "station": "Demo station A",
    "stations": ["Demo station " + letter for letter in "ABCDEFGH"],
    "distribution": {"Excellent": 2, "Good": 12, "Poor": 38, "Very Poor": 68, "Unsuitable": 120},
    "parameters": [dict(name=n, label=l, unit=u, value=v, status=s) for n, l, u, v, s in PARAMETERS],
    "decision": {"overall_priority": "High", "recommended_actions": {"action_items": ACTIONS}},
    "trends": [{"month": m, "value": v} for m, v in zip(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], [5.5, 5.1, 5.3, 4.9, 4.6, 4.8]
    )],
}
