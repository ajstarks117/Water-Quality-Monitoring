import ParameterDistribution from "./ParameterDistribution";
import AssessmentComparison from "./AssessmentComparison";
import React from "react";

export default function DecisionView({
  page,
  result,
  history,
  historyError,
  Card,
}) {
  const d = result.decision;
  const rules = d.parameter_interpretation;
  const abnormal = [
    ...new Map(
      [
        ...rules.abnormal_contributing_parameters,
        ...rules.additional_abnormal_parameters,
      ].map((r) => [r.rule_details.parameter, r.rule_details]),
    ).values(),
  ];
  if (page === "parameters")
    return (
      <>
        {history ? (
          <ParameterDistribution
            data={history}
            sample={result.measurements}
            Card={Card}
          />
        ) : (
          <p>
            {historyError
              ? "Historical distribution unavailable."
              : "Loading historical distribution…"}
          </p>
        )}
        <Card
          title="Measurement review"
          note="Four measured parameters and one derived fecal coliform feature"
        >
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {[
                    "Parameter",
                    "Value",
                    "Unit",
                    "Reference",
                    "Status",
                    "Interpretation",
                    "SHAP contribution",
                  ].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.parameters.map((p) => (
                  <tr key={p.name}>
                    <th scope="row">{p.name}</th>
                    <td>{Number(p.value.toPrecision(7))}</td>
                    <td>{p.unit}</td>
                    <td>
                      {p.minimum !== null && p.maximum !== null
                        ? p.minimum + "–" + p.maximum
                        : p.minimum !== null
                          ? "≥ " + p.minimum
                          : "≤ " + p.maximum}
                      <small>{p.source}</small>
                    </td>
                    <td>{p.abnormal ? "Abnormal" : "Normal"}</td>
                    <td>
                      {abnormal.find((r) => r.parameter === p.name)
                        ?.interpretation || "Within the configured reference."}
                    </td>
                    <td>
                      {result.contributions
                        .find((f) => f.feature === p.name)
                        ?.contribution.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    );
  return (
    <div className="decision-grid">
      <Card title="Assessment context">
        <strong>{d.prediction.predicted_class}</strong>
        <AssessmentComparison result={result} />
        <mark>Priority · {d.overall_priority}</mark>
      </Card>
      <Card
        title="Contributing parameters"
        note="Largest local SHAP contributions to the predicted class"
      >
        {d.model_level_evidence.top_contributing_features.map((f) => (
          <div className="analysis" key={f.feature}>
            <b>{f.feature}</b>
            <span>{f.contribution.toFixed(4)}</span>
          </div>
        ))}
      </Card>
      <Card title="Parameter interpretation">
        {abnormal.map((r) => (
          <div key={r.parameter}>
            <h3>{r.parameter}</h3>
            <p>{r.interpretation}</p>
          </div>
        ))}
        {rules.shap_important_normal_parameters.map((f) => (
          <p key={f.feature}>
            {f.feature} contributed to the prediction within its reference
            range.
          </p>
        ))}
        {!abnormal.length && (
          <p>All five values fall within their configured references.</p>
        )}
      </Card>
      <Card title="Recommended actions">
        {d.recommended_actions.action_items.map((a) => (
          <div className="action" key={a.parameter + a.recommendation}>
            <mark>{a.priority}</mark>
            <div>
              <b>{a.parameter}</b>
              <p>{a.recommendation}</p>
            </div>
          </div>
        ))}
        {!d.recommended_actions.action_items.length && (
          <p>No follow-up action was triggered by the configured rules.</p>
        )}
      </Card>
      <p className="evidence-note">
        {d.causation_caveat} <a href="#parameters">Review parameter values →</a>
      </p>
    </div>
  );
}
