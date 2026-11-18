import React from "react";

export default function AssessmentComparison({ result }) {
  const measurements = Object.keys(result.measurements);
  const withinLimits = measurements.length === 4 && measurements.every(name =>
    result.parameters.some(parameter => parameter.name === name && parameter.abnormal === false)
  );
  if (!withinLimits || !["Poor", "Very Poor", "Unsuitable"].includes(result.predicted_class)) return null;
  return (
    <p role="note"><strong>Model and parameter assessments differ.</strong>{" "}
      All four measurements are within their configured limits. The model predicts {result.predicted_class} using
      the project’s weighted WQI target. Values within individual limits can still produce an adverse index class.
      Compare regulatory reports using their own index and sampling details.</p>
  );
}
