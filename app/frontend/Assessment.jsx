import React, { useEffect, useState } from "react";

export default function Assessment({ onPrediction, history }) {
  const [fields, setFields] = useState([]);
  const [values, setValues] = useState({});
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/assessment/schema", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw Error();
        return response.json();
      })
      .then(setFields)
      .catch(() => {
        if (!controller.signal.aborted)
          setError("Could not load measurement ranges.");
      });
    return () => controller.abort();
  }, [attempt]);

  async function submit(event) {
    event.preventDefault();
    const predict = event.nativeEvent.submitter?.value === "predict";
    onPrediction(null);
    if (missing.length || outside.length) {
      setResult(null);
      setError(missing.length ? "Please provide all required parameters." :
        "Input value is outside the allowed data range. Please verify the measurement.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    const measurements = Object.fromEntries(
      fields.map(({ name }) => [
        name,
        values[name] ? Number(values[name]) : null,
      ]),
    );
    try {
      const response = await fetch(
        predict ? "/api/assessment/predict" : "/api/assessment/preprocess",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(measurements),
        },
      );
      const body = await response.json();
      if (response.ok && predict) {
        onPrediction(body);
        if (location.hash === "#input") location.hash = "prediction";
      } else if (response.ok) setResult(body);
      else setError(body.error || "Could not preprocess measurements.");
    } catch {
      setError("Assessment processing is unavailable. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const present = fields.filter((f) => values[f.name]).length;
  const missing = fields.filter(f => !values[f.name]);
  const outside = fields.filter(f => values[f.name] && (Number(values[f.name]) < f.min || Number(values[f.name]) > f.max));
  const sample = history?.records.find(r => fields.length && fields.every(f => Number.isFinite(r[f.name]) && r[f.name]>=f.min && r[f.name]<=f.max));
  return (
    <div className="grid assessment">
      <section className="card">
        <h2>Measurement inputs</h2>
        <small>Surface-water sample · manual entry</small>
        <form noValidate onSubmit={submit}>
          <fieldset disabled={busy}>
            {fields.map(({ name, min, max }) => (
              <label key={name}>
                {name}
                <input
                  type="number"
                  step="any"
                  required
                  min={min}
                  max={max}
                  value={values[name] ?? ""}
                  onChange={(event) => {
                    onPrediction(null);
                    setValues({ ...values, [name]: event.target.value });
                    setError("");
                    setResult(null);
                  }}
                />
                <small>
                  Allowed range {min}–{max.toLocaleString()}
                </small>
              </label>
            ))}
            {fields.length > 0 && (
              <div className="form-actions">
                {sample && <button type="button" title={sample.station} onClick={()=>{
                  setValues(Object.fromEntries(fields.map(f=>[f.name,String(sample[f.name])])));
                  setError(""); setResult(null); onPrediction(null);
                }}>Use recorded sample</button>}
                <button
                  type="button"
                  onClick={() => {
                    onPrediction(null);
                    setValues({});
                    setError("");
                    setResult(null);
                  }}
                >
                  Reset
                </button>
                <button type="submit">
                  {busy ? "Preprocessing…" : "Validate and preprocess"}
                </button>
                <button type="submit" value="predict">
                  Generate prediction
                </button>
              </div>
            )}
          </fieldset>
        </form>
        {error && <p role="alert">{error}</p>}
        {!fields.length &&
          (error ? (
            <button
              onClick={() => {
                setError("");
                setAttempt(attempt + 1);
              }}
            >
              Retry
            </button>
          ) : (
            <p>Loading measurement ranges…</p>
          ))}
      </section>
      <div className="card-stack"><section className="card">
        <h2>Input summary</h2>
        {fields.length > 0 && (
          <p>
            {present} of {fields.length} required measurements entered
          </p>
        )}
        <p role="status">
          {result
            ? `Preprocessing complete. ${result.input_count} measurements produced ${result.processed_feature_count} model features.`
            : "Submit the sample to check its ranges and apply the frozen preprocessing pipeline."}
        </p>
      </section>
        <section className="card"><h2>Model readiness</h2><span className="badge">{fields.length ? "Assessment API connected" : "Awaiting API"}</span><p>The frozen decision tree uses these four measurements and derives log fecal coliform during preprocessing.</p></section>
        <section className="card"><h2>Validation states</h2>
        {missing.length>0 && <p>Required measurements remaining · {missing.map(f=>f.name).join(", ")}</p>}
        {outside.length>0 && <p>Outside allowed range · {outside.map(f=>f.name).join(", ")}</p>}
        {fields.length>0 && !missing.length && !outside.length && <p>All four values are within the input ranges.</p>}</section>
      </div>
    </div>
  );
}
