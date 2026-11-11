import globalShap from "../../reports/figures/shap_summary.png";
import Overview from "./Overview";
import Evaluation from "./Evaluation";
import DecisionView from "./DecisionView";
import { HistoricalTrends } from "./HistoricalTrends";
import Assessment from "./Assessment";
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Droplets,
  LayoutDashboard,
  ClipboardCheck,
  ScanLine,
  ChartBar,
  SlidersHorizontal,
  ListChecks,
  History,
  Database,
  Waves,
} from "lucide-react";
import "./style.css";

const pages = [
  [
    "overview",
    "Water Quality Overview",
    LayoutDashboard,
    "Explore surface-water observations, model predictions, and parameter-level insights.",
  ],
  [
    "input",
    "Water Quality Assessment",
    ClipboardCheck,
    "Enter four laboratory measurements to prepare a surface-water assessment.",
  ],
  [
    "prediction",
    "Prediction & Explainability",
    ScanLine,
    "Explore the predicted class and its probability distribution.",
  ],
  [
    "parameters",
    "Parameter Analysis",
    SlidersHorizontal,
    "Compare the sample measurements with their reference conditions.",
  ],
  [
    "decisions",
    "Decision Support",
    ListChecks,
    "Review follow-up actions for the selected sample.",
  ],
  [
    "evaluation",
    "Model Evaluation",
    ChartBar,
    "Compare candidate models on the project evaluation split.",
  ],
  [
    "trends",
    "Historical Trends",
    History,
    "Compare historical measurements and WQI classes by month, state, and station.",
  ],
];
function Card({ title, note, children, wide = false }) {
  return (
    <section className={wide ? "card wide" : "card"}>
      <h2>{title}</h2>
      {note && <small>{note}</small>}
      {children}
    </section>
  );
}
function Bars({ rows, max = 1 }) {
  return (
    <div className="bars">
      {rows.map(([name, value, label]) => (
        <div key={name}>
          <div className="row">
            <span>{name}</span>
            <b>{label}</b>
          </div>
          <div className="track">
            <i
              style={{
                width: (Math.abs(value) / max) * 100 + "%",
                background: value < 0 ? "#bd7839" : undefined,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
function PredictionView({ result }) {
  const [scope, setScope] = useState("local");
  const [selected, setSelected] = useState(null);
  if (!result)
    return (
      <Card title="No assessment yet">
        <a href="#input">Enter measurements to generate a prediction →</a>
      </Card>
    );
  const percent = (value) =>
    new Intl.NumberFormat(undefined, {
      style: "percent",
      maximumFractionDigits: 2,
    }).format(value);
  const active =
    result.contributions.find((f) => f.feature === selected) ||
    result.contributions[0];
  const shap = (
    <Card
      title={
        scope === "local"
          ? "Local SHAP contributions"
          : "Global SHAP contributions"
      }
      note={`Baseline model probability: ${percent(result.base_value)}`}
    >
      <div className="chart-tabs">
        {["local", "global"].map((id) => (
          <button
            type="button"
            key={id}
            aria-pressed={scope === id}
            onClick={() => setScope(id)}
          >
            {id === "local" ? "Individual explanation" : "Global explanation"}
          </button>
        ))}
      </div>
      {scope === "global" ? (
        <img
          className="evaluation-matrix"
          src={globalShap}
          alt="Global SHAP feature importance from the saved Phase 7 analysis"
        />
      ) : (
        <div className="bars">
          {result.contributions.map(({ feature, contribution }) => (
            <button
              type="button"
              className="feature-bar"
              key={feature}
              aria-pressed={active.feature === feature}
              onClick={() => setSelected(feature)}
            >
              <div className="row">
                <span>{feature}</span>
                <b>
                  {contribution > 0 ? "+" : ""}
                  {contribution.toFixed(4)}
                </b>
              </div>
              <div className="signed-track">
                <i
                  style={{
                    width: Math.abs(contribution) * 50 + "%",
                    [contribution < 0 ? "right" : "left"]: "50%",
                    background: contribution < 0 ? "#087d75" : "#3469dd",
                  }}
                />
              </div>
            </button>
          ))}
        </div>
      )}
      <small>{result.caveat}</small>
    </Card>
  );
  const probabilities = (
    <Card title="Class probabilities">
      <Bars
        rows={Object.entries(result.class_probabilities).map(([name, p]) => [
          name,
          p,
          percent(p),
        ])}
      />
    </Card>
  );
  return (
    <>
      <div className="grid prediction-summary">
        <Card title="Predicted WQI class">
          <strong>{result.predicted_class}</strong>
          <p>Model probability: {percent(result.model_probability)}</p>
          {result.uncertain && (
            <p role="status">
              Prediction is uncertain. Consider additional measurement or
              laboratory verification.
            </p>
          )}
        </Card>
        <Card title="Assessment measurements">
          <div className="measurement-tiles">
            {Object.entries(result.measurements).map(([name, v]) => (
              <div key={name}>
                <small>{name}</small>
                <b>{v}</b>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="grid prediction-detail">
        {shap}
        <div className="card-stack">
          {probabilities}
          <Card title="Feature interpretation">
            <b>{active.feature}</b>
            <p>
              Observed value ·{" "}
              {active.feature_value.toLocaleString(undefined, {
                maximumFractionDigits: 4,
              })}
            </p>
            <p>
              {Math.abs(active.contribution * 100).toFixed(2)} percentage points{" "}
              {active.contribution < 0
                ? "below"
                : active.contribution > 0
                  ? "above"
                  : "from"}{" "}
              baseline attributable to this feature for {result.predicted_class}
              .
            </p>
          </Card>
        </div>
      </div>
      <div className="card result-actions">
        <a href="#input">Edit measurements →</a>
        <a href="#decisions">View recommendations →</a>
      </div>
    </>
  );
}
function currentPage() {
  const hash = location.hash === "#explanation" ? "#prediction" : location.hash;
  return pages.find(([id]) => "#" + id === hash) || pages[0];
}
function App() {
  const [state, setState] = useState("");
  const [page, setPage] = useState(currentPage);
  const [prediction, setPrediction] = useState(null);
  const liveResult = [
    "prediction",
    "explanation",
    "parameters",
    "decisions",
  ].includes(page[0]);
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [attempt, retry] = useState(0);
  useEffect(() => {
    const update = () => setPage(currentPage());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setError(false);
    fetch("/api/dashboard", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw Error("API unavailable");
        return r.json();
      })
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setError(true);
      });
    return () => controller.abort();
  }, [attempt]);
  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <Droplets />
          AquaInsight
        </div>
        <nav aria-label="Dashboard sections">
          {pages.map(([id, title, Icon]) => (
            <a
              key={id}
              aria-current={id === page[0] ? "page" : undefined}
              href={"#" + id}
            >
              <Icon size={17} />
              {id === "overview" ? "Overview" : title}
            </a>
          ))}
        </nav>
        <div className="asideNote">
          <Waves />
          <b>Surface-water monitoring</b>
          <small>Historical water quality in India</small>
        </div>
        <details className="dataset-info">
          <summary>Dataset information</summary>
          <p>
            CPCB labeled surface-water observations from Maharashtra and Uttar
            Pradesh.
          </p>
          <a href="#trends">Explore sampling history →</a>
        </details>
      </aside>
      <main>
        <header>
          <Database size={17} /> CPCB surface-water dataset{" "}
          <span className="badge">
            {page[0] === "input"
              ? "Manual entry"
              : liveResult
                ? "Live model"
                : page[0] === "evaluation"
                  ? "Evaluation report"
                  : data
                    ? "Dataset connected"
                    : error
                      ? "Dataset unavailable"
                      : "Loading dataset"}
          </span>
          {page[0] === "overview" && data && (
            <label className="region-picker">
              State{" "}
              <select
                aria-label="State"
                value={state}
                onChange={(e) => setState(e.target.value)}
              >
                <option value="">All states</option>
                {[...new Set(data.stations.map((s) => s.State))].map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
          )}
        </header>
        <article>
          <h1>{page[1]}</h1>
          <p className="intro">{page[3]}</p>
          <div hidden={page[0] !== "input"}>
            <Assessment onPrediction={setPrediction} history={data} />
          </div>
          {page[0] === "input" ? null : prediction &&
            ["parameters", "decisions"].includes(page[0]) ? (
            <DecisionView
              page={page[0]}
              result={prediction}
              history={data}
              historyError={error}
              Card={Card}
            />
          ) : liveResult ? (
            <PredictionView result={prediction} />
          ) : page[0] === "evaluation" ? (
            <Evaluation Card={Card} />
          ) : error ? (
            <section role="alert">
              Dashboard data is unavailable.{" "}
              <button onClick={() => retry(attempt + 1)}>Retry</button>
            </section>
          ) : !data ? (
            <p role="status">Loading sample data…</p>
          ) : page[0] === "overview" ? (
            <Overview data={data} state={state} Card={Card} />
          ) : (
            <HistoricalTrends data={data} Card={Card} Bars={Bars} />
          )}
        </article>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
