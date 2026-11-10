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
    "Prediction",
    ScanLine,
    "Explore the predicted class and its probability distribution.",
  ],
  [
    "explanation",
    "Explainability",
    ChartBar,
    "Inspect the contributions to the predicted class probability.",
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
    "trends",
    "Historical Trends",
    History,
    "Compare historical measurements and WQI classes by month, state, and station.",
  ],
];
const colors = ["#5aab9b", "#0e7974", "#e3b053", "#cf803b", "#5366a4"];
function Card({ title, note, children, wide = false }) {
  return (
    <section className={wide ? "card wide" : "card"}>
      <h2>{title}</h2>
      {note && <small>{note}</small>}
      {children}
    </section>
  );
}
function Overview({ d }) {
  const counts = Object.entries(d.distribution);
  const total = counts.reduce((sum, [, n]) => sum + n, 0);
  let angle = 0;
  const stops = counts.map(([, n], i) => {
    const start = angle;
    angle += (n / total) * 360;
    return colors[i] + " " + start + "deg " + angle + "deg";
  });
  const mapped = d.stations.filter(
    (s) => Number.isFinite(s.Latitude) && Number.isFinite(s.Longitude),
  );
  const xs = mapped.map((s) => s.Longitude),
    ys = mapped.map((s) => s.Latitude);
  const west = Math.min(...xs),
    south = Math.min(...ys);
  const width = Math.max(...xs) - west || 1,
    height = Math.max(...ys) - south || 1;
  return (
    <>
      <div className="metrics">
        {[
          ["Total valid observations", total],
          ["Unique stations", d.stations.length],
          ["Parameters analyzed", d.features.length],
          ["Supported target classes", counts.length],
        ].map(([label, n]) => (
          <section className="card" key={label}>
            <small>{label}</small>
            <strong>{n}</strong>
          </section>
        ))}
      </div>
      <div className="grid">
        <Card
          wide
          title="Water-station distribution"
          note="Station locations by longitude and latitude"
        >
          <div className="map">
            {mapped.map((s) => (
              <span
                className="dot"
                key={s.State + s.Station}
                title={s.Station + ", " + s.State}
                style={{
                  left: 4 + (90 * (s.Longitude - west)) / width + "%",
                  top: 94 - (90 * (s.Latitude - south)) / height + "%",
                }}
              />
            ))}
          </div>
        </Card>
        <Card title="Class distribution" note="Labeled CPCB observations">
          <div
            className="ring"
            style={{ background: "conic-gradient(" + stops.join(",") + ")" }}
          >
            <b>
              {total}
              <small>records</small>
            </b>
          </div>
          {counts.map(([c, n], i) => (
            <div className="legend" key={c}>
              <i style={{ background: colors[i] }} />
              {c}
              <b>{n}</b>
            </div>
          ))}
        </Card>
      </div>
      <div className="bottom">
        <Card title="Sampling coverage">
          {[...new Set(d.stations.map((s) => s.State))].map((state) => (
            <p key={state}>
              {state} · {d.stations.filter((s) => s.State === state).length}{" "}
              stations
            </p>
          ))}
        </Card>
        <Card title="Data source">
          <p>CPCB surface-water monitoring records</p>
          <a href="#trends">Explore historical measurements →</a>
        </Card>
        <Card title="Model readiness">
          <p>Frozen decision tree</p>
          <a href="#prediction">View prediction →</a>
        </Card>
      </div>
    </>
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
function PredictionView({ result, explanationFirst }) {
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
  const shap = (
    <Card
      title="Local SHAP contributions"
      note={`Baseline model probability: ${percent(result.base_value)}`}
    >
      <div className="bars">
        {result.contributions.map(({ feature, contribution }) => (
          <div key={feature}>
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
          </div>
        ))}
      </div>
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
      <div className="grid">
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
          {Object.entries(result.measurements).map(([name, v]) => (
            <div className="row" key={name}>
              <span>{name}</span>
              <b>{v}</b>
            </div>
          ))}
        </Card>
      </div>
      <div className="grid">
        {explanationFirst ? shap : probabilities}
        {explanationFirst ? probabilities : shap}
      </div>
      <a href="#input">Edit measurements →</a>
    </>
  );
}
function currentPage() {
  return pages.find(([id]) => "#" + id === location.hash) || pages[0];
}
function App() {
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
      </aside>
      <main>
        <header>
          <Database size={17} /> CPCB surface-water dataset{" "}
          <span className="badge">
            {page[0] === "input"
              ? "Manual entry"
              : liveResult
                ? "Live model"
                : "Historical records"}
          </span>
        </header>
        <article>
          <h1>{page[1]}</h1>
          <p className="intro">{page[3]}</p>
          <div hidden={page[0] !== "input"}>
            <Assessment onPrediction={setPrediction} />
          </div>
          {page[0] === "input" ? null : prediction &&
            ["parameters", "decisions"].includes(page[0]) ? (
            <DecisionView page={page[0]} result={prediction} Card={Card} />
          ) : liveResult ? (
            <PredictionView
              result={prediction}
              explanationFirst={page[0] === "explanation"}
            />
          ) : error ? (
            <section role="alert">
              Dashboard data is unavailable.{" "}
              <button onClick={() => retry(attempt + 1)}>Retry</button>
            </section>
          ) : !data ? (
            <p role="status">Loading sample data…</p>
          ) : page[0] === "overview" ? (
            <Overview d={data} />
          ) : (
            <HistoricalTrends data={data} Card={Card} Bars={Bars} />
          )}
        </article>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
