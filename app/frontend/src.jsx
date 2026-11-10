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
    "Follow the sample station’s dissolved oxygen over time.",
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
function Measures({ d }) {
  return (
    <dl className="measures">
      {d.parameters.map((p) => (
        <div key={p.name}>
          <dt>{p.label}</dt>
          <dd>{p.value}</dd>
          <small>{p.unit}</small>
        </div>
      ))}
    </dl>
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
  const positions = [
    [36, 60],
    [16, 24],
    [28, 38],
    [42, 35],
    [48, 68],
    [62, 47],
    [75, 29],
    [84, 74],
  ];
  return (
    <>
      <div className="metrics">
        {[
          ["Total valid observations", total],
          ["Unique stations", d.stations.length],
          ["Parameters analyzed", d.parameters.length],
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
          note="Illustrative stations · schematic locations"
        >
          <div className="map">
            <svg
              viewBox="0 0 800 260"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <path
                className="land"
                d="M0 82L110 55 220 100 332 75 450 95 570 18 800 55M110 55L82 0M220 100L166 260M450 95L385 260M570 18L635 260"
              />
              <path d="M-10 0C62 84 110 41 177 111S310 214 390 152 510 35 625 86 735 50 820 82M390 152Q530 98 590 179T666 270" />
            </svg>
            {d.stations.map((s, i) => (
              <span
                title={s}
                className="dot"
                key={s}
                style={{
                  left: positions[i][0] + "%",
                  top: positions[i][1] + "%",
                }}
              />
            ))}
            <div className="mapLabel">
              <b>{d.station}</b>
              <small>{d.state}</small>
            </div>
          </div>
        </Card>
        <Card title="Class distribution" note="Illustrative WQI classes">
          <div
            className="ring"
            style={{ background: "conic-gradient(" + stops.join(",") + ")" }}
          >
            <b>
              {total}
              <small>demo records</small>
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
        <Card title="Selected measurements" note={d.station}>
          <Measures d={d} />
        </Card>
        <Card title="Data source">
          <p>Illustrative surface-water records for {d.state}.</p>
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
function Detail({ id, d }) {
  switch (id) {
    case "parameters":
      return (
        <Card title="Measurement review">
          {d.parameters.map((p) => (
            <div className="analysis" key={p.name}>
              <b>{p.name}</b>
              <span>
                {p.value} {p.unit}
              </span>
              <small>{p.status}</small>
            </div>
          ))}
        </Card>
      );
    case "decisions":
      return (
        <Card
          title="Recommended actions"
          note={"Overall priority · " + d.decision.overall_priority}
        >
          {d.decision.recommended_actions.action_items.map((a) => (
            <div className="action" key={a.recommendation}>
              <mark>{a.priority}</mark>
              {a.recommendation}
            </div>
          ))}
        </Card>
      );
    case "trends":
      return (
        <Card
          title="Dissolved oxygen"
          note={d.station + " · monthly measurements (mg/L)"}
        >
          <Bars
            max={Math.max(...d.trends.map((t) => t.value))}
            rows={d.trends.map((t) => [t.month, t.value, t.value + " mg/L"])}
          />
        </Card>
      );
  }
}
function currentPage() {
  return pages.find(([id]) => "#" + id === location.hash) || pages[0];
}
function App() {
  const [page, setPage] = useState(currentPage);
  const [prediction, setPrediction] = useState(null);
  const liveResult = ["prediction", "explanation"].includes(page[0]);
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
                : data?.mode || "Demo"}
          </span>
          {!liveResult && page[0] !== "input" && <b>{data?.state}</b>}
        </header>
        <article>
          <h1>{page[1]}</h1>
          <p className="intro">{page[3]}</p>
          <div hidden={page[0] !== "input"}>
            <Assessment onPrediction={setPrediction} />
          </div>
          {page[0] === "input" ? null : liveResult ? (
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
            <Detail id={page[0]} d={data} />
          )}
        </article>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
