import React, { useMemo, useState } from "react";
import StationMap from "./StationMap";

const classes = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"];
const colors = ["#559688", "#0d7974", "#d7a34a", "#ca7840", "#5266a5"];
export default function Overview({ data, state, Card }) {
  const [choice, select] = useState(null);
  const stations = useMemo(
    () => data.stations.filter((s) => !state || s.State === state),
    [data, state],
  );
  const records = data.records.filter((r) => !state || r.state === state);
  const selected = stations.includes(choice) ? choice : stations[0];
  const samples = records.filter(
    (r) => r.station === selected?.Station && r.state === selected?.State,
  );
  const sample = samples.reduce(
    (last, r) =>
      !last || (r.date || r.month || "") >= (last.date || last.month || "")
        ? r
        : last,
    null,
  );
  const counts = classes.map(
    (c) => records.filter((r) => r.category === c).length,
  );
  let angle = 0;
  const stops = counts.map((n, i) => {
    const start = angle;
    angle += (360 * n) / (records.length || 1);
    return colors[i] + " " + start + "deg " + angle + "deg";
  });
  const missing = records.reduce(
    (n, r) => n + data.features.filter((f) => r[f] == null).length,
    0,
  );
  return (
    <>
      <div className="metrics">
        {[
          ["Total valid observations", records.length],
          ["Unique stations", stations.length],
          ["Parameters analyzed", data.features.length],
          ["Supported target classes", classes.length],
        ].map(([label, n]) => (
          <section key={label} className="card">
            <small>{label}</small>
            <strong>{n}</strong>
          </section>
        ))}
      </div>
      <div className="grid">
        <Card
          wide
          title="Water-station distribution"
          note="Select a station to inspect its sampling record."
        >
          <StationMap
            stations={stations}
            selected={selected}
            onSelect={select}
          />
          <label className="station-picker">
            Station
            <select
              aria-label="Station"
              value={stations.indexOf(selected)}
              onChange={(e) => select(stations[Number(e.target.value)])}
            >
              {stations.map((s, i) => (
                <option key={s.State + s.Station} value={i}>
                  {s.Station} · {s.State}
                </option>
              ))}
            </select>
          </label>
        </Card>
        <Card title="Class distribution" note="Historical WQI labels">
          <div
            className="ring"
            style={{ background: "conic-gradient(" + stops.join(",") + ")" }}
          >
            <b>
              {records.length}
              <small>records</small>
            </b>
          </div>
          {classes.map((c, i) => (
            <div className="legend" key={c}>
              <i style={{ background: colors[i] }} />
              {c}
              <b>
                {counts[i]} ·{" "}
                {((100 * counts[i]) / (records.length || 1)).toFixed(1)}%
              </b>
            </div>
          ))}
        </Card>
      </div>
      <div className="bottom">
        <Card title="Selected measurements" note={selected?.Station}>
          {sample && (
            <>
              <small>
                {sample.date || sample.month} · {sample.category} ·{" "}
                {samples.length} station records
              </small>
              <dl className="measures">
                {data.features.map((f, i) => (
                  <div key={f}>
                    <dt title={f}>{["pH", "DO", "BOD", "FC"][i]}</dt>
                    <dd>{sample[f] ?? "—"}</dd>
                    <small>{["pH", "mg/L", "mg/L", "MPN/100mL"][i]}</small>
                  </div>
                ))}
              </dl>
            </>
          )}
        </Card>
        <Card title="Data quality" note="Missing raw measurements">
          <strong>
            {(
              (100 * missing) /
              (records.length * data.features.length || 1)
            ).toFixed(1)}
            %
          </strong>
          <small>{missing} missing values in this selection</small>
        </Card>
        <Card title="Model readiness">
          <p>Frozen decision tree</p>
          <a href="#evaluation">View model evaluation →</a>
        </Card>
      </div>
    </>
  );
}
