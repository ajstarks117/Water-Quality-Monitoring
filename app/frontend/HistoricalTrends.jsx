import React, { useState } from "react";

export function HistoricalTrends({ data, Card, Bars }) {
  const [state, setState] = useState("");
  const [station, setStation] = useState("");
  const [feature, setFeature] = useState(data.features[1]);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const dayPrecision = data.records.some((r) => r.date);
  const dateOf = (r) => (dayPrecision ? r.date : r.month);
  const available = data.records.map(dateOf).filter(Boolean).sort();
  const classes = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"];
  const colors = ["#559688", "#0d7974", "#d7a34a", "#ca7840", "#5266a5"];
  const stations = data.stations.filter((s) => !state || s.State === state);
  const records = data.records.filter(
    (r) =>
      dateOf(r) &&
      (!from || dateOf(r) >= from) &&
      (!to || dateOf(r) <= to) &&
      (!state || r.state === state) &&
      (!station || r.station === station),
  );
  const months = new Map();
  for (const r of records) {
    if (!months.has(r.month))
      months.set(r.month, { sum: 0, count: 0, classes: {}, total: 0 });
    const m = months.get(r.month);
    m.total++;
    m.classes[r.category] = (m.classes[r.category] || 0) + 1;
    if (r[feature] !== null) {
      m.sum += r[feature];
      m.count++;
    }
  }
  const periods = [...months.entries()].sort(([a], [b]) => a.localeCompare(b));
  const means = periods
    .filter(([, m]) => m.count)
    .map(([month, m]) => [
      month,
      m.sum / m.count,
      (m.sum / m.count).toLocaleString(undefined, {
        maximumFractionDigits: 2,
      }) +
        " · n=" +
        m.count,
    ]);
  return (
    <>
      <section className="card filters">
        <label>
          State
          <select
            aria-label="State"
            value={state}
            onChange={(e) => {
              setState(e.target.value);
              setStation("");
            }}
          >
            <option value="">All states</option>
            {[...new Set(data.stations.map((s) => s.State))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Station
          <select
            aria-label="Station"
            value={station}
            onChange={(e) => setStation(e.target.value)}
          >
            <option value="">All stations</option>
            {[...new Set(stations.map((s) => s.Station))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Parameter
          <select
            aria-label="Parameter"
            value={feature}
            onChange={(e) => setFeature(e.target.value)}
          >
            {data.features.map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </label>
        <div className="date-controls">
          <label>
            From
            <input
              type={dayPrecision ? "date" : "month"}
              aria-label="Sampling from"
              min={available[0]}
              max={to || available.at(-1)}
              value={from}
              onChange={(e) => setFrom(e.target.value)}
            />
          </label>
          <label>
            Through
            <input
              type={dayPrecision ? "date" : "month"}
              aria-label="Sampling through"
              min={from || available[0]}
              max={available.at(-1)}
              value={to}
              onChange={(e) => setTo(e.target.value)}
            />
          </label>
          <button
            type="button"
            onClick={() => {
              setState("");
              setStation("");
              setFeature(data.features[1]);
              setFrom("");
              setTo("");
            }}
          >
            Reset filters
          </button>
        </div>
        {from && to && from > to && (
          <p role="alert">Choose an end date on or after the start date.</p>
        )}
      </section>
      <div className="grid">
        <Card
          title="Measurements over time"
          note={feature + " · monthly mean of available measurements"}
        >
          {means.length ? (
            <Bars rows={means} max={Math.max(...means.map((r) => r[1])) || 1} />
          ) : (
            <p>No dated measurements in this selection.</p>
          )}
        </Card>
        <Card
          title="Monthly class distribution"
          note={records.length + " dated observations"}
        >
          {periods.map(([month, m]) => (
            <details className="period" key={month}>
              <summary>
                {month}
                <span className="class-stack">
                  {classes.map((c, i) => (
                    <i
                      key={c}
                      title={c + " · " + (m.classes[c] || 0)}
                      style={{
                        background: colors[i],
                        width: (100 * (m.classes[c] || 0)) / m.total + "%",
                      }}
                    />
                  ))}
                </span>
              </summary>
              {classes.map((c) => (
                <div className="row" key={c}>
                  {c}
                  <b>{m.classes[c] || 0}</b>
                </div>
              ))}
            </details>
          ))}
          {periods.length > 0 && (
            <div className="class-key">
              {classes.map((c, i) => (
                <span key={c}>
                  <i style={{ background: colors[i] }} />
                  {c}
                </span>
              ))}
            </div>
          )}
          {!periods.length && <p>No dated classes in this selection.</p>}
        </Card>
      </div>
      <div className="grid prediction-summary">
        <Card title="Sampling coverage">
          <p>
            {records.length} observations across {periods.length} sampled months
          </p>
          <p>
            {records.filter((r) => r[feature] == null).length} missing parameter
            values
          </p>
        </Card>
        <Card
          title="Sampling period"
          note="Monthly means use the observations available in each period."
        >
          <p>
            {periods.length
              ? periods[0][0] + " to " + periods.at(-1)[0]
              : "No observations for these filters."}
          </p>
        </Card>
      </div>
    </>
  );
}
