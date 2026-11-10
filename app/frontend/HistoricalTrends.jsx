import React, { useState } from "react";

export function HistoricalTrends({ data, Card, Bars }) {
  const [state, setState] = useState("");
  const [station, setStation] = useState("");
  const [feature, setFeature] = useState(data.features[1]);
  const stations = data.stations.filter((s) => !state || s.State === state);
  const records = data.records.filter(
    (r) =>
      r.month &&
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
          <select aria-label="State"
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
          <select aria-label="Station" value={station} onChange={(e) => setStation(e.target.value)}>
            <option value="">All stations</option>
            {[...new Set(stations.map((s) => s.Station))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Parameter
          <select aria-label="Parameter" value={feature} onChange={(e) => setFeature(e.target.value)}>
            {data.features.map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </label>
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
            <div key={month}>
              <h3>{month}</h3>
              <Bars
                rows={Object.entries(m.classes).map(([c, n]) => [
                  c,
                  n / m.total,
                  String(n),
                ])}
              />
            </div>
          ))}
        </Card>
      </div>
    </>
  );
}
