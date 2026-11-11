import React, { useState } from "react";

export default function ParameterDistribution({ data, sample, Card }) {
  const [feature, setFeature] = useState(data.features[2]);
  const [state, setState] = useState("");
  const [station, setStation] = useState("");
  const records = data.records.filter(
    (r) => (!state || r.state === state) && (!station || r.station === station),
  );
  const values = records
    .map((r) => r[feature])
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  const low = values[0],
    high = values.at(-1);
  const bins = Array(low === high ? 1 : 7).fill(0);
  const width = (high - low) / bins.length || 1;
  const index = (v) => Math.min(bins.length - 1, Math.floor((v - low) / width));
  for (const v of values) bins[index(v)]++;
  const observed = sample[feature];
  const fmt = (n) =>
    n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  const median = values.length
    ? (values[Math.floor((values.length - 1) / 2)] +
        values[Math.floor(values.length / 2)]) /
      2
    : null;
  return (
    <>
      <section className="card filters">
        <label>
          Parameter
          <select
            aria-label="Distribution parameter"
            value={feature}
            onChange={(e) => setFeature(e.target.value)}
          >
            {data.features.map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </label>
        <label>
          State
          <select
            aria-label="Distribution state"
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
            aria-label="Distribution station"
            value={station}
            onChange={(e) => setStation(e.target.value)}
          >
            <option value="">All stations</option>
            {[
              ...new Set(
                data.stations
                  .filter((s) => !state || s.State === state)
                  .map((s) => s.Station),
              ),
            ].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </section>
      <div className="grid">
        <Card title="Parameter distribution" note={feature}>
          {values.length ? (
            <div className="histogram">
              {bins.map((count, i) => {
                const label =
                  fmt(low + i * width) +
                  "–" +
                  fmt(i === bins.length - 1 ? high : low + (i + 1) * width);
                return (
                  <div key={i} title={label + " · " + count + " observations"}>
                    <b>{count}</b>
                    <i
                      style={{
                        height: 4 + (100 * count) / Math.max(...bins),
                        background:
                          observed >= low &&
                          observed <= high &&
                          index(observed) === i
                            ? "#255edb"
                            : undefined,
                      }}
                    />
                    <small>{label}</small>
                  </div>
                );
              })}
            </div>
          ) : (
            <p>No measurements for this selection.</p>
          )}
          <small>Submitted measurement · {fmt(observed)}</small>
        </Card>
        <Card title="Distribution summary">
          {[
            ["Median", median],
            ["Minimum", low],
            ["Maximum", high],
            ["Missing measurements", records.length - values.length],
          ].map(([label, v]) => (
            <p className="row" key={label}>
              {label}
              <b>{fmt(v)}</b>
            </p>
          ))}
        </Card>
      </div>
    </>
  );
}
