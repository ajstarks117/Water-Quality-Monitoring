import React from "react";
import experiment from "../../reports/m61_imbalance_experiment.json";
import scores from "../../reports/model_comparison.csv?raw";
import matrix from "../../reports/figures/confusion_matrices/confusion_matrix_decision_tree.png";

export default function Evaluation({ Card }) {
  const [header, ...records] = scores
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(","));
  const selected = records.find((row) => row[0] === "Decision Tree");
  return (
    <>
      <Card title="Selected model">
        <strong>Decision Tree</strong>
        <small>Selected by macro F1 on the project evaluation split.</small>
      </Card>
      <Card title="Candidate models">
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {header.map((h) => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((row) => (
                <tr key={row[0]}>
                  {row.map((v, i) => (
                    <td key={header[i]}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="evaluation-panels">
        <Card title="Confusion matrix">
          <img
            className="evaluation-matrix"
            src={matrix}
            alt="Decision tree confusion matrix by actual and predicted WQI class"
          />
        </Card>
        <Card title="Precision / recall / F1">
          {[2, 3, 4].map((i) => (
            <p key={i}>
              {header[i]} · {selected[i]}
            </p>
          ))}
        </Card>
        <Card
          title="Class-wise performance"
          note="Recall on the held-out split"
        >
          {Object.entries(
            experiment["Baseline (class_weight=balanced)"]["Decision Tree"]
              .per_class_recall,
          ).map(([name, value]) => (
            <p key={name}>
              {name} · {(100 * value).toFixed(1)}%
            </p>
          ))}
        </Card>
      </div>
    </>
  );
}
