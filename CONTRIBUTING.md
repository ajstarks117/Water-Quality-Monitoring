# Contributing & Collaboration Guidelines

Welcome to the **AI-Driven Water Quality Monitoring & Potability Prediction** project repository. 

To maintain velocity during this 3-week crunch, all 4 team members adhere to the streamlined branching, review, and interface contract rules described below.

---

## 👥 Track Ownership & Governance

| Track | Scope / Domain | Track Lead | Branch Name |
|---|---|---|---|
| **Track A** | Data Ingestion, EDA, Preprocessing & Cleaned Data Contract | Track A Owner | `feature/data-pipeline` |
| **Track B** | ML Model Training, Baselines, Hyperparameter Tuning & Evaluation | Track B Owner | `feature/ml-models` |
| **Track C** | Explainable AI (SHAP), Feature Importance & Decision Support Engine | Track C Owner | `feature/explainability-decision-support` |
| **Track D** | Streamlit Web Application, Real-Time Inference UI & System Integration | Track D Owner | `feature/dashboard` |

### Tie-Breaker Policy
For cross-track technical disputes (e.g. data schema modifications, package upgrades, or deployment targets), the **Project Lead / Track A Owner** acts as the definitive tie-breaker after a short timeboxed discussion (maximum 15 minutes).

---

## 🌿 Branching Model

1. **Default Branch (`main`)**:
   - `main` represents the stable, integrated baseline.
   - Every push/merge to `main` must have passing tests.
2. **Dedicated Feature Branches**:
   - Each track works primarily on its assigned branch:
     - `feature/data-pipeline`
     - `feature/ml-models`
     - `feature/explainability-decision-support`
     - `feature/dashboard`
   - Commit locally as frequently as you like.
   - Sync with `main` regularly (`git pull origin main` or `git rebase main`) to avoid stale drifts.
3. **Pull Requests (PR) & Merging**:
   - Open a PR from your feature branch into `main` once a milestone passes all its designated tests.
   - PRs must reference the Milestone ID (e.g., `M1.2: Cleaned Data Pipeline`).

---

## ⏱️ Code Review Rules (2-Hour SLA)

- **Standard Rule**: Each PR requires **1 approving review** from a teammate before merging.
- **Crunch SLA Fallback**: If no teammate responds within **2 hours during working hours**, the author is authorized to self-merge, provided all automated tests (`pytest tests/`) pass locally and on GitHub.
- After self-merging, the author must post a brief announcement in the team channel summarizing the merge.

---

## 🤝 Hard Interface Contracts

All 4 tracks commit to two rigid interface agreements detailed in [`docs/interface_contracts.md`](docs/interface_contracts.md):

1. **Cleaned Data Contract (M1.2 Freeze)**:
   - Fixed schema, column names, dtypes, and target column (`Potability`).
   - Saved at `data/processed/` and configured via `config/config.yaml`.
   - Cannot be modified without team consensus.
2. **Model Artifact Contract (M6.3 Freeze)**:
   - Serialized pipeline stored at `models/best_model.pkl`.
   - Exposes `.predict(df)` and `.predict_proba(df)` expecting the exact frozen feature columns.
   - Tracks C and D develop against mock predictors until M6.3 is frozen.

---

## 📋 Definition of Done for Milestones

Before opening a PR or pushing to `main`:
1. All milestone test cases pass (`pytest tests/ -v`).
2. Code follows PEP 8 standards and imports without circular dependencies.
3. No secrets (`.env`), raw datasets (`data/raw/`), or unwanted binary checkpoints are tracked in git.
4. The "Known Issues / Blockers" register is updated if any risks remain open.
