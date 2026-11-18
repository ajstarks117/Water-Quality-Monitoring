# M10.3 dashboard performance

Measured on 18 November 2026 in the Windows development/demo environment using Microsoft Edge, Vite and the local Python backend with the frozen model and local dataset.

The Playwright test in app/frontend/tests/reliability.spec.js submits pH 7.8, DO 6.3 mg/L, BOD 2.6 mg/L and fecal coliform 4 MPN/100 mL ten times. Each cycle starts just before clicking Generate prediction, waits for a real HTTP 200 and the SHAP heading, opens Decision Support, verifies all four output sections, then waits two animation frames. Every cycle includes a new backend prediction, SHAP calculation and decision-support response.

| Cycle | Elapsed milliseconds |
| --- | --- |
| 1 | 179 |
| 2 | 149 |
| 3 | 148 |
| 4 | 149 |
| 5 | 171 |
| 6 | 170 |
| 7 | 160 |
| 8 | 157 |
| 9 | 140 |
| 10 | 168 |

Minimum **140 ms**, median **158.5 ms**, maximum **179 ms**.

TC-10.3-04 passed. All ten cycles were below the approximately 2–3-second dashboard target. These are local measurements after server startup and earlier tests warmed the model and history cache. They exclude startup, manual typing and external map-tile loading. They cover one repeated sample and one user, and do not establish remote-network or concurrent-load performance.

Reproduce with the project Python environment active, from app/frontend:

```powershell
npm.cmd test -- tests/reliability.spec.js --workers=1
```

The test prints all ten timings and asserts median latency below 3,000 ms. Its recovery test additionally verifies a real server-side configuration failure produces a friendly UI error and a subsequent normal prediction succeeds.
