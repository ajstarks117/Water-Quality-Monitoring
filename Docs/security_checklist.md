# M10.3 reliability and security checklist

Verified on 18 November 2026 in the local Windows demo environment. PR #6 was confirmed merged into main at b818b3c before this feature branch started. The running application is React/Vite with the existing Python standard-library HTTP backend in app/api.py.

| Case | Evidence | Outcome |
| --- | --- | --- |
| TC-10.3-01 | Source scan for eval/exec calls in src and app found zero matches. JSON parsing and explicit numeric validation handle assessment input. | Pass |
| TC-10.3-02 | An isolated test backend uses the production HTTP handler and prediction path with a threshold loader that raises an exception containing a private marker and path. Edge receives its real HTTP 500, renders the friendly message and recovers on retry against the normal backend. HTTP regression tests also cover history failures and serialization failures. | Pass |
| TC-10.3-03 | Both frozen files have the Windows ReadOnly attribute set. Opening either with Python r+b raises PermissionError. Model bytes match Git HEAD. Threshold content matches HEAD after Windows CRLF normalization. | Pass |
| TC-10.3-04 | Ten live prediction-through-decision-support browser cycles measured 140–179 ms, median 158.5 ms. | Pass |

## Validation and error handling

The existing form checks required fields and allowed numeric ranges before submission. Browser tests confirm missing and impossible measurements produce messages and zero prediction requests. The API independently rejects malformed JSON, empty and oversized bodies, non-object JSON, missing fields, additional fields, booleans, strings including code-like text, collections, NaN, infinity and out-of-range numbers. The new HTTP tests verify rejected requests never load the model.

The assessment payload cap is 8,192 bytes. This is a local demonstration server. Slow clients, concurrent load, authentication and public-network deployment are outside this checklist's verification.

Two gaps were corrected. API access logs previously included the request URL, so query strings could reach logs. Successful and failed application requests now log status codes without request URLs. Unexpected history-loading errors and response serialization failures now return generic JSON errors. Exception details and input bodies are omitted from those application error messages.

## Data and credential checks

- The dashboard accepts measurement fields only. Its historical response selects station, state, dates, coordinates, classes and measurements. It does not expose whole source rows, source files or local configuration. Assessments are held in React state, and this request path does not persist them.
- The labeled dataset header contains no personal identifier fields such as email, phone, person, birth date or Aadhaar. Station and geographic information are intentional environmental data. This is a schema and application-path review rather than a certification of every free-text cell.
- Tracked source/configuration/document text was scanned for private-key blocks, GitHub tokens, AWS access-key patterns and Mapbox token patterns. No matches were found. Raw/interim CSVs, local environment files and log files are absent from the tracked file list.
- The local ignored frontend environment file contains a public Mapbox token. The frontend intentionally sends it to Mapbox for map tiles, and Vite includes frontend environment values in browser assets. It is a browser-visible public token, not a backend secret. Its value was withheld from reports and commits. Provider account restrictions were outside this review.
- Local pipeline diagnostics and dependency warnings can contain filesystem paths in the server terminal. They were absent from the tested browser/API errors. The test exception marker and query sentinel were absent from both the HTTP error response and captured application logs.
- Existing project Office documents can contain author metadata, and team/project documents may identify contributors. They are repository documentation outside dashboard data routes. A claim that every stored project file is anonymous would be inaccurate. The user's modified playbook was excluded from this change.

## Read-only demo verification

The Windows attribute is local filesystem state and is not preserved by Git. On another Windows demo checkout, apply it from the repository root before starting the API:

```powershell
(Get-Item -LiteralPath models/best_model.pkl).IsReadOnly = $true
(Get-Item -LiteralPath config/thresholds.yaml).IsReadOnly = $true
```

This prevents accidental file writes through ordinary file opens. The owner can deliberately clear the attribute. The API binds to loopback, and its routes perform inference and data reads only.

The model SHA-256 is c1c42b4c625e1fda25852ab63c54e94c9caa842f17d67d022dfd8796e529c812.

## Regression results

- Full local Python suite with official data: 245 passed, zero failures, 1,359 warnings in 84.71 seconds.
- Playwright with installed Microsoft Edge and one worker: 15 passed in 26.4 seconds.
- Vite production build: passed.
- Existing scikit-learn artifact/runtime version warnings remain. The tests exercise the supplied frozen artifact in this environment.

Run `python -m pytest -q` at the project root with the virtual environment active. In app/frontend, run `npm run build` and `npm test -- --workers=1`. The browser configuration starts an additional isolated failure-test server on loopback port 18742. The test redirects only the failure scenario to that process. The performance test uses the normal backend on port 18741 with actual inference, SHAP and decision support.
