# AquaInsight dashboard

From the repository root, run `python -m app.api` in the project Python environment. In a second terminal, change to `app/frontend`, run `npm ci` and `npm run dev`, and open the local URL printed by Vite. Use Node 22.12 or later. Vite forwards `/api` requests to the local Python server on port 18741.

The assessment accepts the four frozen raw measurements. It uses the inclusive plausible ranges in the data dictionary so valid measurements outside the historical sample can be entered. The Python API validates the sample and transforms it with the preprocessing steps from `models/best_model.pkl`. Prediction, explanation, and other dashboard sections still display illustrative data.

Run `npm run build` for a production build. To verify all sections in Chromium, run `npm run test:e2e` with the project Python interpreter on PATH, after installing the Playwright Chromium browser with `npx playwright install chromium`. On Windows the test configuration uses installed Microsoft Edge.
