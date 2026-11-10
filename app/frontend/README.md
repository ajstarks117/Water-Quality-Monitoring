# AquaInsight dashboard

From the repository root, run `python -m app.api` in the project Python environment. In a second terminal, change to `app/frontend`, run `npm ci` and `npm run dev`, and open the local URL printed by Vite. Use Node 22.12 or later. Vite forwards `/api/dashboard` to the local Python server on port 18741.

M9.1 provides seven navigable sections with fixed examples from `app/mock_data.py`. The assessment displays the inputs for the example prediction. Named integration points in that file identify the later preprocessing, prediction, explanation and decision-support work.

Run `npm run build` for a production build. To verify all sections in Chromium, run `npm run test:e2e` with the project Python interpreter on PATH, after installing the Playwright Chromium browser with `npx playwright install chromium`. On Windows the test configuration uses installed Microsoft Edge.
