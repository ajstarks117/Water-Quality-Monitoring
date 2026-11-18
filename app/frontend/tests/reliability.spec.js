import { test, expect } from "@playwright/test";

async function fillSample(page) {
  await page.goto("/#input");
  for (const [index, value] of [7.8, 6.3, 2.6, 4].entries())
    await page.getByRole("spinbutton").nth(index).fill(String(value));
}

test("real backend configuration failure displays a friendly error and permits recovery", async ({ page }) => {
  await page.route("**/api/assessment/predict", async route => {
    const response = await route.fetch({ url: "http://127.0.0.1:18742/api/assessment/predict" });
    expect(response.status()).toBe(500);
    expect(await response.json()).toEqual({
      error: "Assessment processing is unavailable. Please try again.",
    });
    await route.fulfill({ response });
  });
  await fillSample(page);
  await page.getByRole("button", { name: "Generate prediction" }).click();
  await expect(page.getByRole("alert")).toHaveText("Assessment processing is unavailable. Please try again.");
  await expect(page.locator("body")).not.toContainText("PRIVATE_SENTINEL");
  await expect(page.locator("body")).not.toContainText("Traceback");
  await expect(page.locator("body")).not.toContainText("C:/private");
  await page.unroute("**/api/assessment/predict");
  await page.getByRole("button", { name: "Generate prediction" }).click();
  await expect(page.getByRole("heading", { name: "Local SHAP contributions" })).toBeVisible();
});

test("ten real prediction to decision render cycles", async ({ page }) => {
  const timings = [];
  for (let cycle = 0; cycle < 10; cycle++) {
    await fillSample(page);
    const start = performance.now();
    const responsePromise = page.waitForResponse(response =>
      response.url().endsWith("/api/assessment/predict") && response.request().method() === "POST");
    await page.getByRole("button", { name: "Generate prediction" }).click();
    expect((await responsePromise).status()).toBe(200);
    await expect(page.getByRole("heading", { name: "Local SHAP contributions" })).toBeVisible();
    await page.getByRole("navigation").getByRole("link", { name: "Decision Support", exact: true }).click();
    for (const name of ["Assessment context", "Contributing parameters", "Parameter interpretation", "Recommended actions"])
      await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    timings.push(Math.round(performance.now() - start));
  }
  const sorted = [...timings].sort((a, b) => a - b);
  const median = (sorted[4] + sorted[5]) / 2;
  console.log("M10.3 prediction-to-decision milliseconds", JSON.stringify({
    cycles: timings, min: sorted[0], median, max: sorted[9],
  }));
  expect(median).toBeLessThan(3000);
});
