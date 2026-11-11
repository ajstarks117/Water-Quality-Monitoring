import { expect, test } from "@playwright/test";

const endpoint = "/api/assessment/predict";
const names = ["Potential of Hydrogen (pH)", "Dissolved oxygen (mg/L)",
  "Biochemical Oxygen Demand (mg/L)", "Fecal Coliform (MPN/100mL)"];

async function enter(page, values) {
  await page.goto("/#input");
  for (const [index, value] of values.entries()) {
    await page.getByRole("spinbutton").nth(index).fill(String(value));
  }
}

async function predict(page, values) {
  await enter(page, values);
  const response = page.waitForResponse(r => r.url().endsWith(endpoint));
  await page.getByRole("button", { name: "Generate prediction" }).click();
  const result = await response;
  expect(result.status()).toBe(200);
  await expect(page.getByRole("heading", { name: "Local SHAP contributions" })).toBeVisible();
  return result.json();
}

for (const [scenario, values, message] of [
  ["missing", [7, 7, 2, ""], "Please provide all required parameters."],
  ["impossible", [20, 7, 2, 100], "Input value is outside the allowed data range. Please verify the measurement."],
]) {
  test(scenario + " measurement blocks prediction in browser and API", async ({ page, request }) => {
    const submissions = [];
    page.on("request", r => { if (r.url().endsWith(endpoint)) submissions.push(r); });
    await enter(page, values);
    await page.getByRole("button", { name: "Generate prediction" }).click();
    await expect(page.getByRole("alert")).toHaveText(message);
    expect(submissions).toHaveLength(0);
    const payload = Object.fromEntries(names.map((name, i) => [name, values[i]]));
    const response = await request.post(endpoint, { data: payload });
    expect(response.status()).toBe(422);
    expect(await response.json()).toEqual({ error: message });
  });
}

test("one severe BOD reading remains visible in decision support", async ({ page }) => {
  const result = await predict(page, [7, 7, 90, 100]);
  await expect(page.getByText("Model and parameter assessments differ.")).toHaveCount(0);
  expect(result.parameters.filter(p => p.abnormal).map(p => p.name)).toEqual([names[2]]);
  await page.getByRole("navigation").getByRole("link", { name: "Decision Support" }).click();
  await expect(page.getByRole("heading", { name: names[2] })).toBeVisible();
  await expect(page.locator(".action").filter({ hasText: names[2] }).getByText("High", { exact: true }).first()).toBeVisible();
});

test("real borderline holdout observation has a visible sensitivity warning", async ({ page }) => {
  const result = await predict(page, [8.44, 8.7, 2.3, 800]);
  expect(result.predicted_class).toBe("Poor");
  expect(result.model_probability).toBe(1);
  await expect(page.getByRole("status").filter({ hasText: "sensitive to small measurement changes" }))
    .toContainText("8.45 gives Very Poor");
  const neighbor = await predict(page, [8.45, 8.7, 2.3, 800]);
  expect(neighbor.predicted_class).toBe("Very Poor");
});

test("accepted pH beyond training support is flagged", async ({ page }) => {
  await predict(page, [10, 7, 2, 100]);
  await expect(page.getByRole("status").filter({ hasText: "outside the observed training range" }))
    .toHaveText("Potential of Hydrogen (pH) is outside the observed training range (5.45–9.2).");
});

test("Krishna sample preserves the ML class and the independent rule assessment", async ({ page }) => {
  const result = await predict(page, [7.8, 6.3, 2.6, 4]);
  expect(result.predicted_class).toBe("Very Poor");
  expect(result.class_probabilities["Very Poor"]).toBe(1);
  expect(result.parameters.every(p => p.abnormal === false)).toBe(true);
  expect(result.base_value + result.contributions.reduce((sum, f) => sum + f.contribution, 0)).toBeCloseTo(1, 6);
  await expect(page.getByText("Very Poor", { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/Even 100% does not guarantee/)).toBeVisible();
  const notice = page.getByText(/All four measurements are within their configured limits/);
  await expect(notice).toBeVisible();
  await page.getByRole("navigation").getByRole("link", { name: "Decision Support" }).click();
  await expect(notice).toBeVisible();
});
