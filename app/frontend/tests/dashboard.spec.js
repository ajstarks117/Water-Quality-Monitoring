import { test, expect } from "@playwright/test";
const sections = [
  ["Overview", "Water-station distribution"],
  ["Water Quality Assessment", "Measurement inputs"],
  ["Prediction & Explainability", "No assessment yet"],
  ["Model Evaluation", "Candidate models"],
  ["Parameter Analysis", "No assessment yet"],
  ["Decision Support", "No assessment yet"],
  ["Historical Trends", "Measurements over time"],
];
for (const width of [390, 1440]) {
  test("all seven sections render at width " + width, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    for (const [label, heading] of sections) {
      await page
        .getByRole("navigation")
        .getByRole("link", { name: label, exact: true })
        .click();
      await expect(
        page.getByRole("heading", { name: heading, exact: true }),
      ).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
    expect(errors).toEqual([]);
  });
}
test("assessment validates missing, impossible and valid samples", async ({
  page,
  request,
}) => {
  const fields = await (await request.get("/api/assessment/schema")).json();
  await page.goto("/#input");
  const inputs = page.getByRole("spinbutton");
  await expect(inputs).toHaveCount(4);
  const submit = page.getByRole("button", { name: "Validate and preprocess" });
  await submit.click();
  await expect(page.getByRole("alert")).toHaveText(
    "Please provide all required parameters.",
  );
  for (const [i, field] of fields.entries()) {
    const input = inputs.nth(i);
    await expect(input).toHaveAccessibleName(
      new RegExp(field.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
    );
    await expect(input).toHaveAttribute("min", String(field.min));
    await expect(input).toHaveAttribute("max", String(field.max));
    await input.fill(String([20, 4.8, 6.2, 320][i]));
  }
  await submit.click();
  await expect(page.getByRole("alert")).toHaveText(
    "Input value is outside the allowed data range. Please verify the measurement.",
  );
  await inputs.first().fill("7.2");
  await submit.click();
  await expect(page.getByRole("status")).toHaveText(
    "Preprocessing complete. 4 measurements produced 5 model features.",
  );
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(inputs.first()).toHaveValue("");
  await expect(page.getByRole("status")).not.toContainText(
    "Preprocessing complete",
  );
});

for (const uncertain of [false, true]) {
  test(
    uncertain
      ? "ambiguous response retains class and warning"
      : "live class and SHAP render within three seconds",
    async ({ page }) => {
      if (uncertain) {
        await page.route("**/api/assessment/predict", async (route) => {
          const real = await route.fetch();
          const data = await real.json();
          await route.fulfill({
            json: {
              ...data,
              model_probability: 0.55,
              uncertain: true,
              class_probabilities: {
                ...data.class_probabilities,
                Unsuitable: 0.55,
                Poor: 0.45,
              },
            },
          });
        });
      }
      await page.goto("/#input");
      const fields = page.getByRole("spinbutton");
      for (const [i, value] of [7.2, 4.8, 6.2, 320].entries())
        await fields.nth(i).fill(String(value));
      const start = Date.now();
      await page.getByRole("button", { name: "Generate prediction" }).click();
      await expect(
        page.getByRole("heading", { name: "Local SHAP contributions" }),
      ).toBeVisible();
      await expect(
        page.getByText("Unsuitable", { exact: true }).first(),
      ).toBeVisible();
      await expect(
        page.getByText("Model probability: " + (uncertain ? "55%" : "100%")),
      ).toBeVisible();
      if (uncertain)
        await expect(
          page.getByText("Prediction is uncertain.", { exact: false }),
        ).toBeVisible();
      else {
        await expect(page.getByText("+0.7124", { exact: true })).toBeVisible();
        const elapsed = Date.now() - start;
        console.log("Live submission to rendered result (ms)", elapsed);
        expect(elapsed).toBeLessThan(3000);
      }
      const bar = page.locator(".feature-bar").last();
      await bar.click();
      await expect(bar).toHaveAttribute("aria-pressed", "true");
      await page.getByRole("button", { name: "Global explanation" }).click();
      await expect(
        page.getByRole("img", { name: /Global SHAP/ }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "Individual explanation" })
        .click();
      await page.getByRole("link", { name: "Edit measurements" }).click();
      await expect(fields.first()).toHaveValue("7.2");
      await fields.first().fill("7.4");
      await page
        .getByRole("navigation")
        .getByRole("link", { name: "Prediction & Explainability", exact: true })
        .click();
      await expect(
        page.getByRole("heading", { name: "No assessment yet" }),
      ).toBeVisible();
    },
  );
}
test("walk four assessments through all seven sections", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  const nav = (name) =>
    page
      .getByRole("navigation")
      .getByRole("link", { name, exact: true })
      .click();
  for (const [values, expectedClass, priority] of [
    [[7, 14, 0, 1], "Good", "None"],
    [[7, 6, 3, 200], "Very Poor", "None"],
    [[7, 3, 8, 1500], "Unsuitable", "High"],
    [[6.5, 4.9, 3.1, 250], null, "High"],
  ]) {
    await nav("Overview");
    await expect(
      page.getByRole("heading", { name: "Water-station distribution" }),
    ).toBeVisible();
    await nav("Water Quality Assessment");
    for (const [i, v] of values.entries())
      await page.getByRole("spinbutton").nth(i).fill(String(v));
    await page.getByRole("button", { name: "Generate prediction" }).click();
    await expect(
      page.getByRole("heading", { name: "Predicted WQI class" }),
    ).toBeVisible();
    if (expectedClass)
      await expect(page.locator("article strong").first()).toHaveText(
        expectedClass,
      );
    await nav("Prediction & Explainability");
    await expect(
      page.getByRole("heading", { name: "Local SHAP contributions" }),
    ).toBeVisible();
    await nav("Parameter Analysis");
    await expect(page.locator("tbody tr")).toHaveCount(5);
    await page.getByLabel("Distribution state").selectOption("Uttar Pradesh");
    await expect(page.locator(".histogram i")).toHaveCount(7);
    await expect(
      page.getByRole("columnheader", { name: "Interpretation", exact: true }),
    ).toBeVisible();
    await nav("Decision Support");
    for (const name of [
      "Assessment context",
      "Contributing parameters",
      "Parameter interpretation",
      "Recommended actions",
    ])
      await expect(
        page.getByRole("heading", { name, exact: true }),
      ).toBeVisible();
    await expect(
      page.getByText("Priority · " + priority, { exact: true }),
    ).toBeVisible();
    console.log(
      "Walkthrough",
      values.join(", "),
      await page.locator(".decision-grid .card").first().locator(":scope > strong").innerText(),
      priority,
      await page.locator(".decision-grid .card").nth(2).innerText(),
    );
    await nav("Historical Trends");
    await page.getByLabel("State", { exact: true }).selectOption("Maharashtra");
    await page
      .getByLabel("Station", { exact: true })
      .selectOption({ index: 1 });
    await expect(
      page.getByRole("heading", { name: "Monthly class distribution" }),
    ).toBeVisible();
    await expect(page.getByText(/^[1-9]\d* dated observations$/)).toBeVisible();
  }
  const through = page.getByLabel("Sampling through");
  const last = await through.getAttribute("max");
  await through.fill(last);
  await page.getByRole("button", { name: "Reset filters" }).click();
  await expect(through).toHaveValue("");
  await expect(page.getByLabel("State", { exact: true })).toHaveValue("");
  await page.locator(".period summary").first().click();
  await expect(page.locator(".period[open] .row")).toHaveCount(5);
  expect(errors).toEqual([]);
});
test("overview filters, station selection and evaluation use real data", async ({
  page,
  request,
}) => {
  const data = await (await request.get("/api/dashboard")).json();
  await page.goto("/");
  const state = "Uttar Pradesh";
  await page.getByLabel("State", { exact: true }).selectOption(state);
  const records = data.records.filter((r) => r.state === state);
  await expect(page.locator(".metrics strong").first()).toHaveText(
    String(records.length),
  );
  const stations = data.stations.filter((s) => s.State === state);
  await page
    .getByRole("combobox", { name: "Station", exact: true })
    .selectOption("1");
  await expect(page.locator(".bottom section").first()).toContainText(
    stations[1].Station,
  );
  const firstMarker = page.locator(".station-map .leaflet-interactive").first();
  await firstMarker.dispatchEvent("click");
  await expect(
    page
      .locator(".station-map .leaflet-popup-content")
      .filter({ hasText: stations[0].Station }),
  ).toContainText(stations[0].Station);
  await expect(page.locator(".bottom section").first()).toContainText(
    stations[0].Station,
  );
  await page
    .getByRole("link", { name: "Model Evaluation", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Candidate models" }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "0.9067", exact: true }),
  ).toBeVisible();
});
