import { test, expect } from "@playwright/test";
const sections = [
  ["Overview", "Water-station distribution"],
  ["Water Quality Assessment", "Measurement inputs"],
  ["Prediction", "No assessment yet"],
  ["Explainability", "No assessment yet"],
  ["Parameter Analysis", "Measurement review"],
  ["Decision Support", "Recommended actions"],
  ["Historical Trends", "Dissolved oxygen"],
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
      await page.getByRole("link", { name: "Edit measurements" }).click();
      await expect(fields.first()).toHaveValue("7.2");
      await fields.first().fill("7.4");
      await page
        .getByRole("navigation")
        .getByRole("link", { name: "Prediction", exact: true })
        .click();
      await expect(
        page.getByRole("heading", { name: "No assessment yet" }),
      ).toBeVisible();
    },
  );
}
