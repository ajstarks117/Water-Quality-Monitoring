import { test, expect } from "@playwright/test";
const sections = [
  ["Overview", "Water-station distribution"],
  ["Water Quality Assessment", "Sample inputs"],
  ["Prediction", "Class probabilities"],
  ["Explainability", "Local SHAP contributions"],
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
test("Python fixture supplies the displayed input and prediction", async ({
  page,
  request,
}) => {
  const response = await request.get("/api/dashboard");
  expect(response.ok()).toBe(true);
  const data = await response.json();
  await page.goto("/#input");
  await expect(page.getByRole("textbox")).toHaveCount(data.parameters.length);
  for (const p of data.parameters)
    await expect(page.getByLabel(p.name)).toHaveValue(String(p.value));
  await page.getByRole("link", { name: "View example prediction" }).click();
  await expect(page.locator("strong")).toHaveText(
    data.prediction.predicted_class,
  );
});
