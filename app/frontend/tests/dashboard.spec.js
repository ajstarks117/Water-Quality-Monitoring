import { test, expect } from "@playwright/test";
const sections = [
  ["Overview", "Water-station distribution"],
  ["Water Quality Assessment", "Measurement inputs"],
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
test("assessment validates missing, impossible and valid samples", async ({page, request}) => {
  const fields = await (await request.get("/api/assessment/schema")).json();
  await page.goto("/#input");
  const inputs = page.getByRole("spinbutton");
  await expect(inputs).toHaveCount(4);
  const submit = page.getByRole("button", {name:"Validate and preprocess"});
  await submit.click();
  await expect(page.getByRole("alert")).toHaveText("Please provide all required parameters.");
  for (const [i, field] of fields.entries()) {
    const input = inputs.nth(i);
    await expect(input).toHaveAccessibleName(new RegExp(field.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    await expect(input).toHaveAttribute("min", String(field.min));
    await expect(input).toHaveAttribute("max", String(field.max));
    await input.fill(String([20, 4.8, 6.2, 320][i]));
  }
  await submit.click();
  await expect(page.getByRole("alert")).toHaveText("Input value is outside the allowed data range. Please verify the measurement.");
  await inputs.first().fill("7.2");
  await submit.click();
  await expect(page.getByRole("status")).toHaveText("Preprocessing complete. 4 measurements produced 5 model features.");
  await page.getByRole("button", {name:"Reset"}).click();
  await expect(inputs.first()).toHaveValue("");
  await expect(page.getByRole("status")).not.toContainText("Preprocessing complete");
});
