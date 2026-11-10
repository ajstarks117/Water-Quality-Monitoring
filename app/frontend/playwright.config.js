import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: {
    baseURL: "http://127.0.0.1:5175",
    channel: process.platform === "win32" ? "msedge" : undefined,
  },
  webServer: [
    {
      command: "python -m app.api",
      cwd: "../..",
      url: "http://127.0.0.1:18741/api/dashboard",
      reuseExistingServer: true,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5175 --strictPort",
      url: "http://127.0.0.1:5175",
    },
  ],
});
