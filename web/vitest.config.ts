import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.{test,spec}.{ts,tsx}"],
    // Zero network in tests — deterministic fixtures only.
    testTimeout: 5000,
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: ["components/**/*.tsx", "lib/**/*.ts"],
    },
  },
  resolve: {
    alias: {
      "@": new URL(".", import.meta.url).pathname,
    },
  },
});
