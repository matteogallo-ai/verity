// Next.js 15 ESLint flat config — inherits Next's recommended rules and adds
// a couple of Verity-specific guards.
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({
  baseDirectory: import.meta.dirname,
});

export default [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // The whole S6 point: never surface something the API didn't return.
      // `console.warn` is fine (dev), but `console.log` in production code
      // suggests debug leftovers.
      "no-console": ["warn", { allow: ["warn", "error"] }],
    },
    ignores: [".next/**", "coverage/**"],
  },
];
