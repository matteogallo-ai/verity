import { describe, expect, it } from "vitest";
import { errorDetail } from "@/lib/api";

describe("errorDetail", () => {
  it("reads a string detail and appends the request id", () => {
    expect(errorDetail({ detail: "Upload too large.", request_id: "r1" }, "x")).toBe(
      "Upload too large. (request r1)"
    );
  });

  it("reads the message of a 503 outage detail object", () => {
    const body = {
      detail: { error: "provider_unavailable", message: "Retry later." },
      request_id: "r2",
    };
    expect(errorDetail(body, "x")).toBe("Retry later. (request r2)");
  });

  it("falls back to the status text for a non-JSON or empty body", () => {
    expect(errorDetail(null, "Bad Gateway")).toBe("Bad Gateway");
    expect(errorDetail({ detail: [{ msg: "bad" }] }, "Unprocessable")).toBe("Unprocessable");
  });
});
