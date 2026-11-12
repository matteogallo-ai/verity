/**
 * The question box: its text follows the corpus language, and its one-click questions
 * are the ones /status serves for the corpus (none when the corpus folder has none).
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AskBox } from "@/components/AskBox";

describe("AskBox", () => {
  it("offers the questions it is given and asks one on click", () => {
    const onAsk = vi.fn();
    render(<AskBox questions={["Combien de jours de vacances ?"]} language="fr" onAsk={onAsk} />);
    fireEvent.click(screen.getByRole("button", { name: "Combien de jours de vacances ?" }));
    expect(onAsk).toHaveBeenCalledWith("Combien de jours de vacances ?");
  });

  it("offers no question when it is given none", () => {
    render(<AskBox questions={[]} language="en" onAsk={vi.fn()} />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
  });

  it("is in French when the corpus is", () => {
    render(<AskBox questions={[]} language="fr" onAsk={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: "Question" })).toHaveAttribute(
      "placeholder",
      "Posez une question sur les documents…",
    );
    expect(screen.getByRole("button", { name: "Demander" })).toHaveTextContent("Demander ↵");
  });

  it("stays in English by default, with the name the demo recorder clicks", () => {
    render(<AskBox questions={[]} language="en" onAsk={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: "Question" })).toHaveAttribute(
      "placeholder",
      "Ask a question about the ingested documents…",
    );
    expect(screen.getByRole("button", { name: "Ask" })).toHaveTextContent("Ask ↵");
  });
});
