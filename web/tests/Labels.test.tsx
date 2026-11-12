import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnswerCard } from "@/components/AnswerCard";
import { ConfidenceReadout } from "@/components/ConfidenceReadout";
import { RefusalState } from "@/components/RefusalState";
import { SourcePanel } from "@/components/SourcePanel";
import { labelsFor, type Labels } from "@/lib/labels";
import { CITED_ANSWER, REFUSED_ANSWER } from "./fixtures";

// VERITY_LANGUAGE=fr: the API answers and refuses in French, so the text around a
// refusal and its citations is French too. English stays the default.

describe("labels in the corpus language", () => {
  it("falls back to English for an unknown or missing language", () => {
    expect(labelsFor(undefined)).toBe(labelsFor("en"));
    expect(labelsFor("de")).toBe(labelsFor("en"));
    expect(labelsFor("fr")).not.toBe(labelsFor("en"));
  });

  it("explains a refusal in French", () => {
    render(
      <RefusalState
        answer={REFUSED_ANSWER.answer}
        provenance={REFUSED_ANSWER.provenance}
        language="fr"
      />
    );
    const region = screen.getByRole("region");
    expect(region).toHaveTextContent("refus");
    expect(region).toHaveTextContent("L'agent a refusé plutôt que de livrer le brouillon");
    expect(region).toHaveTextContent("justification");
    expect(region).toHaveTextContent("appels LLM");
    expect(region).not.toHaveTextContent("The agent declined");
    expect(region).not.toHaveTextContent("Score is passed through");
  });

  it("explains a refusal in English by default", () => {
    render(
      <RefusalState answer={REFUSED_ANSWER.answer} provenance={REFUSED_ANSWER.provenance} />
    );
    expect(screen.getByRole("region")).toHaveTextContent("The agent declined");
  });

  it("labels the confidence signals in French", () => {
    render(
      <ConfidenceReadout
        answer={REFUSED_ANSWER.answer}
        provenance={REFUSED_ANSWER.provenance}
        language="fr"
      />
    );
    const group = screen.getByRole("group");
    expect(group).toHaveTextContent("confiance");
    expect(group).toHaveTextContent("passages examinés");
    expect(group).toHaveTextContent("oui");
    expect(group).not.toHaveTextContent("evidence hits");
  });

  it("labels the citations of an answer in French", () => {
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={() => {}}
        language="fr"
      />
    );
    const region = screen.getByRole("region");
    expect(region).toHaveTextContent("réponse");
    expect(screen.getByTestId("citation-row-1")).toHaveTextContent("passage");
    expect(region).not.toHaveTextContent("llm calls");
  });

  it("labels the evidence panel in French", () => {
    render(
      <SourcePanel
        answer={CITED_ANSWER.answer}
        activeChunkId={null}
        onChunkClick={() => {}}
        language="fr"
      />
    );
    expect(screen.getAllByText("citée").length).toBeGreaterThan(0);
    expect(screen.queryByText(/^reranked:/)).toBeNull();
    expect(screen.getAllByText(/^reclassement\s:/).length).toBeGreaterThan(0);
    expect(screen.queryByText("cited")).toBeNull();
    expect(screen.queryByText("not-cited")).toBeNull();
  });

  it("explains the empty evidence panel in French", () => {
    render(
      <SourcePanel answer={null} activeChunkId={null} onChunkClick={() => {}} language="fr" />
    );
    expect(screen.getByText(/s'affichent ici/)).toBeInTheDocument();
  });
});

describe("French labels for assistive technology and typography", () => {
  it("names the readout and usage groups in French", () => {
    render(
      <RefusalState
        answer={REFUSED_ANSWER.answer}
        provenance={REFUSED_ANSWER.provenance}
        language="fr"
      />
    );
    expect(screen.getByRole("group", { name: /lecture de la confiance/i })).toBeInTheDocument();
    expect(screen.getByLabelText("indicateurs")).toBeInTheDocument();
    expect(screen.getByLabelText("Fonctionnement")).toBeInTheDocument();
    expect(screen.queryByLabelText(/underlying signals|Operational/)).toBeNull();
  });

  it("keeps the English labels for assistive technology by default", () => {
    render(
      <RefusalState answer={REFUSED_ANSWER.answer} provenance={REFUSED_ANSWER.provenance} />
    );
    expect(screen.getByLabelText("underlying signals")).toBeInTheDocument();
    expect(screen.getByLabelText("Operational")).toBeInTheDocument();
  });

  it("never lets a French line break before : ; ! ? » or after «", () => {
    const fr: Labels = labelsFor("fr");
    const strings = Object.values(fr).flatMap((v) =>
      typeof v === "string" ? [v] : Object.values(v as Record<string, string>)
    );
    for (const s of strings) {
      expect(s, s).not.toMatch(/ [:;!?»]|« /);
    }
  });
});
