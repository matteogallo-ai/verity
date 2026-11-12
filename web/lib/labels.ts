/**
 * The text of the question box, and around an answer, a refusal and their evidence, in
 * the corpus language the API reports on /status (VERITY_LANGUAGE). The answer and the
 * refusal are already in that language; a French refusal under an English explanation
 * reads as a broken page.
 * Any language the API names that is not listed here falls back to English.
 */
export type Language = "en" | "fr";

export interface Labels {
  presentationTitle: string;
  questionPlaceholder: string;
  ask: string;
  asking: string;
  askFailed: string;
  ready: string;
  readyHint: string;
  readyHintWithQuestions: string;
  answer: string;
  refused: string;
  refusalNote: string;
  confidence: string;
  agentReported: string;
  refusedSignal: string;
  yes: string;
  no: string;
  citations: string;
  topRerankScore: string;
  notAvailable: string;
  evidenceHits: string;
  rationale: string;
  scoreNote: string;
  citation: string;
  sources: string;
  sourcesNote: string;
  sourcesEmpty: string;
  cited: string;
  notCited: string;
  noSection: string;
  readoutGroup: string;
  signalsGroup: string;
  usageGroup: string;
  operationalGroup: string;
  hitKind: Record<string, string>;
  chunk: string;
  doc: string;
  chars: string;
  latency: string;
  llmCalls: string;
  inTokens: string;
  outTokens: string;
  cost: string;
  trace: string;
}

const EN: Labels = {
  presentationTitle: "Agreement assistant",
  questionPlaceholder: "Ask a question about the ingested documents\u2026",
  ask: "Ask",
  asking: "asking\u2026",
  askFailed: "ask failed:",
  ready: "Ready.",
  readyHint: "Ask a question about the ingested documents (or upload a doc first).",
  readyHintWithQuestions:
    "Ask a question about the ingested documents. Try one of the demo questions above " +
    "(or upload a doc first).",
  answer: "answer",
  refused: "refused",
  refusalNote:
    "The agent declined rather than ship the draft answer: a citation check failed, the " +
    "scorer refused, or its score fell below the threshold. The rationale above names the " +
    "cause; the hits used to make this decision are still shown on the right panel for " +
    "auditability.",
  confidence: "confidence",
  agentReported: "agent-reported",
  refusedSignal: "refused",
  yes: "true",
  no: "false",
  citations: "citations",
  topRerankScore: "top rerank score",
  notAvailable: "n/a",
  evidenceHits: "evidence hits",
  rationale: "rationale",
  scoreNote:
    "Score is passed through verbatim from the confidence scorer for this request — no " +
    "client-side composite, no rescaling. Interpret alongside the signals above.",
  citation: "Citation",
  sources: "sources",
  sourcesNote: "rerank score · cited flag from API set-membership",
  sourcesEmpty: "The evidence used by the agent will appear here after you ask a question.",
  cited: "cited",
  notCited: "not-cited",
  noSection: "(no section)",
  readoutGroup: "Confidence readout (agent-reported score plus underlying signals)",
  signalsGroup: "underlying signals",
  usageGroup: "Usage",
  operationalGroup: "Operational",
  hitKind: {},
  chunk: "chunk",
  doc: "doc",
  chars: "chars",
  latency: "latency",
  llmCalls: "llm calls",
  inTokens: "in tokens",
  outTokens: "out tokens",
  cost: "cost",
  trace: "trace",
};

const FR: Labels = {
  presentationTitle: "Assistant CCT",
  questionPlaceholder: "Posez une question sur les documents\u2026",
  ask: "Demander",
  asking: "recherche en cours\u2026",
  askFailed: "échec de la question\u202f:",
  ready: "Prêt.",
  readyHint: "Posez une question sur les documents (ou ajoutez d'abord un document).",
  readyHintWithQuestions:
    "Posez une question sur les documents. Essayez l'une des questions proposées " +
    "ci-dessus (ou ajoutez d'abord un document).",
  answer: "réponse",
  refused: "refus",
  refusalNote:
    "L'agent a refusé plutôt que de livrer le brouillon : une vérification des citations a " +
    "échoué, l'évaluateur a refusé, ou son score était sous le seuil. La justification " +
    "ci-dessus en donne la cause ; les passages examinés restent affichés dans le panneau " +
    "de droite, pour vérification.",
  confidence: "confiance",
  agentReported: "fourni par l'agent",
  refusedSignal: "refusé",
  yes: "oui",
  no: "non",
  citations: "citations",
  topRerankScore: "meilleur score de pertinence",
  notAvailable: "n/d",
  evidenceHits: "passages examinés",
  rationale: "justification",
  scoreNote:
    "Le score est repris tel quel de l'évaluateur de confiance pour cette question, sans " +
    "calcul ni mise à l'échelle par l'interface. À lire avec les indicateurs ci-dessus.",
  citation: "Citation",
  sources: "sources",
  sourcesNote: "score de pertinence · « citée » selon l'API",
  sourcesEmpty: "Les passages utilisés par l'agent s'affichent ici après une question.",
  cited: "citée",
  notCited: "non citée",
  noSection: "(sans section)",
  readoutGroup: "Lecture de la confiance (score fourni par l'agent et indicateurs)",
  signalsGroup: "indicateurs",
  usageGroup: "Consommation",
  operationalGroup: "Fonctionnement",
  hitKind: { dense: "dense", sparse: "lexical", fused: "fusion", reranked: "reclassement" },
  chunk: "passage",
  doc: "doc",
  chars: "car.",
  latency: "latence",
  llmCalls: "appels LLM",
  inTokens: "jetons d'entrée",
  outTokens: "jetons de sortie",
  cost: "coût",
  trace: "trace",
};

const LABELS: Record<Language, Labels> = { en: EN, fr: FR };

/** The language /status reports, as one the UI has labels for. */
export function labelsLanguage(language: string | undefined): Language {
  return language === "fr" ? "fr" : "en";
}

export function labelsFor(language: string | undefined): Labels {
  return LABELS[labelsLanguage(language)];
}

/** "reranked: 4.038" in English, "reclassement : 4.038" in French (a narrow no-break
 * space before the colon); an unknown kind as sent. */
export function hitScoreLabel(labels: Labels, kind: string): string {
  const name = labels.hitKind[kind];
  return name === undefined ? `${kind}:` : `${name} :`;
}
