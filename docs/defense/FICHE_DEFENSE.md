# Verity : fiche de défense (entretien)

**1. Pourquoi, pour qui.** Sur des contrats, des politiques internes ou des rapports financiers, une démo RAG qui répond avec assurance et se trompe juste assez souvent est dangereuse : une réponse fluide, qui a l'air citée, sans vrai ancrage. Verity vise une équipe qui lit la clause citée avant d'agir (`docs/SYSTEM_CARD.md:23-24`). À ma connaissance, personne ne s'en sert en production.

**2. En une phrase.** Un agent RAG sur des documents d'entreprise qui refuse (« I don't know ») quand ses preuves sont faibles, et qui publie à quelle fréquence il a raison de refuser.

**3. Trois décisions clés**

| Décision | Gain | Coût |
|---|---|---|
| Le refus est une fonctionnalité mesurée, pas un cas d'erreur ([ADR 0003](../adr/0003-refusal-as-headline-feature.md)) | Dans le pipeline actuel, une réponse n'est donnée que si les contrôles déterministes passent (couverture et pertinence des citations, marqueurs d'injection) et que le score dépasse le seuil. Le jeu d'éval contient exprès 33 questions hors corpus sur 95. | Le score est l'auto-évaluation du modèle scoreur, et il n'est pas calibré : ECE 0,116 sur n = 36 (C21, `docs/CLAIMS.md:35`). 5 refus sur 22 sont de trop, et 3 questions hors corpus sur 20 obtiennent quand même une réponse. |
| Chaque citation est vérifiée dans le code, pas par un modèle | Une citation dont le texte cité n'est pas une sous-chaîne exacte du chunk référencé est rejetée (`_validate_citations`, `src/verity/agent/synthesizer.py:125`). Le modèle propose, le code valide. | Une citation de la taille d'un paragraphe peut couvrir plusieurs affirmations : le juge la marque alors « non exacte » alors que l'affirmation est soutenue (`docs/SYSTEM_CARD.md:117`). |
| Les chiffres publiés viennent d'artefacts enregistrés, pas d'un nouveau tirage ([ADR 0007](../adr/0007-reproducibility-from-artefacts.md)) | Une seule exécution réelle par split, jamais relancée, épinglée par `SHA256SUMS`. Le tableau du README et la system card sont générés depuis le scorecard, et la CI vérifie qu'ils sont à jour. | n = 56 sur le split de test : intervalles larges. On ne peut pas relancer pour « améliorer » un chiffre. |

**4. Chiffres** (exécution réelle `eval_969e1d5_test.json`, split de test, agent claude-sonnet-4-6, 2 nov. 2026)

| Chiffre | Source | Comment il a été mesuré |
|---|---|---|
| Rappel du refus : **17/20**, IC 95 % [0,640 ; 0,948] | `docs/SYSTEM_CARD.md:74` | Questions hors corpus que l'agent refuse. Seuils choisis sur le split dev seulement (39 questions). Intervalle de Wilson. |
| Précision du refus : **17/22**, [0,566 ; 0,899] | `docs/SYSTEM_CARD.md:75` | Même exécution : 5 refus portent sur des questions qui avaient une réponse dans le corpus. |
| Fidélité par affirmation : **108/109** (jugée) | `docs/SYSTEM_CARD.md:78` | Juge LLM (claude-opus-5-5, prompt v0.1.0). Les affirmations d'une même réponse sont corrélées, donc l'intervalle est trop étroit. |
| Accord juge / humain : **κ 0,71** sur 60 affirmations, **0,48** sur les 40 réelles seules | `README.md:136-137` | Étiquettes humaines posées à l'aveugle, sans l'origine ni le verdict du juge. Le κ sur les affirmations réelles est fragile : une seule des 40 est non soutenue. |
| Coût par réponse : **0,0132 $** (moyenne, n = 56) | `docs/SYSTEM_CARD.md:88` | Tokens renvoyés par l'API multipliés par la table `model_prices` de `verity.config`. Ce n'est pas une facture. Le coût du juge est exclu (il ne sert qu'à l'éval). |

**5. Une erreur réelle attrapée par le process.** Les labels de retrieval (les IDs des chunks de référence) avaient été calculés sur mon Mac. La première CI de la PR #35 a échoué ([run 41813056725](https://github.com/matteogallo-ai/verity/actions/runs/41813056725)) : sur le runner GitHub, le modèle de layout de docling classait autrement les blocs des 4 PDF du corpus. Les titres numérotés devenaient des items de liste, les lignes clé-valeur des puces, et un paragraphe était coupé en deux. Le diff texte est dans le log du test de fixture, [run 41825483395](https://github.com/matteogallo-ai/verity/actions/runs/41825483395).
- Les offsets bougeaient, donc 9 IDs de chunk aussi. 18 questions perdaient leur chunk de référence et le recall tombait à 0,629, sous le seuil de 0,75 (log du job eval smoke, run 41813056725).
- Mes reproductions locales, non commitées, n'ont pas reproduit l'écart : sous Linux amd64 émulé (Docker), avec 10 puis 4 threads, j'obtenais exactement le parse du Mac. J'en déduis que seul le CPU réel du runner différait. C'est une inférence, pas une mesure.
- Correction : l'éval recalcule les chunks de référence à partir des citations de preuve, sur les chunks qu'elle ingère (`src/verity/eval/labels.py`), et sort en code 2 si une citation est introuvable. Commits `9725027`, `c8b6263`, `3ae7fdc`.
- Le test qui reproduit l'échec sans docling (`tests/unit/test_eval_labels.py`, IDs commités périmés) était rouge avant la correction, avec `recall@k 0.000 < floor 1.000`, dans mon exécution locale.
- Cette PR (branche `test/fixture-digests`) ajoute l'empreinte sha256 de chaque PDF et DOCX à la fixture docling, pour qu'un fichier modifié ne passe plus inaperçu.
- La même PR #35 a montré deux autres trous :
  - le contrôle des claims ne vérifiait que l'existence des fichiers de test, pas celle des fonctions (`ce8e42f`) ;
  - j'ai d'abord cru le slow lane bloqué et j'ai annulé le run 41813056725, alors qu'il était seulement lent : environ 16 tests relançaient chacun une éval complète, et la sortie n'affichait rien (`ae2b6bf`).

**6. Ce que je ferais autrement**
- Faire passer en CI, dès le premier PDF du corpus, les labels et la fixture produits par docling. Les labels, construits sur mon Mac, n'ont rencontré le runner qu'à la PR #35.
- Rendre la sortie du slow lane lisible ligne par ligne dès le début, avec un délai par test, plutôt que de deviner un blocage.
- Mesurer la calibration du score de confiance avant de baser le refus sur un seuil.

**7. Questions pièges**
1. *« 17/20, c'est sur 20 questions. »* → Oui. L'intervalle va de 0,64 à 0,95, et je le publie avec chaque chiffre. Je n'ai pas relancé pour obtenir mieux : c'est la seule exécution réelle sur le split de test.
2. *« Le juge est un LLM : pourquoi le croire ? »* → Il a été validé contre des étiquettes humaines posées à l'aveugle : κ 0,71 sur 60 affirmations. Sur les affirmations réelles seules, κ vaut 0,48 et reste fragile. Un prompt de juge défendu (v0.2.0) a été mesuré à part : κ 0,60, contre 0,64 pour la v0.1.0 sur les mêmes 52 affirmations. Aucune exécution de référence ne l'utilise.
3. *« Le score de confiance, c'est une probabilité ? »* → Non. C'est l'auto-évaluation du modèle, non calibrée (ECE 0,116, C21 en roadmap). Le seuil a été réglé sur le split dev, et je ne présente pas le score comme une probabilité.
4. *« Tu as écrit le code ? »* → J'ai dirigé des agents de code. Ma part :
   - les critères de « done » et les contrôles CI (claims ↔ tests, blocs générés, absence de récit périmé) ;
   - le choix des corrections (par exemple, recalculer les chunks de référence depuis l'evidence plutôt que figer le parse) ;
   - un relecteur à contexte neuf avant chaque remise.

   Je peux expliquer `labels.py` et la chaîne d'IDs de chunk ligne par ligne.
5. *« Et la prompt injection ? »* → Le corpus contient un document piégé (`vendor_onboarding_note.md`). La synthèse balise les preuves (datamarking, prompt v0.3.0). Sa mesure, faite sur l'ancien corpus de 6 documents, n'a détourné aucun cas, avec ou sans la balise (0/12, [0 ; 0,243], `docs/releases/1.1.0.md:29`). Elle ne montre donc pas l'effet de la balise. Un document qui affirme un fait faux reste cru : les défenses visent les instructions, pas les mensonges.
