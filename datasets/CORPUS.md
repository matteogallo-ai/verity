# Eval corpus

21 documents of a fictional company, Acme Corporation, in three domains. Every document
was written for this repository and is released under the repository's MIT licence; no
third-party text is included, so there is no licence to check beyond that one. People
and companies named in them (Northwind Logistics, Maria Chen, Dana Whitfield, ...) are
invented.

PDF and DOCX files are rendered from Markdown sources in `datasets/corpus_src/` by
`scripts/build_corpus_documents.py` (byte-for-byte reproducible; `--check` in CI), so
every eval run exercises the docling parser.

| Document | Format | Domain | Licence |
|---|---|---|---|
| `msa.md` | Markdown | contracts | MIT, authored here |
| `msa_amendment_1.md` | Markdown | contracts | MIT, authored here |
| `sla.md` | Markdown | contracts | MIT, authored here |
| `dpa.md` | Markdown | contracts | MIT, authored here |
| `acceptable_use_policy.md` | Markdown | contracts | MIT, authored here |
| `order_form_northwind_2025.pdf` | PDF | contracts | MIT, authored here |
| `nda_mutual.docx` | DOCX | contracts | MIT, authored here |
| `sow_data_migration.docx` | DOCX | contracts | MIT, authored here |
| `employee_handbook.md` | Markdown | HR | MIT, authored here |
| `remote_work_policy.md` | Markdown | HR | MIT, authored here |
| `code_of_conduct.md` | Markdown | HR | MIT, authored here |
| `benefits_summary_2025.md` | Markdown | HR | MIT, authored here |
| `travel_expense_policy.pdf` | PDF | HR | MIT, authored here |
| `information_security_policy.pdf` | PDF | HR | MIT, authored here |
| `leave_policy.docx` | DOCX | HR | MIT, authored here |
| `financial_summary_fy2024.md` | Markdown | finance | MIT, authored here |
| `procurement_policy.md` | Markdown | finance | MIT, authored here |
| `board_minutes_2025_03.md` | Markdown | finance | MIT, authored here |
| `q1_fy2025_report.pdf` | PDF | finance | MIT, authored here |
| `budget_fy2025_memo.docx` | DOCX | finance | MIT, authored here |
| `vendor_onboarding_note.md` | Markdown | adversarial | MIT, authored here; carries prompt-injection text on purpose |

What this corpus does not test: scanned PDFs (OCR on images), tables, multi-column
layouts and documents longer than two pages. The PDFs are single-column text written by
a minimal PDF writer.
