# Compact technical report

The editable source is [technical-report.md](../technical-report.md). This is a separate, shorter systems report; the 97-page interview reference is unchanged. The PDF contains 10 A4 pages including references, with 11-point main text, two original diagrams and three tables. It is not a submitted or peer-reviewed paper.

## Build and validate

Requires `uv`, the local `book-publish` skill, and Poppler. From the repository root:

```sh
python3 "$HOME/.agents/skills/book-publish/scripts/md_lint.py" --check docs/technical-report.md
mkdir -p "$HOME/output/book/xai-rag-technical-report"
uv run --script "$HOME/.agents/skills/book-publish/scripts/md2pdf.py" docs/technical-report.md --paper a4 --font-size 11 --no-mermaid --stylesheet docs/technical-report-assets/print.css -o "$HOME/output/book/xai-rag-technical-report/xai-rag-technical-report.pdf"
uv run --script "$HOME/.agents/skills/book-publish/scripts/validate_pdf.py" "$HOME/output/book/xai-rag-technical-report/xai-rag-technical-report.pdf" --paper a4 --min-pages 8 --max-pages 10 --require-text 'Abstract' --require-text 'Evaluation methodology' --require-text '24 tests' --require-text 'References' --json
```

The user-specific skill installation is a publishing dependency, not an application dependency. Rendering used WeasyPrint 70.0 with Liberation Serif and Liberation Sans. No separate cover or contents pages consume the page budget. Page count should be rechecked after content, renderer or font changes.

## Research and editorial choices

The supplied [Scholar profile](https://scholar.google.com/citations?user=swapQ0kAAAAJ&hl=en) was inspected on 23 September 2026 using rendered extraction. It listed eight publications. Two were selected for structural comparison:

- Singh and Garain, [An Analysis of Transformer-based Models for Code-mixed Conversational Hate-speech Identification](https://ceur-ws.org/Vol-3395/T7-4.pdf), FIRE 2022, nine pages: problem → related work → data → methods → results → conclusion/references. The report adopts that explicit separation, but does not borrow claims, results or wording.
- Seth et al., [xai_evals: A Framework for Evaluating Post-Hoc Local Explanation Methods](https://arxiv.org/html/2502.03014v1), 2025: useful distinction between a framework's capabilities, evaluation methods and future work. This paper has 12 PDF pages, so it was not the page-budget template.

Technical research used the primary RAG and RAGAs papers, official RAGAS metric definitions, Elastic's RRF definition, and the Lost in the Middle paper's stated findings. All appear in the manuscript bibliography. Version-specific details were checked against the repository and installed RAGAS 0.4.3 source. The style papers are not cited as technical evidence for this unrelated RAG implementation.

The shorter edition removes the original 132-question bank, extended spoken script, exhaustive file-by-file explanations, and interactive labs. It preserves the system's data path, main algorithms, score semantics, evidence limits and evaluation priorities. No benchmark chart was manufactured from the two-query synthetic fixture. Both SVGs are original source-derived schematics; no third-party figure was copied.

## Validation boundary

Fresh checks on 23 September 2026: 24 focused tests passed with one dependency deprecation warning; the synthetic evaluator emitted 160 entries at K=1,2,3,5,10 and all 12 gates passed. These counts are not representative answer-quality evidence. Tests used real metric construction plus fake scorer outputs; no paid judge calls were performed. Historical integration and latency figures remain explicitly dated.

The manuscript contains suggested 15-/30-minute revision routes, not a timed reading experiment. The PDF passed page-count, nonblank-page, size and required-text validation. Structural inspection checked text geometry, bookmark and link presence; rendered pages were visually inspected. All original long-form files and user changes were preserved.
