# Interview reference publishing

The canonical prose is [interview-preparation.md](../interview-preparation.md). These assets publish the same three-part reference as a printable PDF and a self-contained interactive HTML reader. No application code is changed.

## Rebuild and check

Run from the repository root with `uv` installed:

```sh
uv run --script docs/interview-assets/validate_reference.py
uv run --script docs/interview-assets/build_reader.py
```

The second command regenerates the six SVGs and writes `~/output/html/xai-rag/interview-preparation.html`. Use `--assets-only` to regenerate just the illustrations. Python dependencies are pinned in each script's inline metadata; plain system Python may lack them.

PDF publication uses the local `book-publish` skill's WeasyPrint-based builder:

```sh
mkdir -p "$HOME/output/book/xai-rag-interview"
uv run --script "$HOME/.agents/skills/book-publish/scripts/md2pdf.py" docs/interview-preparation.md --paper a4 --font-size 10 --toc --no-mermaid --stylesheet docs/interview-assets/print.css --cover-title 'XAI-RAG: The Complete Interview Reference' --cover-subtitle 'Implementation · 30-minute walkthrough · 132 descriptive questions' --cover-date '17 September 2026' -o "$HOME/output/book/xai-rag-interview/xai-rag-interview.pdf"
uv run --script "$HOME/.agents/skills/book-publish/scripts/validate_pdf.py" "$HOME/output/book/xai-rag-interview/xai-rag-interview.pdf" --paper a4 --min-pages 60 --max-pages 200 --require-text 'Part I' --require-text 'Part II' --require-text 'Part III' --require-text 'Q22.6' --json
```

This PDF command assumes that skill is installed at the given location. It is not a repository dependency. The HTML build is independently reproducible from the repository assets.

## Figures and interactions

All six SVG illustrations and five interactive teaching panels are original, derived from the repository's implementation. Numerical chart values are illustrative, never benchmark evidence. The PDF contains static figures; JavaScript interactions belong to the HTML edition. The HTML embeds its CSS, scripts and SVGs without remote assets; source-reference hyperlinks can still lead to external pages when clicked.

## Verified edition

On 17 September 2026, the source validator counted three major sections, 22 component chapters, 132 answers, 104 valid pinned source references, and 4,151 spoken-section words (29.6 minutes at 140 words/minute; not a timed rehearsal). The generated PDF had 97 A4 pages. Browser checks exercised architecture selection, fusion, ranking metrics, evaluation status branches, latency controls, search and mobile overflow. PDF validation checked required text and page geometry; representative pages were visually inspected, not every page.
