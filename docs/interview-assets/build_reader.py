# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown==3.10.2"]
# ///
"""Build original illustrations and a standalone interactive interview reader.

Run: uv run --script docs/interview-assets/build_reader.py --assets-only
Then after the Markdown exists, run without the flag to publish HTML.
The Markdown is the canonical prose; the reader never loads network assets.
"""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
import re

import markdown

ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS.parent / "interview-preparation.md"
OUTPUT = Path.home() / "output/html/xai-rag/interview-preparation.html"
INK = "#18312f"
TEAL = "#157a73"
GOLD = "#ac681a"


def svg(title: str, body: str, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img" aria-label="{html.escape(title)}">
<title>{html.escape(title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker></defs>
<rect width="900" height="{height}" rx="16" fill="#f4f7f4"/>
<g font-family="DejaVu Sans, sans-serif" fill="{INK}"><text x="32" y="40" font-size="23" font-weight="bold">{html.escape(title)}</text>{body}</g></svg>'''


def box(x: int, y: int, w: int, label: str, sub: str = "") -> str:
    title_size = min(18, (w - 22) / max(1, len(label)) / 0.59)
    sub_size = min(14, (w - 20) / max(1, len(sub)) / 0.54)
    return f'<rect x="{x}" y="{y}" width="{w}" height="68" rx="8" fill="white" stroke="{TEAL}" stroke-width="2"/><text x="{x+w/2}" y="{y+27}" text-anchor="middle" font-size="{title_size}" font-weight="bold">{html.escape(label)}</text><text x="{x+w/2}" y="{y+50}" text-anchor="middle" font-size="{sub_size}">{html.escape(sub)}</text>'


def arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{TEAL}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'


def figures() -> None:
    body = box(30, 80, 240, "HTTP / CLI", "API limits at HTTP boundary")
    body += box(330, 80, 240, "RAGService", "Shared pipeline orchestration")
    body += box(630, 80, 240, "Telemetry", "Spans, timings, safe attributes")
    body += arrow(270, 114, 330, 114) + arrow(570, 114, 630, 114)
    body += box(30, 220, 240, "Chroma + Elasticsearch", "Dense and lexical retrieval")
    body += box(330, 220, 240, "Fusion + reranking", "Preserve each stage's scores")
    body += box(630, 220, 240, "Generator", "Bounded context, cited output")
    body += arrow(400, 148, 155, 220) + arrow(270, 254, 330, 254) + arrow(570, 254, 630, 254)
    body += box(200, 360, 240, "Optional NLI", "Claim-to-child diagnostics")
    body += box(510, 360, 240, "Optional RAGAS", "Response + retrieved contexts")
    body += arrow(750, 288, 630, 360) + arrow(705, 288, 320, 360)
    body += '<text x="32" y="470" font-size="16">Offline evaluator separately consumes recorded datasets; it does not call the live judge.</text>'
    (ASSETS / "architecture.svg").write_text(svg("System architecture and execution boundaries", body, 500))

    body = ""
    steps = [("Discover + bound", "Root, file count, byte limits"), ("Parse + chunk", "Exact source spans; chosen strategy"), ("Embed children", "Pinned retrieval model and dimension"), ("Write Chroma", "Stable IDs; replace stale source records"), ("Write Elasticsearch", "Same identities; lexical index"), ("Check parity", "IDs and content hashes; report divergence")]
    for i, (label, sub) in enumerate(steps):
        y = 75 + i * 95
        body += box(220, y, 460, label, sub)
        if i < len(steps)-1:
            body += arrow(450, y+68, 450, y+93)
    body += '<text x="40" y="685" font-size="16">Failure between writes can leave divergence. This is not a cross-store transaction.</text>'
    (ASSETS / "ingestion.svg").write_text(svg("Document ingestion: ordered writes and visible failure", body, 720))

    body = box(285, 75, 330, "Ranked retrieval results", "Child content + optional parent text")
    for x, title, sub in [(30, "Generator", "Parent/text, character truncation"), (330, "RAGAS", "Full parent/text list"), (630, "NLI", "Claim-scoped child text")]:
        body += box(x, 210, 240, title, sub) + arrow(450, 143, x+120, 210)
    body += '<text x="40" y="325" font-size="17">These evidence views can differ. A score must be interpreted with its exact inputs.</text>'
    body += box(240, 375, 420, "Proposed capture improvement", "Persist the exact generator-visible spans")
    (ASSETS / "evidence.svg").write_text(svg("Three evidence views: an important evaluation boundary", body, 480))

    ranks = [("A", 1, 3), ("B", 2, 1), ("C", 3, None), ("D", None, 2)]
    body = '<text x="32" y="75" font-size="16">Illustrative ranks; k = 60. Score = sum of 1 / (k + rank) where present.</text>'
    for i, (label, dense, lexical) in enumerate(ranks):
        y = 112 + i*62
        a = 0 if dense is None else 1/(60+dense)
        b = 0 if lexical is None else 1/(60+lexical)
        body += f'<text x="35" y="{y+25}" font-size="18">{label}</text><rect x="100" y="{y}" width="{a*17000}" height="32" fill="{TEAL}"/><rect x="{100+a*17000}" y="{y}" width="{b*17000}" height="32" fill="{GOLD}"/><text x="720" y="{y+23}" font-size="18">{a+b:.6f}</text>'
    body += '<text x="32" y="402" font-size="16">Teal: dense contribution. Ochre: lexical contribution. B outranks A in this example.</text>'
    (ASSETS / "rrf.svg").write_text(svg("Reciprocal rank fusion: contributions remain distinct", body, 440))

    grades = {"A":3,"B":2,"C":1}
    order = ["D","A","B","E","C"]
    points_p, points_r = [], []
    body = '<text x="32" y="75" font-size="16">Illustrative ranking D, A, B, E, C. Relevant set: A, B, C.</text>'
    for value in [0, .25, .5, .75, 1]:
        y=330-value*220
        body += f'<line x1="100" y1="{y}" x2="760" y2="{y}" stroke="#cedbd8"/><text x="42" y="{y+6}" font-size="16">{value:.2f}</text>'
    for k in range(1,6):
        x=100+(k-1)*165
        hits=len(set(order[:k]) & set(grades))
        points_p.append(f"{x},{330-hits/k*220}")
        points_r.append(f"{x},{330-hits/3*220}")
        body += f'<text x="{x}" y="365" text-anchor="middle" font-size="18">{k}</text>'
    body += f'<polyline points="{" ".join(points_p)}" fill="none" stroke="{TEAL}" stroke-width="4"/><polyline points="{" ".join(points_r)}" fill="none" stroke="{GOLD}" stroke-width="4" stroke-dasharray="9 5"/>'
    body += '<text x="32" y="405" font-size="16">Cutoff k →   Solid teal: precision@k. Dashed ochre: recall@k.</text>'
    (ASSETS / "retrieval-metrics.svg").write_text(svg("Why precision and recall tell different stories", body, 445))

    body = box(30,90,250,"Request RAGAS?","include_ragas flag")+box(325,90,250,"Server allows it?","allow_request_evaluation")+box(620,90,250,"Dependencies + judge","Construct and score metrics")
    body += arrow(280,124,325,124)+arrow(575,124,620,124)
    body += box(30,260,250,"Not requested","Optional NLI aggregate or zero")+box(325,260,250,"Disabled","Request rejected")+box(620,260,250,"Computed / unavailable / error","Inspect status before the score")
    body += arrow(155,158,155,260)+arrow(450,158,450,260)+arrow(745,158,745,260)
    body += '<text x="32" y="375" font-size="16">Success: valid RAGAS faithfulness supplies overall_confidence.</text><text x="32" y="403" font-size="16">Requested failure: overall_confidence is zero; that is not a measured judge verdict.</text>'
    (ASSETS / "evaluation-flow.svg").write_text(svg("RAGAS request decisions and score interpretation",body,440))


def reader() -> None:
    md = SOURCE.read_text()
    processor = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"], extension_configs={"toc":{"toc_depth":"2-3"}})
    rendered = processor.convert(md)
    def embed(match: re.Match) -> str:
        relative = match.group(1)
        image = (SOURCE.parent / relative).resolve()
        if not image.is_relative_to(ASSETS):
            raise ValueError(f"Image outside illustration directory: {image}")
        return 'src="data:image/svg+xml;base64,'+base64.b64encode(image.read_bytes()).decode()+'"'
    rendered = re.sub(r'src="(interview-assets/[^\"]+\.svg)"', embed, rendered)
    template = (ASSETS / "reader-template.html").read_text()
    result = template.replace("<!--BOOK_CONTENT-->", rendered).replace("<!--BOOK_TOC-->", processor.toc)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(result)
    print(OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    figures()
    if not args.assets_only:
        reader()
