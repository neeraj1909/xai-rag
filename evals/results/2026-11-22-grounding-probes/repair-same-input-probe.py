"""Diagnostic real-NLI check of candidate claims and evidence boundaries."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from xai_rag.config import settings
from xai_rag.explainability import faithfulness
from xai_rag.generation.release import claims_are_valid, render_supported_claims
from xai_rag.models import Claim, RankedResult

source = Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt')
raw = source.read_bytes()
premise = raw.decode().split('Premise (exact):\n', 1)[1].strip()
original = ('RAG couples a retrieval system with a language model so the model generates answers '
            'grounded in fetched evidence rather than relying solely on parametric memory.')
repair = 'RAG combines retrieval and language models.'
checker = faithfulness.FaithfulnessChecker()
full = RankedResult(id='chunk-1', content=premise)
truncated = RankedResult(id='chunk-1', content=premise[:70])
cases = [
    ('initial', original, ['chunk-1'], (full,), 'authored-supported'),
    ('stub-repair', repair, ['chunk-1'], (full,), 'authored-supported'),
    ('negation', 'RAG never uses a retrieval system or language model.', ['chunk-1'], (full,), 'authored-contradicted'),
    ('truncated', repair, ['chunk-1'], (truncated,), 'withhold; incomplete visible evidence'),
    ('wrong-id', repair, ['other-chunk'], (full,), 'reject invalid citation'),
    ('no-evidence', repair, ['chunk-1'], (), 'reject missing citation evidence'),
]
metadata = {'model': settings.nli_model, 'revision': settings.nli_model_revision,
            'thresholds': {'entailment': settings.nli_entailment_threshold, 'contradiction': settings.nli_contradiction_threshold},
            'checker_sha256': hashlib.sha256(Path(faithfulness.__file__).read_bytes()).hexdigest(),
            'evidence_sha256': hashlib.sha256(raw).hexdigest(),
            'torch': importlib.metadata.version('torch'), 'transformers': importlib.metadata.version('transformers'),
            'note': 'Local NLI and citation checks only. No generation, judge or HTTP request.'}
print(json.dumps({'metadata': metadata}), flush=True)
for name,text,ids,evidence,expected in cases:
    started=time.monotonic()
    claim = Claim(text=text, source_chunk_ids=ids)
    valid=claims_are_valid([claim], evidence)
    row={'case':name,'claim':text,'expected_handling':expected,'visible_chars':len(evidence[0].content) if evidence else 0,
         'citation_valid':valid}
    if valid:
        try:
            verdict=checker._check_single_claim(claim, list(evidence))
            row['verdict']=verdict.model_dump(mode='json')
            row['rendered_answer']=render_supported_claims([claim],[verdict])
        except Exception as exc:
            row['error_type']=type(exc).__name__
    row['elapsed_seconds']=round(time.monotonic()-started,3)
    print(json.dumps(row,ensure_ascii=False),flush=True)
