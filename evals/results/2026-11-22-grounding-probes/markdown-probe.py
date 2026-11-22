import json, re
from pathlib import Path
from xai_rag.explainability.faithfulness import FaithfulnessChecker
text=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text().split('Premise (exact):\n',1)[1].strip()
claim='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
cases=[('strip_heading_markers',re.sub(r'(?m)^#{1,6}\s+', '', text),claim),('remove_heading_lines',re.sub(r'(?m)^#{1,6}\s+.*(?:\n|$)', '', text),claim),('refuted_claim',f'The following statement is false: {claim} RAG never uses a retrieval system.',claim)]
c=FaithfulnessChecker()
for name,premise,hypothesis in cases:
 print(json.dumps({'case':name,'scores':c._score_pair(premise,hypothesis),'premise':premise}),flush=True)
