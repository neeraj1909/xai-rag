import json
from pathlib import Path
from xai_rag.explainability.faithfulness import FaithfulnessChecker
text=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text().split('Premise (exact):\n',1)[1].strip()
claim='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
cases=[('original', text,claim),('bare_identical',claim,claim),('bare_negation',claim,'RAG does not couple a retrieval system with a language model.'),('quoted_refuted','The following statement is false: '+claim+' RAG never uses a retrieval system.',claim),('numeric_refuted','It is not true that the fee is 10 dollars. The fee is 20 dollars.','The fee is 10 dollars.')]
models=[('original','MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli','6f5cf0a2b59cabb106aca4c287eed12e357e90eb'),('multilingual','MoritzLaurer/mDeBERTa-v3-base-mnli-xnli','8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c')]
for name,model,rev in models:
 c=FaithfulnessChecker(model,rev)
 for case,premise,hypothesis in cases:
  score=c._score_pair(premise,hypothesis)
  print(json.dumps({'model':name,'case':case,'scores':score,'literal_match':hypothesis in premise}),flush=True)
