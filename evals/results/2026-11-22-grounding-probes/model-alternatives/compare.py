"""Evaluate a pinned three-class NLI checkpoint on authored regression cases."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--revision', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
fixture_path = Path('evals/nli/stress-cases.json')
fixture = fixture_path.read_bytes()
items = json.loads(fixture)['cases']
probe = Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text()
premise = probe.split('Premise (exact):\n', 1)[1].strip()
claim = 'RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
extra = [
 ('E2E', [premise], claim, 'supported'),
 ('literal-refuted', [f'The following statement is false: {claim} RAG never uses a retrieval system.'], claim, 'not_supported'),
 ('negated', [claim], 'RAG does not couple a retrieval system with a language model.', 'not_supported'),
 ('heading-refuted', ['# Incorrect statement\nThe fee is 10 dollars.\nThe correct fee is 20 dollars.'], 'The fee is 10 dollars.', 'not_supported'),
 ('heading-prohibited', ['# Prohibited actions\nPassword sharing is permitted.'], 'Password sharing is permitted.', 'not_supported'),
 ('conditional', ['If the discount is approved, the fee is 10 dollars. The discount is not approved.'], 'The fee is 10 dollars.', 'neutral'),
 ('cross-source', ['The fee is 10 dollars.', 'The fee is 20 dollars, not 10 dollars.'], 'The fee is 10 dollars.', 'not_supported'),
]
cases = extra + [(c['id'], [c['premise']], c['hypothesis'], c['expected']) for c in items]
start=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
model=AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision, trust_remote_code=False).eval()
labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
if set(labels) != {'entailment','neutral','contradiction'}:
 raise RuntimeError(f'Unexpected label mapping: {labels}')
results=[]
for case_id,premises,hypothesis,expected in cases:
 case_start=time.monotonic(); pair_scores=[]
 for text in premises:
  encoded=tokenizer(text,hypothesis,return_tensors='pt',truncation=True,max_length=512)
  raw_token_count=len(tokenizer(text,hypothesis,truncation=False)['input_ids'])
  with torch.inference_mode(): probs=torch.softmax(model(**encoded).logits,dim=-1)[0]
  pair_scores.append({'entailment':float(probs[labels['entailment']]),'neutral':float(probs[labels['neutral']]),'contradiction':float(probs[labels['contradiction']]),'input_tokens_untruncated':raw_token_count})
 e=max(x['entailment'] for x in pair_scores); c=max(x['contradiction'] for x in pair_scores)
 observed='not_supported' if c>=.5 and c>=e else ('supported' if e>=.5 else 'neutral')
 row={'id':case_id,'expected':expected,'observed':observed,'pair_scores':pair_scores,'seconds':round(time.monotonic()-case_start,3)}
 results.append(row); print(json.dumps(row),flush=True)
report={'model':args.model,'revision':args.revision,'label_mapping':labels,'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'python':platform.python_version(),'torch':torch.__version__,'planned':len(cases),'scored':len(results),'seconds':round(time.monotonic()-start,3),'results':results}
with args.output.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
