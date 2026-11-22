"""Compare fixed, authored cases across pinned NLI models; no provider calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai_rag.generation.release import claims_are_valid
from xai_rag.models import Claim, RankedResult

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--revision', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
fixture_path = Path(__file__).with_name('cases.json')
raw = fixture_path.read_bytes()
cases = json.loads(raw)['cases']
started = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision, trust_remote_code=False, local_files_only=True).eval()
labels = {str(v).lower(): int(k) for k,v in model.config.id2label.items()}
if set(labels) != {'entailment','neutral','contradiction'}:
    raise ValueError(f'Unexpected mapping {labels}')
results=[]
for case in cases:
    started_case=time.monotonic()
    claim=Claim(text=case['claim'],source_chunk_ids=case['citations'])
    chunks=[RankedResult(id=c['id'],content=c['text']) for c in case['evidence']]
    valid=claims_are_valid([claim],chunks)
    row={'id':case['id'],'expected':case['expected'],'citation_valid':valid,'pair_scores':[]}
    if not valid:
        row['observed']='invalid_citation'
    else:
        for chunk in chunks:
            if chunk.id not in claim.source_chunk_ids:
                continue
            inputs=tokenizer(chunk.content,claim.text,return_tensors='pt',truncation=True,max_length=512)
            count=len(tokenizer(chunk.content,claim.text,truncation=False)['input_ids'])
            with torch.inference_mode(): logits=model(**inputs).logits[0]; probs=torch.softmax(logits,dim=-1)
            row['pair_scores'].append({'chunk_id':chunk.id,'entailment':float(probs[labels['entailment']]),'neutral':float(probs[labels['neutral']]),'contradiction':float(probs[labels['contradiction']]),'logits':logits.tolist(),'input_tokens_untruncated':count})
        e=max(p['entailment'] for p in row['pair_scores']); c=max(p['contradiction'] for p in row['pair_scores'])
        row['observed']='not_supported' if c >= .5 and c >= e else ('supported' if e >= .5 else 'neutral')
    row['seconds']=round(time.monotonic()-started_case,3)
    results.append(row)
    print(json.dumps({'id':row['id'],'expected':row['expected'],'observed':row['observed']}),flush=True)
report={'model':args.model,'revision':args.revision,'fixture_sha256':hashlib.sha256(raw).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'thresholds':{'entailment':.5,'contradiction':.5},'labels':labels,'planned':len(cases),'scored':len(results),'seconds':round(time.monotonic()-started,3),'results':results}
args.output.write_text(json.dumps(report,indent=2)+'\n')
