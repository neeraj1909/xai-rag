"""Rerun missing cases, with checksummed prior scores for byte-identical cases."""
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

parser=argparse.ArgumentParser()
parser.add_argument('--name',choices=['xsmall','small','minilm'],required=True)
parser.add_argument('--model',required=True)
parser.add_argument('--revision',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
here=Path(__file__).parent
prior=Path('evals/results/2026-11-22-grounding-probes/model-alternatives')
manifest=json.loads((prior/'sha256.json').read_bytes())
for path,expected in manifest['inputs'].items():
    assert hash_file(Path(path))==expected, path
assert manifest[args.name]['revision']==args.revision
snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+args.model.replace('/','--'))/'snapshots'/args.revision
for filename,expected in manifest[args.name]['files'].items():
    assert hash_file(snapshot/filename)==expected, filename
old_path=prior/(args.name+'.json')
old=json.loads(old_path.read_bytes())
assert (old['model'],old['revision'],old['fixture_sha256'])==(args.model,args.revision,manifest['inputs']['evals/nli/stress-cases.json'])
old_results={r['id']:r for r in old['results']}
fixture_path=here/'cases.json'
cases=json.loads(fixture_path.read_bytes())['cases']
original_cases={x['id']:x for x in json.loads(Path('evals/nli/stress-cases.json').read_bytes())['cases']}
# Only reuse the 42 stress cases and E2E, whose exact inputs are independently verifiable.
# Other near-duplicate probes are rerun to avoid subtle changes in evidence wording.
reusable={}
for c in cases:
    if c['id'] in original_cases:
        old_case=original_cases[c['id']]
        assert c['claim']==old_case['hypothesis'] and c['expected']==old_case['expected']
        assert c['evidence']==[{'id':'chunk-1','text':old_case['premise']}] and c['citations']==['chunk-1']
        reusable[c['id']]=old_results[c['id']]
    elif c['id']=='original':
        probe=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text().split('Premise (exact):\n',1)[1].strip()
        assert c['evidence']==[{'id':'chunk-1','text':probe}] and c['citations']==['chunk-1']
        assert c['claim']=='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
        reusable[c['id']]=old_results['E2E']
    elif c['id'] in {'quoted_negation','prohibited_heading','conditional','contradictory_chunks'}:
        original='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
        specs={
          'quoted_negation':('literal-refuted',[f'The following statement is false: {original} RAG never uses a retrieval system.'],original),
          'prohibited_heading':('heading-prohibited',['# Prohibited actions\nPassword sharing is permitted.'],'Password sharing is permitted.'),
          'conditional':('conditional',['If the discount is approved, the fee is 10 dollars. The discount is not approved.'],'The fee is 10 dollars.'),
          'contradictory_chunks':('cross-source',['The fee is 10 dollars.','The fee is 20 dollars, not 10 dollars.'],'The fee is 10 dollars.'),
        }
        old_id,texts,claim=specs[c['id']]
        assert c['claim']==claim and [x['text'] for x in c['evidence']]==texts
        assert c['citations']==[x['id'] for x in c['evidence']]
        assert old_results[old_id]['expected']==c['expected']
        reusable[c['id']]=old_results[old_id]
started=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(args.model,revision=args.revision,trust_remote_code=False,local_files_only=True).eval()
labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
assert labels==old['label_mapping'] and set(labels)=={'entailment','neutral','contradiction'}
results=[]
for c in cases:
    if c['id'] in reusable:
        prior_row=reusable[c['id']]
        row={'id':c['id'],'expected':c['expected'],'observed':prior_row['observed'],'citation_valid':True,'pair_scores':prior_row['pair_scores'],'provenance':'reused_from_checksummed_prior_report'}
    else:
        claim=Claim(text=c['claim'],source_chunk_ids=c['citations'])
        chunks=[RankedResult(id=x['id'],content=x['text']) for x in c['evidence']]
        valid=claims_are_valid([claim],chunks)
        row={'id':c['id'],'expected':c['expected'],'citation_valid':valid,'pair_scores':[],'provenance':'new_inference_or_citation_check'}
        if not valid: row['observed']='invalid_citation'
        else:
            for chunk in chunks:
                if chunk.id not in claim.source_chunk_ids: continue
                inputs=tokenizer(chunk.content,claim.text,return_tensors='pt',truncation=True,max_length=512)
                count=len(tokenizer(chunk.content,claim.text,truncation=False)['input_ids'])
                with torch.inference_mode(): logits=model(**inputs).logits[0]; probs=torch.softmax(logits,dim=-1)
                row['pair_scores'].append({'chunk_id':chunk.id,'entailment':float(probs[labels['entailment']]),'neutral':float(probs[labels['neutral']]),'contradiction':float(probs[labels['contradiction']]),'logits':logits.tolist(),'input_tokens_untruncated':count})
            e=max(p['entailment'] for p in row['pair_scores']); cmax=max(p['contradiction'] for p in row['pair_scores'])
            row['observed']='not_supported' if cmax>=.5 and cmax>=e else ('supported' if e>=.5 else 'neutral')
    results.append(row)
    print(json.dumps({'id':row['id'],'expected':row['expected'],'observed':row['observed'],'provenance':row['provenance']}),flush=True)
report={'model':args.model,'revision':args.revision,'fixture_sha256':hash_file(fixture_path),'runner_sha256':hash_file(Path(__file__)),'prior_report_sha256':hash_file(old_path),'prior_manifest_sha256':hash_file(prior/'sha256.json'),'python':platform.python_version(),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'thresholds':{'entailment':.5,'contradiction':.5},'labels':labels,'planned':len(cases),'scored':len(results),'reused':len(reusable),'new':len(cases)-len(reusable),'seconds':round(time.monotonic()-started,3),'results':results}
args.output.write_text(json.dumps(report,indent=2)+'\n')
