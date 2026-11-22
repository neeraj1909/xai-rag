"""Replay new controls and reuse exact prior inputs with recorded provenance."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xai_rag.generation.release import claims_are_valid
from xai_rag.models import Claim, RankedResult

here=Path(__file__).parent
fixture_path=here/'cases.json'
hash_bytes=lambda v:hashlib.sha256(v).hexdigest()
cases=json.loads(fixture_path.read_bytes())['cases']
stress_path=Path('evals/nli/stress-cases.json')
stress_data=stress_path.read_bytes()
stress={c['id']:c for c in json.loads(stress_data)['cases']}
old_path=Path('evals/results/2026-11-22-nli/stress-report.json')
old=json.loads(old_path.read_bytes())
assert old['fixture_sha256']==hash_bytes(stress_data)
assert old['model']=='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli' and old['revision']=='6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
assert old['entailment_threshold']==old['contradiction_threshold']==.5
old_code=subprocess.check_output(['git','show','c375903:src/xai_rag/explainability/faithfulness.py'])
new_code=Path('src/xai_rag/explainability/faithfulness.py').read_bytes()
assert hash_bytes(old_code)==old['checker_sha256']
# The only change since the old checker is a model-load lock; inference is identical.
def scoring_methods(code):
    tree=ast.parse(code)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FaithfulnessChecker')
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'_score_pair','_check_single_claim'}}
assert len(scoring_methods(old_code))==2 and scoring_methods(old_code)==scoring_methods(new_code)
prior_path=Path('evals/results/2026-11-22-grounding-probes/repair-same-input-results.jsonl')
lines=[json.loads(x) for x in prior_path.read_text().splitlines()]
meta=lines[0]['metadata']; prior={x['case']:x for x in lines[1:]}
assert meta['checker_sha256']==hash_bytes(new_code) and meta['model']==old['model'] and meta['revision']==old['revision']
probe=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_bytes()
assert meta['evidence_sha256']==hash_bytes(probe)
premise=probe.decode().split('Premise (exact):\n',1)[1].strip()
mapping={'original':'initial','scripted_repair':'stub-repair','direct_negation':'negation','truncation':'truncated','wrong_citation':'wrong-id','missing_evidence':'no-evidence'}
old_rows={r['id']:r for r in old['cases']}
reused={}
for c in cases:
    if c['id'] in stress:
        item=stress[c['id']]; row=old_rows[c['id']]
        assert c['claim']==item['hypothesis'] and c['expected']==item['expected'] and c['evidence']==[{'id':'chunk-1','text':item['premise']}] and c['citations']==['chunk-1']
        assert row['premise']==item['premise'] and row['hypothesis']==item['hypothesis'] and row['expected']==item['expected']
        reused[c['id']]={'id':c['id'],'expected':c['expected'],'observed':row['observed'],'citation_valid':True,'pair_scores':[{'entailment':row['entailment'],'contradiction':row['contradiction']}],'provenance':'prior_42_case_report'}
    elif c['id'] in mapping:
        row=prior[mapping[c['id']]]
        assert c['claim']==row['claim']
        if c['id']=='missing_evidence': assert c['evidence']==[] and c['citations']==['chunk-1']
        else:
            expected_text=premise[:70] if c['id']=='truncation' else premise
            assert c['evidence']==[{'id':'chunk-1','text':expected_text}]
            assert c['citations']==(['other'] if c['id']=='wrong_citation' else ['chunk-1'])
        result={'id':c['id'],'expected':c['expected'],'citation_valid':row['citation_valid'],'pair_scores':[],'provenance':'prior_current_checker_same_input_probe'}
        if row['citation_valid']:
            v=row['verdict']; result['observed']=v['verdict']; result['pair_scores']=[{'entailment':v['nli_entailment'],'contradiction':v['nli_contradiction']}]
        else: result['observed']='invalid_citation'
        reused[c['id']]=result
started=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(old['model'],revision=old['revision'],trust_remote_code=False,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(old['model'],revision=old['revision'],trust_remote_code=False,local_files_only=True).eval()
labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
assert set(labels)=={'entailment','neutral','contradiction'}
results=[]
for case in cases:
    if case['id'] in reused: row=reused[case['id']]
    else:
        claim=Claim(text=case['claim'],source_chunk_ids=case['citations'])
        chunks=[RankedResult(id=c['id'],content=c['text']) for c in case['evidence']]
        assert claims_are_valid([claim],chunks)
        scores=[]
        for chunk in chunks:
            inputs=tokenizer(chunk.content,claim.text,return_tensors='pt',truncation=True,max_length=512)
            count=len(tokenizer(chunk.content,claim.text,truncation=False)['input_ids'])
            with torch.inference_mode(): logits=model(**inputs).logits[0]; probs=torch.softmax(logits,dim=-1)
            scores.append({'chunk_id':chunk.id,'entailment':float(probs[labels['entailment']]),'neutral':float(probs[labels['neutral']]),'contradiction':float(probs[labels['contradiction']]),'logits':logits.tolist(),'input_tokens_untruncated':count})
        e=max(p['entailment'] for p in scores); cmax=max(p['contradiction'] for p in scores)
        observed='not_supported' if cmax>=.5 and cmax>=e else ('supported' if e>=.5 else 'neutral')
        row={'id':case['id'],'expected':case['expected'],'observed':observed,'citation_valid':True,'pair_scores':scores,'provenance':'new_inference'}
    results.append(row)
    print(json.dumps({'id':row['id'],'expected':row['expected'],'observed':row['observed'],'provenance':row['provenance']}),flush=True)
report={'model':old['model'],'revision':old['revision'],'fixture_sha256':hash_bytes(fixture_path.read_bytes()),'runner_sha256':hash_bytes(Path(__file__).read_bytes()),'prior_stress_report_sha256':hash_bytes(old_path.read_bytes()),'prior_same_input_report_sha256':hash_bytes(prior_path.read_bytes()),'python':platform.python_version(),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'thresholds':{'entailment':.5,'contradiction':.5},'labels':labels,'planned':len(cases),'scored':len(results),'reused':len(reused),'new':len(cases)-len(reused),'seconds':round(time.monotonic()-started,3),'results':results}
(here/'base.json').write_text(json.dumps(report,indent=2)+'\n')
