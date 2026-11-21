"""Frozen pairwise synthetic question-aware judge experiment.
No provider calls without the explicit --live flag and coordinator lease.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def literals(path):
 values={}
 for n in ast.parse(path.read_text()).body:
  if isinstance(n, ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name) and target.id=='_SYSTEM': values['_SYSTEM']=ast.literal_eval(n.value)
  if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_SCHEMA': values['_SCHEMA']=ast.literal_eval(n.value)
 if set(values)!={'_SYSTEM','_SCHEMA'}: raise ValueError('missing required prompt/schema')
 return values

async def run(args):
 from dotenv import dotenv_values
 from openai import AsyncOpenAI
 key=dotenv_values('/home/neeraj/Documents/xai-rag/.env').get('XAI_RAG_OPENAI_API_KEY')
 if not key: raise SystemExit('Project-scoped credential unavailable')
 fixture=json.loads((HERE/'fixture.json').read_text())
 arms={name:literals(HERE/(name+'.py')) for name in ('baseline','candidate')}
 if arms['baseline']['_SCHEMA'] != arms['candidate']['_SCHEMA']: raise ValueError('Arms must share schema')
 if args.attempts < 2*len(fixture): raise SystemExit('Insufficient attempt budget')
 records=[]
 async with AsyncOpenAI(api_key=key,base_url='https://api.openai.com/v1',timeout=45,max_retries=0) as client:
  for row in fixture:
   for name,arm in arms.items():
    record={'id':row['id'],'arm':name,'expected_supported':row['expected_supported'],'expected_all_parts_addressed':row['expected_all_parts_addressed'],'model_requested':args.model}
    claims=row['claims'] if 'claims' in row else [row['claim']]
    payload={'question':row['question'],'evidence':row['evidence'],'claims':claims}
    start=time.monotonic()
    try:
     res=await client.chat.completions.create(model=args.model,max_completion_tokens=1024,response_format=arm['_SCHEMA'],messages=[{'role':'system','content':arm['_SYSTEM']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
     choice=res.choices[0]
     if choice.finish_reason!='stop' or choice.message.refusal or choice.message.content is None: raise ValueError('incomplete_or_refused')
     parsed=json.loads(choice.message.content)
     if set(parsed)!={'verdicts','all_parts_addressed'} or type(parsed['all_parts_addressed']) is not bool: raise ValueError('invalid_decision')
     vs=parsed['verdicts']
     if len(vs)!=len(claims) or any(set(v)!={'index','supported'} or type(v['index']) is not int or type(v['supported']) is not bool for v in vs) or {v['index'] for v in vs}!=set(range(len(claims))): raise ValueError('invalid_verdict')
     record.update({'individual_verdicts':vs,'observed_supported':all(v['supported'] for v in vs),'observed_all_parts_addressed':parsed['all_parts_addressed'],'model_returned':res.model,'usage':res.usage.model_dump() if res.usage else None,'status':'ok'})
    except Exception as exc:
     record.update({'status':'error','error_type':type(exc).__name__})
    record['elapsed_seconds']=round(time.monotonic()-start,4)
    records.append(record)
    with (HERE/'results.jsonl').open('a') as out: out.write(json.dumps(record)+'\n')
 meta={'model':args.model,'fixture_sha256':hashlib.sha256((HERE/'fixture.json').read_bytes()).hexdigest(),'prompt_sha256':{k:hashlib.sha256(v['_SYSTEM'].encode()).hexdigest() for k,v in arms.items()},'attempts':len(records),'results':records}
 (HERE/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
 for arm in arms:
  rows=[r for r in records if r['arm']==arm]
  errors=sum(r['status']!='ok' for r in rows)
  false_accept=sum(r.get('observed_supported') is True and not r['expected_supported'] for r in rows)
  false_reject=sum(r.get('observed_supported') is False and r['expected_supported'] for r in rows)
  coverage_error=sum(r['status']=='ok' and r['expected_all_parts_addressed'] is not None and r['observed_all_parts_addressed']!=r['expected_all_parts_addressed'] for r in rows)
  print(f'{arm}: attempts={len(rows)} errors={errors} false_accept={false_accept} false_reject={false_reject} coverage_error={coverage_error}')

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--live',action='store_true')
 parser.add_argument('--attempts',type=int,required=True)
 parser.add_argument('--model',default='gpt-6-luna')
 args=parser.parse_args()
 if not args.live: raise SystemExit('No provider calls: --live and coordinator lease are required')
 if (HERE/'results.jsonl').exists(): raise SystemExit('Existing results preserved; use a new directory')
 asyncio.run(run(args))
