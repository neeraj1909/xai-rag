"""Direct relevance, safety and coverage probe; fixed fixture, no retries."""
import asyncio,hashlib,json,time
from pathlib import Path
from openai import AsyncOpenAI
from xai_rag.config import settings
from xai_rag.generation.grounding_judge import _SCHEMA,_SYSTEM,_Verdicts
ROOT=Path('/tmp/xai-safety-coverage-eval');FIXTURE=ROOT/'fixture.json';LEDGER=ROOT/'attempts.jsonl'
async def main():
 if LEDGER.exists():raise RuntimeError('Refusing to overwrite prior attempts')
 cases=json.loads(FIXTURE.read_text())
 if len(cases)!=25:raise RuntimeError('Unexpected fixture length')
 async with AsyncOpenAI(api_key=settings.openai_api_key,base_url='https://api.openai.com/v1',timeout=40,max_retries=0) as client:
  for case in cases:
   claims=case.get('claims') or [case['claim']]
   data={'question':case['question'],'evidence':case['evidence'],'claims':claims}
   row={'id':case['id'],'expected_release':bool(case['expected_supported'] and case.get('expected_all_parts_addressed') is not False),'status':'failed'};started=time.monotonic()
   try:
    response=await client.chat.completions.create(model='gpt-6-luna',max_completion_tokens=1024,response_format=_SCHEMA,messages=[{'role':'system','content':_SYSTEM},{'role':'user','content':json.dumps(data,ensure_ascii=False)}])
    choice=response.choices[0]
    if choice.finish_reason!='stop' or choice.message.refusal or not choice.message.content:raise ValueError('incomplete response')
    parsed=_Verdicts.model_validate_json(choice.message.content)
    if len(parsed.verdicts)!=len(claims) or {v.index for v in parsed.verdicts}!=set(range(len(claims))):raise ValueError('invalid verdict indices')
    release=parsed.all_parts_addressed and all(v.supported and v.relevant and v.safe_to_release for v in parsed.verdicts)
    usage=response.usage
    row.update(status='ok',release=release,model_returned=response.model,raw=json.loads(choice.message.content),input_tokens=usage.prompt_tokens if usage else None,output_tokens=usage.completion_tokens if usage else None)
   except Exception as exc:row['error_type']=type(exc).__name__
   row['elapsed_seconds']=round(time.monotonic()-started,3)
   with LEDGER.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 rows=[json.loads(x) for x in LEDGER.read_text().splitlines()]
 summary={'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),'model':'gpt-6-luna','attempt_cap':25,'actual_attempts':len(rows),'successes':sum(r['status']=='ok' for r in rows),'correct':sum(r.get('release')==r['expected_release'] for r in rows if r['status']=='ok'),'false_releases':[r['id'] for r in rows if r.get('release') is True and not r['expected_release']],'false_refusals':[r['id'] for r in rows if r.get('release') is False and r['expected_release']],'input_tokens':sum(r.get('input_tokens') or 0 for r in rows),'output_tokens':sum(r.get('output_tokens') or 0 for r in rows),'system_prompt_sha256':hashlib.sha256(_SYSTEM.encode()).hexdigest(),'schema_sha256':hashlib.sha256(json.dumps(_SCHEMA,sort_keys=True).encode()).hexdigest(),'label':'author-labeled synthetic and adaptively selected; not held-out or human reviewed'}
 summary['estimated_usd_standard']=(summary['input_tokens']*0.1+summary['output_tokens']*0.5)/1e6
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
asyncio.run(main())
