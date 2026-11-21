"""Repeat live generator cases against an immutable repair snapshot; save every attempt."""
from __future__ import annotations
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from openai import AsyncOpenAI
from xai_rag.config import settings
from xai_rag.generation.generator import RAGGenerator, GenerationContractError
from xai_rag.models import RankedResult

HERE=Path(__file__).parent

def load_module(name, file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

gen_judge=load_module('generation_judge','generation-judge-smoke.py')
guard_judge=load_module('guard_judge','guardrail-judge-smoke.py')
guards=[json.loads(x) for x in (HERE/'guardrail-cases.jsonl').read_text().splitlines() if x.strip()]
cases=[c for c in guards if c['id'] in {'G01','G03','G04','G06'}]
baseline=json.loads((HERE/'generation-judge-baseline.json').read_text())
q06=next(x for x in baseline['rows'] if x['id']=='q06')
cases.append({'id':'q06','question':q06['question'],'reference_answer':q06['reference_answer'],'context':[{'id':x['id'],'text':x['content']} for x in q06['context']]})

async def main():
    client=AsyncOpenAI(api_key=settings.openai_api_key,base_url='https://api.openai.com/v1',timeout=60,max_retries=0)
    original=client.chat.completions.create; calls=[]
    async def capture(*args,**kwargs):
        r=await original(*args,**kwargs)
        calls.append({'model':r.model,'repair_prompt': 'Your previous response did not satisfy the citation contract.' in kwargs['messages'][0]['content'],'response':json.loads(r.choices[0].message.content),'usage':r.usage.model_dump() if r.usage else None})
        return r
    client.chat.completions.create=capture
    report={'source_snapshot_sha256':os.environ.get('XAI_SOURCE_HASH','4cef90bc211b37a069eb09a79ee7330716f8ff19c63ffca6a244814f94b7f1ec'),'generator_model_alias':'gpt-4o-mini','judge_model_alias':'gpt-4.1-mini','repeat_count':3,'judge_rubrics':{'guardrail':guard_judge.RUBRIC,'answer':gen_judge.JUDGE_SYSTEM},'rows':[]}
    try:
        for c in cases:
            for iteration in range(1,4):
                row={'id':c['id'],'iteration':iteration,'question':c['question'],'context':c['context'],'reference_answer':c['reference_answer']}
                chunks=[RankedResult(id=x['id'],content=x['text']) for x in c['context']]
                before=len(calls)
                try:
                    result=await RAGGenerator(client=SimpleNamespace(chat=client.chat),model='gpt-4o-mini').generate(c['question'],chunks)
                    row.update({'answer':result.answer,'claims':[x.model_dump() for x in result.claims],'sources':result.sources})
                    row['provider_calls']=calls[before:]
                    if c['id']=='q06':
                        payload={'question':c['question'],'reference_answer':c['reference_answer'],'retrieved_context':c['context'],'answer':result.answer};rubric=gen_judge.JUDGE_SYSTEM;schema=gen_judge.SCORE_SCHEMA
                    else:
                        payload={'question':c['question'],'context':c['context'],'reference_answer':c['reference_answer'],'expected_behavior':c['expected_behavior'],'checks':c['checks'],'answer':result.answer,'sources':result.sources};rubric=guard_judge.RUBRIC;schema=guard_judge.SCHEMA
                    judge=await original(model='gpt-4.1-mini',temperature=0,max_tokens=500,response_format=schema,messages=[{'role':'system','content':rubric},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
                    row.update({'judge_model':judge.model,'judge_usage':judge.usage.model_dump() if judge.usage else None,'judge':json.loads(judge.choices[0].message.content)})
                except Exception as e:
                    row['provider_calls']=calls[before:]
                    row['error']={'type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
                    if isinstance(e,GenerationContractError):row['error']['message']=str(e)
                report['rows'].append(row)
                (HERE/os.environ.get('XAI_RESULTS_FILE','focused-repair-results.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
                print(c['id'],iteration,'success' if 'answer' in row else row['error']['type'],'calls',len(row['provider_calls']),'repair',any(x['repair_prompt'] for x in row['provider_calls']),flush=True)
    finally:await client.close()

if __name__=='__main__':asyncio.run(main())
