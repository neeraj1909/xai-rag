"""Direct synthetic judge-coverage experiment; maximum 24 attempts, no retry."""
import asyncio
import hashlib
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from xai_rag.config import settings
from xai_rag.generation.grounding_judge import _SCHEMA, _SYSTEM, _Verdicts

ROOT = Path('/tmp/xai-coverage-eval')
FIXTURE = ROOT / 'cases.json'
MODEL = 'gpt-6-luna'
MAX_ATTEMPTS = 24


async def main():
    cases = json.loads(FIXTURE.read_text())
    ledger = ROOT / 'attempts-official.jsonl'
    if ledger.exists() and ledger.stat().st_size:
        raise RuntimeError('Refusing to overwrite prior attempts')
    attempts = 0
    async with AsyncOpenAI(api_key=settings.openai_api_key, base_url='https://api.openai.com/v1', timeout=40, max_retries=0) as client:
        for repeat in range(2):
            for case in cases:
                if attempts >= MAX_ATTEMPTS:
                    break
                attempts += 1
                data = {'question': case['question'],
                        'evidence': [{'id':'source-1', 'text': case['evidence']}],
                        'claims': [{'index':0, 'claim':case['claim'], 'cited_chunk_id':'source-1'}]}
                result = {'case_id':case['id'], 'repeat':repeat + 1,
                          'expected_all_parts':case['expected_all_parts'], 'model_requested':MODEL,
                          'status':'failed'}
                started = time.monotonic()
                try:
                    response = await client.chat.completions.create(
                        model=MODEL, max_completion_tokens=1024, response_format=_SCHEMA,
                        messages=[{'role':'system','content':_SYSTEM},
                                  {'role':'user','content':json.dumps(data,ensure_ascii=False)}])
                    choice = response.choices[0]
                    if choice.finish_reason != 'stop' or choice.message.refusal or not choice.message.content:
                        raise ValueError('incomplete response')
                    parsed = _Verdicts.model_validate_json(choice.message.content)
                    if len(parsed.verdicts) != 1 or parsed.verdicts[0].index != 0:
                        raise ValueError('unexpected verdict indices')
                    usage=response.usage
                    result.update(status='ok', model_returned=response.model,
                                  supported=parsed.verdicts[0].supported,
                                  all_parts_addressed=parsed.all_parts_addressed,
                                  prompt_tokens=usage.prompt_tokens if usage else None,
                                  completion_tokens=usage.completion_tokens if usage else None,
                                  raw_response=json.loads(choice.message.content))
                except Exception as exc:
                    result['error_type'] = type(exc).__name__
                result['elapsed_seconds']=round(time.monotonic()-started, 3)
                with ledger.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
    rows=[json.loads(line) for line in ledger.read_text().splitlines()]
    out={'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
         'model': MODEL,'attempt_limit':MAX_ATTEMPTS, 'actual_attempts':len(rows),
         'system_prompt_sha256':hashlib.sha256(_SYSTEM.encode()).hexdigest(),
         'schema_sha256':hashlib.sha256(json.dumps(_SCHEMA,sort_keys=True).encode()).hexdigest(),
         'successes':sum(r['status']=='ok' for r in rows),
         'coverage_correct':sum(r.get('all_parts_addressed') == r['expected_all_parts']
                                for r in rows if r['status']=='ok'),
         'input_tokens':sum(r.get('prompt_tokens') or 0 for r in rows),
         'output_tokens':sum(r.get('completion_tokens') or 0 for r in rows),
         'source_note':'synthetic author-labeled; adaptively selected observed case, no human review'}
    out['estimated_usd_standard'] = (out['input_tokens'] * 0.1 + out['output_tokens'] * 0.5) / 1000000
    (ROOT/'summary-official.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))

asyncio.run(main())
