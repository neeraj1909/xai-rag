import json, statistics
from pathlib import Path
p=Path('/tmp/xai-verifier-research/luna'); cases=json.loads((p/'fixture.json').read_text()); run=json.loads((p/'results.json').read_text()); rows={r['id']:r for r in run['results']}
nli=json.loads(Path('/home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation/nli-stress-results.json').read_text()); nr={r['id']:r for r in nli['cases']}
def counts(items,fn):
 fa=[];fr=[]; err=[]; subtypes=[]
 for c in items:
  if 'error_type' in rows[c['id']]: err.append(c['id']); continue
  expected=c['expected']=='supported'; release=fn(c)
  if release and not expected: fa.append(c['id'])
  if expected and not release: fr.append(c['id'])
  expected_v={'supported':'supported','not_supported':'contradicted','neutral':'insufficient'}[c['expected']]
  if rows[c['id']]['verdict']!=expected_v: subtypes.append((c['id'],c['expected'],rows[c['id']]['verdict']))
 return {'n':len(items),'false_accepts':fa,'false_rejects':fr,'errors':err,'subtype_mismatches':subtypes}
common=cases[:42]; judge=lambda c:rows[c['id']]['verdict']=='supported'; nli_gate=lambda c:nr[c['id']]['observed']=='supported'
print('fixture SHA256',run['fixture_sha256']);print('judge common',counts(common,judge));print('judge adversarial',counts(cases[42:],judge));print('AND common',counts(common,lambda c:judge(c) and nli_gate(c)))
for i in ('N12','N14','A06'):
 r=rows[i];print(i, 'input',next(c for c in cases if c['id']==i),'output',r['raw']['choices'][0]['message']['content'])
usages=[r['raw']['usage'] for r in run['results'] if 'raw' in r]; probe=json.loads((p/'probe-N12.json').read_text())['raw']['usage']; usages.append(probe)
x=sum(u['prompt_tokens'] for u in usages);y=sum(u['completion_tokens'] for u in usages);cached=sum((u.get('prompt_tokens_details') or {}).get('cached_tokens',0) for u in usages);cache_write=sum((u.get('prompt_tokens_details') or {}).get('cache_write_tokens',0) for u in usages)
print('51 attempts: input',x,'output',y,'cached',cached,'cache write',cache_write,'estimated USD',((x-cached-cache_write)*.1+cached*.01+cache_write*.125+y*.5)/1e6)
l=[r['elapsed_seconds'] for r in run['results']];print('latency median',statistics.median(l),'min',min(l),'max',max(l));print('models',set(r['raw']['model'] for r in run['results'] if 'raw' in r))
