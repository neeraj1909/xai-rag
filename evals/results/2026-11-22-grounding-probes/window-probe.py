import json,re
from pathlib import Path
import torch
from xai_rag.explainability.faithfulness import FaithfulnessChecker
orig=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text().split('Premise (exact):\n',1)[1].strip()
claim='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
fixture=json.loads(Path('evals/nli/stress-cases.json').read_text())['cases']
ids={'N01','N02','N03','N07','N08','N13','N14','N19','N20','N22','N23','N31','N32'}
cases=[{'id':x['id'],'premises':[x['premise']],'claim':x['hypothesis'],'expected':x['expected']} for x in fixture if x['id'] in ids]
cases += [
{'id':'e2e','premises':[orig],'claim':claim,'expected':'supported'},
{'id':'quoted-refuted','premises':[f'The following statement is false: {claim} RAG never uses a retrieval system.'],'claim':claim,'expected':'not_supported'},
{'id':'heading-refuted','premises':['# Incorrect statement\nThe fee is 10 dollars.\nThe correct fee is 20 dollars.'],'claim':'The fee is 10 dollars.','expected':'not_supported'},
{'id':'heading-not-authorized','premises':['# Prohibited actions\nPassword sharing is permitted.'],'claim':'Password sharing is permitted.','expected':'not_supported'},
{'id':'not-claim','premises':['It is not true that the fee is 10 dollars. The fee is 20 dollars.'],'claim':'The fee is 10 dollars.','expected':'not_supported'},
{'id':'two-sources-conflict','premises':['The fee is 10 dollars.','The fee is 20 dollars, not 10 dollars.'],'claim':'The fee is 10 dollars.','expected':'not_supported'}]
# This deliberately simple splitter is a hypothesis, not a safe parser.
def split(p):
 return [s.strip() for line in p.splitlines() for s in re.split(r'(?<=[.!?])\s+',line) if s.strip()]
def label(e,c):
 return 'not_supported' if c>=.5 and c>=e else ('supported' if e>=.5 else 'neutral')
priority=['e2e','quoted-refuted','heading-refuted','heading-not-authorized','not-claim','two-sources-conflict','N19','N20','N07','N08']
cases=sorted((c for c in cases if c['id'] in priority),key=lambda c:priority.index(c['id']))
checker=FaithfulnessChecker(); model,tok=checker._get_model_and_tokenizer()
for case in cases:
 ps=case['premises']; windows=[]
 for p in ps:
  ss=split(p)
  windows.extend(ss)
  windows.extend(ss[i]+'\n'+ss[i+1] for i in range(len(ss)-1))
 pairs=list(dict.fromkeys(ps+windows)); scores=[]
 for i in range(0,len(pairs),4):
  batch=pairs[i:i+4]; tokens=tok(batch,[case['claim']]*len(batch),return_tensors='pt',padding=True,truncation=True,max_length=512)
  with torch.no_grad(): probs=torch.softmax(model(**tokens).logits,dim=-1).tolist()
  scores.extend({'text':p,'entailment':row[0],'contradiction':row[2]} for p,row in zip(batch,probs))
 bytext={x['text']:x for x in scores}
 def aggregate(items):
  e=max(bytext[x]['entailment'] for x in items); c=max(bytext[x]['contradiction'] for x in items)
  return {'label':label(e,c),'entailment':e,'contradiction':c}
 print(json.dumps({'id':case['id'],'expected':case['expected'],'baseline':aggregate(ps),'windows':aggregate(windows),'window_scores':[bytext[x] for x in windows]},ensure_ascii=False),flush=True)
