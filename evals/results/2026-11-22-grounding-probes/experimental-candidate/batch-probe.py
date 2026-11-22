import json,re
from pathlib import Path
import torch
from xai_rag.explainability.faithfulness import FaithfulnessChecker
cases=json.loads(Path('evals/nli/stress-cases.json').read_text())['cases']
items=[]
for case in cases:
 p=case['premise']; hyp=case['hypothesis']
 bits=[s.strip() for line in p.splitlines() for s in re.split(r'(?<=[.!?])\s+',line) if s.strip()]
 windows=list(dict.fromkeys(bits+[bits[i]+'\n'+bits[i+1] for i in range(len(bits)-1)]))
 pairs=list(dict.fromkeys([p]+windows))
 for j,text in enumerate(pairs):items.append((case['id'],hyp,text,j==0))
checker=FaithfulnessChecker(); model,tok=checker._get_model_and_tokenizer(); labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
records={c['id']:{'id':c['id'],'expected':c['expected'],'scores':[]} for c in cases}
for i in range(0,len(items),8):
 batch=items[i:i+8]
 tokens=tok([x[2] for x in batch],[x[1] for x in batch],return_tensors='pt',padding=True,truncation=True,max_length=512)
 with torch.no_grad(): rows=torch.softmax(model(**tokens).logits,dim=-1).tolist()
 for item,row in zip(batch,rows):records[item[0]]['scores'].append({'e':row[labels['entailment']],'c':row[labels['contradiction']],'full':item[3]})
 print(f'{i+len(batch)}/{len(items)}',flush=True)
 Path('/tmp/xai-grounding-batch.json').write_text(json.dumps(list(records.values())))
