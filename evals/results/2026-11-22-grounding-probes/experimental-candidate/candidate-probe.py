import json,time
from pathlib import Path
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.models import Claim,RankedResult
orig=Path('evals/results/2026-11-22-docker-e2e/nli-probe.txt').read_text().split('Premise (exact):\n',1)[1].strip()
text='RAG couples a retrieval system with a language model so the model generates answers grounded in fetched evidence rather than relying solely on parametric memory.'
cases=[('e2e',[orig],text,'supported'),('refuted',[f'The following statement is false: {text} RAG never uses a retrieval system.'],text,'not_supported'),('heading-refuted',['# Incorrect statement\nThe fee is 10 dollars.\nThe correct fee is 20 dollars.'],'The fee is 10 dollars.','not_supported'),('heading-prohibited',['# Prohibited actions\nPassword sharing is permitted.'],'Password sharing is permitted.','not_supported'),('negated',['It is not true that the fee is 10 dollars. The fee is 20 dollars.'],'The fee is 10 dollars.','not_supported'),('conflict',['The fee is 10 dollars.','The fee is 20 dollars, not 10 dollars.'],'The fee is 10 dollars.','not_supported'),('conditional',['If the discount is approved, the fee is 10 dollars. The discount was not approved.'],'The fee is 10 dollars.','not_supported'),('separate',['Package A costs 12 dollars. Package B costs 21 dollars.'],'Package A costs 12 dollars.','supported')]
c=FaithfulnessChecker()
for ident,premises,claim,expected in cases:
 t=time.monotonic(); ids=[str(i) for i in range(len(premises))]; out=c._check_single_claim(Claim(text=claim,source_chunk_ids=ids),[RankedResult(id=str(i),content=p) for i,p in enumerate(premises)])
 print(json.dumps({'id':ident,'expected':expected,'verdict':out.verdict.value,'entailment':out.nli_entailment,'contradiction':out.nli_contradiction,'elapsed_s':round(time.monotonic()-t,1)}),flush=True)
