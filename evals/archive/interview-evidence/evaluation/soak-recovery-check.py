import datetime,json,time,pathlib
import httpx,psutil
base=pathlib.Path('/home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation')
start=datetime.datetime.fromisoformat(json.loads((base/'soak-summary.json').read_text())['measurement_started_utc'])
api=[]
for p in psutil.process_iter(['pid','cmdline']):
 c=p.info['cmdline'] or []
 if 'uvicorn' in c and '18069' in c: api.append(p)
assert len(api)==1
p=api[0]; key=p.environ()['XAI_RAG_API_KEY']; fixture=pathlib.Path(p.cwd()).parent/'fixtures'
def save(name,response,latency,passed):
 now=datetime.datetime.now(datetime.timezone.utc)
 row={'utc':now.isoformat(),'elapsed_seconds':round((now-start).total_seconds(),3),'kind':'recovery_check','name':name,'status':response.status_code,'passed':passed,'latency_ms':round(latency*1000,2),'body':response.json()}
 with (base/'soak-intervals.jsonl').open('a') as f: f.write(json.dumps(row,separators=(',',':'))+'\n')
 print(name,'status',row['status'],'passed',passed,'latency_ms',row['latency_ms'])
with httpx.Client(trust_env=False,timeout=90) as client:
 h={'X-API-Key':key};t=time.monotonic();r=client.post('http://127.0.0.1:18069/query',headers=h,json={'query':'What is the refund deadline?','top_k':1});b=r.json();save('post_es_recovery_query',r,time.monotonic()-t,r.status_code==200 and '30 days' in b.get('answer','') and bool(b.get('sources')))
 t=time.monotonic();r=client.post('http://127.0.0.1:18069/ingest',headers=h,params={'source_path':str(fixture),'strategy':'fixed'});b=r.json();save('post_es_recovery_ingestion',r,time.monotonic()-t,r.status_code==200 and b.get('status')=='success' and b.get('ingested_files')==1 and b.get('total_chunks')==1 and b.get('parity_consistent') is True)
