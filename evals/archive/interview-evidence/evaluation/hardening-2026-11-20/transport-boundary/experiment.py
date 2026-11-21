import asyncio, json, logging, socket, time, platform
from datetime import datetime, timezone
from pathlib import Path
import fastapi, httpx, uvicorn

out=Path('/tmp/xai-transport-boundary'); out.mkdir(exist_ok=True)
logger=logging.getLogger('httpcore'); logger.setLevel(logging.DEBUG)
handler=logging.FileHandler(out/'httpcore-debug.log', mode='w'); handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(message)s')); logger.addHandler(handler)
app=fastapi.FastAPI()
@app.get('/ready')
async def ready(): return {'status':'healthy'}
@app.post('/query')
async def query():
    await asyncio.sleep(.02)
    return {'answer':'synthetic'}
async def main():
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(128); port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',log_level='error',access_log=False,timeout_keep_alive=5,lifespan='off'))
    st=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if st.done(): raise RuntimeError('server failed')
        await asyncio.sleep(.02)
    result={'python':platform.python_version(),'httpx':httpx.__version__,'uvicorn':uvicorn.__version__,'fastapi':fastapi.__version__,'port':port,'started_utc':datetime.now(timezone.utc).isoformat(),'interval':5,'samples_per_arm':49,'pool':'httpx default 100 max / 20 keepalive / 5 sec expiry','server_keepalive':5,'query_every_seconds':30,'note':'Loopback stub, no data or providers. Different from original app/stores/faults.'}
    records=[]
    async def arm(name,headers):
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',timeout=120,headers=headers,trust_env=False) as c:
            start=time.monotonic()
            async def request(kind,seq):
                begin=time.monotonic(); row={'arm':name,'kind':kind,'seq':seq,'ts':datetime.now(timezone.utc).isoformat()}
                try:
                    r=await (c.get('/ready',timeout=6) if kind=='ready' else c.post('/query',json={'query':'synthetic'},timeout=90))
                    row.update(status=r.status_code,ok=r.status_code==200)
                except Exception as exc: row.update(ok=False,error_type=type(exc).__name__)
                row['latency_ms']=round((time.monotonic()-begin)*1000,2); records.append(row)
                if not row['ok']: print(json.dumps(row),flush=True)
            async def ready_loop():
                for i in range(49):
                    await asyncio.sleep(max(0,start+i*5-time.monotonic())); await request('ready',i)
            async def other_loop():
                for i in range(9):
                    await asyncio.sleep(max(0,start+i*30-time.monotonic())); await request('query',i)
            await asyncio.gather(ready_loop(),other_loop())
    try:
        await asyncio.gather(arm('pooled',{}),arm('close',{'Connection':'close'}))
    finally:
        server.should_exit=True; await st; sock.close()
        (out/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        result['counts']={arm:{kind:{'total':len(sub:=[r for r in records if r['arm']==arm and r['kind']==kind]),'failures':sum(not x['ok'] for x in sub)} for kind in ('ready','query')} for arm in ('pooled','close')}
        result['finished_utc']=datetime.now(timezone.utc).isoformat(); (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
asyncio.run(main())
