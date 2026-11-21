import hashlib,http.server,json,subprocess,threading
from pathlib import Path
root=Path('/tmp/xai-security-prep-selftest')
policy=json.loads((root/'policy.json').read_text())
scopes={p['key_sha256']:p for p in policy['principals']}
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):
  if self.path.startswith('/ingest'):body={'status':'success','parity_consistent':True}
  else:
   inp=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   principal=scopes[hashlib.sha256(self.headers['X-API-Key'].encode()).hexdigest()]
   ev=[{'id':str(i),'content':'synthetic','source_file':s} for i,s in enumerate(principal['sources'])]
   with (root/'capture.jsonl').open('a') as f:f.write(json.dumps({'query':inp['query'],'evidence':ev})+'\n')
   body={'query':inp['query'],'answer':'Synthetic [0]','claims':[{'source_chunk_ids':['0']}],'sources':['0']}
  data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 p=subprocess.run(['python3','/tmp/xai-http-evaluation/run_security.py','--port',str(server.server_port),'--root',str(root),'--output','/tmp/xai-http-evaluation/security-selftest.json','--max-cases','2'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0,(p.stdout,p.stderr)
 out=json.load(open('/tmp/xai-http-evaluation/security-selftest.json'))
 assert len(out['queries'])==2 and all(r['all_case_sources_visible'] for r in out['queries'])
 print('PASS: mocked security HTTP queries use case-specific granted source mapping')
finally:server.shutdown();server.server_close()
