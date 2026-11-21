import http.server,json,subprocess,threading,hashlib
from pathlib import Path
root=Path('/tmp/xai-eval-prep-selftest')
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_POST(self):
  if self.path.startswith('/ingest'):
   body={'status':'success','parity_consistent':True}
  else:
   inp=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   with (root/'capture.jsonl').open('a') as f:f.write(json.dumps({'query':inp['query'],'evidence':[{'id':'chunk','content':'synthetic','source_file':'rail_base.txt'}]})+'\n')
   body={'query':inp['query'],'answer':'Synthetic [chunk]','claims':[{'source_chunk_ids':['chunk']}],'sources':['chunk'],'retrieval_explanations':[{'chunk_id':'chunk'}]}
  data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
try:
 p=subprocess.run(['python3','/tmp/xai-http-evaluation/run_http.py','--port',str(server.server_port),'--root',str(root),'--output','/tmp/xai-http-evaluation/http-selftest.json','--cases','H01','H04'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0, (p.stdout,p.stderr)
 out=json.load(open('/tmp/xai-http-evaluation/http-selftest.json'))
 assert len(out['ingestion'])==25 and len(out['queries'])==2 and all(x['evidence_capture_matches']==1 for x in out['queries'])
 print('PASS: HTTP runner ingests 25 documents and maps captured evidence for 2 mocked cases')
finally:server.shutdown();server.server_close()
