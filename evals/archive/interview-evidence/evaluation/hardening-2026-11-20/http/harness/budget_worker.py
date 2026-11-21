import asyncio,httpx
from openai import AsyncOpenAI
from call_budget import install
install()
async def handle(request):
 await asyncio.sleep(.02)
 return httpx.Response(200,json={'id':'x','object':'chat.completion','created':0,'model':'mock','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'ok'}}]})
async def run():
 async with AsyncOpenAI(api_key='dummy',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as c:
  try:await c.chat.completions.create(model='mock',messages=[{'role':'user','content':'x'}])
  except RuntimeError:pass
asyncio.run(run())
