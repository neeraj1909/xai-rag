from contextlib import asynccontextmanager

from xai_rag.api import app as module
from xai_rag.models import XAIRAGResponse


class Chroma:
    def heartbeat(self):
        return 1


class Cluster:
    async def health(self, **kwargs):
        return {"status": "green", "timed_out": False}


class ES:
    cluster = Cluster()

    async def info(self):
        return {"version": {"number": "test"}}


class FakeService:
    async def query(self, request, *, event_sink=None):
        return XAIRAGResponse(
            query=request.query,
            answer="Synthetic response [fixture-1]",
            sources=["fixture-1"],
        )


@asynccontextmanager
async def isolated_lifespan(app):
    module.state.chroma_client = Chroma()
    module.state.es_client = ES()
    yield


module.app.router.lifespan_context = isolated_lifespan
module._build_service = FakeService
app = module.app
