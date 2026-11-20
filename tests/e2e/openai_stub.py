"""Deterministic, network-local OpenAI chat-completions substitute for E2E tests."""

from __future__ import annotations

import html
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CHUNK_PATTERN = re.compile(
    r'<retrieved_chunk id="([^"\n]+)">\s*(.*?)\s*</retrieved_chunk>',
    re.DOTALL,
)
_FIXTURE_CLAIM = (
    "RAG couples a retrieval system with a language model so the model generates answers "
    "grounded in fetched evidence rather than relying solely on parametric memory."
)


def _first_sentence(text: str) -> str | None:
    """Select an intact prose sentence, not a Markdown heading or fragment."""
    in_code = False
    for line in html.unescape(text).splitlines():
        line = line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith(("#", "|", "-", "*", ">")):
            continue
        match = re.match(r"([A-Z][^\n]*?[.!?])(?:\s|$)", line)
        if match and len(match.group(1).split()) >= 4:
            return match.group(1)
    return None


class Handler(BaseHTTPRequestHandler):
    """Serve only the tiny surface the OpenAI client exercises."""

    server_version = "xai-rag-e2e-stub/1"

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        self._write_json(404, {"error": {"type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"type": "not_found"}})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 1_000_000:
            self._write_json(400, {"error": {"type": "invalid_request"}})
            return

        request = json.loads(self.rfile.read(content_length))
        user_text = next(
            (
                str(message.get("content", ""))
                for message in request.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
        schema = request.get("response_format", {}).get("json_schema", {}).get("name")
        chosen = next(
            (
                (match.group(1), sentence)
                for match in _CHUNK_PATTERN.finditer(user_text)
                if (sentence := _first_sentence(match.group(2)))
            ),
            None,
        )
        if schema == "grounding_verdicts":
            data = json.loads(user_text)
            response_content = {
                "verdicts": [
                    {
                        "index": claim["index"],
                        "supported": True,
                        "relevant": True,
                        "safe_to_release": True,
                    }
                    for claim in data["claims"]
                ],
                "all_parts_addressed": True,
            }
        elif chosen is None:
            response_content = {
                "answer": "I don't have enough information to answer this question.",
                "claims": [],
                "sources": [],
            }
        else:
            chunk_id, claim = chosen
            if claim == _FIXTURE_CLAIM and any(
                message.get("role") == "system"
                and "Your previous response could not be verified."
                in str(message.get("content", ""))
                for message in request.get("messages", [])
            ):
                claim = "RAG combines retrieval and language models."
            response_content = {
                "answer": f"{claim} [{chunk_id}]",
                "claims": [{"text": claim, "source_chunk_ids": [chunk_id]}],
                "sources": [chunk_id],
            }

        self._write_json(
            200,
            {
                "id": "chatcmpl-xai-rag-e2e",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "deterministic-local-stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(response_content),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        """Avoid logging prompts or request headers from the test boundary."""
        del format, args


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
