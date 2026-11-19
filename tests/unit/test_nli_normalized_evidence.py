"""The NLI premise decodes only complete, model-visible prompt text."""

import html

import pytest

from xai_rag.generation.generator import _format_context
from xai_rag.models import RankedResult

pytestmark = pytest.mark.unit


def test_unescape_once_preserves_only_bounded_parent_source_prefix() -> None:
    parent = "Parent x < 5 & literal &lt; and &#60; > end café 🚀"
    chunk = RankedResult(id="id", content="child NOT shown", context_content=parent)
    omitted = RankedResult(id="later", content="other evidence")
    seen_parent = False
    for budget in range(1, 180):
        prompt, snapshots = _format_context([chunk, omitted], budget)
        assert len(prompt) <= budget
        if not snapshots or snapshots[0].id != "id":
            continue
        literal = snapshots[0].content
        normalized = html.unescape(literal)
        assert parent.startswith(normalized)
        assert html.escape(normalized, quote=False) == literal
        assert snapshots[0].context_content is None
        if normalized == parent:
            seen_parent = True
            assert "&lt;" in normalized
            assert "&#60;" in normalized
            break
    assert seen_parent
