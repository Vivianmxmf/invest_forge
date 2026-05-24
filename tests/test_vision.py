"""Tests for the multimodal vision feature — all offline / network-free.

Coverage:
  * ChatMessage.images default + backward-compat
  * _to_openai_messages: empty-images path (plain str) vs vision-parts path
  * FakeLLMClient handles image-bearing messages + dispatches VISION_PROMPT
  * make_vision_node: client=None, no images, fake client + data-url
  * make_researcher_node: vision_analysis present vs absent (no-regression lock)
  * build_vision_client: None when not local / no model; client when configured
  * API: invalid image → 400; valid base64 PNG → 200 with final_recommendation
  * graph end-to-end: vision_client + input_images → vision_analysis flows into memo
"""
from __future__ import annotations

import base64
from io import BytesIO

import pytest

from invest_forge.agents.graph import GraphDeps, run_pipeline_inline
from invest_forge.agents.nodes import make_researcher_node, make_vision_node
from invest_forge.agents.prompts import RESEARCHER_PROMPT, VISION_PROMPT
from invest_forge.agents.state import empty_state
from invest_forge.common.config import LLMConfig
from invest_forge.llm.client import ChatMessage, build_vision_client
from invest_forge.llm.fake import FakeLLMClient
from invest_forge.llm.openai_client import _to_openai_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_png() -> bytes:
    """Return a minimal valid 2×2 PNG image."""
    from PIL import Image

    buf = BytesIO()
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


_TINY_PNG = _make_tiny_png()
_TINY_DATA_URL = _make_data_url(_TINY_PNG)


# ---------------------------------------------------------------------------
# §1  ChatMessage.images
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chat_message_default_images_empty():
    """Existing construction (role + content only) must have images == ()."""
    msg = ChatMessage(role="user", content="hello")
    assert msg.images == ()


@pytest.mark.unit
def test_chat_message_with_images():
    msg = ChatMessage(role="user", content="describe", images=("data:image/png;base64,abc",))
    assert len(msg.images) == 1
    assert msg.images[0].startswith("data:")


# ---------------------------------------------------------------------------
# §2  _to_openai_messages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_openai_messages_no_images_plain_string():
    """When no message has images the output is byte-identical to the old format."""
    msgs = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Tell me about AAPL"),
    ]
    result = _to_openai_messages(msgs)
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me about AAPL"},
    ]


@pytest.mark.unit
def test_to_openai_messages_with_images_vision_format():
    """Messages carrying images must use the OpenAI vision multipart format."""
    data_url = "data:image/png;base64,iVBOR"
    msgs = [
        ChatMessage(role="user", content="Analyse this chart", images=(data_url,)),
    ]
    result = _to_openai_messages(msgs)
    assert len(result) == 1
    content = result[0]["content"]
    # Must be a list of parts, not a plain string.
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Analyse this chart"}
    assert content[1] == {"type": "image_url", "image_url": {"url": data_url}}


@pytest.mark.unit
def test_to_openai_messages_mixed():
    """Text-only and vision messages can coexist; each is handled independently."""
    data_url = "data:image/png;base64,XYZ"
    msgs = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="img msg", images=(data_url,)),
        ChatMessage(role="assistant", content="reply"),
    ]
    result = _to_openai_messages(msgs)
    assert result[0] == {"role": "system", "content": "sys"}
    assert isinstance(result[1]["content"], list)
    assert result[2] == {"role": "assistant", "content": "reply"}


# ---------------------------------------------------------------------------
# §3  FakeLLMClient with image-bearing messages + VISION_PROMPT dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fake_llm_handles_image_bearing_message():
    """FakeLLMClient must not crash when a message carries images."""
    fake = FakeLLMClient()
    msg = ChatMessage(role="user", content="plain text", images=(_TINY_DATA_URL,))
    # Should complete without error; content is still matched on text.
    resp = fake.complete([msg])
    assert resp.text  # non-empty fallback or matched response


@pytest.mark.unit
def test_fake_llm_dispatches_vision_prompt():
    """VISION_PROMPT must hit the [视觉]/[Vision] pattern in FakeLLMClient."""
    fake = FakeLLMClient()
    msg = ChatMessage(role="user", content=VISION_PROMPT, images=(_TINY_DATA_URL,))
    resp = fake.complete([msg])
    # The canned response must mention image analysis keywords.
    assert "图像" in resp.text or "K 线" in resp.text or "财务" in resp.text


# ---------------------------------------------------------------------------
# §4  make_vision_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vision_node_client_none_returns_empty():
    """make_vision_node(None) → always {}."""
    node = make_vision_node(None)
    state = empty_state()
    state["input_images"] = [_TINY_DATA_URL]
    assert node(state) == {}


@pytest.mark.unit
def test_vision_node_no_images_returns_empty():
    """make_vision_node with a real client but no images → {}."""
    fake = FakeLLMClient()
    node = make_vision_node(fake)
    state = empty_state()
    # input_images is empty (default from empty_state)
    assert node(state) == {}


@pytest.mark.unit
def test_vision_node_with_image_populates_vision_analysis():
    """make_vision_node with fake client + data-url → sets non-empty vision_analysis."""
    fake = FakeLLMClient()
    node = make_vision_node(fake)
    state = empty_state()
    state["input_images"] = [_TINY_DATA_URL]
    patch = node(state)
    assert "vision_analysis" in patch
    assert patch["vision_analysis"]  # non-empty


@pytest.mark.unit
def test_vision_node_message_carries_image():
    """The ChatMessage sent to the VLM must include the data-url in images."""
    captured: list[list[ChatMessage]] = []

    class CapturingFake(FakeLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[override]
            captured.append(list(messages))
            return super().complete(messages, **kwargs)

    node = make_vision_node(CapturingFake())
    state = empty_state()
    state["input_images"] = [_TINY_DATA_URL]
    node(state)
    assert captured
    msg = captured[0][-1]  # last message
    assert _TINY_DATA_URL in msg.images


# ---------------------------------------------------------------------------
# §5  make_researcher_node — no-regression invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_researcher_node_no_vision_prompt_identical():
    """When vision_analysis is absent/empty, the researcher prompt is exactly
    RESEARCHER_PROMPT.format(...) with no extra content."""
    import json

    captured_prompts: list[str] = []

    class PromptCapture(FakeLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[override]
            captured_prompts.append(messages[-1].content)
            return super().complete(messages, **kwargs)

    node = make_researcher_node(PromptCapture())
    state = empty_state()
    state["fundamental_data"] = {}
    state["macro_context"] = {}
    state["news_items"] = []
    state["rag_evidence"] = []
    # vision_analysis is empty (default)
    node(state)

    expected = RESEARCHER_PROMPT.format(
        fundamentals=json.dumps({}, ensure_ascii=False, indent=2),
        macro=json.dumps({}, ensure_ascii=False, indent=2),
        news_summary="(无相关新闻)",
        rag_evidence="(无 RAG 命中)",
    )
    assert captured_prompts[0] == expected, "Researcher prompt must be byte-identical when no vision"


@pytest.mark.unit
def test_researcher_node_appends_vision_analysis():
    """When vision_analysis is non-empty the researcher prompt gets the extra block."""
    import json

    captured_prompts: list[str] = []

    class PromptCapture(FakeLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[override]
            captured_prompts.append(messages[-1].content)
            return super().complete(messages, **kwargs)

    node = make_researcher_node(PromptCapture())
    state = empty_state()
    state["fundamental_data"] = {}
    state["macro_context"] = {}
    state["news_items"] = []
    state["rag_evidence"] = []
    state["vision_analysis"] = "图像分析: 营收 432 亿元."
    node(state)

    assert "图像分析证据" in captured_prompts[0]
    assert "图像分析: 营收 432 亿元." in captured_prompts[0]


# ---------------------------------------------------------------------------
# §6  build_vision_client
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_vision_client_none_when_fake_provider():
    cfg = LLMConfig(
        provider="fake",
        model="fake-llm",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url=None,
        local_model=None,
        local_vision_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    assert build_vision_client(cfg) is None


@pytest.mark.unit
def test_build_vision_client_none_when_local_no_vision_model():
    cfg = LLMConfig(
        provider="local",
        model="Qwen/Qwen2.5-7B-Instruct",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url="http://localhost:8000/v1",
        local_model="Qwen/Qwen2.5-7B-Instruct",
        local_vision_model=None,
    )
    assert build_vision_client(cfg) is None


@pytest.mark.unit
def test_build_vision_client_returns_client_when_local_vision_model_set():
    """build_vision_client returns an OpenAILLMClient when properly configured.

    We do NOT call the network — just assert the type and attributes.
    """
    from unittest.mock import patch

    from invest_forge.llm.openai_client import OpenAILLMClient

    cfg = LLMConfig(
        provider="local",
        model="Qwen/Qwen2.5-7B-Instruct",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url="http://localhost:8000/v1",
        local_model="Qwen/Qwen2.5-7B-Instruct",
        local_vision_model="Qwen/Qwen2.5-VL-7B-Instruct",
        local_vision_base_url="http://localhost:8002/v1",
    )
    # Patch OpenAI SDK so __post_init__ doesn't fail without the package.
    with patch("invest_forge.llm.openai_client.OpenAILLMClient.__post_init__"):
        client = build_vision_client(cfg)

    assert client is not None
    assert isinstance(client, OpenAILLMClient)
    assert client.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert client.base_url == "http://localhost:8002/v1"


@pytest.mark.unit
def test_build_vision_client_falls_back_to_local_base_url_when_vision_base_url_absent():
    """When local_vision_base_url is None, fall back to local_base_url."""
    from unittest.mock import patch

    from invest_forge.llm.openai_client import OpenAILLMClient

    cfg = LLMConfig(
        provider="local",
        model="Qwen/Qwen2.5-7B-Instruct",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url="http://localhost:8000/v1",
        local_model="Qwen/Qwen2.5-7B-Instruct",
        local_vision_model="Qwen/Qwen2.5-VL-7B-Instruct",
        local_vision_base_url=None,
    )
    with patch("invest_forge.llm.openai_client.OpenAILLMClient.__post_init__"):
        client = build_vision_client(cfg)

    assert isinstance(client, OpenAILLMClient)
    assert client.base_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# §7  API tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_api_invalid_image_returns_400():
    """A URL pointing to a private/loopback host must return HTTP 400."""
    from fastapi.testclient import TestClient

    from invest_forge.api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/analyze",
            json={
                "ts_code": "688981.SH",
                "images": [{"kind": "url", "value": "https://127.0.0.1/evil.png"}],
            },
        )
    assert resp.status_code == 400
    # The body must not expose internal details (just a generic message).
    assert "invalid image input" in resp.json().get("detail", "").lower()


@pytest.mark.unit
def test_api_valid_base64_image_returns_200(monkeypatch):
    """A valid base64 PNG must pass validation and return a 200 recommendation."""
    from fastapi.testclient import TestClient

    # Encode a real tiny PNG.
    b64 = base64.b64encode(_TINY_PNG).decode("ascii")

    from invest_forge.api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/analyze",
            json={
                "ts_code": "688981.SH",
                "images": [{"kind": "base64", "value": b64, "mime": "image/png"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "final_recommendation" in data
    assert data["final_recommendation"]


# ---------------------------------------------------------------------------
# §8  Graph end-to-end: vision_client + input_images → vision_analysis in memo
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graph_vision_flows_into_research_memo(
    fake_data, lexicon_sentiment, tiny_retriever
):
    """When vision_client and input_images are provided, vision_analysis must be
    populated and the researcher node must receive the 图像分析证据 block in its prompt."""
    captured_prompts: list[str] = []

    class CapturingFake(FakeLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[override]
            captured_prompts.append(messages[-1].content)
            return super().complete(messages, **kwargs)

    vision_fake = FakeLLMClient()
    main_fake = CapturingFake()

    deps = GraphDeps(
        llm=main_fake,
        data_provider=fake_data,
        sentiment=lexicon_sentiment,
        retriever=tiny_retriever,
        vision_client=vision_fake,
    )

    state = run_pipeline_inline(
        deps,
        ts_code="TEST.SH",
        input_images=[_TINY_DATA_URL],
    )

    # Vision analysis must have been computed and stored in state.
    assert state.get("vision_analysis"), "vision_analysis should be non-empty"

    # The prompt sent to the researcher must contain the vision evidence block.
    # (research_memo is the LLM *output*, not the prompt — we check the prompt.)
    researcher_prompt = next(
        (p for p in captured_prompts if "[研究員]" in p or "[研究员]" in p or "[Researcher]" in p),
        captured_prompts[0] if captured_prompts else "",
    )
    assert "图像分析证据" in researcher_prompt, (
        "Researcher prompt must include the vision evidence block when vision_analysis is set"
    )


@pytest.mark.unit
def test_graph_no_vision_client_unchanged(fake_data, lexicon_sentiment, tiny_retriever):
    """Without a vision_client the graph behaves exactly as before — no vision_analysis."""
    deps = GraphDeps(
        llm=FakeLLMClient(),
        data_provider=fake_data,
        sentiment=lexicon_sentiment,
        retriever=tiny_retriever,
        vision_client=None,
    )
    state = run_pipeline_inline(deps, ts_code="TEST.SH")
    # vision_analysis should be empty / absent.
    assert not state.get("vision_analysis")
