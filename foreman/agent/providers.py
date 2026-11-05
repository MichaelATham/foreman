"""
Model providers. The agent loop speaks one internal dialect (Anthropic-style
content blocks: text / image / tool_use / tool_result); each provider turns
that into its own API and back.

    provider.stream(system, messages, tools) -> context manager yielding
        objects with .type == "text" and .text for streamed prose, and
        .get_final_message() -> object with .content (blocks) and .stop_reason

Selection (see `make_provider`): FOREMAN_PROVIDER=anthropic|openai, otherwise
whichever API key is set (Anthropic first).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-6.1-sol"}


def make_provider(model: str | None = None):
    name = (os.environ.get("FOREMAN_PROVIDER") or "").lower()
    if not name:
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "anthropic"
    model = model or os.environ.get("FOREMAN_MODEL") or DEFAULT_MODELS[name]
    if name == "openai":
        return OpenAIProvider(model)
    return AnthropicProvider(model)


# --------------------------------------------------------------------------- #
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, client=None):
        self.model = model
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def stream(self, system: str, messages: list[dict], tools: list[dict]):
        return self.client.messages.stream(model=self.model, max_tokens=8000, system=system,
                                           tools=tools, messages=messages)


# --------------------------------------------------------------------------- #
class _OpenAIResponsesStream:
    """Consumes a Responses API event stream: text deltas are yielded as they
    arrive; function calls are assembled from output_item events."""

    def __init__(self, provider, kwargs):
        self._p = provider
        self._kwargs = kwargs
        self._text: list[str] = []
        self._calls: dict[int, dict] = {}
        self._response_id = None
        self._done = False

    def __enter__(self):
        self._events = self._p.client.responses.create(stream=True, **self._kwargs)
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        if self._done:
            return
        for ev in self._events:
            t = getattr(ev, "type", "")
            if t == "response.output_text.delta":
                self._text.append(ev.delta)
                yield SimpleNamespace(type="text", text=ev.delta)
            elif t == "response.output_item.added" and getattr(ev.item, "type", "") == "function_call":
                self._calls[ev.output_index] = {"id": ev.item.call_id, "name": ev.item.name, "args": [ev.item.arguments or ""]}
            elif t == "response.function_call_arguments.delta":
                slot = self._calls.setdefault(ev.output_index, {"id": None, "name": None, "args": []})
                slot["args"].append(ev.delta)
            elif t == "response.output_item.done" and getattr(ev.item, "type", "") == "function_call":
                # authoritative final arguments
                self._calls[ev.output_index] = {"id": ev.item.call_id, "name": ev.item.name, "args": [ev.item.arguments or "{}"]}
            elif t == "response.completed":
                self._response_id = ev.response.id
            elif t in ("response.failed", "response.incomplete", "error"):
                msg = getattr(getattr(ev, "response", None), "error", None) or getattr(ev, "error", None) or t
                raise RuntimeError(f"OpenAI response {t}: {msg}")
        self._done = True

    def get_final_message(self):
        for _ in self:
            pass
        blocks = []
        text = "".join(self._text)
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
        for idx in sorted(self._calls):
            c = self._calls[idx]
            raw = "".join(c["args"]) or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
            blocks.append(SimpleNamespace(type="tool_use", id=c["id"] or f"call_{idx}", name=c["name"], input=args))
        # remember server-side state so reasoning items survive between turns
        self._p._last_response_id = self._response_id
        self._p._sent_messages = self._p._pending_sent
        return SimpleNamespace(content=blocks, stop_reason="tool_use" if self._calls else "end_turn")


class OpenAIProvider:
    """OpenAI via the Responses API (required for tool calling on GPT-6 Astra /
    GPT-6.1 Sol; also works for gpt-4.1 etc.). The conversation is chained with
    `previous_response_id`, so only the messages added since the last call are
    sent and the model's reasoning items persist server-side."""
    name = "openai"

    def __init__(self, model: str, client=None, reasoning_effort: str | None = None):
        self.model = model
        if client is None:
            import openai
            client = openai.OpenAI()
        self.client = client
        self.reasoning_effort = reasoning_effort or os.environ.get("FOREMAN_REASONING", "medium")
        self._last_response_id = None
        self._sent_messages = 0
        self._pending_sent = 0

    def stream(self, system: str, messages: list[dict], tools: list[dict]):
        # Only chain when the history we already sent is a prefix of the current one.
        chain = self._last_response_id is not None and self._sent_messages <= len(messages)
        # When chaining, the server already holds our previous assistant output
        # (text, function calls, reasoning), so send only the new user-side items.
        new_msgs = [m for m in messages[self._sent_messages:] if m["role"] == "user"] if chain else messages
        kwargs = {
            "model": self.model,
            "instructions": system,
            "input": self._convert(new_msgs),
            "tools": [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"], "strict": False} for t in tools],
            "max_output_tokens": 8000,
            "store": True,
        }
        if chain:
            kwargs["previous_response_id"] = self._last_response_id
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        self._pending_sent = len(messages)
        return _OpenAIResponsesStream(self, kwargs)

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        """Internal blocks -> Responses API input items."""
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text = "".join(b["text"] for b in content if b.get("type") == "text")
                if text:
                    out.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
                for b in content:
                    if b.get("type") == "tool_use":
                        out.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                    "arguments": json.dumps(b["input"])})
                continue
            parts: list[dict] = []
            for b in content:
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "input_text", "text": b["text"]})
                elif t == "image":
                    parts.append(_img_part(b))
                elif t == "tool_result":
                    if parts:
                        out.append({"role": "user", "content": parts})
                        parts = []
                    texts, imgs = [], []
                    for rb in b["content"]:
                        if rb.get("type") == "text":
                            texts.append(rb["text"])
                        elif rb.get("type") == "image":
                            imgs.append(_img_part(rb))
                    out.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                                "output": "\n".join(texts) or "(no text)"})
                    if imgs:
                        out.append({"role": "user", "content": [{"type": "input_text",
                                    "text": f"[image attached to the result of tool call {b['tool_use_id']}]"}] + imgs})
            if parts:
                out.append({"role": "user", "content": parts})
        return out


def _img_part(block: dict) -> dict:
    src = block["source"]
    return {"type": "input_image", "image_url": f"data:{src['media_type']};base64,{src['data']}", "detail": "high"}
