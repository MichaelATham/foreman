"""The OpenAI Responses-API adapter, exercised with a fake client so the input
conversion, streaming assembly and previous_response_id chaining are tested
offline."""
import json
from types import SimpleNamespace as NS

from foreman.agent.providers import OpenAIProvider
from foreman.agent.tools import TOOLS


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["stream"] is True

        def events():
            yield NS(type="response.output_text.delta", delta="Plan: ")
            yield NS(type="response.output_text.delta", delta="build it.")
            yield NS(type="response.output_item.added", output_index=1,
                     item=NS(type="function_call", call_id="call_1", name="build", arguments=""))
            yield NS(type="response.function_call_arguments.delta", output_index=1, delta='{"script": "part = b')
            yield NS(type="response.function_call_arguments.delta", output_index=1, delta='ox((1,1,1))", "note": "x"}')
            yield NS(type="response.output_item.done", output_index=1,
                     item=NS(type="function_call", call_id="call_1", name="build", arguments='{"script": "part = box((1,1,1))", "note": "x"}'))
            yield NS(type="response.completed", response=NS(id="resp_42"))
        return events()


def _provider():
    fake = FakeResponses()
    return OpenAIProvider("gpt-6.1-sol", client=NS(responses=fake), reasoning_effort="medium"), fake


def test_responses_stream_and_conversion():
    p, fake = _provider()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "make a box"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "c0", "name": "measure", "input": {"kind": "probe", "point": [1, 2, 3]}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c0", "content": [
            {"type": "text", "text": "result"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]},
    ]
    with p.stream("sys", messages, TOOLS) as st:
        texts = [e.text for e in st if e.type == "text"]
        final = st.get_final_message()
    assert "".join(texts) == "Plan: build it."
    assert final.stop_reason == "tool_use"
    tu = [b for b in final.content if b.type == "tool_use"][0]
    assert tu.id == "call_1" and tu.name == "build" and tu.input == {"script": "part = box((1,1,1))", "note": "x"}

    kw = fake.calls[0]
    assert kw["model"] == "gpt-6.1-sol" and kw["instructions"] == "sys" and kw["reasoning"] == {"effort": "medium"}
    assert "previous_response_id" not in kw
    inp = kw["input"]
    assert inp[0] == {"role": "user", "content": [{"type": "input_text", "text": "make a box"}]}
    assert inp[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
    assert inp[2]["type"] == "function_call" and json.loads(inp[2]["arguments"])["kind"] == "probe"
    assert inp[3] == {"type": "function_call_output", "call_id": "c0", "output": "result"}
    assert inp[4]["role"] == "user" and inp[4]["content"][1]["type"] == "input_image"
    assert inp[4]["content"][1]["image_url"].startswith("data:image/png;base64,")
    tools = kw["tools"]
    assert tools[0]["type"] == "function" and {t["name"] for t in tools} == {"plan", "build", "look", "measure", "finish"}


def test_responses_chaining_sends_only_new_user_items():
    p, fake = _provider()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "brief"}]}]
    with p.stream("sys", msgs, TOOLS) as st:
        st.get_final_message()
    # the loop appends the assistant turn and the tool results, then calls again
    msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "build", "input": {}}]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "BUILD OK"}]}]})
    with p.stream("sys", msgs, TOOLS) as st:
        st.get_final_message()
    kw = fake.calls[1]
    assert kw["previous_response_id"] == "resp_42"
    assert kw["input"] == [{"type": "function_call_output", "call_id": "call_1", "output": "BUILD OK"}]
