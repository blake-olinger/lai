"""OpenAI-compatible JSON serialization, response parsing, and SSE processing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from lai.domain.chat import TokenUsage
from lai.domain.errors import InferenceBackendError
from lai.domain.inference import (
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.inference.thinking import ParsedDelta, ThinkingStreamParser, split_thinking_content


def as_mapping(value: object) -> Mapping[str, object] | None:
    """Narrow an unknown JSON value to a string-keyed mapping."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def as_list(value: object) -> list[object]:
    """Narrow an unknown JSON value to a list."""
    if isinstance(value, list):
        return cast("list[object]", value)
    return []


def as_optional_str(value: object) -> str | None:
    """Return value if it is a str, otherwise None."""
    return value if isinstance(value, str) else None


def to_int(value: object, *, default: int = 0) -> int:
    """Convert an integer-like JSON scalar to int safely."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def parse_json_mapping(raw_text: str, *, context: str) -> Mapping[str, object]:
    """Decode JSON text and require a top-level mapping."""
    try:
        parsed: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {context}: {exc}"
        raise InferenceBackendError(msg) from exc
    mapping = as_mapping(parsed)
    if mapping is None:
        msg = f"Expected JSON object in {context}."
        raise InferenceBackendError(msg)
    return mapping


def parse_usage(raw_usage: object) -> TokenUsage | None:
    """Parse OpenAI token usage object into TokenUsage."""
    usage = as_mapping(raw_usage)
    if usage is None:
        return None
    prompt = max(0, to_int(usage.get("prompt_tokens"), default=0))
    completion = max(0, to_int(usage.get("completion_tokens"), default=0))
    total = to_int(usage.get("total_tokens"), default=prompt + completion)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=max(0, total),
    )


def _extract_error_detail(frame: Mapping[str, object]) -> str | None:
    """Extract an error message if an OpenAI JSON frame carries an error field."""
    err = frame.get("error")
    if err is None:
        return None
    err_map = as_mapping(err)
    if err_map is not None:
        msg = as_optional_str(err_map.get("message"))
        return msg or str(dict(err_map))
    return str(err)


@dataclass(slots=True)
class _PartialToolCall:
    tool_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)


class ToolCallAccumulator:
    """Accumulates streamed OpenAI delta.tool_calls chunks by index."""

    def __init__(self) -> None:
        self._entries: dict[int, _PartialToolCall] = {}

    def feed(self, raw_calls: list[object]) -> None:
        """Ingest one frame's delta.tool_calls array."""
        for item in raw_calls:
            item_map = as_mapping(item)
            if item_map is None:
                continue
            idx = to_int(item_map.get("index"), default=len(self._entries))
            entry = self._entries.setdefault(idx, _PartialToolCall())
            tool_id = as_optional_str(item_map.get("id"))
            if tool_id:
                entry.tool_id = tool_id
            self._ingest_function(entry, as_mapping(item_map.get("function")))

    @staticmethod
    def _ingest_function(entry: _PartialToolCall, fn_map: Mapping[str, object] | None) -> None:
        if fn_map is None:
            return
        fn_name = as_optional_str(fn_map.get("name"))
        if fn_name:
            entry.name = fn_name
        raw_args = fn_map.get("arguments")
        if isinstance(raw_args, str):
            entry.arguments_parts.append(raw_args)
        elif isinstance(raw_args, Mapping):
            mapped = dict(cast("Mapping[str, object]", raw_args))
            entry.arguments_parts.append(json.dumps(mapped))

    def flush(self) -> tuple[InferenceToolCall, ...]:
        """Return completed InferenceToolCall objects and reset buffer."""
        completed: list[InferenceToolCall] = []
        for idx in sorted(self._entries):
            entry = self._entries[idx]
            if not entry.name:
                continue
            completed.append(
                InferenceToolCall(
                    id=entry.tool_id or f"call_{idx}",
                    name=entry.name,
                    arguments_json="".join(entry.arguments_parts),
                )
            )
        self._entries.clear()
        return tuple(completed)


def serialize_chat_request(
    request: InferenceRequest,
    *,
    stream: bool,
    include_enable_thinking_kwarg: bool = False,
) -> dict[str, Any]:
    """Serialize an InferenceRequest into an OpenAI Chat Completions payload."""
    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        item: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
        if msg.name is not None:
            item["name"] = msg.name
        if msg.tool_call_id is not None:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in msg.tool_calls
            ]
        messages.append(item)

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": (
                        dict(tool.parameters_schema)
                        if tool.parameters_schema
                        else {"type": "object", "properties": {}}
                    ),
                },
            }
            for tool in request.tools
        ]
    if include_enable_thinking_kwarg:
        payload["chat_template_kwargs"] = {"enable_thinking": request.enable_thinking}
    elif not request.enable_thinking:
        payload["reasoning_effort"] = "none"
    return payload


def parse_complete_response(payload: Mapping[str, object]) -> InferenceResponse:
    """Parse a non-streaming /v1/chat/completions JSON body."""
    err_detail = _extract_error_detail(payload)
    if err_detail is not None:
        msg = f"Inference backend returned error: {err_detail}"
        raise InferenceBackendError(msg)

    choices = as_list(payload.get("choices"))
    if not choices:
        msg = "Inference response contained no choices."
        raise InferenceBackendError(msg)
    choice0 = as_mapping(choices[0])
    message = as_mapping(choice0.get("message")) if choice0 is not None else None
    if choice0 is None or message is None:
        msg = "Inference response missing choice message."
        raise InferenceBackendError(msg)

    raw_content = as_optional_str(message.get("content")) or ""
    raw_reasoning = as_optional_str(message.get("reasoning_content")) or as_optional_str(
        message.get("reasoning")
    )
    parsed = split_thinking_content(raw_content, raw_reasoning)

    acc = ToolCallAccumulator()
    acc.feed(as_list(message.get("tool_calls")))
    return InferenceResponse(
        content=parsed.content_delta,
        thinking=parsed.thinking_delta,
        tool_calls=acc.flush(),
        finish_reason=as_optional_str(choice0.get("finish_reason")),
        usage=parse_usage(payload.get("usage")) or TokenUsage(),
    )


class SseStreamProcessor:
    """Stateful OpenAI SSE frame processor for content, thinking, tools, and usage."""

    def __init__(self) -> None:
        self._thinking = ThinkingStreamParser()
        self._tools = ToolCallAccumulator()
        self.done = False

    def process_data(self, data_str: str) -> InferenceChunk | None:
        """Process a single SSE `data:` payload string."""
        if data_str == "[DONE]":
            self.done = True
            return self.flush_remaining()

        frame = parse_json_mapping(data_str, context="SSE chunk")
        err_detail = _extract_error_detail(frame)
        if err_detail is not None:
            msg = f"Inference stream error: {err_detail}"
            raise InferenceBackendError(msg)

        usage = parse_usage(frame.get("usage"))
        choices = as_list(frame.get("choices"))
        if not choices:
            return InferenceChunk(usage=usage) if usage is not None else None

        choice0 = as_mapping(choices[0])
        if choice0 is None:
            return None
        return self._build_choice_chunk(choice0, usage)

    def _build_choice_chunk(
        self,
        choice: Mapping[str, object],
        usage: TokenUsage | None,
    ) -> InferenceChunk | None:
        finish_reason = as_optional_str(choice.get("finish_reason"))
        delta = as_mapping(choice.get("delta")) or {}
        raw_content = as_optional_str(delta.get("content"))
        raw_reasoning = as_optional_str(delta.get("reasoning_content")) or as_optional_str(
            delta.get("reasoning")
        )
        parsed = self._thinking.feed(raw_content, raw_reasoning)
        raw_tcs = as_list(delta.get("tool_calls"))
        if raw_tcs:
            self._tools.feed(raw_tcs)

        content_delta, thinking_delta, tool_calls = self._finalize_if_needed(
            parsed,
            finish_reason,
        )
        if not (
            content_delta or thinking_delta or tool_calls or finish_reason or usage is not None
        ):
            return None
        return InferenceChunk(
            content_delta=content_delta,
            thinking_delta=thinking_delta,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _finalize_if_needed(
        self,
        parsed: ParsedDelta,
        finish_reason: str | None,
    ) -> tuple[str, str, tuple[InferenceToolCall, ...]]:
        if finish_reason is None:
            return parsed.content_delta, parsed.thinking_delta, ()
        tail = self._thinking.finalize()
        return (
            parsed.content_delta + tail.content_delta,
            parsed.thinking_delta + tail.thinking_delta,
            self._tools.flush(),
        )

    def flush_remaining(self) -> InferenceChunk | None:
        """Flush any buffered thinking partial tags or un-emitted tool calls."""
        tail = self._thinking.finalize()
        tool_calls = self._tools.flush()
        if not (tail.content_delta or tail.thinking_delta or tool_calls):
            return None
        return InferenceChunk(
            content_delta=tail.content_delta,
            thinking_delta=tail.thinking_delta,
            tool_calls=tool_calls,
        )
