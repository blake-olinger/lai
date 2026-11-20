"""Conversation history structural repair and atomic tool-pair budget trimming."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from lai.domain.chat import MessageRole
from lai.domain.inference import InferenceMessage
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.chat import Message

_ELIGIBLE_HISTORY_ROLES = frozenset({MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL})
_MISSING_TOOL_RESULT_JSON = json.dumps({"success": False, "error": "tool_call_not_completed"})


def _estimate_message_tokens(msg: InferenceMessage) -> int:
    """Estimate conservative token cost for a single InferenceMessage."""
    tool_text = "".join(f"{tc.name}{tc.arguments_json}" for tc in msg.tool_calls)
    return max(1, estimate_tokens(msg.content) + estimate_tokens(tool_text) + 4)


def repair_tool_call_history(
    messages: Sequence[InferenceMessage],
) -> tuple[InferenceMessage, ...]:
    """Ensure every ASSISTANT(tool_calls) is immediately followed by matching TOOL replies.

    Drops orphaned TOOL messages, orders TOOL replies to match assistant.tool_calls order,
    and synthesizes an error TOOL message for any unanswered tool_call_id so Gemma 4's
    Jinja chat_template never emits an unclosed <|tool_response> block.
    """
    repaired: list[InferenceMessage] = []
    idx = 0
    total = len(messages)

    while idx < total:
        msg = messages[idx]
        if msg.role is MessageRole.TOOL:
            idx += 1
            continue

        if msg.role is MessageRole.ASSISTANT and msg.tool_calls:
            repaired.append(msg)
            idx += 1
            replies_by_id: dict[str, InferenceMessage] = {}
            while idx < total and messages[idx].role is MessageRole.TOOL:
                reply = messages[idx]
                if reply.tool_call_id and reply.tool_call_id not in replies_by_id:
                    replies_by_id[reply.tool_call_id] = reply
                idx += 1

            for tc in msg.tool_calls:
                existing_reply = replies_by_id.get(tc.id)
                if existing_reply is not None:
                    repaired.append(
                        InferenceMessage(
                            role=MessageRole.TOOL,
                            content=existing_reply.content,
                            name=existing_reply.name or tc.name,
                            tool_call_id=tc.id,
                        )
                    )
                else:
                    repaired.append(
                        InferenceMessage(
                            role=MessageRole.TOOL,
                            content=_MISSING_TOOL_RESULT_JSON,
                            name=tc.name,
                            tool_call_id=tc.id,
                        )
                    )
            continue

        repaired.append(msg)
        idx += 1

    return tuple(repaired)


def _group_atomic_units(
    messages: Sequence[InferenceMessage],
) -> list[tuple[list[InferenceMessage], int]]:
    """Group each ASSISTANT(tool_calls) + its consecutive TOOL replies into one atomic unit."""
    units: list[tuple[list[InferenceMessage], int]] = []
    idx = 0
    total = len(messages)

    while idx < total:
        msg = messages[idx]
        if msg.role is MessageRole.ASSISTANT and msg.tool_calls:
            group = [msg]
            tokens = _estimate_message_tokens(msg)
            idx += 1
            while idx < total and messages[idx].role is MessageRole.TOOL:
                reply = messages[idx]
                group.append(reply)
                tokens += _estimate_message_tokens(reply)
                idx += 1
            units.append((group, tokens))
            continue

        units.append(([msg], _estimate_message_tokens(msg)))
        idx += 1

    return units


def trim_history_to_budget(
    messages: Sequence[Message],
    *,
    max_history_tokens: int,
) -> tuple[InferenceMessage, ...]:
    """Retain recent history within token budget without splitting tool_call/TOOL pairs."""
    if max_history_tokens <= 0 or not messages:
        return ()

    converted = [
        InferenceMessage(
            role=msg.role,
            content=msg.content,
            name=msg.tool_name,
            tool_call_id=msg.tool_call_id,
            tool_calls=msg.tool_calls,
        )
        for msg in messages
        if msg.role in _ELIGIBLE_HISTORY_ROLES
    ]
    repaired = repair_tool_call_history(converted)
    units = _group_atomic_units(repaired)

    selected_units: list[list[InferenceMessage]] = []
    used_tokens = 0
    for group, group_tokens in reversed(units):
        if used_tokens + group_tokens > max_history_tokens:
            break
        used_tokens += group_tokens
        selected_units.append(group)

    selected_units.reverse()
    flattened: list[InferenceMessage] = []
    for group in selected_units:
        flattened.extend(group)
    return tuple(flattened)
