"""Stateful parser separating Gemma 4 <thought> tags and reasoning_content."""

from __future__ import annotations

from dataclasses import dataclass

OPEN_THOUGHT_TAG = "<thought>"
CLOSE_THOUGHT_TAG = "</thought>"


@dataclass(frozen=True, slots=True)
class ParsedDelta:
    """Separated content and thinking text from a streaming chunk or full response."""

    content_delta: str = ""
    thinking_delta: str = ""


def _longest_prefix_suffix(text: str, tag: str) -> int:
    """Return length of longest non-empty suffix of text matching a prefix of tag."""
    max_len = min(len(text), len(tag) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ThinkingStreamParser:
    """Incremental state machine splitting <thought>...</thought> across stream deltas."""

    def __init__(
        self,
        *,
        open_tag: str = OPEN_THOUGHT_TAG,
        close_tag: str = CLOSE_THOUGHT_TAG,
        start_in_thought: bool = False,
    ) -> None:
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._in_thought = start_in_thought
        self._buffer = ""

    @property
    def in_thought(self) -> bool:
        """Return whether the parser is currently inside a thought block."""
        return self._in_thought

    def _strip_tags(self, text: str) -> str:
        """Remove thought boundary tags from pre-separated reasoning text."""
        return text.replace(self._open_tag, "").replace(self._close_tag, "")

    def _consume_in_thought(self, buf: str, thinking_parts: list[str]) -> str | None:
        """Consume buffer while inside a thought block; return remaining text or None."""
        close_idx = buf.find(self._close_tag)
        if close_idx != -1:
            if close_idx > 0:
                thinking_parts.append(buf[:close_idx])
            self._in_thought = False
            return buf[close_idx + len(self._close_tag) :]

        hold = _longest_prefix_suffix(buf, self._close_tag)
        if hold > 0:
            if len(buf) > hold:
                thinking_parts.append(buf[:-hold])
            self._buffer = buf[-hold:]
        elif buf:
            thinking_parts.append(buf)
        return None

    def _consume_outside_thought(self, buf: str, content_parts: list[str]) -> str | None:
        """Consume buffer while outside thought tags; return remaining text or None."""
        open_idx = buf.find(self._open_tag)
        close_idx = buf.find(self._close_tag)
        if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
            if open_idx > 0:
                content_parts.append(buf[:open_idx])
            self._in_thought = True
            return buf[open_idx + len(self._open_tag) :]
        if close_idx != -1:
            if close_idx > 0:
                content_parts.append(buf[:close_idx])
            return buf[close_idx + len(self._close_tag) :]

        hold = max(
            _longest_prefix_suffix(buf, self._open_tag),
            _longest_prefix_suffix(buf, self._close_tag),
        )
        if hold > 0:
            if len(buf) > hold:
                content_parts.append(buf[:-hold])
            self._buffer = buf[-hold:]
        elif buf:
            content_parts.append(buf)
        return None

    def feed(
        self,
        content_delta: str | None = None,
        reasoning_delta: str | None = None,
    ) -> ParsedDelta:
        """Process content and reasoning deltas without leaking thought tags."""
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        if reasoning_delta:
            cleaned = self._strip_tags(reasoning_delta)
            if cleaned:
                thinking_parts.append(cleaned)

        if content_delta:
            remaining: str | None = self._buffer + content_delta
            self._buffer = ""
            while remaining:
                if self._in_thought:
                    remaining = self._consume_in_thought(remaining, thinking_parts)
                else:
                    remaining = self._consume_outside_thought(remaining, content_parts)

        return ParsedDelta(
            content_delta="".join(content_parts),
            thinking_delta="".join(thinking_parts),
        )

    def finalize(self) -> ParsedDelta:
        """Flush any buffered partial tag prefix at end-of-stream."""
        pending = self._buffer
        self._buffer = ""
        if not pending:
            return ParsedDelta()
        if self._in_thought:
            return ParsedDelta(thinking_delta=pending)
        return ParsedDelta(content_delta=pending)


def split_thinking_content(
    content: str,
    reasoning_content: str | None = None,
    *,
    start_in_thought: bool = False,
) -> ParsedDelta:
    """Separate <thought> blocks and reasoning_content for a complete response."""
    parser = ThinkingStreamParser(start_in_thought=start_in_thought)
    first = parser.feed(content_delta=content, reasoning_delta=reasoning_content)
    tail = parser.finalize()
    return ParsedDelta(
        content_delta=first.content_delta + tail.content_delta,
        thinking_delta=first.thinking_delta + tail.thinking_delta,
    )
