"""Deterministic passthrough and regex-based guardrail providers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.guardrails import GuardrailDecision, GuardrailVerdict
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text
from lai.ports.guardrails import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Sequence

_MS_PER_SECOND = 1000.0
_RAW_MATCH_PREVIEW_CHARS = 120


@dataclass(frozen=True, slots=True)
class RegexPolicyRule:
    """Single deterministic regex guardrail rule bound to a named policy."""

    policy_name: str
    pattern: re.Pattern[str]
    reason: str = "Matched deterministic guardrail rule."


DEFAULT_REGEX_RULES: tuple[RegexPolicyRule, ...] = (
    RegexPolicyRule(
        policy_name="No Prompt Injection",
        pattern=re.compile(
            r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?|rules)\b"
        ),
        reason="Prompt attempts to override prior system instructions.",
    ),
    RegexPolicyRule(
        policy_name="No Prompt Injection",
        pattern=re.compile(
            r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|system)\s+"
            r"(?:instructions|prompts?|directives)\b"
        ),
        reason="Prompt attempts to disregard system directives.",
    ),
    RegexPolicyRule(
        policy_name="No Prompt Injection",
        pattern=re.compile(
            r"(?i)\breveal\s+(?:your\s+)?(?:hidden\s+|full\s+)?(?:system\s+prompt|instructions)\b"
        ),
        reason="Prompt attempts to exfiltrate hidden system instructions.",
    ),
    RegexPolicyRule(
        policy_name="No Prompt Injection",
        pattern=re.compile(r"(?i)\byou\s+are\s+now\s+(?:DAN|in\s+developer\s+mode)\b"),
        reason="Prompt uses explicit jailbreak persona framing.",
    ),
)


def _normalize_rule(rule: RegexPolicyRule | tuple[str, re.Pattern[str] | str]) -> RegexPolicyRule:
    """Convert tuple rule specifications into compiled RegexPolicyRule objects."""
    if isinstance(rule, RegexPolicyRule):
        return rule
    policy_name, raw_pattern = rule
    compiled = re.compile(raw_pattern) if isinstance(raw_pattern, str) else raw_pattern
    return RegexPolicyRule(
        policy_name=policy_name,
        pattern=compiled,
        reason=f"Matched deterministic guardrail policy '{policy_name}'.",
    )


class AllowAllGuardrailProvider(GuardrailProviderPort):
    """Permissive guardrail provider used only when guardrails are explicitly disabled."""

    def __init__(self, *, max_logged_prompt_chars: int = 2000) -> None:
        self._max_logged_prompt_chars = max_logged_prompt_chars

    @property
    def name(self) -> str:
        """Return the identifier for this guardrail provider."""
        return "allow-all"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        """Always allow the prompt while returning a scrubbed audit-ready decision."""
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            reason="Guardrails explicitly disabled.",
            latency_ms=0.0,
            prompt_sha256=fingerprint_prompt(prompt),
            prompt_chars=len(prompt),
            scrubbed_prompt=scrub_sensitive_text(
                prompt,
                max_chars=self._max_logged_prompt_chars,
            ),
        )


class RegexPolicyGuardrailProvider(GuardrailProviderPort):
    """Lightweight deterministic regex rule provider for pluggable pre-flight composition."""

    def __init__(
        self,
        rules: Sequence[RegexPolicyRule | tuple[str, re.Pattern[str] | str]] | None = None,
        *,
        max_logged_prompt_chars: int = 2000,
        provider_name: str = "regex-policy",
    ) -> None:
        raw_rules = DEFAULT_REGEX_RULES if rules is None else rules
        self._rules = tuple(_normalize_rule(rule) for rule in raw_rules)
        self._max_logged_prompt_chars = max_logged_prompt_chars
        self._name = provider_name

    @property
    def name(self) -> str:
        """Return the identifier for this guardrail provider."""
        return self._name

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        """Evaluate prompt sequentially against deterministic regex rules."""
        started = time.perf_counter()
        prompt_sha256 = fingerprint_prompt(prompt)
        prompt_chars = len(prompt)
        scrubbed = scrub_sensitive_text(prompt, max_chars=self._max_logged_prompt_chars)

        for rule in self._rules:
            match = rule.pattern.search(prompt)
            if match is not None:
                elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
                return GuardrailDecision(
                    verdict=GuardrailVerdict.BLOCK,
                    provider=self.name,
                    violated_policy=rule.policy_name,
                    reason=rule.reason,
                    raw_output=scrub_sensitive_text(
                        match.group(0),
                        max_chars=_RAW_MATCH_PREVIEW_CHARS,
                    ),
                    latency_ms=elapsed_ms,
                    prompt_sha256=prompt_sha256,
                    prompt_chars=prompt_chars,
                    scrubbed_prompt=scrubbed,
                )

        elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            latency_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
            prompt_chars=prompt_chars,
            scrubbed_prompt=scrubbed,
        )
