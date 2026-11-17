"""Map-Reduce aggregator for holistic analysis across oversized project files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import Citation, OrchestratorEvent, ProgressEvent, TokenUsage
from lai.domain.inference import InferenceRequest
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.ingestion.detector import estimate_tokens
from lai.observability.logging import get_component_logger
from lai.orchestrator._helpers import add_token_usage, extract_citations, read_file_text
from lai.orchestrator._stream import (
    StreamedInferenceAccumulator,
    TurnOutcome,
    stream_leased_inference,
)
from lai.orchestrator.prompts import (
    build_compression_messages,
    build_map_window_messages,
    build_reduce_messages,
    format_file_summary,
)
from lai.orchestrator.windower import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_WINDOW_TOKENS,
    TextWindow,
    slice_text_into_windows,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.inference import InferenceMessage, ModelProfile
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.scheduler import InferenceSpoolerPort
    from lai.ports.storage import BlobStorePort

DEFAULT_COMPRESSION_THRESHOLD_TOKENS = 10_000
_COMPRESSION_BATCH_LIMIT_TOKENS = 14_000
_MAP_TEMPERATURE = 0.1
_MAP_MAX_TOKENS = 1024
_COMPRESS_MAX_TOKENS = 2048


@dataclass(slots=True)
class _ReduceParams:
    """Bundled inputs for the final Map-Reduce synthesis turn."""

    turn_input: TurnInput
    principal: AuthPrincipal
    model_profile: ModelProfile
    files: Sequence[FileRecord]
    file_summaries: Sequence[str]
    findings: Sequence[str]
    fallback_citations: Sequence[Citation]
    session_system_prompt: str | None
    history_messages: Sequence[InferenceMessage]


def _pack_finding_batches(findings: Sequence[str], *, batch_limit_tokens: int) -> list[list[str]]:
    """Group findings into token-bounded batches for hierarchical compression."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for item in findings:
        item_tokens = max(1, estimate_tokens(item) + 4)
        if current and current_tokens + item_tokens > batch_limit_tokens:
            batches.append(current)
            current = [item]
            current_tokens = item_tokens
        else:
            current.append(item)
            current_tokens += item_tokens
    if current:
        batches.append(current)
    return batches


class MapReduceAggregator:
    """Executes Map-Reduce over oversized files with per-window GPU lease interleaving."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        blob_store: BlobStorePort,
        inference_provider: InferenceProviderPort,
        spooler: InferenceSpoolerPort,
        file_repo: FileRepositoryPort | None = None,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        compression_threshold_tokens: int = DEFAULT_COMPRESSION_THRESHOLD_TOKENS,
    ) -> None:
        self._blob_store = blob_store
        self._inference_provider = inference_provider
        self._spooler = spooler
        self._file_repo = file_repo
        self._window_tokens = window_tokens
        self._overlap_tokens = overlap_tokens
        self._compression_threshold_tokens = compression_threshold_tokens
        self._log = get_component_logger("orchestrator")

    async def _collect_windows_and_summaries(
        self,
        files: Sequence[FileRecord],
    ) -> tuple[list[tuple[TextWindow, str]], list[str]]:
        """Build deterministic summaries and line-preserving windows across target files."""
        window_items: list[tuple[TextWindow, str]] = []
        file_summaries: list[str] = []
        for record in files:
            summary_text = format_file_summary(record)
            file_summaries.append(summary_text)
            raw_text = await read_file_text(self._blob_store, record, file_repo=self._file_repo)
            windows = slice_text_into_windows(
                record.id,
                record.filename,
                raw_text,
                window_tokens=self._window_tokens,
                overlap_tokens=self._overlap_tokens,
            )
            window_items.extend((window, summary_text) for window in windows)
        return window_items, file_summaries

    async def _run_map_window(
        self,
        *,
        turn_input: TurnInput,
        principal: AuthPrincipal,
        model_profile: ModelProfile,
        window: TextWindow,
        file_summary_text: str,
    ) -> tuple[str, TokenUsage]:
        """Acquire one MAP_REDUCE lease and extract concise findings for one window."""
        lease_req = LeaseRequest(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            principal_id=principal.principal_id,
            priority=JobPriority.MAP_REDUCE,
        )
        request = InferenceRequest(
            model=model_profile.name,
            messages=build_map_window_messages(
                turn_input.user_message,
                window,
                file_summary_text=file_summary_text,
            ),
            temperature=_MAP_TEMPERATURE,
            top_p=model_profile.top_p,
            max_tokens=min(_MAP_MAX_TOKENS, model_profile.max_tokens),
            stream=False,
            enable_thinking=False,
        )
        async with self._spooler.lease(lease_req):
            response = await self._inference_provider.chat_complete(request)
        header = f"[{window.filename}:L{window.start_line}-L{window.end_line}]"
        return f"{header}\n{response.content.strip()}", response.usage

    async def _compress_findings(
        self,
        *,
        turn_input: TurnInput,
        principal: AuthPrincipal,
        model_profile: ModelProfile,
        findings: list[str],
    ) -> tuple[list[str], TokenUsage]:
        """Perform one hierarchical compression pass over oversized map findings."""
        batches = _pack_finding_batches(
            findings,
            batch_limit_tokens=_COMPRESSION_BATCH_LIMIT_TOKENS,
        )
        compressed: list[str] = []
        usage = TokenUsage()
        total_batches = len(batches)
        for idx, batch in enumerate(batches):
            lease_req = LeaseRequest(
                session_id=turn_input.session_id,
                project_id=turn_input.project_id,
                principal_id=principal.principal_id,
                priority=JobPriority.MAP_REDUCE,
            )
            request = InferenceRequest(
                model=model_profile.name,
                messages=build_compression_messages(
                    turn_input.user_message,
                    batch,
                    batch_index=idx,
                    total_batches=total_batches,
                ),
                temperature=_MAP_TEMPERATURE,
                top_p=model_profile.top_p,
                max_tokens=min(_COMPRESS_MAX_TOKENS, model_profile.max_tokens),
                stream=False,
                enable_thinking=False,
            )
            async with self._spooler.lease(lease_req):
                response = await self._inference_provider.chat_complete(request)
            compressed.append(response.content.strip())
            usage = add_token_usage(usage, response.usage)
        return compressed, usage

    async def _stream_reduce(
        self,
        params: _ReduceParams,
        total_usage: TokenUsage,
    ) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
        """Acquire final MAP_REDUCE lease, stream Reduce synthesis, and yield TurnOutcome."""
        turn_input = params.turn_input
        profile = params.model_profile
        reduce_messages = build_reduce_messages(
            turn_input.user_message,
            params.file_summaries,
            params.findings,
            session_system_prompt=params.session_system_prompt,
            history_messages=params.history_messages,
        )
        effective_temp = (
            turn_input.temperature if turn_input.temperature is not None else profile.temperature
        )
        effective_max_tokens = (
            turn_input.max_tokens if turn_input.max_tokens is not None else profile.max_tokens
        )
        lease_req = LeaseRequest(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            principal_id=params.principal.principal_id,
            priority=JobPriority.MAP_REDUCE,
        )
        request = InferenceRequest(
            model=profile.name,
            messages=reduce_messages,
            temperature=effective_temp,
            top_p=profile.top_p,
            max_tokens=effective_max_tokens,
            stream=True,
            enable_thinking=turn_input.enable_thinking,
        )
        acc = StreamedInferenceAccumulator()
        async for event in stream_leased_inference(
            spooler=self._spooler,
            inference_provider=self._inference_provider,
            lease_request=lease_req,
            inference_request=request,
            acc=acc,
        ):
            yield event

        final_content = acc.content
        findings_citations = extract_citations("\n\n".join(params.findings), params.files)
        fallback = findings_citations or tuple(params.fallback_citations)
        citations = extract_citations(final_content, params.files, fallback_citations=fallback)
        yield TurnOutcome(
            content=final_content,
            thinking=acc.thinking,
            citations=citations,
            usage=add_token_usage(total_usage, acc.usage),
        )

    async def run_stream(  # noqa: PLR0913
        self,
        *,
        turn_input: TurnInput,
        principal: AuthPrincipal,
        model_profile: ModelProfile,
        files: Sequence[FileRecord],
        session_system_prompt: str | None = None,
        history_messages: Sequence[InferenceMessage] = (),
    ) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
        """Run Map, optional hierarchical compression, and Reduce synthesis as an event stream."""
        window_items, file_summaries = await self._collect_windows_and_summaries(files)
        total_windows = len(window_items)
        self._log.info("map_reduce_started", file_count=len(files), window_count=total_windows)

        findings: list[str] = []
        total_usage = TokenUsage()
        for idx, (window, summary_text) in enumerate(window_items):
            yield ProgressEvent(
                stage="map_window",
                message=(
                    f"Mapping {window.filename} lines {window.start_line}-{window.end_line} "
                    f"({idx + 1}/{total_windows})."
                ),
                completed_steps=idx,
                total_steps=total_windows,
            )
            finding, map_usage = await self._run_map_window(
                turn_input=turn_input,
                principal=principal,
                model_profile=model_profile,
                window=window,
                file_summary_text=summary_text,
            )
            findings.append(finding)
            total_usage = add_token_usage(total_usage, map_usage)

        if findings and estimate_tokens("\n\n".join(findings)) > self._compression_threshold_tokens:
            yield ProgressEvent(
                stage="compress_findings",
                message="Compressing oversized intermediate Map findings.",
                completed_steps=total_windows,
                total_steps=total_windows,
            )
            findings, compress_usage = await self._compress_findings(
                turn_input=turn_input,
                principal=principal,
                model_profile=model_profile,
                findings=findings,
            )
            total_usage = add_token_usage(total_usage, compress_usage)

        yield ProgressEvent(
            stage="reduce",
            message="Synthesizing unified final response.",
            completed_steps=total_windows,
            total_steps=total_windows,
        )
        window_fallbacks = [
            Citation(
                file_id=w.file_id,
                filename=w.filename,
                start_line=w.start_line,
                end_line=w.end_line,
            )
            for w, _ in window_items
        ] or [Citation(file_id=f.id, filename=f.filename) for f in files]

        params = _ReduceParams(
            turn_input=turn_input,
            principal=principal,
            model_profile=model_profile,
            files=files,
            file_summaries=file_summaries,
            findings=findings,
            fallback_citations=window_fallbacks,
            session_system_prompt=session_system_prompt,
            history_messages=history_messages,
        )
        async for event in self._stream_reduce(params, total_usage):
            yield event
