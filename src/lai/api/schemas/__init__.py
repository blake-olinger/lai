"""Pydantic v2 API request and response schemas."""

from __future__ import annotations

from lai.api.schemas.common import (
    AlertsListResponse,
    DeleteResponse,
    ErrorDetail,
    ErrorResponse,
    GpuStatusDTO,
    HealthResponse,
    SystemStatusResponse,
)
from lai.api.schemas.files import FileChunkListResponse, FileListResponse
from lai.api.schemas.openai import (
    OpenAIChatChoice,
    OpenAIChatChoiceMessage,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatMessageInput,
    OpenAIDeltaPayload,
    OpenAIModelInfo,
    OpenAIModelListResponse,
    OpenAIStreamChoice,
    OpenAIUsage,
)
from lai.api.schemas.projects import ProjectCreateRequest, ProjectListResponse
from lai.api.schemas.sessions import (
    SessionCreateRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageRequest,
)
from lai.api.schemas.tools import ResolveApprovalRequest, ToolListResponse

__all__ = [
    "AlertsListResponse",
    "DeleteResponse",
    "ErrorDetail",
    "ErrorResponse",
    "FileChunkListResponse",
    "FileListResponse",
    "GpuStatusDTO",
    "HealthResponse",
    "OpenAIChatChoice",
    "OpenAIChatChoiceMessage",
    "OpenAIChatCompletionChunk",
    "OpenAIChatCompletionRequest",
    "OpenAIChatCompletionResponse",
    "OpenAIChatMessageInput",
    "OpenAIDeltaPayload",
    "OpenAIModelInfo",
    "OpenAIModelListResponse",
    "OpenAIStreamChoice",
    "OpenAIUsage",
    "ProjectCreateRequest",
    "ProjectListResponse",
    "ResolveApprovalRequest",
    "SessionCreateRequest",
    "SessionDetailResponse",
    "SessionListResponse",
    "SessionMessageRequest",
    "SystemStatusResponse",
    "ToolListResponse",
]
