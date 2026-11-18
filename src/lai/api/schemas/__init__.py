"""Pydantic v2 API request and response schemas."""

from __future__ import annotations

from lai.api.schemas.client_apps import (
    ClientAppListResponse,
    ClientAppResponse,
    CreateClientAppRequest,
    CreatedClientAppResponse,
)
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
    OpenAIDeltaToolCall,
    OpenAIModelInfo,
    OpenAIModelListResponse,
    OpenAIStreamChoice,
    OpenAIToolCall,
    OpenAIToolFunction,
    OpenAIUsage,
)
from lai.api.schemas.projects import ProjectCreateRequest, ProjectListResponse
from lai.api.schemas.sessions import (
    SessionCreateRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageRequest,
)
from lai.api.schemas.tools import (
    ResolveApprovalRequest,
    SubmitToolResultRequest,
    ToolListResponse,
)

__all__ = [
    "AlertsListResponse",
    "ClientAppListResponse",
    "ClientAppResponse",
    "CreateClientAppRequest",
    "CreatedClientAppResponse",
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
    "OpenAIDeltaToolCall",
    "OpenAIModelInfo",
    "OpenAIModelListResponse",
    "OpenAIStreamChoice",
    "OpenAIToolCall",
    "OpenAIToolFunction",
    "OpenAIUsage",
    "ProjectCreateRequest",
    "ProjectListResponse",
    "ResolveApprovalRequest",
    "SessionCreateRequest",
    "SessionDetailResponse",
    "SessionListResponse",
    "SessionMessageRequest",
    "SubmitToolResultRequest",
    "SystemStatusResponse",
    "ToolListResponse",
]
