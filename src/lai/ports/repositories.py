"""Persistence repository protocols (PEP 544)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from lai.domain.auth import ApiKeyRecord
    from lai.domain.chat import Message, Session
    from lai.domain.client_apps import ClientAppRecord
    from lai.domain.files import FileChunk, FileRecord, ScoredChunk
    from lai.domain.projects import Project
    from lai.domain.tools import ApprovalStatus, ToolApprovalRecord, ToolAuditRecord


@runtime_checkable
class ApiKeyRepositoryPort(Protocol):
    """Persistence contract for hashed API key records."""

    async def create(self, record: ApiKeyRecord) -> ApiKeyRecord:
        """Persist a new API key record."""
        ...

    async def get_by_id(self, key_id: str) -> ApiKeyRecord | None:
        """Fetch an API key record by ID."""
        ...

    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        """Fetch an API key record by its credential hash."""
        ...

    async def list_all(self) -> list[ApiKeyRecord]:
        """List all persisted API key records."""
        ...

    async def touch_last_used(self, key_id: str, used_at: datetime) -> None:
        """Update the last_used_at timestamp for a key."""
        ...

    async def revoke(self, key_id: str, revoked_at: datetime) -> bool:
        """Mark an API key as revoked, returning True if found."""
        ...


@runtime_checkable
class ClientAppRepositoryPort(Protocol):
    """Persistence contract for registered external client applications."""

    async def create(self, record: ClientAppRecord) -> ClientAppRecord:
        """Persist a new client application record."""
        ...

    async def update(self, record: ClientAppRecord) -> ClientAppRecord:
        """Persist updated metadata for an existing non-revoked client application."""
        ...

    async def get_by_id(self, app_id: str) -> ClientAppRecord | None:
        """Fetch a client application record by ID."""
        ...

    async def get_by_oauth_client_id(self, oauth_client_id: str) -> ClientAppRecord | None:
        """Fetch a client application record by its bound OAuth client ID."""
        ...

    async def list_all(self) -> list[ClientAppRecord]:
        """List all persisted client application records."""
        ...

    async def touch_last_used(self, app_id: str, used_at: datetime) -> None:
        """Update the last_used_at timestamp for a client application."""
        ...

    async def revoke(self, app_id: str, revoked_at: datetime) -> bool:
        """Mark a client application as revoked, returning True if found."""
        ...


@runtime_checkable
class ProjectRepositoryPort(Protocol):
    """Persistence contract for workspace projects."""

    async def create(self, project: Project) -> Project:
        """Persist a new project."""
        ...

    async def get(self, project_id: str) -> Project | None:
        """Fetch a project by ID."""
        ...

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Project]:
        """List projects ordered by creation or update time."""
        ...

    async def list_for_owner(
        self,
        owner_principal_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Project]:
        """List projects owned by a specific principal."""
        ...

    async def count_for_owner(self, owner_principal_id: str) -> int:
        """Count projects owned by a specific principal."""
        ...

    async def delete(self, project_id: str) -> bool:
        """Delete a project and return True if it existed."""
        ...


@runtime_checkable
class SessionRepositoryPort(Protocol):
    """Persistence contract for chat sessions and messages."""

    async def create_session(self, session: Session) -> Session:
        """Persist a new chat session."""
        ...

    async def get_session(self, session_id: str) -> Session | None:
        """Fetch a chat session by ID."""
        ...

    async def list_sessions(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        """List chat sessions belonging to a project."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and return True if it existed."""
        ...

    async def add_message(self, message: Message) -> Message:
        """Append a message to a chat session."""
        ...

    async def list_messages(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Message]:
        """List messages within a session in chronological order."""
        ...

    async def list_recent_messages(
        self,
        session_id: str,
        limit: int = 200,
    ) -> list[Message]:
        """List the most recent messages within a session in chronological order."""
        ...


@runtime_checkable
class FileRepositoryPort(Protocol):
    """Persistence and FTS5 search contract for uploaded project files and chunks."""

    async def upsert_file(self, record: FileRecord) -> FileRecord:
        """Create or update a file metadata record."""
        ...

    async def get_file(self, file_id: str) -> FileRecord | None:
        """Fetch a file metadata record by ID."""
        ...

    async def list_files(
        self,
        project_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[FileRecord]:
        """List files belonging to a project."""
        ...

    async def replace_chunks(self, file_id: str, chunks: Sequence[FileChunk]) -> None:
        """Atomically replace all indexed chunks for a file."""
        ...

    async def list_chunks(
        self,
        file_id: str,
        start_line: int | None = None,
        end_line: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[FileChunk]:
        """Return ordered chunks for a file, optionally filtered by line range and paginated."""
        ...

    async def search_fts(
        self,
        project_id: str,
        query: str,
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        """Run full-text search across chunks scoped to a project."""
        ...

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file record and its chunks, returning True if found."""
        ...

    async def sum_bytes_for_owner(self, owner_principal_id: str) -> int:
        """Return total file bytes stored across projects owned by a principal."""
        ...


@runtime_checkable
class ToolAuditRepositoryPort(Protocol):
    """Persistence contract for tool approvals and execution audit trails."""

    async def create_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        """Persist a pending tool approval request."""
        ...

    async def get_approval(self, approval_id: str) -> ToolApprovalRecord | None:
        """Fetch a tool approval record by ID."""
        ...

    async def list_pending_approvals(
        self,
        project_id: str,
        session_id: str | None = None,
    ) -> list[ToolApprovalRecord]:
        """List pending approval requests for a project or session."""
        ...

    async def update_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        """Persist an updated approval decision state."""
        ...

    async def transition_approval_status(
        self,
        record: ToolApprovalRecord,
        *,
        expected_status: ApprovalStatus,
    ) -> bool:
        """Atomically update approval state only when current status matches expected_status."""
        ...

    async def record_audit(self, record: ToolAuditRecord) -> ToolAuditRecord:
        """Append an immutable tool execution audit entry."""
        ...

    async def list_audits(
        self,
        project_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolAuditRecord]:
        """List recent tool audit entries for a project or session."""
        ...
