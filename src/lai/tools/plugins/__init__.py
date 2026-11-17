"""Phase 1 read-only project file inspection plugins."""

from __future__ import annotations

from lai.tools.plugins.file_overview import GetFileOverviewPlugin
from lai.tools.plugins.read_lines import DEFAULT_MAX_LINES_PER_CALL, ReadFileLinesPlugin
from lai.tools.plugins.search_files import SearchFilesPlugin

__all__ = [
    "DEFAULT_MAX_LINES_PER_CALL",
    "GetFileOverviewPlugin",
    "ReadFileLinesPlugin",
    "SearchFilesPlugin",
]
