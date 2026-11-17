"""In-memory catalog of registered tool plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.ports.tools import ToolPluginPort, ToolRegistryPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.tools import ToolSpec


class InMemoryToolRegistry(ToolRegistryPort):
    """In-memory registry mapping tool names to ToolPluginPort instances."""

    def __init__(self, plugins: Sequence[ToolPluginPort] = ()) -> None:
        self._plugins: dict[str, ToolPluginPort] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: ToolPluginPort) -> None:
        """Register or replace a tool plugin by its specification name."""
        self._plugins[plugin.spec.name] = plugin

    def get(self, name: str) -> ToolPluginPort | None:
        """Look up a registered tool plugin by name."""
        return self._plugins.get(name)

    def list_specs(self) -> list[ToolSpec]:
        """Return specifications for all registered tool plugins."""
        return [plugin.spec for plugin in self._plugins.values()]
