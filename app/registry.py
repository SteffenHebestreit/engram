"""Lightweight strategy registries for the pluggable pipeline stages.

Each customizable stage (chunker, metadata extractor, fusion, proximity,
expander, ...) owns a `Registry`. Built-in implementations register their
default under a string key; `Settings` selects one by key, so a deployment can
swap a stage without editing the core, and a third-party package can add a new
strategy by importing the registry and registering against it.

Kept deliberately tiny: no plugin framework, just a name -> object map with a
decorator. `load_entrypoints()` optionally discovers external strategies
declared under the `engram.plugins` entry-point group.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar, overload

T = TypeVar("T")


class Registry(Generic[T]):
    """A named map of strategy implementations of one kind."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...
    @overload
    def register(self, name: str, item: T) -> T: ...

    def register(self, name: str, item: T | None = None):
        """Register `item` under `name`.

        Usable directly (`reg.register("x", impl)`) or as a decorator
        (`@reg.register("x")`).
        """
        if item is not None:
            self._items[name] = item
            return item

        def decorator(obj: T) -> T:
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def load_entrypoints(group: str = "engram.plugins") -> int:
    """Import packages declaring an `engram.plugins` entry point.

    A plugin package exposes a callable that registers its strategies on import.
    Returns the number of entry points loaded. Failures are swallowed so a
    broken third-party plugin cannot take the service down.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib >=3.8
        return 0

    loaded = 0
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - older selection API
        eps = entry_points().get(group, [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            hook = ep.load()
            if callable(hook):
                hook()
            loaded += 1
        except Exception:  # pragma: no cover - defensive
            continue
    return loaded
