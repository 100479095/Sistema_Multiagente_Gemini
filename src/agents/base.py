"""Agent base class, the ``@tool`` decorator, and JSON-state helpers.

A *tool* is a typed method on an :class:`Agent`. The ``@tool`` decorator
attaches a :class:`ToolSpec` that (a) builds a pydantic model from the method
signature for argument validation and (b) emits the JSON schema the LLM tool-API
expects. Adding a new tool is therefore just writing a typed, decorated method
(PDR RNF-6).

``untrusted_fields`` declares which keys of a tool's structured result carry
*untrusted* content (e.g. an event title or email body). Provenance tagging in a
later phase reads this metadata; the data is **never** filtered (PDR §6/§8.0d).
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

# --------------------------------------------------------------------------- #
# JSON state helpers (the simulated data stores are plain local JSON files)
# --------------------------------------------------------------------------- #


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Tool specification + decorator
# --------------------------------------------------------------------------- #


def _strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove pydantic-generated ``title`` keys for a leaner tool schema."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def _build_args_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a pydantic model describing the callable's keyword arguments.

    Annotations are resolved with :func:`typing.get_type_hints` so that string
    annotations (under ``from __future__ import annotations``) such as
    ``Literal["on", "off"]`` become real types pydantic can validate against.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - defensive; unresolved hints -> Any
        hints = {}
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    return create_model(f"{func.__name__}_Args", **fields)  # type: ignore[call-overload]


@dataclass
class ToolSpec:
    """Metadata describing one callable tool exposed to the orchestrator."""

    name: str
    description: str
    args_model: type[BaseModel]
    func_name: str
    untrusted_fields: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        """OpenAI-format tool schema for the LLM tools API."""
        params = _strip_titles(self.args_model.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate/coerce raw arguments; raises pydantic ValidationError."""
        return self.args_model(**arguments).model_dump()


def tool(
    description: str | None = None,
    *,
    untrusted_fields: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark an agent method as an LLM-invocable tool.

    ``description`` defaults to the method's docstring first line. The argument
    schema is derived from the method's type hints.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        desc = description or (inspect.getdoc(func) or "").split("\n", 1)[0]
        func.__tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            name=func.__name__,
            description=desc,
            args_model=_build_args_model(func),
            func_name=func.__name__,
            untrusted_fields=untrusted_fields,
        )
        return func

    return decorator


# --------------------------------------------------------------------------- #
# Agent base class
# --------------------------------------------------------------------------- #


class Agent:
    """Base for simulated agents. Collects its decorated tools."""

    #: Human-readable agent name (used in logs).
    name: str = "agent"

    def tool_specs(self) -> list[ToolSpec]:
        """Return the :class:`ToolSpec` of every ``@tool`` method on this agent."""
        specs: list[ToolSpec] = []
        for attr_name in dir(type(self)):
            attr = getattr(type(self), attr_name, None)
            spec = getattr(attr, "__tool_spec__", None)
            if spec is not None:
                specs.append(spec)
        return specs

    def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Validate arguments and invoke the named tool on this instance."""
        method = getattr(self, tool_name, None)
        spec = getattr(method, "__tool_spec__", None)
        if method is None or spec is None:
            raise KeyError(f"{self.name!r} has no tool {tool_name!r}")
        validated = spec.validate_arguments(arguments)
        return method(**validated)
