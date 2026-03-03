"""
Generator registry system.

Allows generators to declare their parameters as metadata,
enabling the UI to build itself automatically.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Param:
    """One UI parameter for a generator.

    Args:
        name: kwarg name passed to the generator function
        label: UI display label
        type: "float", "int", "bool", "str", "color", "range", or "image"
        default: Default value (type depends on param type)
        min: Minimum value (float/int sliders)
        max: Maximum value (float/int sliders)
        step: Step size (float/int sliders)
        choices: List of choices (str dropdown)
        range_default: Default (min, max) tuple for range type
        range_min: Minimum for both range sliders
        range_max: Maximum for both range sliders
    """
    name: str
    label: str
    type: str
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None
    range_default: Optional[Tuple[float, float]] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None


@dataclass
class GeneratorInfo:
    """Complete metadata for one registered generator."""
    func: Callable
    name: str
    label: str
    category: str
    params: List[Param] = field(default_factory=list)
    export_name: str = ""


# Global registry - insertion order preserved
_registry: Dict[str, GeneratorInfo] = {}

# Category ordering (controls top-level tab order)
CATEGORIES = ["Rocks", "Vegetation", "Buildings", "Terrain", "Props", "Furniture", "Instruments"]


def register(
    name: str,
    label: str,
    category: str,
    params: List[Param],
    export_name: str = "",
):
    """Register a generator function with its UI metadata.

    Can be used as a decorator or called directly:

        # As decorator:
        @register(name="rock", label="Rock", category="Rocks", params=[...])
        def generate_rock(...):
            ...

        # Direct call:
        register(name="rock", label="Rock", category="Rocks", params=[...])(generate_rock)
    """
    def decorator(func: Callable) -> Callable:
        _registry[name] = GeneratorInfo(
            func=func,
            name=name,
            label=label,
            category=category,
            params=params,
            export_name=export_name or name,
        )
        return func
    return decorator


def get_registry() -> Dict[str, GeneratorInfo]:
    """Return the full registry dict."""
    return _registry


def get_by_category() -> Dict[str, List[GeneratorInfo]]:
    """Return generators grouped by category, in CATEGORIES order."""
    grouped = {cat: [] for cat in CATEGORIES}
    for info in _registry.values():
        grouped.setdefault(info.category, []).append(info)
    return {k: v for k, v in grouped.items() if v}
