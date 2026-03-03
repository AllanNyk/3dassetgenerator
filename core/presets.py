"""
Preset system for saving and loading generator parameters.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Default presets directory
PRESETS_DIR = Path(__file__).parent.parent / "presets"


def ensure_presets_dir():
    """Ensure presets directory exists."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def save_preset(
    name: str,
    asset_type: str,
    parameters: Dict[str, Any],
    description: str = "",
    overwrite: bool = False
) -> str:
    """
    Save a preset to JSON file.

    Args:
        name: Preset name (used as filename)
        asset_type: Type of asset (rock, tree, building, etc.)
        parameters: Dictionary of generator parameters
        description: Optional description
        overwrite: Whether to overwrite existing preset

    Returns:
        Path to saved preset file
    """
    ensure_presets_dir()

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
    filepath = PRESETS_DIR / f"{safe_name}.json"

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"Preset '{name}' already exists. Use overwrite=True to replace.")

    preset_data = {
        "name": name,
        "asset_type": asset_type,
        "parameters": parameters,
        "description": description,
        "created": datetime.now().isoformat(),
        "version": "1.0"
    }

    with open(filepath, 'w') as f:
        json.dump(preset_data, f, indent=2)

    return str(filepath)


def load_preset(name: str) -> Dict[str, Any]:
    """
    Load a preset from JSON file.

    Args:
        name: Preset name (without .json extension)

    Returns:
        Preset data dictionary
    """
    ensure_presets_dir()

    # Try with and without .json extension
    filepath = PRESETS_DIR / f"{name}.json"
    if not filepath.exists():
        filepath = PRESETS_DIR / name
    if not filepath.exists():
        raise FileNotFoundError(f"Preset '{name}' not found")

    with open(filepath, 'r') as f:
        return json.load(f)


def list_presets(asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available presets.

    Args:
        asset_type: Filter by asset type (optional)

    Returns:
        List of preset info dictionaries
    """
    ensure_presets_dir()

    presets = []
    for filepath in PRESETS_DIR.glob("*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if asset_type is None or data.get("asset_type") == asset_type:
                    presets.append({
                        "name": data.get("name", filepath.stem),
                        "asset_type": data.get("asset_type", "unknown"),
                        "description": data.get("description", ""),
                        "filepath": str(filepath)
                    })
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(presets, key=lambda x: x["name"])


def delete_preset(name: str) -> bool:
    """
    Delete a preset.

    Args:
        name: Preset name

    Returns:
        True if deleted, False if not found
    """
    ensure_presets_dir()

    filepath = PRESETS_DIR / f"{name}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def get_default_parameters(asset_type: str) -> Dict[str, Any]:
    """
    Get default parameters for an asset type.

    Args:
        asset_type: Type of asset

    Returns:
        Default parameters dictionary
    """
    defaults = {
        "rock": {
            "base_size": 1.0,
            "irregularity": 0.3,
            "subdivisions": 2
        },
        "tree": {
            "trunk_height": 2.0,
            "trunk_radius": 0.15,
            "canopy_radius": 1.2,
            "canopy_style": "spherical"
        },
        "building": {
            "width": 4.0,
            "depth": 4.0,
            "height": 6.0,
            "roof_style": "pointed"
        },
        "crystal": {
            "height": 2.0,
            "radius": 0.5,
            "points": 6
        },
        "barrel": {
            "height": 1.5,
            "radius": 0.5,
            "bulge": 0.1
        },
        "crate": {
            "size": 1.0
        },
        "terrain": {
            "width": 10.0,
            "depth": 10.0,
            "height": 2.0,
            "resolution": 32,
            "noise_scale": 3.0
        }
    }

    return defaults.get(asset_type, {})
