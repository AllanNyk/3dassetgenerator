"""
Rock and stone generators with vertex color support.
"""

import numpy as np
import trimesh
import random
from typing import Optional, Tuple

from core.registry import register, Param


def set_seed(seed: Optional[int] = None):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def apply_color_variation(
    mesh: trimesh.Trimesh,
    base_color: Tuple[int, int, int, int],
    variation: float = 0.1
) -> trimesh.Trimesh:
    """Apply vertex colors with slight random variation."""
    colors = np.tile(base_color, (len(mesh.vertices), 1)).astype(np.float32)

    for i in range(3):
        noise = np.random.uniform(-variation, variation, len(mesh.vertices)) * base_color[i]
        colors[:, i] = np.clip(colors[:, i] + noise, 0, 255)

    mesh.visual.vertex_colors = colors.astype(np.uint8)
    return mesh


def generate_rock(
    base_size: float = 1.0,
    irregularity: float = 0.3,
    subdivisions: int = 2,
    squash: Tuple[float, float] = (0.6, 0.9),
    color: Tuple[int, int, int] = (128, 128, 128),
    color_variation: float = 0.15,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate a procedural rock mesh with color.

    Args:
        base_size: Base size of the rock
        irregularity: How bumpy/irregular (0.0 to 1.0)
        subdivisions: Detail level (1-4 recommended)
        squash: Z-axis squash range (min, max)
        color: RGB color for the rock
        color_variation: Amount of color variation (0-1)
        seed: Random seed for reproducibility

    Returns:
        trimesh.Trimesh object with vertex colors
    """
    set_seed(seed)

    color_rgba = (*color, 255)

    # Start with an icosphere
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=base_size)
    vertices = mesh.vertices.copy()

    # Apply noise displacement
    for i in range(len(vertices)):
        direction = vertices[i] / np.linalg.norm(vertices[i])
        displacement = np.random.uniform(-irregularity, irregularity) * base_size
        vertices[i] += direction * displacement

    # Squash to make less spherical
    vertices[:, 2] *= np.random.uniform(*squash)

    mesh.vertices = vertices
    mesh.fix_normals()

    # Apply color
    apply_color_variation(mesh, color_rgba, color_variation)

    return mesh


def generate_rock_pile(
    num_rocks: int = 5,
    spread: float = 2.0,
    size_range: Tuple[float, float] = (0.3, 1.0),
    color: Tuple[int, int, int] = (128, 128, 128),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a pile of rocks with color."""
    set_seed(seed)

    meshes = []
    for i in range(num_rocks):
        size = np.random.uniform(*size_range)
        # Slightly vary color for each rock
        varied_color = (
            int(np.clip(color[0] + np.random.randint(-20, 20), 0, 255)),
            int(np.clip(color[1] + np.random.randint(-20, 20), 0, 255)),
            int(np.clip(color[2] + np.random.randint(-20, 20), 0, 255))
        )
        rock = generate_rock(base_size=size, irregularity=0.35, color=varied_color, seed=None)

        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        z = size * 0.3

        rock.apply_translation([x, y, z])
        meshes.append(rock)

    return trimesh.util.concatenate(meshes)


def generate_boulder(
    size: float = 2.0,
    color: Tuple[int, int, int] = (100, 100, 100),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a large boulder with color."""
    return generate_rock(
        base_size=size,
        irregularity=0.25,
        subdivisions=3,
        squash=(0.5, 0.7),
        color=color,
        seed=seed
    )


def generate_pebbles(
    num_pebbles: int = 20,
    spread: float = 1.0,
    size_range: Tuple[float, float] = (0.05, 0.15),
    color: Tuple[int, int, int] = (140, 140, 140),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate small scattered pebbles with color."""
    set_seed(seed)

    meshes = []
    for _ in range(num_pebbles):
        size = np.random.uniform(*size_range)
        varied_color = (
            int(np.clip(color[0] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(color[1] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(color[2] + np.random.randint(-15, 15), 0, 255))
        )
        pebble = generate_rock(base_size=size, irregularity=0.15, subdivisions=1, color=varied_color, seed=None)

        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        z = size * 0.3

        pebble.apply_translation([x, y, z])
        meshes.append(pebble)

    return trimesh.util.concatenate(meshes)


# --- Registry ---

register(
    name="rock", label="Single Rock", category="Rocks",
    params=[
        Param("base_size", "Size", "float", default=1.0, min=0.5, max=5.0),
        Param("irregularity", "Irregularity", "float", default=0.3, min=0.0, max=0.8),
        Param("subdivisions", "Detail", "int", default=2, min=1, max=4, step=1),
        Param("color", "Color", "color", default="#787878"),
    ],
)(generate_rock)

register(
    name="rock_pile", label="Rock Pile", category="Rocks",
    params=[
        Param("num_rocks", "Count", "int", default=5, min=2, max=20, step=1),
        Param("spread", "Spread", "float", default=3.0, min=1.0, max=10.0),
        Param("size_range", "Size Range", "range", range_default=(0.3, 1.0), range_min=0.1, range_max=2.0),
        Param("color", "Color", "color", default="#808080"),
    ],
)(generate_rock_pile)

register(
    name="boulder", label="Boulder", category="Rocks",
    params=[
        Param("size", "Size", "float", default=2.0, min=1.0, max=5.0),
        Param("color", "Color", "color", default="#646464"),
    ],
)(generate_boulder)

register(
    name="pebbles", label="Pebbles", category="Rocks",
    params=[
        Param("num_pebbles", "Count", "int", default=20, min=5, max=50, step=1),
        Param("spread", "Spread", "float", default=1.0, min=0.5, max=5.0),
        Param("size_range", "Size Range", "range", range_default=(0.05, 0.15), range_min=0.01, range_max=0.5),
        Param("color", "Color", "color", default="#8C8C8C"),
    ],
)(generate_pebbles)
