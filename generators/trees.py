"""
Tree and vegetation generators with vertex color support.
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


def apply_color(mesh: trimesh.Trimesh, color: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    """Apply a solid vertex color to a mesh."""
    colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    return mesh


def apply_color_variation(
    mesh: trimesh.Trimesh,
    base_color: Tuple[int, int, int, int],
    variation: float = 0.1
) -> trimesh.Trimesh:
    """Apply vertex colors with slight random variation for natural look."""
    colors = np.tile(base_color, (len(mesh.vertices), 1)).astype(np.float32)

    # Add random variation to RGB channels
    for i in range(3):
        noise = np.random.uniform(-variation, variation, len(mesh.vertices)) * base_color[i]
        colors[:, i] = np.clip(colors[:, i] + noise, 0, 255)

    mesh.visual.vertex_colors = colors.astype(np.uint8)
    return mesh


def generate_tree(
    trunk_height: float = 2.0,
    trunk_radius: float = 0.15,
    canopy_radius: float = 1.2,
    canopy_style: str = "spherical",
    trunk_color: Tuple[int, int, int] = (101, 67, 33),
    canopy_color: Tuple[int, int, int] = (34, 139, 34),
    color_variation: float = 0.15,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate a procedural tree mesh with colors.

    Args:
        trunk_height: Height of the trunk
        trunk_radius: Radius of the trunk
        canopy_radius: Radius of the foliage
        canopy_style: "spherical", "conical", or "layered"
        trunk_color: RGB color for trunk (default: brown)
        canopy_color: RGB color for canopy (default: forest green)
        color_variation: Amount of color variation (0-1)
        seed: Random seed

    Returns:
        trimesh.Trimesh object with vertex colors
    """
    set_seed(seed)

    trunk_rgba = (*trunk_color, 255)
    canopy_rgba = (*canopy_color, 255)

    meshes = []

    # Create trunk with brown color
    trunk = trimesh.creation.cylinder(
        radius=trunk_radius,
        height=trunk_height,
        sections=8
    )
    trunk.apply_translation([0, 0, trunk_height / 2])
    apply_color_variation(trunk, trunk_rgba, color_variation)
    meshes.append(trunk)

    # Create canopy with green color
    canopy_base_z = trunk_height * 0.7

    if canopy_style == "spherical":
        canopy = trimesh.creation.icosphere(subdivisions=2, radius=canopy_radius)
        canopy.vertices[:, 2] *= 0.7
        canopy.apply_translation([0, 0, canopy_base_z + canopy_radius * 0.5])
        apply_color_variation(canopy, canopy_rgba, color_variation)
        meshes.append(canopy)

    elif canopy_style == "conical":
        canopy = trimesh.creation.cone(
            radius=canopy_radius,
            height=canopy_radius * 2
        )
        canopy.apply_translation([0, 0, canopy_base_z + canopy_radius])
        apply_color_variation(canopy, canopy_rgba, color_variation)
        meshes.append(canopy)

    elif canopy_style == "layered":
        for i in range(3):
            layer_radius = canopy_radius * (1 - i * 0.2)
            layer = trimesh.creation.cone(
                radius=layer_radius,
                height=layer_radius * 0.8
            )
            layer_z = canopy_base_z + i * (canopy_radius * 0.5)
            layer.apply_translation([0, 0, layer_z])
            # Slightly different green for each layer
            layer_color = (
                canopy_color[0] + i * 10,
                canopy_color[1] - i * 15,
                canopy_color[2] + i * 5,
                255
            )
            apply_color_variation(layer, layer_color, color_variation)
            meshes.append(layer)

    combined = trimesh.util.concatenate(meshes)
    return combined


def generate_forest_patch(
    num_trees: int = 10,
    spread: float = 10.0,
    trunk_color: Tuple[int, int, int] = (101, 67, 33),
    canopy_color: Tuple[int, int, int] = (34, 139, 34),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a patch of varied trees with colors."""
    set_seed(seed)

    styles = ["spherical", "conical", "layered"]
    meshes = []

    for i in range(num_trees):
        style = random.choice(styles)
        height = np.random.uniform(1.5, 3.0)

        # Vary the canopy color slightly for each tree
        varied_canopy = (
            int(np.clip(canopy_color[0] + np.random.randint(-20, 20), 0, 255)),
            int(np.clip(canopy_color[1] + np.random.randint(-30, 30), 0, 255)),
            int(np.clip(canopy_color[2] + np.random.randint(-20, 20), 0, 255))
        )

        tree = generate_tree(
            trunk_height=height,
            canopy_radius=height * 0.5,
            canopy_style=style,
            trunk_color=trunk_color,
            canopy_color=varied_canopy
        )

        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        tree.apply_translation([x, y, 0])
        meshes.append(tree)

    return trimesh.util.concatenate(meshes)


def generate_bush(
    radius: float = 0.8,
    height: float = 0.6,
    color: Tuple[int, int, int] = (34, 120, 34),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a simple bush with color."""
    set_seed(seed)

    color_rgba = (*color, 255)
    meshes = []

    # Main body
    main = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    main.vertices[:, 2] *= height / radius
    main.apply_translation([0, 0, height * 0.5])
    apply_color_variation(main, color_rgba, 0.2)
    meshes.append(main)

    # Add some random bumps
    for _ in range(3):
        bump_radius = radius * np.random.uniform(0.3, 0.5)
        bump = trimesh.creation.icosphere(subdivisions=1, radius=bump_radius)

        angle = np.random.uniform(0, 2 * np.pi)
        dist = radius * 0.5
        x = np.cos(angle) * dist
        y = np.sin(angle) * dist
        z = height * np.random.uniform(0.3, 0.7)

        bump.apply_translation([x, y, z])
        # Slightly different shade
        bump_color = (
            color[0] + np.random.randint(-15, 15),
            color[1] + np.random.randint(-20, 20),
            color[2] + np.random.randint(-10, 10),
            255
        )
        apply_color_variation(bump, bump_color, 0.15)
        meshes.append(bump)

    return trimesh.util.concatenate(meshes)


def generate_stump(
    radius: float = 0.3,
    height: float = 0.4,
    color: Tuple[int, int, int] = (101, 67, 33),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a tree stump with color."""
    set_seed(seed)

    color_rgba = (*color, 255)

    # Main stump cylinder
    stump = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=12
    )
    stump.apply_translation([0, 0, height / 2])
    apply_color_variation(stump, color_rgba, 0.15)

    meshes = [stump]

    for i in range(4):
        angle = i * np.pi / 2 + np.random.uniform(-0.3, 0.3)
        root = trimesh.creation.cylinder(
            radius=radius * 0.3,
            height=radius * 0.8,
            sections=6
        )
        root.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 3, [0, 1, 0]
        ))
        root.apply_transform(trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1]
        ))
        root.apply_translation([np.cos(angle) * radius * 0.7, np.sin(angle) * radius * 0.7, 0.1])
        apply_color_variation(root, color_rgba, 0.15)
        meshes.append(root)

    return trimesh.util.concatenate(meshes)


# --- Registry ---

register(
    name="tree", label="Tree", category="Vegetation",
    params=[
        Param("trunk_height", "Trunk Height", "float", default=2.0, min=1.0, max=10.0),
        Param("canopy_radius", "Canopy Size", "float", default=1.2, min=0.5, max=5.0),
        Param("canopy_style", "Style", "str", default="Spherical",
              choices=["Spherical", "Conical", "Layered"]),
        Param("trunk_color", "Trunk Color", "color", default="#654321"),
        Param("canopy_color", "Canopy Color", "color", default="#228B22"),
    ],
)(generate_tree)

register(
    name="forest", label="Forest Patch", category="Vegetation",
    params=[
        Param("num_trees", "Trees", "int", default=10, min=3, max=30, step=1),
        Param("spread", "Spread", "float", default=10.0, min=5.0, max=30.0),
        Param("trunk_color", "Trunk Color", "color", default="#654321"),
        Param("canopy_color", "Canopy Color", "color", default="#228B22"),
    ],
)(generate_forest_patch)

register(
    name="bush", label="Bush", category="Vegetation",
    params=[
        Param("radius", "Radius", "float", default=0.8, min=0.3, max=2.0),
        Param("height", "Height", "float", default=0.6, min=0.2, max=1.5),
        Param("color", "Bush Color", "color", default="#227822"),
    ],
)(generate_bush)

register(
    name="stump", label="Stump", category="Vegetation",
    params=[
        Param("radius", "Radius", "float", default=0.3, min=0.1, max=1.0),
        Param("height", "Height", "float", default=0.4, min=0.1, max=1.0),
        Param("color", "Color", "color", default="#654321"),
    ],
)(generate_stump)
