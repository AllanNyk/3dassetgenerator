"""
Terrain and heightmap generators.
"""

import numpy as np
import trimesh
import random
from typing import Optional, Tuple
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from core.noise import noise_2d_grid, PerlinNoise, fractal_noise

from core.registry import register, Param


def set_seed(seed: Optional[int] = None):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def apply_terrain_colors(
    mesh: trimesh.Trimesh,
    base_color: Tuple[int, int, int, int],
    variation: float = 0.1
) -> trimesh.Trimesh:
    """Apply vertex colors with height-based variation."""
    verts = mesh.vertices
    z = verts[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    t = (z - z_min) / z_range  # 0 at bottom, 1 at top

    colors = np.tile(base_color, (len(verts), 1)).astype(np.float32)
    # Darken lower areas, lighten higher
    for i in range(3):
        height_shift = (t - 0.5) * 0.3 * base_color[i]
        noise = np.random.uniform(-variation, variation, len(verts)) * base_color[i]
        colors[:, i] = np.clip(colors[:, i] + height_shift + noise, 0, 255)
    mesh.visual.vertex_colors = colors.astype(np.uint8)
    return mesh


def generate_heightmap(
    width: int = 64,
    height: int = 64,
    noise_scale: float = 3.0,
    octaves: int = 4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a 2D heightmap using fractal noise.

    Args:
        width, height: Grid dimensions
        noise_scale: Scale of the noise (higher = more features)
        octaves: Number of noise layers
        seed: Random seed

    Returns:
        2D numpy array with values in [0, 1]
    """
    return noise_2d_grid(width, height, scale=noise_scale, octaves=octaves, seed=seed)


def generate_terrain(
    width: float = 10.0,
    depth: float = 10.0,
    height: float = 2.0,
    resolution: int = 32,
    noise_scale: float = 3.0,
    octaves: int = 4,
    color: Tuple[int, int, int] = (90, 140, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate a terrain mesh from noise.

    Args:
        width: Terrain width (X axis)
        depth: Terrain depth (Y axis)
        height: Maximum terrain height (Z axis)
        resolution: Grid resolution (vertices per side)
        noise_scale: Noise scale
        octaves: Noise octaves
        seed: Random seed

    Returns:
        trimesh.Trimesh terrain mesh
    """
    set_seed(seed)

    # Generate heightmap
    heightmap = generate_heightmap(resolution, resolution, noise_scale, octaves, seed)

    # Flatten the center region with a smooth falloff
    center = (resolution - 1) / 2.0
    flat_val = heightmap[int(center), int(center)]
    flat_radius = 0.25  # fraction of grid that is fully flat
    fade_width = 0.15   # fraction over which it blends back to noise
    for j in range(resolution):
        for i in range(resolution):
            dx = (i - center) / center  # -1 to 1
            dy = (j - center) / center
            dist = np.sqrt(dx * dx + dy * dy)
            blend = np.clip((dist - flat_radius) / fade_width, 0, 1)
            heightmap[j, i] = flat_val * (1 - blend) + heightmap[j, i] * blend

    # Create vertices
    vertices = []
    for j in range(resolution):
        for i in range(resolution):
            x = (i / (resolution - 1) - 0.5) * width
            y = (j / (resolution - 1) - 0.5) * depth
            z = heightmap[j, i] * height
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create faces (two triangles per grid cell)
    faces = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            # Vertex indices
            v0 = j * resolution + i
            v1 = j * resolution + (i + 1)
            v2 = (j + 1) * resolution + (i + 1)
            v3 = (j + 1) * resolution + i

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    apply_terrain_colors(mesh, (*color, 255), 0.1)

    return mesh


def generate_terrain_island(
    radius: float = 5.0,
    height: float = 2.0,
    resolution: int = 32,
    noise_scale: float = 3.0,
    falloff: float = 2.0,
    color: Tuple[int, int, int] = (80, 130, 50),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate an island-style terrain that falls off at edges.

    Args:
        radius: Island radius
        height: Maximum height
        resolution: Grid resolution
        noise_scale: Noise scale
        falloff: Edge falloff strength
        seed: Random seed

    Returns:
        trimesh.Trimesh terrain mesh
    """
    set_seed(seed)

    # Generate base heightmap
    heightmap = generate_heightmap(resolution, resolution, noise_scale, 4, seed)

    # Apply circular falloff
    center = resolution / 2
    for j in range(resolution):
        for i in range(resolution):
            # Distance from center (normalized)
            dx = (i - center) / center
            dy = (j - center) / center
            dist = np.sqrt(dx*dx + dy*dy)

            # Falloff function
            falloff_value = max(0, 1 - (dist ** falloff))
            heightmap[j, i] *= falloff_value

    # Create mesh (similar to regular terrain)
    vertices = []
    diameter = radius * 2

    for j in range(resolution):
        for i in range(resolution):
            x = (i / (resolution - 1) - 0.5) * diameter
            y = (j / (resolution - 1) - 0.5) * diameter
            z = heightmap[j, i] * height
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            v0 = j * resolution + i
            v1 = j * resolution + (i + 1)
            v2 = (j + 1) * resolution + (i + 1)
            v3 = (j + 1) * resolution + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    apply_terrain_colors(mesh, (*color, 255), 0.1)

    return mesh


def generate_terrain_plateau(
    width: float = 10.0,
    depth: float = 10.0,
    base_height: float = 1.0,
    plateau_height: float = 0.5,
    resolution: int = 32,
    noise_scale: float = 2.0,
    color: Tuple[int, int, int] = (100, 90, 70),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate terrain with a flat plateau on top.

    Args:
        width, depth: Terrain dimensions
        base_height: Height of the plateau base
        plateau_height: Additional height variation on top
        resolution: Grid resolution
        noise_scale: Noise scale
        seed: Random seed

    Returns:
        trimesh.Trimesh terrain mesh
    """
    set_seed(seed)

    # Generate heightmap
    heightmap = generate_heightmap(resolution, resolution, noise_scale, 3, seed)

    # Create plateau effect - flatten the middle heights
    for j in range(resolution):
        for i in range(resolution):
            h = heightmap[j, i]
            # Squash heights toward the middle
            if h > 0.3 and h < 0.7:
                heightmap[j, i] = 0.5 + (h - 0.5) * 0.3

    # Scale heights
    heightmap = base_height + heightmap * plateau_height

    # Create mesh
    vertices = []
    for j in range(resolution):
        for i in range(resolution):
            x = (i / (resolution - 1) - 0.5) * width
            y = (j / (resolution - 1) - 0.5) * depth
            z = heightmap[j, i]
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            v0 = j * resolution + i
            v1 = j * resolution + (i + 1)
            v2 = (j + 1) * resolution + (i + 1)
            v3 = (j + 1) * resolution + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    apply_terrain_colors(mesh, (*color, 255), 0.1)

    return mesh


# --- Registry ---

register(
    name="terrain", label="Terrain", category="Terrain",
    params=[
        Param("width", "Width", "float", default=10.0, min=5.0, max=50.0),
        Param("depth", "Depth", "float", default=10.0, min=5.0, max=50.0),
        Param("height", "Height", "float", default=2.0, min=0.5, max=10.0),
        Param("resolution", "Resolution", "int", default=32, min=8, max=64, step=1),
        Param("noise_scale", "Noise Scale", "float", default=3.0, min=1.0, max=10.0),
        Param("color", "Color", "color", default="#5A8C3C"),
    ],
)(generate_terrain)

register(
    name="island", label="Island", category="Terrain",
    params=[
        Param("radius", "Radius", "float", default=5.0, min=2.0, max=20.0),
        Param("height", "Height", "float", default=2.0, min=0.5, max=5.0),
        Param("resolution", "Resolution", "int", default=32, min=16, max=64, step=1),
        Param("noise_scale", "Noise Scale", "float", default=3.0, min=1.0, max=10.0),
        Param("color", "Color", "color", default="#508232"),
    ],
)(generate_terrain_island)

register(
    name="plateau", label="Plateau", category="Terrain",
    params=[
        Param("width", "Width", "float", default=10.0, min=5.0, max=50.0),
        Param("depth", "Depth", "float", default=10.0, min=5.0, max=50.0),
        Param("base_height", "Base Height", "float", default=1.0, min=0.5, max=5.0),
        Param("plateau_height", "Plateau Height", "float", default=0.5, min=0.1, max=3.0),
        Param("resolution", "Resolution", "int", default=32, min=8, max=64, step=1),
        Param("noise_scale", "Noise Scale", "float", default=2.0, min=1.0, max=10.0),
        Param("color", "Color", "color", default="#645A46"),
    ],
)(generate_terrain_plateau)
