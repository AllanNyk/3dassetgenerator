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


def generate_oak_tree(
    height: float = 5.0,
    trunk_color: Tuple[int, int, int] = (90, 58, 30),
    leaf_color: Tuple[int, int, int] = (45, 122, 30),
    seed: Optional[int] = None
) -> trimesh.Scene:
    """Generate an oak tree with a thick trunk and billboard leaf sprites."""
    set_seed(seed)

    trunk_rgba = (*trunk_color, 255)
    trunk_meshes = []

    trunk_radius = height * 0.06
    fork_z = height * 0.45

    # --- Smooth tapered trunk (manual mesh, extends past fork for branch overlap) ---
    trunk_top = fork_z + height * 0.08  # extend past fork so branches connect
    n_rings = 14
    sections = 10
    trunk_verts = []
    trunk_faces = []
    wobble_amt = trunk_radius * 0.08

    for ri in range(n_rings + 1):
        t = ri / n_rings
        z = t * trunk_top
        r = trunk_radius * (1.0 - t * 0.5)
        # Root flare at base
        if t < 0.15:
            r *= 1.0 + (0.15 - t) * 2.5
        # Slight bulge at fork zone where branches emerge
        fork_t = fork_z / trunk_top
        dist_to_fork = abs(t - fork_t)
        if dist_to_fork < 0.15:
            r *= 1.0 + (0.15 - dist_to_fork) * 1.2
        for si in range(sections):
            a = si * 2 * np.pi / sections
            x = r * np.cos(a) + np.random.uniform(-wobble_amt, wobble_amt)
            y = r * np.sin(a) + np.random.uniform(-wobble_amt, wobble_amt)
            trunk_verts.append([x, y, z])

    for ri in range(n_rings):
        for si in range(sections):
            ns = (si + 1) % sections
            v0 = ri * sections + si
            v1 = ri * sections + ns
            v2 = (ri + 1) * sections + ns
            v3 = (ri + 1) * sections + si
            trunk_faces.extend([[v0, v1, v2], [v0, v2, v3]])

    trunk_mesh = trimesh.Trimesh(
        vertices=np.array(trunk_verts), faces=np.array(trunk_faces))
    trunk_mesh.fix_normals()
    apply_color_variation(trunk_mesh, trunk_rgba, 0.15)
    trunk_meshes.append(trunk_mesh)

    # --- Helper to orient a cylinder along a direction ---
    def _orient_branch(mesh, base, direction, length):
        d = np.array(direction, dtype=np.float64)
        d /= np.linalg.norm(d)
        z_axis = np.array([0.0, 0.0, 1.0])
        ax = np.cross(z_axis, d)
        ax_len = np.linalg.norm(ax)
        if ax_len > 1e-6:
            ax /= ax_len
            ang = np.arccos(np.clip(np.dot(z_axis, d), -1, 1))
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(ang, ax))
        mesh.apply_translation(base + d * length / 2)

    # --- Primary branches ---
    n_primary = 3 + int(np.random.uniform(0, 3))

    for bi in range(n_primary):
        azimuth = bi * (2 * np.pi / n_primary) + np.random.uniform(-0.3, 0.3)
        spread = np.random.uniform(0.4, 0.85)
        branch_len = height * np.random.uniform(0.22, 0.38)
        branch_r = trunk_radius * np.random.uniform(0.35, 0.55)

        dx = np.sin(spread) * np.cos(azimuth)
        dy = np.sin(spread) * np.sin(azimuth)
        dz = np.cos(spread)
        direction = np.array([dx, dy, dz])

        base = np.array([0.0, 0.0, fork_z + np.random.uniform(-0.3, 0.1)])
        tip = base + direction / np.linalg.norm(direction) * branch_len

        branch = trimesh.creation.cylinder(
            radius=branch_r, height=branch_len, sections=7)
        _orient_branch(branch, base, direction, branch_len)
        apply_color_variation(branch, trunk_rgba, 0.12)
        trunk_meshes.append(branch)

        # --- Secondary branches off each primary ---
        n_secondary = np.random.randint(2, 4)
        for si in range(n_secondary):
            frac = np.random.uniform(0.35, 0.85)
            sec_base = base + direction / np.linalg.norm(direction) * branch_len * frac
            sec_spread = np.random.uniform(0.4, 1.0)
            sec_azimuth = azimuth + np.random.uniform(-1.0, 1.0)
            sec_dx = np.sin(sec_spread) * np.cos(sec_azimuth)
            sec_dy = np.sin(sec_spread) * np.sin(sec_azimuth)
            sec_dz = np.cos(sec_spread)
            sec_dir = np.array([sec_dx, sec_dy, sec_dz])
            sec_len = branch_len * np.random.uniform(0.3, 0.6)
            sec_r = branch_r * np.random.uniform(0.3, 0.6)

            sec_branch = trimesh.creation.cylinder(
                radius=sec_r, height=sec_len, sections=6)
            _orient_branch(sec_branch, sec_base, sec_dir, sec_len)
            apply_color_variation(sec_branch, trunk_rgba, 0.12)
            trunk_meshes.append(sec_branch)

            # --- Tertiary twigs off some secondaries ---
            if np.random.random() < 0.5:
                twig_frac = np.random.uniform(0.5, 0.9)
                twig_base = sec_base + sec_dir / np.linalg.norm(sec_dir) * sec_len * twig_frac
                twig_az = sec_azimuth + np.random.uniform(-1.2, 1.2)
                twig_sp = np.random.uniform(0.5, 1.1)
                twig_dir = np.array([
                    np.sin(twig_sp) * np.cos(twig_az),
                    np.sin(twig_sp) * np.sin(twig_az),
                    np.cos(twig_sp)])
                twig_len = sec_len * np.random.uniform(0.3, 0.55)
                twig_r = sec_r * np.random.uniform(0.3, 0.6)

                twig = trimesh.creation.cylinder(
                    radius=twig_r, height=twig_len, sections=5)
                _orient_branch(twig, twig_base, twig_dir, twig_len)
                apply_color_variation(twig, trunk_rgba, 0.12)
                trunk_meshes.append(twig)

    # --- Leaf billboards ---
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    from textures.generator import oak_leaf_sprite

    sprite = oak_leaf_sprite(256, 256, seed)
    diffuse_img = Image.fromarray(sprite.diffuse)

    canopy_center = np.array([0.0, 0.0, fork_z + height * 0.25])
    canopy_rx = height * 0.45
    canopy_ry = height * 0.45
    canopy_rz = height * 0.30
    quad_size_base = height * 0.12

    n_quads = max(150, int(height * 50))
    template = np.array([
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5],
    ])
    template_uvs = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])

    all_verts = []
    all_faces = []
    all_uvs = []

    for i in range(n_quads):
        # Position within ellipsoidal canopy volume
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi * 0.55)
        r = np.random.uniform(0.15, 1.0) ** 0.5

        x = canopy_center[0] + r * canopy_rx * np.sin(phi) * np.cos(theta)
        y = canopy_center[1] + r * canopy_ry * np.sin(phi) * np.sin(theta)
        z = canopy_center[2] + r * canopy_rz * np.cos(phi)

        pos = np.array([x, y, z])

        qs = quad_size_base * np.random.uniform(0.7, 1.3)
        yaw = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        rot_2d = np.array([[cos_a, -sin_a, 0],
                           [sin_a, cos_a, 0],
                           [0, 0, 1]])

        verts = (rot_2d @ (template * qs).T).T + pos

        offset = i * 4
        all_verts.append(verts)
        all_faces.append(np.array([[offset, offset + 1, offset + 2],
                                    [offset, offset + 2, offset + 3]]))
        all_uvs.append(template_uvs)

    leaf_verts = np.vstack(all_verts)
    leaf_faces = np.vstack(all_faces)
    leaf_uvs = np.vstack(all_uvs)

    leaves_mesh = trimesh.Trimesh(vertices=leaf_verts, faces=leaf_faces)
    leaves_mesh.fix_normals()

    mat = PBRMaterial(
        baseColorTexture=diffuse_img,
        alphaMode='MASK',
        alphaCutoff=0.5,
        doubleSided=True,
        roughnessFactor=0.8,
    )
    leaves_mesh.visual = TextureVisuals(uv=leaf_uvs, material=mat)

    # --- Combine into Scene ---
    trunk_combined = trimesh.util.concatenate(trunk_meshes)
    trunk_combined.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(trunk_combined, node_name='trunk')
    scene.add_geometry(leaves_mesh, node_name='leaves')

    return scene


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

register(
    name="oak_tree", label="Oak Tree", category="Vegetation",
    params=[
        Param("height", "Height", "float", default=5.0, min=3.0, max=12.0),
        Param("trunk_color", "Trunk Color", "color", default="#5A3A1E"),
        Param("leaf_color", "Leaf Color", "color", default="#2D7A1E"),
    ],
)(generate_oak_tree)
