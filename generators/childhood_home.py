"""
Childhood Home generator — a variation of the Danish 1870s farmhouse.
Core gameplay building with traversable interior.
"""

import numpy as np
import trimesh
import random
from collections import defaultdict
from typing import Optional, Tuple

from core.registry import register, Param


class MeshCollector:
    """Collects meshes into named material groups for textured mode."""

    def __init__(self):
        self._groups = defaultdict(list)
        self._current_group = 'default'
        self._flat = []

    def set_group(self, name: str):
        self._current_group = name

    def append(self, mesh):
        self._groups[self._current_group].append(mesh)
        self._flat.append(mesh)

    def extend(self, meshes):
        for m in meshes:
            self.append(m)

    @property
    def flat(self):
        return self._flat

    @property
    def groups(self):
        return dict(self._groups)


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


def _wobble_box(extents, wobble=0.015):
    """Create a box with slight vertex displacement for handcrafted look."""
    box = trimesh.creation.box(extents=extents)
    verts = box.vertices.copy()
    max_dim = max(extents)
    for i in range(3):
        verts[:, i] += np.random.uniform(-wobble, wobble, len(verts)) * max_dim
    box.vertices = verts
    return box


def _build_wall_with_opening(
    wall_w: float, wall_h: float, wall_t: float,
    open_cx: float, open_sill_z: float,
    open_w: float, open_h: float,
    outer_color: Tuple[int, int, int, int],
    inner_color: Tuple[int, int, int, int],
    variation: float = 0.04,
    wobble: float = 0.012,
) -> Tuple[list, list]:
    """Build a wall section with a rectangular opening (window or door) as 4 panels.

    Returns (outer_panels, inner_panels). Each panel section has outer + inner slabs.
    open_cx is the opening center X relative to the wall center.
    """
    outers = []
    inners = []
    hw = wall_w / 2
    open_left = open_cx - open_w / 2
    open_right = open_cx + open_w / 2
    open_top = open_sill_z + open_h

    def _dual_slab(ext, pos):
        outer_t = wall_t * 0.55
        inner_t = wall_t * 0.45
        outer = _wobble_box([ext[0], outer_t, ext[2]], wobble=wobble)
        outer.apply_translation([pos[0], pos[1] + (inner_t / 2), pos[2]])
        apply_color_variation(outer, outer_color, variation)
        inner = _wobble_box([ext[0], inner_t, ext[2]], wobble=wobble)
        inner.apply_translation([pos[0], pos[1] - (outer_t / 2), pos[2]])
        apply_color_variation(inner, inner_color, variation)
        outers.append(outer)
        inners.append(inner)

    # Left panel
    left_w = open_left - (-hw)
    if left_w > 0.01:
        cx = -hw + left_w / 2
        _dual_slab(
            [left_w, wall_t, wall_h],
            [cx, 0, wall_h / 2],
        )

    # Right panel
    right_w = hw - open_right
    if right_w > 0.01:
        cx = open_right + right_w / 2
        _dual_slab(
            [right_w, wall_t, wall_h],
            [cx, 0, wall_h / 2],
        )

    # Below opening
    below_h = open_sill_z
    if below_h > 0.01:
        _dual_slab(
            [open_w, wall_t, below_h],
            [open_cx, 0, below_h / 2],
        )

    # Above opening
    above_h = wall_h - open_top
    if above_h > 0.01:
        _dual_slab(
            [open_w, wall_t, above_h],
            [open_cx, 0, open_top + above_h / 2],
        )

    return outers, inners


def _build_smooth_arch(half_w, arch_rise, base_z, fd, ft, n_seg=16):
    """Build a smooth arch mesh from vertices and faces.

    half_w: half the window width (arch X radius)
    arch_rise: arch height (Z radius)
    base_z: Z position of arch base
    fd: frame depth (Y extent)
    ft: frame thickness (radial thickness of arch)
    n_seg: number of angular segments
    Returns a trimesh.Trimesh.
    """
    verts = []
    faces = []
    hy = fd / 2

    for i in range(n_seg + 1):
        angle = np.pi * i / n_seg
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Inner edge of arch
        ix = -half_w * cos_a
        iz = base_z + arch_rise * sin_a
        # Outer edge of arch (offset outward by ft)
        ox = -(half_w + ft) * cos_a
        oz = base_z + (arch_rise + ft) * sin_a

        # 4 verts per step: inner-front, inner-back, outer-front, outer-back
        verts.append([ix, hy, iz])       # 0: inner-front
        verts.append([ix, -hy, iz])      # 1: inner-back
        verts.append([ox, hy, oz])       # 2: outer-front
        verts.append([ox, -hy, oz])      # 3: outer-back

    for i in range(n_seg):
        b = i * 4
        n = (i + 1) * 4
        # Front face (inner-front to outer-front)
        faces.append([b + 0, n + 0, n + 2])
        faces.append([b + 0, n + 2, b + 2])
        # Back face (inner-back to outer-back)
        faces.append([b + 1, b + 3, n + 3])
        faces.append([b + 1, n + 3, n + 1])
        # Outer face
        faces.append([b + 2, n + 2, n + 3])
        faces.append([b + 2, n + 3, b + 3])
        # Inner face
        faces.append([b + 0, b + 1, n + 1])
        faces.append([b + 0, n + 1, n + 0])

    # End caps (left and right ends of the arch)
    # Left end cap (i=0)
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])
    # Right end cap (i=n_seg)
    e = n_seg * 4
    faces.append([e + 0, e + 1, e + 3])
    faces.append([e + 0, e + 3, e + 2])

    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
    mesh.fix_normals()
    return mesh


def _build_arched_window(
    win_w: float, win_h: float, arch_rise: float,
    sill_z: float, wall_t: float,
    frame_color: Tuple[int, int, int, int],
    sill_color: Tuple[int, int, int, int],
    variation: float = 0.05,
    glass_color: Tuple[int, int, int, int] = None,
) -> list:
    """Build a window frame with arch top and protruding sill.

    Returns list of meshes centered at X=0.
    glass_color: optional RGBA tuple for semi-transparent glass pane.
    """
    meshes = []
    ft = 0.06  # frame thickness
    fd = wall_t + 0.02  # frame depth (slightly deeper than wall)

    # Side jambs (2)
    for side in [-1, 1]:
        jamb = trimesh.creation.box(extents=[ft, fd, win_h])
        jamb.apply_translation([side * (win_w / 2 + ft / 2), 0, sill_z + win_h / 2])
        apply_color_variation(jamb, frame_color, variation)
        meshes.append(jamb)

    # Bottom sill (interior flush)
    sill = trimesh.creation.box(extents=[win_w + ft * 2, fd, ft])
    sill.apply_translation([0, 0, sill_z - ft / 2])
    apply_color_variation(sill, frame_color, variation)
    meshes.append(sill)

    # Outer protruding sill
    outer_sill = trimesh.creation.box(extents=[win_w + ft * 3, 0.08, 0.035])
    outer_sill.apply_translation([0, wall_t / 2 + 0.04, sill_z - ft / 2])
    apply_color_variation(outer_sill, sill_color, variation)
    meshes.append(outer_sill)

    # Smooth arched top
    arch = _build_smooth_arch(win_w / 2, arch_rise, sill_z + win_h, fd, ft)
    apply_color_variation(arch, frame_color, variation)
    meshes.append(arch)

    # Glass pane (rectangular + arch fill)
    if glass_color is not None:
        pane_t = 0.005   # pane thickness
        pane_inset = 0.01  # inset from frame inner edges
        pane_w = win_w - pane_inset

        # Rectangular pane
        pane = trimesh.creation.box(extents=[pane_w, pane_t, win_h])
        pane.apply_translation([0, 0, sill_z + win_h / 2])
        pane.visual.vertex_colors = np.tile(
            glass_color, (len(pane.vertices), 1)
        ).astype(np.uint8)
        meshes.append(pane)

        # Arch pane (filled semi-ellipse, front + back faces)
        n_pts = 16
        verts = []
        faces = []
        hy = pane_t / 2
        base_z = sill_z + win_h
        arch_hw = pane_w / 2

        # Front face: center + arc points
        verts.append([0, hy, base_z])
        for i in range(n_pts + 1):
            angle = np.pi * i / n_pts
            x = -arch_hw * np.cos(angle)
            z = arch_rise * np.sin(angle) + base_z
            verts.append([x, hy, z])
        for i in range(n_pts):
            faces.append([0, i + 1, i + 2])

        # Back face: center + arc points (reversed winding)
        b = len(verts)
        verts.append([0, -hy, base_z])
        for i in range(n_pts + 1):
            angle = np.pi * i / n_pts
            x = -arch_hw * np.cos(angle)
            z = arch_rise * np.sin(angle) + base_z
            verts.append([x, -hy, z])
        for i in range(n_pts):
            faces.append([b, b + i + 2, b + i + 1])

        arch_pane = trimesh.Trimesh(
            vertices=np.array(verts), faces=np.array(faces)
        )
        arch_pane.fix_normals()
        arch_pane.visual.vertex_colors = np.tile(
            glass_color, (len(arch_pane.vertices), 1)
        ).astype(np.uint8)
        meshes.append(arch_pane)

    return meshes


def _build_door_frame(
    door_w: float, door_h: float,
    wall_t: float,
    frame_color: Tuple[int, int, int, int],
    variation: float = 0.05,
) -> list:
    """Build an interior door frame (two jambs + lintel). Centered at X=0, base at Z=0."""
    meshes = []
    ft = 0.06  # frame thickness
    fd = wall_t + 0.04  # slightly deeper than wall

    # Side jambs (spread along Y, depth along X to match partition walls)
    for side in [-1, 1]:
        jamb = trimesh.creation.box(extents=[fd, ft, door_h])
        jamb.apply_translation([0, side * (door_w / 2 + ft / 2), door_h / 2])
        apply_color_variation(jamb, frame_color, variation)
        meshes.append(jamb)

    # Lintel (top)
    lintel = trimesh.creation.box(extents=[fd, door_w + ft * 2, ft])
    lintel.apply_translation([0, 0, door_h + ft / 2])
    apply_color_variation(lintel, frame_color, variation)
    meshes.append(lintel)

    return meshes


def _build_front_door_frame(
    door_w: float, door_h: float,
    wall_t: float,
    frame_color: Tuple[int, int, int, int],
    variation: float = 0.05,
) -> list:
    """Build a door frame for a wall running along X (front/back walls).
    Jambs spread along X, depth along Y."""
    meshes = []
    ft = 0.06
    fd = wall_t + 0.04

    for side in [-1, 1]:
        jamb = trimesh.creation.box(extents=[ft, fd, door_h])
        jamb.apply_translation([side * (door_w / 2 + ft / 2), 0, door_h / 2])
        apply_color_variation(jamb, frame_color, variation)
        meshes.append(jamb)

    lintel = trimesh.creation.box(extents=[door_w + ft * 2, fd, ft])
    lintel.apply_translation([0, 0, door_h + ft / 2])
    apply_color_variation(lintel, frame_color, variation)
    meshes.append(lintel)

    return meshes


def _build_plank_surface(
    width: float, depth: float, z: float,
    plank_w: float, axis: str,
    color: Tuple[int, int, int, int],
    variation: float = 0.06,
    thickness: float = 0.04,
) -> list:
    """Build a surface of planks (floor or ceiling).

    axis: 'y' means planks run along Y (each plank spans depth),
          'x' means planks run along X.
    Returns list of meshes.
    """
    meshes = []
    if axis == 'y':
        n = max(1, int(width / plank_w))
        actual_w = width / n
        for i in range(n):
            px = -width / 2 + actual_w * (i + 0.5)
            p = trimesh.creation.box(extents=[actual_w * 0.995, depth, thickness])
            p.apply_translation([px, 0, z])
            c = list(color)
            for ci in range(3):
                c[ci] = int(np.clip(c[ci] + np.random.randint(-6, 6), 0, 255))
            apply_color_variation(p, tuple(c), variation)
            meshes.append(p)
    else:
        n = max(1, int(depth / plank_w))
        actual_w = depth / n
        for i in range(n):
            py = -depth / 2 + actual_w * (i + 0.5)
            p = trimesh.creation.box(extents=[width, actual_w * 0.995, thickness])
            p.apply_translation([0, py, z])
            c = list(color)
            for ci in range(3):
                c[ci] = int(np.clip(c[ci] + np.random.randint(-6, 6), 0, 255))
            apply_color_variation(p, tuple(c), variation)
            meshes.append(p)
    return meshes


def _build_farmhouse_chimney(
    base_w: float, base_d: float, height: float,
    color: Tuple[int, int, int, int],
    variation: float = 0.06,
) -> list:
    """Build a simple brick chimney from stacked wobble boxes."""
    meshes = []
    n_courses = max(3, int(height / 0.25))
    course_h = height / n_courses
    for i in range(n_courses):
        taper = 1.0 - 0.05 * (i / n_courses)
        brick = _wobble_box(
            [base_w * taper, base_d * taper, course_h],
            wobble=0.015,
        )
        brick.apply_translation([0, 0, course_h * (i + 0.5)])
        c = list(color)
        for ci in range(3):
            c[ci] = int(np.clip(c[ci] + np.random.randint(-5, 5), 0, 255))
        apply_color_variation(brick, tuple(c), variation)
        meshes.append(brick)

    # Crown cap (wider)
    cap = _wobble_box([base_w * 1.15, base_d * 1.15, course_h * 0.6], wobble=0.01)
    cap.apply_translation([0, 0, height + course_h * 0.3])
    apply_color_variation(cap, color, variation * 0.5)
    meshes.append(cap)

    return meshes


def _build_gable_thatch(
    mid_z: float, ridge_z: float, D: float, gable_h: float,
    wall_face_x: float, sign: int, roof_t: float,
    overhang: float = 0.3, droop: float = 0.15,
    n_rows: int = 10, n_cols: int = 8,
) -> trimesh.Trimesh:
    """Build a curved thatch panel for the upper half of a gable end.

    Covers the triangular upper gable area with a thick mesh that curves
    gently outward at the lower edge, creating a soft overhanging eave.

    wall_face_x: X position of the outer wall face
    sign: +1 for right end wall, -1 for left end wall
    """
    verts = []
    faces = []
    half_h = ridge_z - mid_z

    # Apex (ridge): single outer + inner vertex
    verts.append([wall_face_x + sign * roof_t, 0, ridge_z])   # 0: outer
    verts.append([wall_face_x, 0, ridge_z])                     # 1: inner

    for row in range(1, n_rows + 1):
        t = row / n_rows
        z_base = ridge_z - t * half_h
        half_w = (D / 2) * (t * half_h) / gable_h

        # Smooth outward curve and slight droop (sin² for gentle onset)
        curve = overhang * np.sin(t * np.pi / 2) ** 2
        z_drop = droop * np.sin(t * np.pi / 2) ** 2

        x_outer = wall_face_x + sign * (roof_t + curve)
        x_inner = wall_face_x

        for col in range(n_cols + 1):
            frac = col / n_cols
            y = -half_w + frac * 2 * half_w
            verts.append([x_outer, y, z_base - z_drop])
            verts.append([x_inner, y, z_base])

    def oi(r, c):
        if r == 0:
            return 0
        return 2 + (r - 1) * (n_cols + 1) * 2 + c * 2

    def ii(r, c):
        if r == 0:
            return 1
        return 2 + (r - 1) * (n_cols + 1) * 2 + c * 2 + 1

    # Outer surface
    for c in range(n_cols):
        faces.append([oi(0, 0), oi(1, c), oi(1, c + 1)])
    for r in range(1, n_rows):
        for c in range(n_cols):
            faces.append([oi(r, c), oi(r + 1, c), oi(r + 1, c + 1)])
            faces.append([oi(r, c), oi(r + 1, c + 1), oi(r, c + 1)])

    # Inner surface (reversed winding)
    for c in range(n_cols):
        faces.append([ii(0, 0), ii(1, c + 1), ii(1, c)])
    for r in range(1, n_rows):
        for c in range(n_cols):
            faces.append([ii(r, c), ii(r + 1, c + 1), ii(r + 1, c)])
            faces.append([ii(r, c), ii(r, c + 1), ii(r + 1, c + 1)])

    # Bottom edge
    for c in range(n_cols):
        faces.append([oi(n_rows, c), oi(n_rows, c + 1), ii(n_rows, c + 1)])
        faces.append([oi(n_rows, c), ii(n_rows, c + 1), ii(n_rows, c)])

    # Left side edge (col=0)
    faces.append([oi(0, 0), ii(0, 0), ii(1, 0)])
    faces.append([oi(0, 0), ii(1, 0), oi(1, 0)])
    for r in range(1, n_rows):
        faces.append([oi(r, 0), ii(r, 0), ii(r + 1, 0)])
        faces.append([oi(r, 0), ii(r + 1, 0), oi(r + 1, 0)])

    # Right side edge (col=n_cols)
    faces.append([oi(0, 0), oi(1, n_cols), ii(1, n_cols)])
    faces.append([oi(0, 0), ii(1, n_cols), ii(0, 0)])
    for r in range(1, n_rows):
        faces.append([oi(r, n_cols), oi(r + 1, n_cols), ii(r + 1, n_cols)])
        faces.append([oi(r, n_cols), ii(r + 1, n_cols), ii(r, n_cols)])

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces),
    )
    mesh.fix_normals()
    return mesh


_TEXTURE_PRESETS = ["None", "Stone", "White Stone", "Wood", "Metal", "Grass", "Dirt", "Thatch", "Dark Thatch", "Brick", "White Brick", "Bark", "Cobblestone"]


def _load_custom_texture(path):
    """Load a user-supplied image file as a TextureSet (256x256)."""
    from PIL import Image
    from textures.generator import TextureSet
    img = Image.open(path).convert("RGB").resize((256, 256))
    diffuse = np.array(img, dtype=np.uint8)
    # Flat normal map (pointing straight out)
    normal = np.full((256, 256, 3), (128, 128, 255), dtype=np.uint8)
    # Uniform mid-roughness
    roughness = np.full((256, 256), 128, dtype=np.uint8)
    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=256, height=256, name="custom")


def _load_texture(name, seed):
    """Load a texture preset by name (case-insensitive)."""
    from textures.generator import (
        stone_texture, wood_texture, metal_texture, grass_texture, dirt_texture,
        thatch_texture, white_stone_texture, dark_thatch_texture,
        brick_texture, white_brick_texture, bark_texture, cobblestone_texture)
    loaders = {
        'stone': stone_texture,
        'white stone': white_stone_texture,
        'wood': wood_texture,
        'metal': metal_texture,
        'grass': grass_texture,
        'dirt': dirt_texture,
        'thatch': thatch_texture,
        'dark thatch': dark_thatch_texture,
        'brick': brick_texture,
        'white brick': white_brick_texture,
        'bark': bark_texture,
        'cobblestone': cobblestone_texture,
    }
    return loaders[name.lower()](256, 256, seed)


def _build_textured_scene(groups, seed, tex_choices, img_overrides=None):
    """Build a trimesh.Scene with PBR-textured materials from grouped meshes.

    tex_choices: dict mapping group name -> (preset_name, uv_scale)
    img_overrides: optional dict mapping group name -> image file path
    """
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    from core.mesh import unweld_mesh, compute_triplanar_uvs

    img_overrides = img_overrides or {}

    # Cache textures so identical presets aren't generated twice
    tex_cache = {}
    tex_map = {}
    for group_name, (preset, uv_scale) in tex_choices.items():
        # "None" means keep vertex colors for this group
        if preset.lower() == "none":
            continue
        # Custom image overrides the preset if provided
        if group_name in img_overrides and img_overrides[group_name]:
            tex_map[group_name] = (_load_custom_texture(img_overrides[group_name]), uv_scale)
            continue
        if preset not in tex_cache:
            tex_cache[preset] = _load_texture(preset, seed)
        tex_map[group_name] = (tex_cache[preset], uv_scale)

    scene = trimesh.Scene()

    for group_name, mesh_list in groups.items():
        if not mesh_list:
            continue
        combined = trimesh.util.concatenate(mesh_list)

        if group_name == 'glass':
            # Glass keeps vertex colors, just set blend material
            combined.visual.material = PBRMaterial(
                alphaMode='BLEND', doubleSided=True)
            scene.add_geometry(combined, node_name=group_name)
            continue

        if group_name not in tex_map:
            scene.add_geometry(combined, node_name=group_name)
            continue

        tex_set, uv_scale = tex_map[group_name]

        unwelded = unweld_mesh(combined)
        uvs = compute_triplanar_uvs(unwelded, scale=uv_scale)

        diffuse_img = Image.fromarray(tex_set.diffuse)
        normal_img = Image.fromarray(tex_set.normal)

        mat = PBRMaterial(
            baseColorTexture=diffuse_img,
            normalTexture=normal_img,
            roughnessFactor=0.7,
        )

        unwelded.visual = TextureVisuals(uv=uvs, material=mat)
        scene.add_geometry(unwelded, node_name=group_name)

    return scene


# ---------------------------------------------------------------------------
# Childhood Home Generator
# ---------------------------------------------------------------------------

def generate_childhood_home(
    size: float = 1.0,
    wall_color: Tuple[int, int, int] = (245, 240, 230),
    interior_color: Tuple[int, int, int] = (235, 195, 140),
    roof_color: Tuple[int, int, int] = (178, 155, 110),
    ridge_color: Tuple[int, int, int] = (178, 155, 110),
    floor_color: Tuple[int, int, int] = (140, 95, 50),
    foundation_color: Tuple[int, int, int] = (120, 115, 105),
    sill_color: Tuple[int, int, int] = (245, 240, 230),
    door_frame_color: Tuple[int, int, int] = (90, 65, 35),
    win_frame_color: Tuple[int, int, int] = (15, 45, 15),
    glass_alpha: int = 80,
    textured: bool = True,
    tex_exterior: str = "White Brick",
    tex_interior: str = "White Stone",
    tex_roof: str = "Dark Thatch",
    tex_ridge: str = "Thatch",
    tex_wood: str = "Wood",
    tex_foundation: str = "Stone",
    tex_sill: str = "None",
    tex_door_frames: str = "None",
    tex_win_frames: str = "None",
    img_exterior: Optional[str] = None,
    img_interior: Optional[str] = None,
    img_roof: Optional[str] = None,
    img_ridge: Optional[str] = None,
    img_wood: Optional[str] = None,
    img_foundation: Optional[str] = None,
    seed: Optional[int] = None,
) -> trimesh.Trimesh:
    """Generate the childhood home — a Danish 1870s farmhouse with 6 rooms,
    thatched roof, and traversable interior."""
    set_seed(seed)

    meshes = MeshCollector()

    # --- Base dimensions ---
    L = 18.0   # length along X
    D = 7.0    # depth along Y
    WALL_H = 2.5
    RIDGE_H = 6.5
    WALL_T = 0.4   # exterior wall thickness
    PART_T = 0.25  # partition wall thickness

    # Door opening dimensions for interior connections
    DOOR_W = 0.9
    DOOR_H = 2.1
    FRONT_DOOR_W = 1.2  # wider front entrance door
    FRAME_T = 0.06  # door/window frame thickness

    # Colors
    ext_rgba = (*wall_color, 255)
    int_rgba = (*interior_color, 255)
    roof_rgba = (*roof_color, 255)
    ridge_rgba = (*ridge_color, 255)
    floor_rgba = (*floor_color, 255)
    tar_rgba = (25, 22, 18, 255)
    stone_rgba = (*foundation_color, 255)
    sill_rgba = (*sill_color, 255)
    win_frame_rgba = (*win_frame_color, 255)
    door_frame_rgba = (*door_frame_color, 255)
    chimney_rgba = (*wall_color, 255)  # white chimneys
    glass_rgba = (180, 210, 230, int(glass_alpha))  # semi-transparent pale blue

    # --- Room layout ---
    # Rooms: 0=Parlor, 1=Kitchen, 2=Entré, 3=Main room, 4=Bedroom1, 5=Bedroom2
    # The entré is room index 2, positioned between kitchen and main room
    room_widths = [2.6, 3.0, 1.8, 3.2, 2.8, 2.4]
    total_interior = L - 2 * WALL_T
    raw_total = sum(room_widths)
    room_widths = [w * total_interior / raw_total for w in room_widths]
    room_x_starts = []
    x_cursor = -L / 2 + WALL_T
    for rw in room_widths:
        room_x_starts.append(x_cursor)
        x_cursor += rw
    room_centers_x = [xs + rw / 2 for xs, rw in zip(room_x_starts, room_widths)]
    # Partition wall X positions (5 walls between 6 rooms)
    partition_xs = [room_x_starts[i + 1] for i in range(5)]

    # --- 1. Foundation (no wobble, split around entrance, inset corners) ---
    meshes.set_group('stone')
    found_h = 0.15
    found_t = 0.08
    entre_cx_f = -L / 2 + WALL_T + sum(room_widths[:2]) + room_widths[2] / 2
    entre_half_f = (FRONT_DOOR_W + 2 * FRAME_T) / 2

    # Front foundation — split around entrance doorway
    front_found_y = D / 2 + found_t / 2
    left_found_end = entre_cx_f - entre_half_f
    left_found_w = left_found_end - (-L / 2)
    if left_found_w > 0.01:
        f = trimesh.creation.box(extents=[left_found_w, found_t, found_h])
        f.apply_translation([-L / 2 + left_found_w / 2, front_found_y, found_h / 2])
        apply_color_variation(f, stone_rgba, 0.08)
        meshes.append(f)
    right_found_start = entre_cx_f + entre_half_f
    right_found_w = L / 2 - right_found_start
    if right_found_w > 0.01:
        f = trimesh.creation.box(extents=[right_found_w, found_t, found_h])
        f.apply_translation([right_found_start + right_found_w / 2, front_found_y, found_h / 2])
        apply_color_variation(f, stone_rgba, 0.08)
        meshes.append(f)

    # Back foundation (full length)
    f = trimesh.creation.box(extents=[L, found_t, found_h])
    f.apply_translation([0, -D / 2 - found_t / 2, found_h / 2])
    apply_color_variation(f, stone_rgba, 0.08)
    meshes.append(f)

    # End wall foundations (full depth to seal corners; overlap is hidden)
    for fx in [-L / 2 - found_t / 2, L / 2 + found_t / 2]:
        f = trimesh.creation.box(extents=[found_t, D + 2 * found_t, found_h])
        f.apply_translation([fx, 0, found_h / 2])
        apply_color_variation(f, stone_rgba, 0.08)
        meshes.append(f)

    # --- 2. Exterior walls (plain, no masonry overlay — texture handles that) ---
    meshes.set_group('exterior')
    WIN_W = 0.8
    WIN_H = 1.1
    WIN_ARCH = 0.1
    WIN_SILL_Z = 0.8
    # Wall opening sized to clear frame/sill (arch curves into wall above — no z-fight)
    WIN_OPEN_W = WIN_W + 2 * FRAME_T
    WIN_OPEN_SILL = WIN_SILL_Z - FRAME_T
    WIN_OPEN_H = WIN_H + 2 * FRAME_T

    front_y = D / 2 - WALL_T / 2
    back_y = -D / 2 + WALL_T / 2

    def _add_wall_panels(outers, inners, translation):
        """Add outer panels to 'exterior' group, inner panels to 'interior' group."""
        meshes.set_group('exterior')
        for p in outers:
            p.apply_translation(translation)
        meshes.extend(outers)
        meshes.set_group('interior')
        for p in inners:
            p.apply_translation(translation)
        meshes.extend(inners)

    for ri in range(6):
        rx_start = room_x_starts[ri]
        rw = room_widths[ri]
        rx_center = room_centers_x[ri]

        # Entré (room 2) has a front door opening instead of a window
        if ri == 2:
            # Front wall with door opening for entré (widened + taller to clear frame)
            door_open_w = FRONT_DOOR_W + 2 * FRAME_T
            door_open_h = 2.1 + FRAME_T
            outers, inners = _build_wall_with_opening(
                rw, WALL_H, WALL_T,
                0, 0, door_open_w, door_open_h,
                ext_rgba, int_rgba, wobble=0,
            )
            _add_wall_panels(outers, inners, [rx_center, front_y, 0])
        else:
            # Normal front wall segment with window
            outers, inners = _build_wall_with_opening(
                rw, WALL_H, WALL_T,
                0, WIN_OPEN_SILL, WIN_OPEN_W, WIN_OPEN_H,
                ext_rgba, int_rgba, wobble=0,
            )
            _add_wall_panels(outers, inners, [rx_center, front_y, 0])

        # Back wall segment with window (all rooms)
        # Swap outers/inners: _dual_slab puts "outer" at +Y, but for the
        # back wall +Y faces inward, so the physical exterior is the -Y slab.
        outers, inners = _build_wall_with_opening(
            rw, WALL_H, WALL_T,
            0, WIN_OPEN_SILL, WIN_OPEN_W, WIN_OPEN_H,
            ext_rgba, int_rgba, wobble=0,
        )
        _add_wall_panels(inners, outers, [rx_center, back_y, 0])

    # End walls (left and right, rectangular + gable)
    for side_x, sign in [(-L / 2 + WALL_T / 2, -1), (L / 2 - WALL_T / 2, 1)]:
        # Rectangular portion (full depth D to seal corners, no wobble)
        meshes.set_group('exterior')
        outer = trimesh.creation.box(extents=[WALL_T * 0.55, D, WALL_H])
        outer.apply_translation([side_x + sign * WALL_T * 0.45 / 2, 0, WALL_H / 2])
        apply_color_variation(outer, ext_rgba, 0.04)
        meshes.append(outer)
        meshes.set_group('interior')
        inner = trimesh.creation.box(extents=[WALL_T * 0.45, D, WALL_H])
        inner.apply_translation([side_x - sign * WALL_T * 0.55 / 2, 0, WALL_H / 2])
        apply_color_variation(inner, int_rgba, 0.04)
        meshes.append(inner)

        # Gable wall (lower half — upper half covered by thatch)
        meshes.set_group('exterior')
        gable_h = RIDGE_H - WALL_H
        inner_d = D
        n_gable = 8
        for gi in range(n_gable // 2):
            frac = gi / n_gable
            next_frac = (gi + 1) / n_gable
            z_bot = WALL_H + frac * gable_h
            z_top = WALL_H + next_frac * gable_h
            course_h = z_top - z_bot
            avg_w = inner_d * (1.0 - (frac + next_frac) / 2)
            if avg_w < 0.05:
                continue
            gable_block = trimesh.creation.box(extents=[WALL_T * 0.55, avg_w, course_h])
            gable_block.apply_translation([
                side_x + sign * WALL_T * 0.45 / 2, 0, z_bot + course_h / 2,
            ])
            apply_color_variation(gable_block, ext_rgba, 0.04)
            meshes.append(gable_block)

        # Gable thatch (upper half — overhang starts at main roof edge)
        meshes.set_group('roof')
        wall_face_x = side_x + sign * WALL_T / 2
        mid_z = WALL_H + gable_h / 2
        thatch = _build_gable_thatch(
            mid_z, RIDGE_H, D, gable_h,
            wall_face_x, sign, roof_t=0.4,
        )
        apply_color_variation(thatch, roof_rgba, 0.05)
        meshes.append(thatch)

    # Tar base strip on exterior walls
    meshes.set_group('stone')
    tar_h = WALL_H * 0.05
    tar_t = 0.03
    entre_cx = room_centers_x[2]
    entre_half = (FRONT_DOOR_W + 2 * FRAME_T) / 2  # half of door opening width

    # Front wall tar strip — split around entrance doorway
    front_tar_y = D / 2 + tar_t / 2
    # Left of entrance
    left_tar_end = entre_cx - entre_half
    left_tar_w = left_tar_end - (-L / 2)
    if left_tar_w > 0.01:
        tar = trimesh.creation.box(extents=[left_tar_w, tar_t, tar_h])
        tar.apply_translation([-L / 2 + left_tar_w / 2, front_tar_y, tar_h / 2])
        apply_color_variation(tar, tar_rgba, 0.02)
        meshes.append(tar)
    # Right of entrance
    right_tar_start = entre_cx + entre_half
    right_tar_w = L / 2 - right_tar_start
    if right_tar_w > 0.01:
        tar = trimesh.creation.box(extents=[right_tar_w, tar_t, tar_h])
        tar.apply_translation([right_tar_start + right_tar_w / 2, front_tar_y, tar_h / 2])
        apply_color_variation(tar, tar_rgba, 0.02)
        meshes.append(tar)

    # Back wall tar strip (full length, no door)
    tar = trimesh.creation.box(extents=[L, tar_t, tar_h])
    tar.apply_translation([0, -D / 2 - tar_t / 2, tar_h / 2])
    apply_color_variation(tar, tar_rgba, 0.02)
    meshes.append(tar)

    # End wall tar strips (inset to avoid overlapping front/back strips)
    end_tar_len = D - 2 * tar_t
    for ex in [-L / 2 - tar_t / 2, L / 2 + tar_t / 2]:
        tar = trimesh.creation.box(extents=[tar_t, end_tar_len, tar_h])
        tar.apply_translation([ex, 0, tar_h / 2])
        apply_color_variation(tar, tar_rgba, 0.02)
        meshes.append(tar)

    # --- 3. Interior partition walls with door openings ---
    meshes.set_group('interior')
    # Each partition gets a door opening so rooms are connected.
    interior_d = D - 2 * WALL_T

    # Doorway offset: 1/3 of interior depth from front wall, pulled 5% towards center
    door_y = interior_d / 3 * 0.95  # towards front (positive Y)

    for pi, px in enumerate(partition_xs):
        # Widened gap so door frame sits without clipping
        gap_w = DOOR_W + 2 * FRAME_T

        # Back part of partition (negative Y side of door — larger section)
        back_d = (door_y - gap_w / 2) + interior_d / 2
        if back_d > 0.01:
            part = trimesh.creation.box(extents=[PART_T, back_d, WALL_H])
            part.apply_translation([px, -interior_d / 2 + back_d / 2, WALL_H / 2])
            apply_color_variation(part, int_rgba, 0.05)
            meshes.append(part)

        # Front part of partition (positive Y side of door — smaller section)
        front_d = interior_d / 2 - (door_y + gap_w / 2)
        if front_d > 0.01:
            part = trimesh.creation.box(extents=[PART_T, front_d, WALL_H])
            part.apply_translation([px, interior_d / 2 - front_d / 2, WALL_H / 2])
            apply_color_variation(part, int_rgba, 0.05)
            meshes.append(part)

        # Above door (starts above lintel to avoid z-fighting)
        above_z = DOOR_H + FRAME_T
        above_h = WALL_H - above_z
        if above_h > 0.01:
            part = trimesh.creation.box(extents=[PART_T, gap_w, above_h])
            part.apply_translation([px, door_y, above_z + above_h / 2])
            apply_color_variation(part, int_rgba, 0.05)
            meshes.append(part)

        # Door frame
        meshes.set_group('door_frames')
        frame_meshes = _build_door_frame(DOOR_W, DOOR_H, PART_T, door_frame_rgba)
        for fm in frame_meshes:
            fm.apply_translation([px, door_y, 0])
        meshes.extend(frame_meshes)
        meshes.set_group('interior')

    # --- 4. Floor (plank strips per room) ---
    meshes.set_group('wood')
    for ri in range(6):
        if ri == 0:
            floor_x_start = -L / 2 + WALL_T
            floor_x_end = partition_xs[0] - PART_T / 2
        elif ri == 5:
            floor_x_start = partition_xs[4] + PART_T / 2
            floor_x_end = L / 2 - WALL_T
        else:
            floor_x_start = partition_xs[ri - 1] + PART_T / 2
            floor_x_end = partition_xs[ri] - PART_T / 2
        floor_w = floor_x_end - floor_x_start
        floor_cx = (floor_x_start + floor_x_end) / 2

        planks = _build_plank_surface(
            floor_w, interior_d, 0.02,
            plank_w=0.25, axis='y',
            color=floor_rgba, variation=0.06,
        )
        for p in planks:
            p.apply_translation([floor_cx, 0, 0])
        meshes.extend(planks)

    # --- 4b. Floor planks under doorframes ---
    for pi, px in enumerate(partition_xs):
        plank = trimesh.creation.box(extents=[PART_T, DOOR_W, 0.04])
        plank.apply_translation([px, door_y, 0.02])
        apply_color_variation(plank, floor_rgba, 0.06)
        meshes.append(plank)

    # Entrance doorway floor plank
    entre_plank = trimesh.creation.box(extents=[FRONT_DOOR_W + 2 * FRAME_T, WALL_T, 0.04])
    entre_plank.apply_translation([room_centers_x[2], front_y, 0.02])
    apply_color_variation(entre_plank, floor_rgba, 0.06)
    meshes.append(entre_plank)

    # --- 4c. Solid base plane under floor to seal gaps ---
    floor_base = trimesh.creation.box(extents=[L - 2 * WALL_T, D - 2 * WALL_T, 0.01])
    floor_base.apply_translation([0, 0, -0.01])
    apply_color_variation(floor_base, floor_rgba, 0.02)
    meshes.append(floor_base)

    # --- 5. Ceiling (plank strips) ---
    meshes.set_group('wood')
    ceil_color = tuple(max(0, c - 15) for c in floor_color) + (255,)
    for ri in range(6):
        if ri == 0:
            ceil_x_start = -L / 2 + WALL_T
            ceil_x_end = partition_xs[0] - PART_T / 2
        elif ri == 5:
            ceil_x_start = partition_xs[4] + PART_T / 2
            ceil_x_end = L / 2 - WALL_T
        else:
            ceil_x_start = partition_xs[ri - 1] + PART_T / 2
            ceil_x_end = partition_xs[ri] - PART_T / 2
        ceil_w = ceil_x_end - ceil_x_start
        ceil_cx = (ceil_x_start + ceil_x_end) / 2

        planks = _build_plank_surface(
            ceil_w, interior_d, WALL_H,
            plank_w=0.25, axis='y',
            color=ceil_color, variation=0.05,
        )
        for p in planks:
            p.apply_translation([ceil_cx, 0, 0])
        meshes.extend(planks)

    # --- 6. Windows ---
    def _add_window_meshes(win_meshes):
        """Sort window sub-meshes into frames and glass groups."""
        for wm in win_meshes:
            vc = wm.visual.vertex_colors
            if vc is not None and len(vc) > 0 and vc.shape[1] >= 4 and (vc[:, 3] < 255).any():
                meshes.set_group('glass')
            else:
                meshes.set_group('frames')
            meshes.append(wm)

    for ri in range(6):
        rx_center = room_centers_x[ri]

        # Back window (all rooms)
        win_meshes = _build_arched_window(
            WIN_W, WIN_H, WIN_ARCH, WIN_SILL_Z, WALL_T,
            win_frame_rgba, sill_rgba, glass_color=glass_rgba,
        )
        for wm in win_meshes:
            wm.apply_translation([rx_center, back_y, 0])
        _add_window_meshes(win_meshes)

        # Front window (skip entré room — it has the front door)
        if ri != 2:
            win_meshes = _build_arched_window(
                WIN_W, WIN_H, WIN_ARCH, WIN_SILL_Z, WALL_T,
                win_frame_rgba, sill_rgba, glass_color=glass_rgba,
            )
            for wm in win_meshes:
                wm.apply_translation([rx_center, front_y, 0])
            _add_window_meshes(win_meshes)

    # Front door frame on entré
    meshes.set_group('door_frames')
    entre_door_frame = _build_front_door_frame(FRONT_DOOR_W, 2.1, WALL_T, door_frame_rgba)
    for fm in entre_door_frame:
        fm.apply_translation([room_centers_x[2], front_y, 0])
    meshes.extend(entre_door_frame)

    # --- 6b. Continuous windowsill ledge along both long walls ---
    meshes.set_group('sill')
    sill_h = 0.04         # ledge thickness
    sill_depth = 0.07     # how far it protrudes from wall
    sill_z = WIN_SILL_Z - FRAME_T / 2  # align with window sill line

    # Back wall — full length, no door interruption
    back_sill = trimesh.creation.box(extents=[L, sill_depth, sill_h])
    back_sill.apply_translation([0, -D / 2 - sill_depth / 2, sill_z])
    apply_color_variation(back_sill, sill_rgba, 0.04)
    meshes.append(back_sill)

    # Front wall — split around entré door opening
    entre_cx_sill = room_centers_x[2]
    entre_half_sill = (FRONT_DOOR_W + 2 * FRAME_T) / 2
    front_sill_y = D / 2 + sill_depth / 2

    # Left section: from left end wall to door opening
    left_sill_end = entre_cx_sill - entre_half_sill
    left_sill_w = left_sill_end - (-L / 2)
    if left_sill_w > 0.01:
        sill_left = trimesh.creation.box(extents=[left_sill_w, sill_depth, sill_h])
        sill_left.apply_translation([-L / 2 + left_sill_w / 2, front_sill_y, sill_z])
        apply_color_variation(sill_left, sill_rgba, 0.04)
        meshes.append(sill_left)

    # Right section: from door opening to right end wall
    right_sill_start = entre_cx_sill + entre_half_sill
    right_sill_w = L / 2 - right_sill_start
    if right_sill_w > 0.01:
        sill_right = trimesh.creation.box(extents=[right_sill_w, sill_depth, sill_h])
        sill_right.apply_translation([right_sill_start + right_sill_w / 2, front_sill_y, sill_z])
        apply_color_variation(sill_right, sill_rgba, 0.04)
        meshes.append(sill_right)

    # --- 7. Simple gabled roof (single slab per slope, light brown for texturing) ---
    meshes.set_group('roof')
    roof_overhang = 0.4
    roof_length = L + 2 * roof_overhang
    gable_h = RIDGE_H - WALL_H
    roof_pitch_angle = np.arctan2(gable_h, D / 2)
    slope_w = (D / 2) / np.cos(roof_pitch_angle) + roof_overhang
    roof_t = 0.12

    ridge_gap = 0.08  # pull each slope back from ridge to avoid overlap
    slope_w_trimmed = slope_w - ridge_gap

    for side in [+1, -1]:
        slab = trimesh.creation.box(extents=[roof_length, slope_w_trimmed, roof_t])
        rot = trimesh.transformations.rotation_matrix(-side * roof_pitch_angle, [1, 0, 0])
        slab.apply_transform(rot)
        # Offset from ridge: half the trimmed width + the gap
        y_off = side * np.cos(roof_pitch_angle) * (slope_w_trimmed / 2 + ridge_gap)
        z_off = RIDGE_H - np.sin(roof_pitch_angle) * (slope_w_trimmed / 2 + ridge_gap)
        slab.apply_translation([0, y_off, z_off])
        apply_color_variation(slab, roof_rgba, 0.05)
        meshes.append(slab)

    # Ridge cap (wider to cover the gap between the two slopes)
    meshes.set_group('ridge')
    ridge_cap_w = 2 * ridge_gap / np.cos(roof_pitch_angle) + 0.25
    ridge = trimesh.creation.box(extents=[roof_length, ridge_cap_w, 0.1])
    ridge.apply_translation([0, 0, RIDGE_H + 0.05])
    apply_color_variation(ridge, ridge_rgba, 0.04)
    meshes.append(ridge)

    # --- 8. Chimneys (2, white, 10% shorter) ---
    meshes.set_group('exterior')
    chimney_h = 0.675
    # Kitchen end
    chim_meshes = _build_farmhouse_chimney(0.5, 0.5, chimney_h, chimney_rgba)
    for cm in chim_meshes:
        cm.apply_translation([room_centers_x[1], 0, RIDGE_H - 0.1])
    meshes.extend(chim_meshes)

    # Last room end
    chim_meshes = _build_farmhouse_chimney(0.45, 0.45, chimney_h, chimney_rgba)
    for cm in chim_meshes:
        cm.apply_translation([room_centers_x[5], 0, RIDGE_H - 0.1])
    meshes.extend(chim_meshes)

    # --- Combine and scale ---
    if textured:
        tex_choices = {
            'exterior':    (tex_exterior, 1.0),
            'interior':    (tex_interior, 1.0),
            'roof':        (tex_roof, 0.7),
            'ridge':       (tex_ridge, 0.7),
            'wood':        (tex_wood, 2.0),
            'stone':       (tex_foundation, 1.5),
            'sill':        (tex_sill, 2.0),
            'door_frames': (tex_door_frames, 2.0),
            'frames':      (tex_win_frames, 2.0),
        }
        img_overrides = {
            'exterior': img_exterior,
            'interior': img_interior,
            'roof':     img_roof,
            'ridge':    img_ridge,
            'wood':     img_wood,
            'stone':    img_foundation,
        }
        result = _build_textured_scene(meshes.groups, seed, tex_choices, img_overrides)
        if size != 1.0:
            for geom in result.geometry.values():
                geom.apply_scale(size)
        return result

    combined = trimesh.util.concatenate(meshes.flat)
    combined.fix_normals()

    if size != 1.0:
        combined.apply_scale(size)

    return combined


# --- Registry ---

register(
    name="childhood_home", label="Childhood Home", category="Childhood Home",
    button_top=True,
    params=[
        Param("size", "Size", "float", default=1.0, min=0.3, max=3.0),
        Param("wall_color", "Ext. Walls", "color", default="#F5F0E6"),
        Param("interior_color", "Int. Walls", "color", default="#EBC38C"),
        Param("roof_color", "Roof Color", "color", default="#B29B6E"),
        Param("ridge_color", "Ridge Color", "color", default="#B29B6E"),
        Param("floor_color", "Floor Color", "color", default="#8C5F32"),
        Param("foundation_color", "Foundation Color", "color", default="#787369"),
        Param("sill_color", "Sill Color", "color", default="#F5F0E6"),
        Param("door_frame_color", "Door Frame Color", "color", default="#5A4123"),
        Param("win_frame_color", "Win. Frame Color", "color", default="#0F2D0F"),
        Param("glass_alpha", "Glass Opacity", "int", default=80, min=0, max=255, step=5),
        Param("textured", "Textured", "bool", default=True),
        Param("tex_exterior", "Ext. Texture", "str", default="White Brick", choices=_TEXTURE_PRESETS),
        Param("tex_interior", "Int. Texture", "str", default="White Stone", choices=_TEXTURE_PRESETS),
        Param("tex_roof", "Roof Texture", "str", default="Dark Thatch", choices=_TEXTURE_PRESETS),
        Param("tex_ridge", "Ridge Texture", "str", default="Thatch", choices=_TEXTURE_PRESETS),
        Param("tex_wood", "Wood Texture", "str", default="Wood", choices=_TEXTURE_PRESETS),
        Param("tex_foundation", "Foundation Tex.", "str", default="Stone", choices=_TEXTURE_PRESETS),
        Param("tex_sill", "Sill Texture", "str", default="None", choices=_TEXTURE_PRESETS),
        Param("tex_door_frames", "Door Frame Tex.", "str", default="None", choices=_TEXTURE_PRESETS),
        Param("tex_win_frames", "Win. Frame Tex.", "str", default="None", choices=_TEXTURE_PRESETS),
        Param("img_exterior", "Ext. Image", "image"),
        Param("img_interior", "Int. Image", "image"),
        Param("img_roof", "Roof Image", "image"),
        Param("img_ridge", "Ridge Image", "image"),
        Param("img_wood", "Wood Image", "image"),
        Param("img_foundation", "Foundation Img.", "image"),
    ],
)(generate_childhood_home)
