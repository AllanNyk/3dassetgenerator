"""
Small prop and decoration generators with vertex color support.
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


def generate_crystal(
    height: float = 2.0,
    radius: float = 0.5,
    points: int = 6,
    color: Tuple[int, int, int] = (130, 50, 200),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a crystal/gem shape."""
    set_seed(seed)

    top = trimesh.creation.cone(radius=radius, height=height * 0.6, sections=points)
    bottom = trimesh.creation.cone(radius=radius, height=height * 0.4, sections=points)

    bottom.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    top.apply_translation([0, 0, height * 0.2])
    bottom.apply_translation([0, 0, height * 0.2])

    crystal = trimesh.util.concatenate([top, bottom])
    crystal.fix_normals()
    apply_color_variation(crystal, (*color, 255), 0.15)

    return crystal


def generate_crate(
    size: float = 1.0,
    color: Tuple[int, int, int] = (160, 120, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a simple crate/box."""
    set_seed(seed)

    w = size * np.random.uniform(0.9, 1.1)
    h = size * np.random.uniform(0.9, 1.1)
    d = size * np.random.uniform(0.9, 1.1)

    crate = trimesh.creation.box(extents=[w, d, h])
    crate.apply_translation([0, 0, h/2])
    apply_color_variation(crate, (*color, 255), 0.1)

    return crate


def generate_barrel(
    height: float = 1.5,
    radius: float = 0.5,
    bulge: float = 0.1,
    color: Tuple[int, int, int] = (140, 90, 40),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a barrel shape."""
    set_seed(seed)

    sections = 16
    height_segments = 8

    vertices = []
    faces = []

    for j in range(height_segments + 1):
        z = (j / height_segments) * height
        t = j / height_segments
        bulge_factor = 1 + bulge * np.sin(t * np.pi)
        r = radius * bulge_factor

        for i in range(sections):
            angle = (i / sections) * 2 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for j in range(height_segments):
        for i in range(sections):
            i_next = (i + 1) % sections
            v0 = j * sections + i
            v1 = j * sections + i_next
            v2 = (j + 1) * sections + i_next
            v3 = (j + 1) * sections + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    bottom_center = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])
    top_center = len(vertices)
    vertices = np.vstack([vertices, [0, 0, height]])

    for i in range(sections):
        i_next = (i + 1) % sections
        faces.append([bottom_center, i_next, i])
        top_ring_start = height_segments * sections
        faces.append([top_center, top_ring_start + i, top_ring_start + i_next])

    faces = np.array(faces)

    barrel = trimesh.Trimesh(vertices=vertices, faces=faces)
    barrel.fix_normals()
    apply_color_variation(barrel, (*color, 255), 0.1)

    return barrel


def generate_lamp(
    pole_height: float = 2.0,
    pole_radius: float = 0.05,
    lamp_radius: float = 0.2,
    color: Tuple[int, int, int] = (60, 60, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a simple street lamp."""
    set_seed(seed)

    meshes = []

    # Pole
    pole = trimesh.creation.cylinder(
        radius=pole_radius,
        height=pole_height,
        sections=8
    )
    pole.apply_translation([0, 0, pole_height / 2])
    apply_color_variation(pole, (*color, 255), 0.05)
    meshes.append(pole)

    # Lamp housing (icosphere squashed)
    lamp = trimesh.creation.icosphere(subdivisions=2, radius=lamp_radius)
    lamp.vertices[:, 2] *= 0.6
    lamp.apply_translation([0, 0, pole_height])
    apply_color_variation(lamp, (255, 240, 180, 255), 0.05)
    meshes.append(lamp)

    # Small cap on top
    cap = trimesh.creation.cone(
        radius=lamp_radius * 1.2,
        height=lamp_radius * 0.5
    )
    cap.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    cap.apply_translation([0, 0, pole_height + lamp_radius * 0.3])
    apply_color_variation(cap, (*color, 255), 0.05)
    meshes.append(cap)

    return trimesh.util.concatenate(meshes)


def generate_sign(
    width: float = 1.0,
    height: float = 0.6,
    pole_height: float = 1.5,
    color: Tuple[int, int, int] = (180, 160, 120),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a simple sign on a pole."""
    set_seed(seed)

    meshes = []

    pole_radius = 0.04

    # Pole
    pole = trimesh.creation.cylinder(
        radius=pole_radius,
        height=pole_height,
        sections=6
    )
    pole.apply_translation([0, 0, pole_height / 2])
    apply_color_variation(pole, (100, 80, 50, 255), 0.05)
    meshes.append(pole)

    # Sign board
    sign = trimesh.creation.box(extents=[width, 0.05, height])
    sign.apply_translation([0, 0, pole_height + height/2])
    apply_color_variation(sign, (*color, 255), 0.1)
    meshes.append(sign)

    return trimesh.util.concatenate(meshes)


def generate_bench(
    length: float = 1.5,
    seat_height: float = 0.45,
    color: Tuple[int, int, int] = (140, 100, 50),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a simple park bench."""
    set_seed(seed)

    meshes = []

    seat_depth = 0.4
    seat_thickness = 0.05
    leg_thickness = 0.08

    # Seat
    seat = trimesh.creation.box(extents=[length, seat_depth, seat_thickness])
    seat.apply_translation([0, 0, seat_height])
    apply_color_variation(seat, (*color, 255), 0.1)
    meshes.append(seat)

    # Back
    back = trimesh.creation.box(extents=[length, seat_thickness, seat_height * 0.6])
    back.apply_translation([0, -seat_depth/2 + seat_thickness/2, seat_height + seat_height * 0.3])
    apply_color_variation(back, (*color, 255), 0.1)
    meshes.append(back)

    # Legs (4 corners)
    leg_positions = [
        (-length/2 + leg_thickness, -seat_depth/2 + leg_thickness),
        (length/2 - leg_thickness, -seat_depth/2 + leg_thickness),
        (-length/2 + leg_thickness, seat_depth/2 - leg_thickness),
        (length/2 - leg_thickness, seat_depth/2 - leg_thickness),
    ]

    for x, y in leg_positions:
        leg = trimesh.creation.box(extents=[leg_thickness, leg_thickness, seat_height])
        leg.apply_translation([x, y, seat_height / 2])
        apply_color_variation(leg, (80, 80, 80, 255), 0.05)
        meshes.append(leg)

    return trimesh.util.concatenate(meshes)


def generate_pot(
    radius: float = 0.3,
    height: float = 0.4,
    color: Tuple[int, int, int] = (180, 100, 50),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a flower pot."""
    set_seed(seed)

    # Tapered cylinder (wider at top)
    sections = 16
    height_segments = 4

    vertices = []

    for j in range(height_segments + 1):
        z = (j / height_segments) * height
        t = j / height_segments
        r = radius * (0.7 + 0.3 * t)  # Taper from 70% to 100%

        for i in range(sections):
            angle = (i / sections) * 2 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for j in range(height_segments):
        for i in range(sections):
            i_next = (i + 1) % sections
            v0 = j * sections + i
            v1 = j * sections + i_next
            v2 = (j + 1) * sections + i_next
            v3 = (j + 1) * sections + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Bottom cap
    bottom_center = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])

    for i in range(sections):
        i_next = (i + 1) % sections
        faces.append([bottom_center, i_next, i])

    faces = np.array(faces)

    pot = trimesh.Trimesh(vertices=vertices, faces=faces)
    pot.fix_normals()
    apply_color_variation(pot, (*color, 255), 0.1)

    return pot


# --- Registry ---

register(
    name="crystal", label="Crystal", category="Props",
    params=[
        Param("height", "Height", "float", default=2.0, min=0.5, max=5.0),
        Param("radius", "Radius", "float", default=0.5, min=0.2, max=2.0),
        Param("points", "Sides", "int", default=6, min=3, max=12, step=1),
        Param("color", "Color", "color", default="#8232C8"),
    ],
)(generate_crystal)

register(
    name="crate", label="Crate", category="Props",
    params=[
        Param("size", "Size", "float", default=1.0, min=0.5, max=5.0),
        Param("color", "Color", "color", default="#A0783C"),
    ],
)(generate_crate)

register(
    name="barrel", label="Barrel", category="Props",
    params=[
        Param("height", "Height", "float", default=1.5, min=0.5, max=4.0),
        Param("radius", "Radius", "float", default=0.5, min=0.2, max=2.0),
        Param("bulge", "Bulge", "float", default=0.1, min=0.0, max=0.3),
        Param("color", "Color", "color", default="#8C5A28"),
    ],
)(generate_barrel)

register(
    name="lamp", label="Lamp", category="Props",
    params=[
        Param("pole_height", "Pole Height", "float", default=2.0, min=1.0, max=5.0),
        Param("lamp_radius", "Lamp Size", "float", default=0.2, min=0.1, max=0.5),
        Param("color", "Pole Color", "color", default="#3C3C3C"),
    ],
)(generate_lamp)

register(
    name="sign", label="Sign", category="Props",
    params=[
        Param("width", "Width", "float", default=1.0, min=0.5, max=3.0),
        Param("height", "Height", "float", default=0.6, min=0.3, max=2.0),
        Param("pole_height", "Pole Height", "float", default=1.5, min=0.5, max=3.0),
        Param("color", "Sign Color", "color", default="#B4A078"),
    ],
)(generate_sign)

register(
    name="bench", label="Bench", category="Props",
    params=[
        Param("length", "Length", "float", default=1.5, min=1.0, max=3.0),
        Param("seat_height", "Seat Height", "float", default=0.45, min=0.3, max=0.6),
        Param("color", "Wood Color", "color", default="#8C6432"),
    ],
)(generate_bench)

register(
    name="pot", label="Flower Pot", category="Props",
    params=[
        Param("radius", "Radius", "float", default=0.3, min=0.1, max=1.0),
        Param("height", "Height", "float", default=0.4, min=0.1, max=1.0),
        Param("color", "Color", "color", default="#B46432"),
    ],
)(generate_pot)


# =============================================================================
# ANTIQUE PROPS
# =============================================================================

def _wobble_box(extents, wobble=0.015):
    """Create a box with slight vertex displacement for handcrafted look."""
    box = trimesh.creation.box(extents=extents)
    verts = box.vertices.copy()
    max_dim = max(extents)
    for i in range(3):
        verts[:, i] += np.random.uniform(-wobble, wobble, len(verts)) * max_dim
    box.vertices = verts
    return box


def generate_candlestick(
    height: float = 0.30,
    color: Tuple[int, int, int] = (180, 155, 70),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique brass candlestick with turned stem and drip tray."""
    set_seed(seed)
    meshes = []

    # Base (wide, flat disc with slight dome)
    base_r = 0.055
    base = trimesh.creation.cylinder(radius=base_r, height=0.012, sections=24)
    base.apply_translation([0, 0, 0.006])
    apply_color_variation(base, (*color, 255), 0.08)
    meshes.append(base)

    # Base rim (torus-like ring)
    rim_pts = []
    for i in range(24):
        a = (i / 24) * 2 * np.pi
        for j in range(8):
            b = (j / 8) * 2 * np.pi
            rx = (base_r + 0.005 * np.cos(b)) * np.cos(a)
            ry = (base_r + 0.005 * np.cos(b)) * np.sin(a)
            rz = 0.012 + 0.005 * np.sin(b)
            rim_pts.append([rx, ry, rz])
    # Use convex hull for the rim (approximate torus)
    rim = trimesh.convex.convex_hull(np.array(rim_pts))
    apply_color_variation(rim, (*color, 255), 0.06)
    meshes.append(rim)

    # Stem (turned profile - built from stacked cylinders)
    stem_sections = 12
    stem_base_z = 0.015
    stem_h = height - 0.06  # leave room for base and top
    for j in range(stem_sections):
        t = j / stem_sections
        z = stem_base_z + t * stem_h
        dz = stem_h / stem_sections

        # Turned profile: narrow in middle, wider at transitions
        r = 0.012 * (0.6 + 0.4 * np.sin(t * np.pi)
                      + 0.15 * np.sin(t * 3 * np.pi))
        r *= np.random.uniform(0.97, 1.03)

        seg = trimesh.creation.cylinder(radius=r, height=dz, sections=16)
        seg.apply_translation([0, 0, z + dz / 2])
        apply_color_variation(seg, (*color, 255), 0.06)
        meshes.append(seg)

    # Drip tray (small dish near top)
    tray_z = stem_base_z + stem_h
    tray = trimesh.creation.cylinder(radius=0.03, height=0.006, sections=20)
    tray.apply_translation([0, 0, tray_z])
    apply_color_variation(tray, (*color, 255), 0.06)
    meshes.append(tray)

    # Candle spike (thin pointed cylinder)
    spike = trimesh.creation.cone(radius=0.004, height=0.02, sections=8)
    spike.apply_translation([0, 0, tray_z + 0.013])
    apply_color_variation(spike, (*color, 255), 0.05)
    meshes.append(spike)

    # Candle (cream colored wax cylinder)
    candle_h = 0.08
    candle = trimesh.creation.cylinder(radius=0.012, height=candle_h, sections=12)
    candle.apply_translation([0, 0, tray_z + candle_h / 2 + 0.003])
    apply_color_variation(candle, (240, 230, 200, 255), 0.04)
    meshes.append(candle)

    # Wick
    wick = trimesh.creation.cylinder(radius=0.001, height=0.01, sections=6)
    wick.apply_translation([0, 0, tray_z + candle_h + 0.008])
    apply_color_variation(wick, (40, 30, 20, 255), 0.02)
    meshes.append(wick)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_book(
    width: float = 0.17,
    height: float = 0.24,
    thickness: float = 0.03,
    color: Tuple[int, int, int] = (120, 40, 30),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an old leather-bound book with visible pages."""
    set_seed(seed)
    meshes = []

    spine_r = thickness * 0.6

    # Pages block (cream/off-white, slightly inset from cover)
    pages_w = width - 0.008
    pages_h = height - 0.006
    pages_t = thickness - 0.006
    pages = _wobble_box([pages_w, pages_h, pages_t], wobble=0.003)
    pages.apply_translation([0.004, 0, 0])
    apply_color_variation(pages, (235, 225, 195, 255), 0.04)
    meshes.append(pages)

    # Front cover
    cover_t = 0.003
    front = _wobble_box([width, height, cover_t], wobble=0.004)
    front.apply_translation([0, 0, thickness / 2])
    apply_color_variation(front, (*color, 255), 0.12)
    meshes.append(front)

    # Back cover
    back = _wobble_box([width, height, cover_t], wobble=0.004)
    back.apply_translation([0, 0, -thickness / 2])
    apply_color_variation(back, (*color, 255), 0.12)
    meshes.append(back)

    # Spine (rounded edge on left side)
    spine = trimesh.creation.cylinder(radius=spine_r, height=height - 0.004, sections=12)
    spine.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    spine.apply_translation([-width / 2, 0, 0])
    apply_color_variation(spine, (*color, 255), 0.1)
    meshes.append(spine)

    # Decorative lines on spine (gold embossing)
    gold = (190, 170, 80, 255)
    for sz in [-height * 0.35, -height * 0.1, height * 0.1, height * 0.35]:
        line = trimesh.creation.cylinder(radius=spine_r + 0.001, height=0.003, sections=12)
        line.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        line.apply_translation([-width / 2, sz, 0])
        apply_color_variation(line, gold, 0.06)
        meshes.append(line)

    # Title area on front cover (raised rectangle)
    title = _wobble_box([width * 0.5, height * 0.15, 0.002], wobble=0.002)
    title.apply_translation([0.01, height * 0.15, thickness / 2 + 0.002])
    apply_color_variation(title, gold, 0.08)
    meshes.append(title)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_clogs(
    size: float = 0.28,
    color: Tuple[int, int, int] = (160, 120, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a pair of antique wooden clogs (Dutch-style) with foot openings."""
    set_seed(seed)
    meshes = []

    length = size
    width = size * 0.40
    h = size * 0.32

    def _clog_profile(t):
        """Return (half_width, height, base_z) for position t along shoe (0=heel, 1=toe)."""
        # Width: widest at ball, narrow at heel and toe
        w = width * (0.55 + 0.45 * np.sin(t * np.pi) ** 0.7)
        if t > 0.8:
            w *= 1.0 - 0.4 * ((t - 0.8) / 0.2)
        # Height: tallest at toe box, lower at heel
        ht = h * (0.6 + 0.4 * np.sin(np.clip(t * 1.1, 0, 1) * np.pi))
        # Toe upturn
        bz = 0.0
        if t > 0.85:
            bz = 0.012 * ((t - 0.85) / 0.15)
        return w / 2, ht, bz

    for shoe_side in [-1, 1]:
        ox = shoe_side * (width * 0.8)
        all_verts = []
        all_faces = []

        n_rings = 18
        n_seg = 16
        wall = 0.010  # shell thickness

        # Build outer + inner shell together for a hollow clog
        # Outer shell
        for ri in range(n_rings):
            t = ri / (n_rings - 1)
            y = (t - 0.5) * length
            hw, ht, bz = _clog_profile(t)

            for si in range(n_seg):
                a = (si / n_seg) * 2 * np.pi
                if np.sin(a) < 0:
                    # Bottom: flat sole
                    rx = hw * np.cos(a)
                    rz = bz
                else:
                    rx = hw * np.cos(a)
                    rz = bz + ht * np.sin(a) * 0.55
                all_verts.append([ox + rx, y, rz])

        # Inner shell (inset, only for the opening region: heel to ~60% along)
        # The opening starts at heel (t=0) and ends around mid-foot (t=0.55)
        inner_rings = 10
        inner_start = len(all_verts)
        for ri in range(inner_rings):
            t = ri / (inner_rings - 1) * 0.55
            y = (t - 0.5) * length
            hw, ht, bz = _clog_profile(t)
            hw_i = hw - wall
            ht_i = ht - wall

            for si in range(n_seg):
                a = (si / n_seg) * 2 * np.pi
                if np.sin(a) < 0:
                    rx = hw_i * np.cos(a)
                    rz = bz + wall
                else:
                    rx = hw_i * np.cos(a)
                    rz = bz + wall + ht_i * np.sin(a) * 0.55
                all_verts.append([ox + rx, y, rz])

        all_verts = np.array(all_verts)

        # Outer shell faces (both windings for double-sided)
        for ri in range(n_rings - 1):
            for si in range(n_seg):
                sn = (si + 1) % n_seg
                v0 = ri * n_seg + si
                v1 = ri * n_seg + sn
                v2 = (ri + 1) * n_seg + sn
                v3 = (ri + 1) * n_seg + si
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])
                all_faces.append([v0, v2, v1])
                all_faces.append([v0, v3, v2])

        # Toe cap (close the toe end)
        toe_ring = (n_rings - 1) * n_seg
        tc_idx = len(all_verts)
        tc = all_verts[toe_ring:toe_ring + n_seg].mean(axis=0)
        all_verts = np.vstack([all_verts, tc])
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            all_faces.append([tc_idx, toe_ring + si, toe_ring + sn])
            all_faces.append([tc_idx, toe_ring + sn, toe_ring + si])

        # Inner shell faces (double-sided)
        for ri in range(inner_rings - 1):
            for si in range(n_seg):
                sn = (si + 1) % n_seg
                v0 = inner_start + ri * n_seg + si
                v1 = inner_start + ri * n_seg + sn
                v2 = inner_start + (ri + 1) * n_seg + sn
                v3 = inner_start + (ri + 1) * n_seg + si
                all_faces.append([v0, v2, v1])
                all_faces.append([v0, v3, v2])
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])

        # Close inner shell toe-ward end
        inner_toe_ring = inner_start + (inner_rings - 1) * n_seg
        itc_idx = len(all_verts)
        itc = all_verts[inner_toe_ring:inner_toe_ring + n_seg].mean(axis=0)
        all_verts = np.vstack([all_verts, itc])
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            all_faces.append([itc_idx, inner_toe_ring + sn, inner_toe_ring + si])
            all_faces.append([itc_idx, inner_toe_ring + si, inner_toe_ring + sn])

        # Connect outer and inner shells at heel opening (rim, ring 0)
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            o0 = si
            o1 = sn
            i0 = inner_start + si
            i1 = inner_start + sn
            all_faces.append([o0, o1, i1])
            all_faces.append([o0, i1, i0])
            all_faces.append([o0, i1, o1])
            all_faces.append([o0, i0, i1])

        # Wobble
        verts = all_verts.copy()
        for i in range(3):
            verts[:, i] += np.random.uniform(-0.0015, 0.0015, len(verts))

        shoe = trimesh.Trimesh(vertices=verts, faces=np.array(all_faces))
        apply_color_variation(shoe, (*color, 255), 0.12)

        # Color inner shell darker
        darker = tuple(max(0, c - 35) for c in color)
        inner_mask = np.arange(len(verts)) >= inner_start
        colors = shoe.visual.vertex_colors.copy().astype(np.float32)
        for vi in np.where(inner_mask)[0]:
            for ci in range(3):
                colors[vi, ci] = darker[ci] + np.random.uniform(-10, 10)
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        shoe.visual.vertex_colors = colors
        meshes.append(shoe)

    result = trimesh.util.concatenate(meshes)
    return result


def generate_broom(
    height: float = 1.2,
    color: Tuple[int, int, int] = (140, 110, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an old-style round broom (besom) with bristles radiating from handle."""
    set_seed(seed)
    meshes = []

    handle_r = 0.014
    handle_h = height * 0.7
    bristle_h = height * 0.3
    brush_r_top = 0.04   # radius at binding
    brush_r_bot = 0.10   # radius at bristle tips (cone shape)

    # Handle (slightly tapered wooden stick)
    sections = 10
    for j in range(sections):
        t = j / sections
        z = bristle_h + t * handle_h
        dz = handle_h / sections
        r = handle_r * (1.0 - 0.1 * t)
        r *= np.random.uniform(0.97, 1.03)
        seg = trimesh.creation.cylinder(radius=r, height=dz, sections=10)
        seg.apply_translation([0, 0, z + dz / 2])
        apply_color_variation(seg, (*color, 255), 0.08)
        meshes.append(seg)

    # Binding (wraps around where bristles meet handle)
    binding = trimesh.creation.cylinder(
        radius=brush_r_top + 0.005, height=0.03, sections=16)
    binding.apply_translation([0, 0, bristle_h + 0.015])
    apply_color_variation(binding, (70, 55, 35, 255), 0.06)
    meshes.append(binding)

    # Wire wraps
    for wz in [bristle_h + 0.005, bristle_h + 0.028]:
        wire = trimesh.creation.cylinder(
            radius=brush_r_top + 0.007, height=0.004, sections=16)
        wire.apply_translation([0, 0, wz])
        apply_color_variation(wire, (50, 45, 40, 255), 0.04)
        meshes.append(wire)

    # Bristle body: round cone shape from binding down to tips
    # Built as stacked rings expanding outward
    bristle_color = (190, 175, 100, 255)
    n_rings = 12
    n_seg = 20
    b_verts = []

    for ri in range(n_rings):
        t = ri / (n_rings - 1)  # 0=top (binding), 1=bottom (tips)
        z = bristle_h * (1.0 - t)
        r = brush_r_top + (brush_r_bot - brush_r_top) * t

        for si in range(n_seg):
            a = (si / n_seg) * 2 * np.pi
            rx = r * np.cos(a)
            ry = r * np.sin(a)
            # Increasing raggedness toward tips
            wobble = 0.002 + 0.008 * t
            rx += np.random.uniform(-wobble, wobble)
            ry += np.random.uniform(-wobble, wobble)
            z_off = np.random.uniform(-0.008, 0.008) * t
            b_verts.append([rx, ry, z + z_off])

    b_verts = np.array(b_verts)
    b_faces = []
    for ri in range(n_rings - 1):
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            v0 = ri * n_seg + si
            v1 = ri * n_seg + sn
            v2 = (ri + 1) * n_seg + sn
            v3 = (ri + 1) * n_seg + si
            # Double-sided
            b_faces.append([v0, v1, v2])
            b_faces.append([v0, v2, v3])
            b_faces.append([v0, v2, v1])
            b_faces.append([v0, v3, v2])

    # Close top (at binding)
    top_c = len(b_verts)
    b_verts = np.vstack([b_verts, [0, 0, bristle_h]])
    for si in range(n_seg):
        sn = (si + 1) % n_seg
        b_faces.append([top_c, si, sn])
        b_faces.append([top_c, sn, si])

    # Close bottom (tips) - slightly concave center
    bot_ring = (n_rings - 1) * n_seg
    bot_c = len(b_verts)
    b_verts = np.vstack([b_verts, [0, 0, -0.005]])
    for si in range(n_seg):
        sn = (si + 1) % n_seg
        b_faces.append([bot_c, bot_ring + sn, bot_ring + si])
        b_faces.append([bot_c, bot_ring + si, bot_ring + sn])

    bristle_mesh = trimesh.Trimesh(vertices=b_verts, faces=np.array(b_faces))
    apply_color_variation(bristle_mesh, bristle_color, 0.1)
    meshes.append(bristle_mesh)

    result = trimesh.util.concatenate(meshes)
    return result


register(
    name="candlestick", label="Candlestick", category="Props",
    params=[
        Param("height", "Height", "float", default=0.30, min=0.15, max=0.50),
        Param("color", "Metal Color", "color", default="#B49B46"),
    ],
)(generate_candlestick)

register(
    name="book", label="Book", category="Props",
    params=[
        Param("width", "Width", "float", default=0.17, min=0.10, max=0.25),
        Param("height", "Height", "float", default=0.24, min=0.15, max=0.35),
        Param("thickness", "Thickness", "float", default=0.03, min=0.01, max=0.06),
        Param("color", "Cover Color", "color", default="#782820"),
    ],
)(generate_book)

register(
    name="clogs", label="Clogs", category="Props",
    params=[
        Param("size", "Size", "float", default=0.28, min=0.20, max=0.35),
        Param("color", "Wood Color", "color", default="#A0783C"),
    ],
)(generate_clogs)

register(
    name="broom", label="Broom", category="Props",
    params=[
        Param("height", "Height", "float", default=1.2, min=0.8, max=1.6),
        Param("color", "Handle Color", "color", default="#8C6E3C"),
    ],
)(generate_broom)


# =============================================================================
# INDOOR PROPS — Danish 1870s style
# =============================================================================

def generate_kitchen_stove(
    width: float = 0.70,
    height: float = 0.85,
    depth: float = 0.50,
    color: Tuple[int, int, int] = (45, 42, 40),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a Danish 1870s cast-iron kitchen stove with cooking plates and oven door."""
    set_seed(seed)
    meshes = []

    leg_h = 0.12
    body_h = height - leg_h
    wall = 0.025
    iron = (*color, 255)
    brass = (170, 145, 60, 255)

    # --- Stove body (main box) ---
    body = _wobble_box([width, depth, body_h], wobble=0.012)
    body.apply_translation([0, 0, leg_h + body_h / 2])
    apply_color_variation(body, iron, 0.08)
    meshes.append(body)

    # --- Flat cooktop (slightly wider, overhanging) ---
    top = _wobble_box([width + 0.02, depth + 0.02, 0.025], wobble=0.008)
    top.apply_translation([0, 0, height])
    apply_color_variation(top, iron, 0.06)
    meshes.append(top)

    # --- Cooking plates (circular rings on top, 2x2 grid) ---
    plate_r = min(width, depth) * 0.16
    plate_positions = [
        (-width * 0.22, -depth * 0.18),
        (width * 0.22, -depth * 0.18),
        (-width * 0.22, depth * 0.18),
        (width * 0.22, depth * 0.18),
    ]
    for px, py in plate_positions:
        # Outer ring
        ring = trimesh.creation.annulus(r_min=plate_r * 0.6, r_max=plate_r,
                                         height=0.008, sections=20)
        ring.apply_translation([px, py, height + 0.016])
        apply_color_variation(ring, (55, 50, 48, 255), 0.05)
        meshes.append(ring)
        # Center cap
        cap = trimesh.creation.cylinder(radius=plate_r * 0.55, height=0.006, sections=16)
        cap.apply_translation([px, py, height + 0.016])
        apply_color_variation(cap, (50, 47, 44, 255), 0.06)
        meshes.append(cap)

    # --- Oven door (front, lower section) ---
    door_w = width * 0.50
    door_h = body_h * 0.45
    door_y = depth / 2 + 0.002
    door_z = leg_h + body_h * 0.28

    door = _wobble_box([door_w, 0.018, door_h], wobble=0.008)
    door.apply_translation([0, door_y, door_z])
    apply_color_variation(door, (40, 37, 35, 255), 0.07)
    meshes.append(door)

    # Door frame (raised border)
    frame_t = 0.012
    for dims, pos in [
        ([door_w + 0.02, 0.006, frame_t], [0, door_y + 0.01, door_z + door_h / 2 + frame_t / 2]),
        ([door_w + 0.02, 0.006, frame_t], [0, door_y + 0.01, door_z - door_h / 2 - frame_t / 2]),
        ([frame_t, 0.006, door_h], [-door_w / 2 - frame_t / 2, door_y + 0.01, door_z]),
        ([frame_t, 0.006, door_h], [door_w / 2 + frame_t / 2, door_y + 0.01, door_z]),
    ]:
        fr = _wobble_box(dims, wobble=0.003)
        fr.apply_translation(pos)
        apply_color_variation(fr, iron, 0.06)
        meshes.append(fr)

    # Door handle (brass knob)
    handle = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
    handle.apply_translation([door_w * 0.35, door_y + 0.025, door_z])
    apply_color_variation(handle, brass, 0.06)
    meshes.append(handle)

    # --- Firebox door (smaller, upper front) ---
    fb_w = width * 0.30
    fb_h = body_h * 0.20
    fb_z = leg_h + body_h * 0.72
    fb_door = _wobble_box([fb_w, 0.015, fb_h], wobble=0.006)
    fb_door.apply_translation([0, door_y, fb_z])
    apply_color_variation(fb_door, (38, 35, 33, 255), 0.07)
    meshes.append(fb_door)

    # Air vent (small horizontal slot)
    vent = _wobble_box([fb_w * 0.6, 0.008, 0.008], wobble=0.002)
    vent.apply_translation([0, door_y + 0.012, fb_z])
    apply_color_variation(vent, (30, 28, 26, 255), 0.04)
    meshes.append(vent)

    # --- Cast iron legs (4 sturdy legs with slight decoration) ---
    leg_r = 0.025
    leg_positions = [
        (-width / 2 + 0.06, -depth / 2 + 0.06),
        (width / 2 - 0.06, -depth / 2 + 0.06),
        (-width / 2 + 0.06, depth / 2 - 0.06),
        (width / 2 - 0.06, depth / 2 - 0.06),
    ]
    for lx, ly in leg_positions:
        # Leg (tapered cylinder sections for cast-iron look)
        for si in range(4):
            t = si / 4
            z = t * leg_h
            dz = leg_h / 4
            r = leg_r * (1.1 - 0.2 * t + 0.1 * np.sin(t * np.pi))
            r *= np.random.uniform(0.97, 1.03)
            seg = trimesh.creation.cylinder(radius=r, height=dz, sections=10)
            seg.apply_translation([lx, ly, z + dz / 2])
            apply_color_variation(seg, iron, 0.06)
            meshes.append(seg)

        # Small foot pad
        pad = trimesh.creation.cylinder(radius=leg_r * 1.3, height=0.008, sections=10)
        pad.apply_translation([lx, ly, 0.004])
        apply_color_variation(pad, iron, 0.05)
        meshes.append(pad)

    # --- Chimney pipe (exits from back top) ---
    pipe_r = 0.045
    pipe_h = height * 0.45
    pipe = trimesh.creation.cylinder(radius=pipe_r, height=pipe_h, sections=14)
    pipe.apply_translation([0, -depth / 2 + 0.04, height + pipe_h / 2])
    apply_color_variation(pipe, (35, 32, 30, 255), 0.06)
    meshes.append(pipe)

    # Pipe collar at base
    collar = trimesh.creation.cylinder(radius=pipe_r + 0.008, height=0.02, sections=14)
    collar.apply_translation([0, -depth / 2 + 0.04, height + 0.01])
    apply_color_variation(collar, iron, 0.05)
    meshes.append(collar)

    # --- Decorative side panel detail (raised rectangle) ---
    for side in [-1, 1]:
        panel = _wobble_box([0.008, depth * 0.6, body_h * 0.5], wobble=0.004)
        panel.apply_translation([side * (width / 2 + 0.003), 0, leg_h + body_h * 0.5])
        apply_color_variation(panel, (50, 46, 43, 255), 0.06)
        meshes.append(panel)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_fireplace(
    width: float = 1.2,
    height: float = 1.1,
    depth: float = 0.45,
    color: Tuple[int, int, int] = (140, 95, 65),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a Danish 1870s indoor brick fireplace with mantel and hearth."""
    set_seed(seed)
    meshes = []

    brick = (*color, 255)
    mortar = (180, 170, 150, 255)
    mantel_color = (90, 55, 30, 255)
    iron_color = (45, 42, 38, 255)

    opening_w = width * 0.55
    opening_h = height * 0.50
    wall_t = 0.08
    hearth_depth = 0.20

    # --- Main brick surround (U-shape around opening) ---
    # Left pillar
    pillar_w = (width - opening_w) / 2
    for side in [-1, 1]:
        pillar = _wobble_box([pillar_w, depth, height * 0.85], wobble=0.015)
        pillar.apply_translation([side * (opening_w / 2 + pillar_w / 2), 0,
                                  height * 0.85 / 2])
        apply_color_variation(pillar, brick, 0.12)
        meshes.append(pillar)

    # Top section (lintel area, above opening)
    lintel_h = height * 0.85 - opening_h
    lintel = _wobble_box([width, depth, lintel_h], wobble=0.012)
    lintel.apply_translation([0, 0, opening_h + lintel_h / 2])
    apply_color_variation(lintel, brick, 0.12)
    meshes.append(lintel)

    # Back wall of firebox
    back = _wobble_box([opening_w + 0.02, wall_t, opening_h], wobble=0.012)
    back.apply_translation([0, -depth / 2 + wall_t / 2, opening_h / 2])
    apply_color_variation(back, (60, 45, 35, 255), 0.1)
    meshes.append(back)

    # Firebox floor
    floor = _wobble_box([opening_w, depth - wall_t, 0.03], wobble=0.008)
    floor.apply_translation([0, wall_t / 2, 0.015])
    apply_color_variation(floor, (55, 42, 32, 255), 0.08)
    meshes.append(floor)

    # --- Brick texture rows (decorative horizontal lines on surround) ---
    brick_h = 0.065
    num_rows = int(height * 0.85 / brick_h)
    for ri in range(num_rows):
        z = ri * brick_h + brick_h / 2
        if z > opening_h or z > height * 0.85:
            # Full width row
            row = _wobble_box([width + 0.003, 0.006, 0.004], wobble=0.002)
            row.apply_translation([0, depth / 2 + 0.002, z])
            apply_color_variation(row, mortar, 0.06)
            meshes.append(row)
        else:
            # Only on pillars
            for side in [-1, 1]:
                row = _wobble_box([pillar_w + 0.003, 0.006, 0.004], wobble=0.002)
                row.apply_translation([side * (opening_w / 2 + pillar_w / 2),
                                       depth / 2 + 0.002, z])
                apply_color_variation(row, mortar, 0.06)
                meshes.append(row)

    # --- Keystone arch above opening ---
    arch_segments = 12
    arch_r = opening_w / 2
    arch_rise = opening_h * 0.12
    for i in range(arch_segments):
        t0 = i / arch_segments
        t1 = (i + 1) / arch_segments
        a0 = t0 * np.pi
        a1 = t1 * np.pi
        x0 = arch_r * np.cos(a0)
        z0 = opening_h + arch_rise * np.sin(a0)
        x1 = arch_r * np.cos(a1)
        z1 = opening_h + arch_rise * np.sin(a1)
        seg_len = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)
        seg = _wobble_box([seg_len, depth * 0.15, 0.04], wobble=0.004)
        mid_x = (x0 + x1) / 2
        mid_z = (z0 + z1) / 2
        seg_angle = np.arctan2(z1 - z0, x1 - x0)
        seg.apply_transform(trimesh.transformations.rotation_matrix(seg_angle, [0, 1, 0]))
        seg.apply_translation([mid_x, depth / 2 - depth * 0.075, mid_z])
        apply_color_variation(seg, brick, 0.1)
        meshes.append(seg)

    # --- Wooden mantel (top shelf) ---
    mantel_h = 0.05
    mantel = _wobble_box([width + 0.08, depth + 0.06, mantel_h], wobble=0.012)
    mantel.apply_translation([0, 0.03, height * 0.85 + mantel_h / 2])
    apply_color_variation(mantel, mantel_color, 0.1)
    meshes.append(mantel)

    # Mantel supports (two small brackets)
    for side in [-1, 1]:
        bracket = _wobble_box([0.04, 0.04, 0.06], wobble=0.005)
        bracket.apply_translation([side * (width / 2 - 0.03), depth / 2 + 0.01,
                                   height * 0.85 - 0.03])
        apply_color_variation(bracket, mantel_color, 0.08)
        meshes.append(bracket)

    # --- Hearth (stone slab extending forward) ---
    hearth = _wobble_box([width + 0.10, hearth_depth, 0.04], wobble=0.01)
    hearth.apply_translation([0, depth / 2 + hearth_depth / 2 - 0.02, 0.02])
    apply_color_variation(hearth, (120, 115, 105, 255), 0.08)
    meshes.append(hearth)

    # --- Fire grate (iron basket inside) ---
    grate_w = opening_w * 0.6
    grate_h = 0.15
    grate_d = depth * 0.4
    grate_z = 0.04

    # Grate bottom (grid of bars)
    for i in range(5):
        bar = trimesh.creation.cylinder(radius=0.005, height=grate_d, sections=6)
        bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        bx = -grate_w / 2 + (i + 0.5) * grate_w / 5
        bar.apply_translation([bx, 0, grate_z])
        apply_color_variation(bar, iron_color, 0.05)
        meshes.append(bar)

    # Grate sides (angled iron bars)
    for side in [-1, 1]:
        side_bar = _wobble_box([0.008, grate_d * 0.8, grate_h], wobble=0.003)
        side_bar.apply_translation([side * grate_w / 2, 0, grate_z + grate_h / 2])
        apply_color_variation(side_bar, iron_color, 0.05)
        meshes.append(side_bar)

    # Grate back
    grate_back = _wobble_box([grate_w, 0.008, grate_h], wobble=0.003)
    grate_back.apply_translation([0, -grate_d / 2, grate_z + grate_h / 2])
    apply_color_variation(grate_back, iron_color, 0.05)
    meshes.append(grate_back)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_chimney(
    width: float = 0.50,
    height: float = 2.0,
    color: Tuple[int, int, int] = (155, 100, 65),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a Danish 1870s brick chimney with cap and mortar lines."""
    set_seed(seed)
    meshes = []

    brick = (*color, 255)
    mortar = (185, 175, 155, 255)
    depth = width * 0.85

    # Slight taper: base is slightly wider
    taper = 0.03

    # --- Main chimney shaft (built from stacked brick courses) ---
    brick_h = 0.065
    num_courses = int(height / brick_h)

    for ci in range(num_courses):
        t = ci / max(num_courses - 1, 1)
        z = ci * brick_h
        w = width - taper * t
        d = depth - taper * 0.8 * t

        course = _wobble_box([w, d, brick_h - 0.003], wobble=0.008)
        course.apply_translation([0, 0, z + brick_h / 2])
        # Vary brick color per course slightly
        var = np.random.uniform(-8, 8, 3).astype(int)
        course_color = tuple(np.clip(np.array(color) + var, 0, 255))
        apply_color_variation(course, (*course_color, 255), 0.1)
        meshes.append(course)

        # Mortar line (thin gap between courses)
        if ci < num_courses - 1:
            mortar_line = _wobble_box([w + 0.002, d + 0.002, 0.004], wobble=0.002)
            mortar_line.apply_translation([0, 0, z + brick_h])
            apply_color_variation(mortar_line, mortar, 0.05)
            meshes.append(mortar_line)

    # --- Crown (wider top cap, overhanging) ---
    top_w = width - taper + 0.06
    top_d = depth - taper * 0.8 + 0.06
    crown_h = 0.06

    # Lower crown course
    crown_low = _wobble_box([top_w - 0.02, top_d - 0.02, crown_h * 0.5], wobble=0.008)
    crown_low.apply_translation([0, 0, height + crown_h * 0.25])
    apply_color_variation(crown_low, brick, 0.08)
    meshes.append(crown_low)

    # Upper crown (overhanging cap)
    crown_top = _wobble_box([top_w, top_d, crown_h * 0.5], wobble=0.01)
    crown_top.apply_translation([0, 0, height + crown_h * 0.75])
    apply_color_variation(crown_top, brick, 0.08)
    meshes.append(crown_top)

    # --- Flue opening (dark hole on top) ---
    flue_w = width * 0.45
    flue_d = depth * 0.45
    flue = _wobble_box([flue_w, flue_d, 0.03], wobble=0.004)
    flue.apply_translation([0, 0, height + crown_h + 0.01])
    apply_color_variation(flue, (30, 25, 22, 255), 0.03)
    meshes.append(flue)

    # --- Vertical mortar lines (every few courses, staggered) ---
    for ci in range(0, num_courses, 2):
        t = ci / max(num_courses - 1, 1)
        z = ci * brick_h
        w = width - taper * t

        # 2-3 vertical lines per face
        num_vlines = max(2, int(w / 0.15))
        for vi in range(num_vlines):
            vx = -w / 2 + (vi + 0.5 + 0.5 * (ci % 4 == 0)) * w / (num_vlines + 0.5)
            vx = np.clip(vx, -w / 2 + 0.02, w / 2 - 0.02)
            vline = _wobble_box([0.004, 0.006, brick_h - 0.003], wobble=0.002)
            vline.apply_translation([vx, depth / 2 * (1 - taper * 0.8 * t / depth) + 0.002,
                                     z + brick_h / 2])
            apply_color_variation(vline, mortar, 0.05)
            meshes.append(vline)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_washing_basin(
    height: float = 0.85,
    color: Tuple[int, int, int] = (120, 78, 42),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a Danish 1870s washing basin on a wooden stand with towel bar."""
    set_seed(seed)
    meshes = []

    wood = (*color, 255)
    ceramic = (230, 225, 215, 255)

    stand_w = 0.55
    stand_d = 0.40
    top_h = height
    leg_radius = 0.018
    leg_inset = 0.04
    apron_h = 0.06

    # --- Tabletop (with basin cutout area) ---
    top_t = 0.03
    top = _wobble_box([stand_w, stand_d, top_t], wobble=0.01)
    top.apply_translation([0, 0, top_h])
    apply_color_variation(top, wood, 0.1)
    meshes.append(top)

    # --- Turned legs (4 legs) ---
    leg_h = top_h - top_t / 2
    leg_positions = [
        (-stand_w / 2 + leg_inset, -stand_d / 2 + leg_inset),
        (stand_w / 2 - leg_inset, -stand_d / 2 + leg_inset),
        (-stand_w / 2 + leg_inset, stand_d / 2 - leg_inset),
        (stand_w / 2 - leg_inset, stand_d / 2 - leg_inset),
    ]
    for lx, ly in leg_positions:
        # Build turned leg similar to furniture.py pattern
        sections = 8
        segments = 10
        l_verts = []
        l_faces = []
        for j in range(sections + 1):
            t = j / sections
            z = t * leg_h
            r = leg_radius * (0.8 + 0.3 * np.sin(t * np.pi)
                              - 0.1 * np.sin(t * 2 * np.pi))
            if t < 0.08:
                r = leg_radius * (1.1 - 0.3 * t / 0.08)
            r *= np.random.uniform(0.97, 1.03)
            for si in range(segments):
                angle = (si / segments) * 2 * np.pi
                l_verts.append([lx + r * np.cos(angle), ly + r * np.sin(angle), z])
        l_verts = np.array(l_verts)
        for j in range(sections):
            for si in range(segments):
                sn = (si + 1) % segments
                v0 = j * segments + si
                v1 = j * segments + sn
                v2 = (j + 1) * segments + sn
                v3 = (j + 1) * segments + si
                l_faces.append([v0, v1, v2])
                l_faces.append([v0, v2, v3])
        # Caps
        bc = len(l_verts)
        l_verts = np.vstack([l_verts, [lx, ly, 0]])
        for si in range(segments):
            sn = (si + 1) % segments
            l_faces.append([bc, sn, si])
        tc = len(l_verts)
        l_verts = np.vstack([l_verts, [lx, ly, leg_h]])
        top_ring = sections * segments
        for si in range(segments):
            sn = (si + 1) % segments
            l_faces.append([tc, top_ring + si, top_ring + sn])
        leg_mesh = trimesh.Trimesh(vertices=l_verts, faces=np.array(l_faces))
        leg_mesh.fix_normals()
        apply_color_variation(leg_mesh, wood, 0.08)
        meshes.append(leg_mesh)

    # --- Apron (frame under top) ---
    apron_z = top_h - top_t / 2 - apron_h / 2
    for y_pos in [-stand_d / 2 + leg_inset, stand_d / 2 - leg_inset]:
        ap = _wobble_box([stand_w - 2 * leg_inset, 0.018, apron_h], wobble=0.006)
        ap.apply_translation([0, y_pos, apron_z])
        apply_color_variation(ap, wood, 0.08)
        meshes.append(ap)
    for x_pos in [-stand_w / 2 + leg_inset, stand_w / 2 - leg_inset]:
        ap = _wobble_box([0.018, stand_d - 2 * leg_inset, apron_h], wobble=0.006)
        ap.apply_translation([x_pos, 0, apron_z])
        apply_color_variation(ap, wood, 0.08)
        meshes.append(ap)

    # --- Lower shelf ---
    shelf_z = leg_h * 0.25
    shelf = _wobble_box([stand_w - 2 * leg_inset - 0.02,
                         stand_d - 2 * leg_inset - 0.02, 0.015], wobble=0.008)
    shelf.apply_translation([0, 0, shelf_z])
    apply_color_variation(shelf, wood, 0.1)
    meshes.append(shelf)

    # --- Ceramic basin (bowl shape on top) ---
    basin_r = 0.16
    basin_h = 0.09
    n_rings = 10
    n_seg = 20
    b_verts = []
    b_faces = []
    for ri in range(n_rings + 1):
        t = ri / n_rings
        z = top_h + top_t / 2 + basin_h * (1.0 - t)
        r = basin_r * np.sin(t * np.pi * 0.5 + np.pi * 0.05)
        r = max(r, 0.01)
        for si in range(n_seg):
            a = (si / n_seg) * 2 * np.pi
            bx = r * np.cos(a) + np.random.uniform(-0.001, 0.001)
            by = r * np.sin(a) + np.random.uniform(-0.001, 0.001)
            b_verts.append([bx, by, z])
    b_verts = np.array(b_verts)
    for ri in range(n_rings):
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            v0 = ri * n_seg + si
            v1 = ri * n_seg + sn
            v2 = (ri + 1) * n_seg + sn
            v3 = (ri + 1) * n_seg + si
            b_faces.append([v0, v1, v2])
            b_faces.append([v0, v2, v3])
            b_faces.append([v0, v2, v1])
            b_faces.append([v0, v3, v2])
    # Bottom center
    bot_c = len(b_verts)
    b_verts = np.vstack([b_verts, [0, 0, top_h + top_t / 2 + 0.005]])
    bot_ring = n_rings * n_seg
    for si in range(n_seg):
        sn = (si + 1) % n_seg
        b_faces.append([bot_c, bot_ring + sn, bot_ring + si])
        b_faces.append([bot_c, bot_ring + si, bot_ring + sn])

    basin_mesh = trimesh.Trimesh(vertices=b_verts, faces=np.array(b_faces))
    basin_mesh.fix_normals()
    apply_color_variation(basin_mesh, ceramic, 0.04)
    meshes.append(basin_mesh)

    # Basin rim (torus-like thickened edge)
    rim_pts = []
    for i in range(24):
        a = (i / 24) * 2 * np.pi
        for j in range(8):
            b = (j / 8) * 2 * np.pi
            rx = (basin_r + 0.008 * np.cos(b)) * np.cos(a)
            ry = (basin_r + 0.008 * np.cos(b)) * np.sin(a)
            rz = top_h + top_t / 2 + basin_h + 0.008 * np.sin(b)
            rim_pts.append([rx, ry, rz])
    rim = trimesh.convex.convex_hull(np.array(rim_pts))
    apply_color_variation(rim, ceramic, 0.03)
    meshes.append(rim)

    # --- Towel bar (wooden rod between back legs) ---
    bar_z = leg_h * 0.55
    bar = trimesh.creation.cylinder(radius=0.008, height=stand_w - 2 * leg_inset, sections=8)
    bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    bar.apply_translation([0, -stand_d / 2 + leg_inset, bar_z])
    apply_color_variation(bar, wood, 0.08)
    meshes.append(bar)

    # --- Splash back (small raised board behind basin) ---
    splash_h = 0.18
    splash = _wobble_box([stand_w * 0.8, 0.015, splash_h], wobble=0.008)
    splash.apply_translation([0, -stand_d / 2 + leg_inset,
                              top_h + top_t / 2 + splash_h / 2])
    apply_color_variation(splash, wood, 0.1)
    meshes.append(splash)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


# =============================================================================
# OUTDOOR PROPS — Danish 1870s rural style
# =============================================================================

def generate_pitchfork(
    height: float = 1.5,
    color: Tuple[int, int, int] = (130, 95, 50),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a hand-forged pitchfork with wooden handle and iron tines."""
    set_seed(seed)
    meshes = []

    wood = (*color, 255)
    iron = (55, 50, 45, 255)
    handle_r = 0.015
    handle_h = height * 0.72
    neck_h = height * 0.08
    tine_h = height * 0.20

    # --- Wooden handle (slightly irregular) ---
    sections = 10
    for j in range(sections):
        t = j / sections
        z = t * handle_h
        dz = handle_h / sections
        r = handle_r * (1.0 - 0.08 * t)
        r *= np.random.uniform(0.96, 1.04)
        seg = trimesh.creation.cylinder(radius=r, height=dz, sections=8)
        seg.apply_translation([0, 0, z + dz / 2])
        apply_color_variation(seg, wood, 0.1)
        meshes.append(seg)

    # Handle end knob
    knob = trimesh.creation.icosphere(subdivisions=1, radius=handle_r * 1.5)
    knob.apply_translation([0, 0, 0])
    apply_color_variation(knob, wood, 0.08)
    meshes.append(knob)

    # --- Metal neck/socket (where handle meets tines) ---
    socket = trimesh.creation.cylinder(radius=handle_r + 0.005, height=neck_h, sections=10)
    socket.apply_translation([0, 0, handle_h + neck_h / 2])
    apply_color_variation(socket, iron, 0.06)
    meshes.append(socket)

    # --- Tines (3 prongs curving forward slightly) ---
    tine_r = 0.005
    n_tines = 3
    tine_spacing = 0.06
    tine_z_start = handle_h + neck_h

    for ti in range(n_tines):
        tx = (ti - (n_tines - 1) / 2) * tine_spacing
        # Build tine from segments with slight forward curve
        n_seg = 8
        prev_pos = np.array([tx, 0, tine_z_start])
        for si in range(n_seg):
            st = (si + 1) / n_seg
            # Slight forward curve
            cur_y = 0.03 * np.sin(st * np.pi * 0.4)
            cur_z = tine_z_start + st * tine_h
            cur_pos = np.array([tx, cur_y, cur_z])

            diff = cur_pos - prev_pos
            seg_len = np.linalg.norm(diff)
            seg = trimesh.creation.cylinder(
                radius=tine_r * (1.0 - 0.3 * st), height=seg_len, sections=6)
            direction = diff / seg_len
            z_axis = np.array([0, 0, 1])
            if abs(np.dot(direction, z_axis)) < 0.999:
                axis = np.cross(z_axis, direction)
                axis = axis / np.linalg.norm(axis)
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                seg.apply_transform(
                    trimesh.transformations.rotation_matrix(rot_angle, axis))
            mid = (prev_pos + cur_pos) / 2
            seg.apply_translation(mid)
            apply_color_variation(seg, iron, 0.06)
            meshes.append(seg)
            prev_pos = cur_pos

    # Cross bar connecting tines at base
    crossbar = _wobble_box([tine_spacing * (n_tines - 1) + 0.02, 0.008, 0.008],
                           wobble=0.003)
    crossbar.apply_translation([0, 0, tine_z_start + 0.01])
    apply_color_variation(crossbar, iron, 0.06)
    meshes.append(crossbar)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_rake(
    height: float = 1.5,
    color: Tuple[int, int, int] = (125, 90, 48),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique wooden rake with iron head and tines."""
    set_seed(seed)
    meshes = []

    wood = (*color, 255)
    iron = (55, 50, 45, 255)
    handle_r = 0.014
    handle_h = height * 0.82
    head_w = 0.40

    # --- Wooden handle ---
    sections = 10
    for j in range(sections):
        t = j / sections
        z = t * handle_h
        dz = handle_h / sections
        r = handle_r * (1.0 - 0.06 * t)
        r *= np.random.uniform(0.96, 1.04)
        seg = trimesh.creation.cylinder(radius=r, height=dz, sections=8)
        seg.apply_translation([0, 0, z + dz / 2])
        apply_color_variation(seg, wood, 0.1)
        meshes.append(seg)

    # Handle end
    knob = trimesh.creation.icosphere(subdivisions=1, radius=handle_r * 1.4)
    knob.apply_translation([0, 0, 0])
    apply_color_variation(knob, wood, 0.08)
    meshes.append(knob)

    # --- Rake head (horizontal bar) ---
    head_h = 0.025
    head_d = 0.025
    head = _wobble_box([head_w, head_d, head_h], wobble=0.008)
    head.apply_translation([0, 0, handle_h + head_h / 2])
    apply_color_variation(head, iron, 0.08)
    meshes.append(head)

    # Socket where handle meets head
    socket = _wobble_box([0.03, 0.03, 0.04], wobble=0.004)
    socket.apply_translation([0, 0, handle_h + 0.01])
    apply_color_variation(socket, iron, 0.06)
    meshes.append(socket)

    # --- Tines (short teeth pointing down from head) ---
    n_tines = 10
    tine_h = 0.07
    tine_r = 0.004
    for ti in range(n_tines):
        tx = -head_w / 2 + (ti + 0.5) * head_w / n_tines
        tine = trimesh.creation.cylinder(
            radius=tine_r * np.random.uniform(0.9, 1.1),
            height=tine_h, sections=6)
        # Tines point slightly forward and down
        tine.apply_transform(
            trimesh.transformations.rotation_matrix(0.15, [1, 0, 0]))
        tine.apply_translation([tx, 0.01, handle_h - tine_h / 2 + 0.01])
        apply_color_variation(tine, iron, 0.06)
        meshes.append(tine)

        # Tine tip (pointed)
        tip = trimesh.creation.cone(radius=tine_r, height=0.015, sections=6)
        tip.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        tip.apply_transform(
            trimesh.transformations.rotation_matrix(0.15, [1, 0, 0]))
        tip.apply_translation([tx, 0.01, handle_h - tine_h + 0.003])
        apply_color_variation(tip, iron, 0.06)
        meshes.append(tip)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_hay_pile(
    radius: float = 1.0,
    height: float = 0.8,
    color: Tuple[int, int, int] = (195, 175, 90),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a loose hay pile with irregular surface and stray strands."""
    set_seed(seed)
    meshes = []

    hay = (*color, 255)

    # --- Main pile body (deformed half-sphere) ---
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    verts = sphere.vertices.copy()

    # Keep only upper half, flatten bottom
    keep_mask = verts[:, 2] > -radius * 0.15
    # Deform into pile shape: flatten top, spread base
    for i in range(len(verts)):
        x, y, z = verts[i]
        if z < 0:
            verts[i, 2] = 0
            # Spread base outward
            spread = 1.2
            verts[i, 0] *= spread
            verts[i, 1] *= spread
        else:
            # Squash height
            verts[i, 2] *= height / radius
            # Random surface bumps for loose hay look
            bump = np.random.uniform(-0.06, 0.06) * radius
            verts[i, 0] += bump
            verts[i, 1] += bump
            verts[i, 2] += np.random.uniform(-0.04, 0.04) * height

    sphere.vertices = verts
    # Remove degenerate faces
    sphere.fix_normals()
    apply_color_variation(sphere, hay, 0.15)
    meshes.append(sphere)

    # --- Loose strands sticking out (thin cylinders) ---
    n_strands = 25
    for _ in range(n_strands):
        angle = np.random.uniform(0, 2 * np.pi)
        r_pos = np.random.uniform(0.2, 0.9) * radius
        sx = r_pos * np.cos(angle)
        sy = r_pos * np.sin(angle)
        # Height on pile surface (approximate)
        t_r = r_pos / radius
        sz = height * np.sqrt(max(0, 1.0 - t_r ** 2)) * np.random.uniform(0.6, 1.0)

        strand_len = np.random.uniform(0.08, 0.25) * radius
        strand = trimesh.creation.cylinder(
            radius=np.random.uniform(0.003, 0.006), height=strand_len, sections=4)

        # Random tilt
        tilt_x = np.random.uniform(-0.6, 0.6)
        tilt_y = np.random.uniform(-0.6, 0.6)
        strand.apply_transform(
            trimesh.transformations.rotation_matrix(tilt_x, [1, 0, 0]))
        strand.apply_transform(
            trimesh.transformations.rotation_matrix(tilt_y, [0, 1, 0]))
        strand.apply_translation([sx, sy, sz + strand_len / 2])

        # Slightly varied hay color
        strand_var = tuple(np.clip(np.array(color) + np.random.randint(-20, 20, 3),
                                   0, 255))
        apply_color_variation(strand, (*strand_var, 255), 0.1)
        meshes.append(strand)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_henhouse(
    width: float = 1.2,
    height: float = 1.0,
    depth: float = 0.9,
    color: Tuple[int, int, int] = (130, 90, 45),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a Danish 1870s wooden henhouse raised on legs with nesting boxes."""
    set_seed(seed)
    meshes = []

    wood = (*color, 255)
    roof_color = (80, 65, 45, 255)
    iron = (50, 45, 40, 255)

    leg_h = 0.30
    body_h = height - leg_h
    wall = 0.02
    roof_overhang = 0.08
    roof_angle = 0.25  # pitch

    # --- Legs (4 posts, slightly irregular) ---
    leg_r = 0.025
    for lx, ly in [
        (-width / 2 + 0.06, -depth / 2 + 0.06),
        (width / 2 - 0.06, -depth / 2 + 0.06),
        (-width / 2 + 0.06, depth / 2 - 0.06),
        (width / 2 - 0.06, depth / 2 - 0.06),
    ]:
        leg = _wobble_box([leg_r * 2, leg_r * 2, leg_h], wobble=0.012)
        leg.apply_translation([lx, ly, leg_h / 2])
        apply_color_variation(leg, wood, 0.1)
        meshes.append(leg)

    # --- Floor ---
    floor = _wobble_box([width, depth, wall], wobble=0.01)
    floor.apply_translation([0, 0, leg_h])
    apply_color_variation(floor, wood, 0.08)
    meshes.append(floor)

    # --- Walls ---
    # Back wall
    back = _wobble_box([width, wall, body_h], wobble=0.012)
    back.apply_translation([0, -depth / 2 + wall / 2, leg_h + body_h / 2])
    apply_color_variation(back, wood, 0.1)
    meshes.append(back)

    # Side walls
    for side in [-1, 1]:
        sw = _wobble_box([wall, depth, body_h], wobble=0.012)
        sw.apply_translation([side * (width / 2 - wall / 2), 0, leg_h + body_h / 2])
        apply_color_variation(sw, wood, 0.1)
        meshes.append(sw)

    # Front wall (with door opening)
    door_w = width * 0.25
    door_h = body_h * 0.55
    # Left section
    front_left_w = (width - door_w) / 2
    fl = _wobble_box([front_left_w, wall, body_h], wobble=0.01)
    fl.apply_translation([-(door_w / 2 + front_left_w / 2), depth / 2 - wall / 2,
                          leg_h + body_h / 2])
    apply_color_variation(fl, wood, 0.1)
    meshes.append(fl)

    # Right section
    fr = _wobble_box([front_left_w, wall, body_h], wobble=0.01)
    fr.apply_translation([(door_w / 2 + front_left_w / 2), depth / 2 - wall / 2,
                          leg_h + body_h / 2])
    apply_color_variation(fr, wood, 0.1)
    meshes.append(fr)

    # Above door
    above_door_h = body_h - door_h
    ad = _wobble_box([door_w, wall, above_door_h], wobble=0.008)
    ad.apply_translation([0, depth / 2 - wall / 2, leg_h + door_h + above_door_h / 2])
    apply_color_variation(ad, wood, 0.1)
    meshes.append(ad)

    # --- Door (simple plank) ---
    door = _wobble_box([door_w - 0.02, 0.012, door_h - 0.02], wobble=0.008)
    door.apply_translation([0, depth / 2 + 0.005, leg_h + door_h / 2])
    darker = tuple(max(0, c - 15) for c in color)
    apply_color_variation(door, (*darker, 255), 0.1)
    meshes.append(door)

    # Door hinge (small iron strips)
    for hz in [leg_h + door_h * 0.25, leg_h + door_h * 0.75]:
        hinge = _wobble_box([0.03, 0.006, 0.015], wobble=0.002)
        hinge.apply_translation([-door_w / 2 + 0.01, depth / 2 + 0.012, hz])
        apply_color_variation(hinge, iron, 0.04)
        meshes.append(hinge)

    # --- Sloped roof (front higher, back lower — simple lean-to) ---
    roof_front_z = leg_h + body_h + roof_angle * depth
    roof_back_z = leg_h + body_h
    roof_w = width + 2 * roof_overhang
    roof_d = depth + 2 * roof_overhang

    # Build as a tilted box
    roof_t = 0.025
    roof_verts = np.array([
        [-roof_w / 2, -roof_d / 2, roof_back_z],
        [roof_w / 2, -roof_d / 2, roof_back_z],
        [roof_w / 2, roof_d / 2, roof_front_z],
        [-roof_w / 2, roof_d / 2, roof_front_z],
        [-roof_w / 2, -roof_d / 2, roof_back_z + roof_t],
        [roof_w / 2, -roof_d / 2, roof_back_z + roof_t],
        [roof_w / 2, roof_d / 2, roof_front_z + roof_t],
        [-roof_w / 2, roof_d / 2, roof_front_z + roof_t],
    ], dtype=np.float64)
    # Wobble roof vertices
    for i in range(3):
        roof_verts[:, i] += np.random.uniform(-0.008, 0.008, len(roof_verts))
    roof_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # back
        [2, 6, 7], [2, 7, 3],  # front
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ])
    roof = trimesh.Trimesh(vertices=roof_verts, faces=roof_faces)
    roof.fix_normals()
    apply_color_variation(roof, roof_color, 0.1)
    meshes.append(roof)

    # --- Nesting boxes (2 boxes inside, against back wall) ---
    box_w = width * 0.35
    box_h = body_h * 0.40
    box_d = depth * 0.45
    for side in [-1, 1]:
        bx = side * width * 0.22
        # Box bottom
        bb = _wobble_box([box_w, box_d, wall * 0.7], wobble=0.005)
        bb.apply_translation([bx, -depth / 2 + wall + box_d / 2,
                              leg_h + wall])
        apply_color_variation(bb, wood, 0.08)
        meshes.append(bb)
        # Box sides
        for s2 in [-1, 1]:
            bs = _wobble_box([wall * 0.7, box_d, box_h], wobble=0.005)
            bs.apply_translation([bx + s2 * box_w / 2,
                                  -depth / 2 + wall + box_d / 2,
                                  leg_h + wall + box_h / 2])
            apply_color_variation(bs, wood, 0.08)
            meshes.append(bs)
        # Box front lip (low)
        lip = _wobble_box([box_w, wall * 0.7, box_h * 0.3], wobble=0.004)
        lip.apply_translation([bx, -depth / 2 + wall + box_d,
                              leg_h + wall + box_h * 0.15])
        apply_color_variation(lip, wood, 0.08)
        meshes.append(lip)

    # --- Ramp (chicken ladder to door) ---
    ramp_len = 0.50
    ramp_w = 0.15
    ramp = _wobble_box([ramp_w, ramp_len, 0.012], wobble=0.008)
    ramp_angle = np.arctan2(leg_h, ramp_len)
    ramp.apply_transform(
        trimesh.transformations.rotation_matrix(-ramp_angle, [1, 0, 0]))
    ramp.apply_translation([0, depth / 2 + ramp_len * 0.4, leg_h * 0.4])
    apply_color_variation(ramp, wood, 0.1)
    meshes.append(ramp)

    # Ramp rungs
    n_rungs = 5
    for ri in range(n_rungs):
        rt = (ri + 0.5) / n_rungs
        ry = depth / 2 + rt * ramp_len * 0.8
        rz = leg_h * (1.0 - rt * 0.9)
        rung = _wobble_box([ramp_w + 0.01, 0.01, 0.01], wobble=0.003)
        rung.apply_translation([0, ry, rz])
        apply_color_variation(rung, wood, 0.08)
        meshes.append(rung)

    # --- Perch bar (inside, horizontal rod) ---
    perch = trimesh.creation.cylinder(radius=0.012, height=width * 0.7, sections=8)
    perch.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    perch.apply_translation([0, depth * 0.1, leg_h + body_h * 0.6])
    apply_color_variation(perch, wood, 0.08)
    meshes.append(perch)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_firewood_stack(
    width: float = 1.0,
    height: float = 0.8,
    depth: float = 0.40,
    color: Tuple[int, int, int] = (110, 75, 40),
    seed: Optional[int] = None
) -> trimesh.Scene:
    """Generate a firewood stack of many small logs stacked on top of each other."""
    set_seed(seed)
    bark_meshes = []
    wood_meshes = []

    log_r_min = 0.03
    log_r_max = 0.06
    log_length = depth

    # Calculate how many logs fit
    avg_r = (log_r_min + log_r_max) / 2
    cols = max(3, int(width / (avg_r * 2.2)))
    rows = max(3, int(height / (avg_r * 2.0)))

    for row in range(rows):
        # Each row offset slightly for stacking
        offset_x = avg_r * (row % 2) * 0.5
        row_z = avg_r + row * avg_r * 1.85

        if row_z > height:
            break

        n_in_row = cols - (row % 2)
        for col in range(n_in_row):
            r = np.random.uniform(log_r_min, log_r_max)
            lx = -width / 2 + r + col * (width - 2 * r) / max(n_in_row - 1, 1)
            lx += offset_x + np.random.uniform(-0.01, 0.01)
            lz = row_z + np.random.uniform(-0.008, 0.008)
            l_len = log_length * np.random.uniform(0.88, 1.05)

            # Log body (cylinder with bark texture via color variation)
            log = trimesh.creation.cylinder(radius=r, height=l_len, sections=8)
            log.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            # Slight random tilt
            log.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.random.uniform(-0.04, 0.04), [1, 0, 0]))
            log.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.random.uniform(-0.04, 0.04), [0, 0, 1]))
            log.apply_translation([lx, 0, lz])

            bark_meshes.append(log)

            # Cross-section end caps (lighter wood color, visible at front and back)
            for cap_side in [-1, 1]:
                cap = trimesh.creation.cylinder(
                    radius=r * 0.92, height=0.004, sections=8)
                cap.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                cap.apply_translation([lx, cap_side * l_len / 2, lz])
                wood_meshes.append(cap)

    # --- Optional: side support stakes (2 vertical posts at ends) ---
    stake_h = height + 0.10
    stake_r = 0.02
    for side in [-1, 1]:
        stake = _wobble_box([stake_r * 2, stake_r * 2, stake_h], wobble=0.01)
        stake.apply_translation([side * (width / 2 + stake_r), 0, stake_h / 2])
        bark_meshes.append(stake)

    # Build textured Scene with PBR wood material
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    from core.mesh import unweld_mesh, compute_triplanar_uvs
    from textures.generator import wood_texture

    tex = wood_texture(256, 256, seed)
    diffuse_img = Image.fromarray(tex.diffuse)
    normal_img = Image.fromarray(tex.normal)

    scene = trimesh.Scene()

    for group_name, group_meshes, uv_scale in [
        ('bark', bark_meshes, 10.0),
        ('wood', wood_meshes, 15.0),
    ]:
        combined = trimesh.util.concatenate(group_meshes)
        unwelded = unweld_mesh(combined)
        uvs = compute_triplanar_uvs(unwelded, scale=uv_scale)
        mat = PBRMaterial(
            baseColorTexture=diffuse_img,
            normalTexture=normal_img,
            roughnessFactor=0.7,
        )
        unwelded.visual = TextureVisuals(uv=uvs, material=mat)
        scene.add_geometry(unwelded, node_name=group_name)

    return scene


def generate_flower_bed(
    width: float = 1.2,
    depth: float = 0.6,
    height: float = 0.25,
    color: Tuple[int, int, int] = (120, 80, 40),
    seed: Optional[int] = None
) -> trimesh.Scene:
    """Generate a rustic raised flower bed with crop rows."""
    set_seed(seed)
    wood_meshes = []
    plant_meshes = []

    soil_rgba = (75, 50, 28, 255)
    plank_t = 0.035

    # Wooden frame — 2 long sides (X) + 2 short sides (Y)
    for side in [-1, 1]:
        plank = _wobble_box([width, plank_t, height], wobble=0.012)
        plank.apply_translation([0, side * (depth / 2 - plank_t / 2), height / 2])
        wood_meshes.append(plank)

    for side in [-1, 1]:
        plank = _wobble_box([plank_t, depth - 2 * plank_t, height], wobble=0.012)
        plank.apply_translation([side * (width / 2 - plank_t / 2), 0, height / 2])
        wood_meshes.append(plank)

    # Corner posts (slightly taller than frame)
    post_s = 0.045
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            post = _wobble_box([post_s, post_s, height + 0.04], wobble=0.01)
            post.apply_translation([
                sx * (width / 2 - post_s / 2),
                sy * (depth / 2 - post_s / 2),
                (height + 0.04) / 2,
            ])
            wood_meshes.append(post)

    # Soil fill
    soil_w = width - 2 * plank_t - 0.01
    soil_d = depth - 2 * plank_t - 0.01
    soil_h = height * 0.75
    soil = trimesh.creation.box(extents=[soil_w, soil_d, soil_h])
    soil.apply_translation([0, 0, soil_h / 2])
    apply_color_variation(soil, soil_rgba, 0.15)
    plant_meshes.append(soil)

    # Crop rows — 50% carrot tops, 50% cabbages
    inner_w = soil_w - 0.04
    inner_d = soil_d - 0.04
    n_rows = max(2, int(inner_d / 0.10))
    plants_per_row = max(3, int(inner_w / 0.07))

    for row in range(n_rows):
        row_y = -inner_d / 2 + (row + 0.5) * inner_d / n_rows
        for p in range(plants_per_row):
            px = -inner_w / 2 + (p + 0.5) * inner_w / plants_per_row
            px += np.random.uniform(-0.012, 0.012)
            py = row_y + np.random.uniform(-0.012, 0.012)

            if np.random.random() < 0.5:
                # --- Carrot top: 4-5 thin grass-like strands ---
                n_strands = np.random.randint(4, 6)
                for _ in range(n_strands):
                    strand_h = np.random.uniform(0.06, 0.12)
                    tilt_x = np.random.uniform(-0.025, 0.025)
                    tilt_y = np.random.uniform(-0.025, 0.025)
                    strand = trimesh.creation.cylinder(
                        radius=0.002, height=strand_h, sections=4)
                    # Tilt the strand outward
                    tilt_angle = np.sqrt(tilt_x**2 + tilt_y**2) * 8
                    if tilt_angle > 0.01:
                        tilt_axis = np.array([-tilt_y, tilt_x, 0.0])
                        tilt_axis /= np.linalg.norm(tilt_axis)
                        strand.apply_transform(
                            trimesh.transformations.rotation_matrix(
                                tilt_angle, tilt_axis))
                    strand.apply_translation([
                        px + tilt_x, py + tilt_y,
                        soil_h + strand_h / 2,
                    ])
                    strand_green = (40 + np.random.randint(-10, 10),
                                    130 + np.random.randint(-20, 20),
                                    25 + np.random.randint(-8, 8), 255)
                    apply_color_variation(strand, strand_green, 0.12)
                    plant_meshes.append(strand)
            else:
                # --- Cabbage: layered spheres ---
                cab_r = np.random.uniform(0.018, 0.028)
                # Core
                core = trimesh.creation.icosphere(subdivisions=1, radius=cab_r * 0.6)
                core.vertices[:, 2] *= 0.7
                core.apply_translation([px, py, soil_h + cab_r * 0.4])
                core_green = (50 + np.random.randint(-10, 10),
                              140 + np.random.randint(-15, 15),
                              40 + np.random.randint(-8, 8), 255)
                apply_color_variation(core, core_green, 0.1)
                plant_meshes.append(core)
                # Outer leaves (3-4 slightly offset spheres)
                for li in range(np.random.randint(3, 5)):
                    a = li * 2 * np.pi / 4 + np.random.uniform(-0.3, 0.3)
                    lr = cab_r * np.random.uniform(0.5, 0.7)
                    leaf = trimesh.creation.icosphere(
                        subdivisions=1, radius=lr)
                    leaf.vertices[:, 2] *= 0.5
                    off_r = cab_r * 0.4
                    leaf.apply_translation([
                        px + np.cos(a) * off_r,
                        py + np.sin(a) * off_r,
                        soil_h + cab_r * 0.25,
                    ])
                    leaf_green = (35 + np.random.randint(-12, 12),
                                  120 + np.random.randint(-25, 25),
                                  30 + np.random.randint(-10, 10), 255)
                    apply_color_variation(leaf, leaf_green, 0.15)
                    plant_meshes.append(leaf)

    # Build textured Scene — wood texture on frame, vertex colors on plants
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    from core.mesh import unweld_mesh, compute_triplanar_uvs
    from textures.generator import wood_texture

    scene = trimesh.Scene()

    # Plants + soil — keep vertex colors
    plants = trimesh.util.concatenate(plant_meshes)
    plants.fix_normals()
    scene.add_geometry(plants, node_name='plants')

    # Wood frame — apply wood texture
    wood = trimesh.util.concatenate(wood_meshes)
    unwelded = unweld_mesh(wood)
    uvs = compute_triplanar_uvs(unwelded, scale=12.0)

    tex = wood_texture(256, 256, seed)
    diffuse_img = Image.fromarray(tex.diffuse)
    normal_img = Image.fromarray(tex.normal)

    mat = PBRMaterial(
        baseColorTexture=diffuse_img,
        normalTexture=normal_img,
        roughnessFactor=0.7,
    )
    unwelded.visual = TextureVisuals(uv=uvs, material=mat)
    scene.add_geometry(unwelded, node_name='wood')

    return scene


def generate_garden_gate(
    width: float = 1.0,
    height: float = 1.2,
    color: Tuple[int, int, int] = (130, 85, 40),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a wooden garden gate with fence posts and handle."""
    set_seed(seed)
    meshes = []

    wood_rgba = (*color, 255)
    post_w = 0.08
    post_d = 0.08
    post_h = height + 0.15

    # Fence posts
    for side in [-1, 1]:
        post = _wobble_box([post_w, post_d, post_h], wobble=0.012)
        post.apply_translation([side * (width / 2 + post_w / 2), 0, post_h / 2])
        apply_color_variation(post, wood_rgba, 0.10)
        meshes.append(post)

        # Post cap
        cap = _wobble_box([post_w + 0.02, post_d + 0.02, 0.025], wobble=0.008)
        cap.apply_translation([side * (width / 2 + post_w / 2), 0, post_h + 0.0125])
        apply_color_variation(cap, wood_rgba, 0.08)
        meshes.append(cap)

    # Gate planks (vertical)
    plank_t = 0.025
    gate_w = width - 0.02
    n_planks = max(4, round(gate_w / 0.09))
    plank_w = gate_w / n_planks
    gate_bottom = 0.05

    for i in range(n_planks):
        px = -gate_w / 2 + plank_w / 2 + i * plank_w
        plank_h = height - gate_bottom + np.random.uniform(-0.015, 0.015)
        plank = _wobble_box([plank_w - 0.005, plank_t, plank_h], wobble=0.01)
        plank.apply_translation([px, 0, gate_bottom + plank_h / 2])
        plank_color = tuple(np.clip(np.array(color) + np.random.randint(-12, 12, 3), 0, 255))
        apply_color_variation(plank, (*plank_color, 255), 0.10)
        meshes.append(plank)

    # Horizontal cross braces on back
    brace_h = 0.05
    brace_z_lo = gate_bottom + (height - gate_bottom) * 0.25
    brace_z_hi = gate_bottom + (height - gate_bottom) * 0.75
    for bz in [brace_z_lo, brace_z_hi]:
        brace = _wobble_box([gate_w, plank_t + 0.008, brace_h], wobble=0.01)
        brace.apply_translation([0, -plank_t / 2 - 0.004, bz])
        brace_color = tuple(np.clip(np.array(color) + np.array([-8, -6, -4]), 0, 255))
        apply_color_variation(brace, (*brace_color, 255), 0.08)
        meshes.append(brace)

    # Diagonal Z-brace
    diag_dz = brace_z_hi - brace_z_lo
    diag_dx = gate_w
    diag_len = np.sqrt(diag_dx**2 + diag_dz**2)
    diag_angle = np.arctan2(diag_dz, diag_dx)
    diag = _wobble_box([diag_len, plank_t + 0.005, 0.035], wobble=0.008)
    diag.apply_transform(
        trimesh.transformations.rotation_matrix(diag_angle, [0, 1, 0]))
    diag.apply_translation([0, -plank_t / 2 - 0.004, (brace_z_lo + brace_z_hi) / 2])
    brace_color = tuple(np.clip(np.array(color) + np.array([-8, -6, -4]), 0, 255))
    apply_color_variation(diag, (*brace_color, 255), 0.08)
    meshes.append(diag)

    # Handle
    handle = _wobble_box([0.035, 0.025, 0.07], wobble=0.006)
    handle.apply_translation([gate_w / 2 - 0.07, -plank_t / 2 - 0.0125, height * 0.5])
    handle_color = (max(0, color[0] - 25), max(0, color[1] - 20),
                    max(0, color[2] - 12), 255)
    apply_color_variation(handle, handle_color, 0.06)
    meshes.append(handle)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


# --- Registry: Indoor props ---

register(
    name="kitchen_stove", label="Kitchen Stove", category="Props",
    params=[
        Param("width", "Width", "float", default=0.70, min=0.50, max=1.0),
        Param("height", "Height", "float", default=0.85, min=0.60, max=1.2),
        Param("depth", "Depth", "float", default=0.50, min=0.35, max=0.70),
        Param("color", "Iron Color", "color", default="#2D2A28"),
    ],
)(generate_kitchen_stove)

register(
    name="fireplace", label="Fireplace", category="Props",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.8, max=2.0),
        Param("height", "Height", "float", default=1.1, min=0.8, max=1.5),
        Param("depth", "Depth", "float", default=0.45, min=0.3, max=0.6),
        Param("color", "Brick Color", "color", default="#8C5F41"),
    ],
)(generate_fireplace)

register(
    name="chimney", label="Chimney", category="Props",
    params=[
        Param("width", "Width", "float", default=0.50, min=0.30, max=0.80),
        Param("height", "Height", "float", default=2.0, min=1.0, max=3.5),
        Param("color", "Brick Color", "color", default="#9B6441"),
    ],
)(generate_chimney)

register(
    name="washing_basin", label="Washing Basin", category="Props",
    params=[
        Param("height", "Height", "float", default=0.85, min=0.65, max=1.0),
        Param("color", "Wood Color", "color", default="#784E2A"),
    ],
)(generate_washing_basin)

# --- Registry: Outdoor props ---

register(
    name="pitchfork", label="Pitchfork", category="Props",
    params=[
        Param("height", "Height", "float", default=1.5, min=1.0, max=2.0),
        Param("color", "Handle Color", "color", default="#825F32"),
    ],
)(generate_pitchfork)

register(
    name="rake", label="Rake", category="Props",
    params=[
        Param("height", "Height", "float", default=1.5, min=1.0, max=2.0),
        Param("color", "Handle Color", "color", default="#7D5A30"),
    ],
)(generate_rake)

register(
    name="hay_pile", label="Hay Pile", category="Props",
    params=[
        Param("radius", "Radius", "float", default=1.0, min=0.5, max=2.5),
        Param("height", "Height", "float", default=0.8, min=0.3, max=1.5),
        Param("color", "Hay Color", "color", default="#C3AF5A"),
    ],
)(generate_hay_pile)

register(
    name="henhouse", label="Henhouse", category="Props",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.8, max=2.0),
        Param("height", "Height", "float", default=1.0, min=0.7, max=1.5),
        Param("depth", "Depth", "float", default=0.9, min=0.6, max=1.2),
        Param("color", "Wood Color", "color", default="#825A2D"),
    ],
)(generate_henhouse)

register(
    name="firewood_stack", label="Firewood Stack", category="Props",
    params=[
        Param("width", "Width", "float", default=1.0, min=0.5, max=2.0),
        Param("height", "Height", "float", default=0.8, min=0.3, max=1.5),
        Param("depth", "Depth", "float", default=0.40, min=0.25, max=0.60),
        Param("color", "Bark Color", "color", default="#6E4B28"),
    ],
)(generate_firewood_stack)

register(
    name="flower_bed", label="Flower Bed", category="Props",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.5, max=2.5),
        Param("depth", "Depth", "float", default=0.6, min=0.3, max=1.2),
        Param("height", "Height", "float", default=0.25, min=0.15, max=0.5),
        Param("color", "Wood Color", "color", default="#785028"),
    ],
)(generate_flower_bed)

register(
    name="garden_gate", label="Garden Gate", category="Props",
    params=[
        Param("width", "Width", "float", default=1.0, min=0.6, max=1.8),
        Param("height", "Height", "float", default=1.2, min=0.8, max=2.0),
        Param("color", "Wood Color", "color", default="#825528"),
    ],
)(generate_garden_gate)
