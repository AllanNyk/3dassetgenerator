"""
Antique furniture generators (1850-1900 style) with handcrafted irregularity.
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


# =============================================================================
# HELPERS — handcrafted antique feel
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


def _turned_leg(height, base_radius=0.02, sections=10, segments=16, style="bulge"):
    """Create a lathe-turned leg typical of antique furniture.

    Builds the leg from stacked cylinder rings with varying radii.
    style: "bulge" (Queen Anne), "taper" (Shaker), "ornate" (Victorian)
    """
    vertices = []
    faces = []

    for j in range(sections + 1):
        t = j / sections
        z = t * height

        # Compute radius based on style
        if style == "bulge":
            # Slight bulge in lower-middle, taper at top and bottom
            r = base_radius * (0.8 + 0.4 * np.sin(t * np.pi)
                               - 0.15 * np.sin(t * 2 * np.pi))
            # Foot at very bottom
            if t < 0.08:
                r = base_radius * (1.1 - 0.3 * t / 0.08)
            # Narrow at very top
            if t > 0.92:
                u = (t - 0.92) / 0.08
                r = r * (1.0 - 0.2 * u)
        elif style == "taper":
            # Simple taper from thicker bottom to thinner top
            r = base_radius * (1.1 - 0.3 * t)
            # Small foot
            if t < 0.05:
                r = base_radius * 1.15
        elif style == "ornate":
            # Victorian style: multiple bulges and rings
            r = base_radius * (0.7 + 0.3 * np.sin(t * np.pi)
                               + 0.12 * np.sin(t * 4 * np.pi)
                               + 0.06 * np.sin(t * 8 * np.pi))
            # Pronounced foot
            if t < 0.1:
                r = base_radius * (1.3 - 0.5 * t / 0.1)
            # Ring at top
            if t > 0.9:
                r = base_radius * 0.9
        else:
            r = base_radius

        # Add handcrafted irregularity
        r *= np.random.uniform(0.97, 1.03)

        for i in range(segments):
            angle = (i / segments) * 2 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for j in range(sections):
        for i in range(segments):
            i_next = (i + 1) % segments
            v0 = j * segments + i
            v1 = j * segments + i_next
            v2 = (j + 1) * segments + i_next
            v3 = (j + 1) * segments + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Bottom cap
    bc = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([bc, i_next, i])

    # Top cap
    tc = len(vertices)
    vertices = np.vstack([vertices, [0, 0, height]])
    top_ring = sections * segments
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([tc, top_ring + i, top_ring + i_next])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def _decorative_arch(width, height, thickness, segments=20):
    """Create a decorative arch shape (for headboards, wardrobe tops)."""
    meshes = []

    # Straight sides
    side_h = height * 0.6
    for side in [-1, 1]:
        post = _wobble_box([thickness, thickness, side_h], wobble=0.008)
        post.apply_translation([side * width / 2, 0, side_h / 2])
        meshes.append(post)

    # Arch across top
    arch_h = height - side_h
    for i in range(segments):
        t = i / segments
        t_next = (i + 1) / segments
        angle = t * np.pi
        angle_next = t_next * np.pi

        x0 = width / 2 * np.cos(angle)
        z0 = side_h + arch_h * np.sin(angle)
        x1 = width / 2 * np.cos(angle_next)
        z1 = side_h + arch_h * np.sin(angle_next)

        seg_len = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)
        seg = _wobble_box([seg_len, thickness, thickness], wobble=0.005)

        # Position and rotate segment
        mid_x = (x0 + x1) / 2
        mid_z = (z0 + z1) / 2
        seg_angle = np.arctan2(z1 - z0, x1 - x0)
        seg.apply_transform(trimesh.transformations.rotation_matrix(seg_angle, [0, 1, 0]))
        seg.apply_translation([mid_x, 0, mid_z])
        meshes.append(seg)

    return meshes


# =============================================================================
# GENERATORS
# =============================================================================

def generate_table(
    width: float = 1.2,
    depth: float = 0.7,
    height: float = 0.78,
    color: Tuple[int, int, int] = (139, 105, 20),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique dining table with turned legs and apron."""
    set_seed(seed)
    meshes = []

    top_thickness = 0.04
    leg_radius = 0.025
    apron_height = 0.08
    apron_thickness = 0.02
    leg_inset = 0.06  # how far legs are inset from edge

    # --- Tabletop ---
    top = _wobble_box([width, depth, top_thickness], wobble=0.01)
    top.apply_translation([0, 0, height])
    apply_color_variation(top, (*color, 255), 0.1)
    meshes.append(top)

    # --- Legs (4 turned legs) ---
    leg_height = height - top_thickness / 2
    leg_positions = [
        (-width / 2 + leg_inset, -depth / 2 + leg_inset),
        (width / 2 - leg_inset, -depth / 2 + leg_inset),
        (-width / 2 + leg_inset, depth / 2 - leg_inset),
        (width / 2 - leg_inset, depth / 2 - leg_inset),
    ]

    for lx, ly in leg_positions:
        leg = _turned_leg(leg_height, base_radius=leg_radius, style="bulge")
        leg.apply_translation([lx, ly, 0])
        apply_color_variation(leg, (*color, 255), 0.08)
        meshes.append(leg)

    # --- Apron (frame under tabletop) ---
    apron_z = height - top_thickness / 2 - apron_height / 2
    # Front and back
    for y_pos in [-depth / 2 + leg_inset, depth / 2 - leg_inset]:
        apron = _wobble_box([width - 2 * leg_inset, apron_thickness, apron_height])
        apron.apply_translation([0, y_pos, apron_z])
        apply_color_variation(apron, (*color, 255), 0.08)
        meshes.append(apron)

    # Left and right
    for x_pos in [-width / 2 + leg_inset, width / 2 - leg_inset]:
        apron = _wobble_box([apron_thickness, depth - 2 * leg_inset, apron_height])
        apron.apply_translation([x_pos, 0, apron_z])
        apply_color_variation(apron, (*color, 255), 0.08)
        meshes.append(apron)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_chair(
    seat_height: float = 0.45,
    backrest_height: float = 0.45,
    color: Tuple[int, int, int] = (92, 58, 30),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique wooden chair with turned legs and slatted back."""
    set_seed(seed)
    meshes = []

    seat_w = 0.42
    seat_d = 0.40
    seat_thickness = 0.03
    leg_radius = 0.018
    leg_inset = 0.04

    # --- Seat (rounded edges using convex hull of offset spheres) ---
    seat_pts = []
    r = 0.025  # corner rounding radius
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                cx = sx * (seat_w / 2 - r)
                cy = sy * (seat_d / 2 - r)
                cz = sz * (seat_thickness / 2 - r * 0.5)
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
                seat_pts.append(sphere.vertices + [cx, cy, cz])
    seat = trimesh.convex.convex_hull(np.vstack(seat_pts))
    # Wobble for handcrafted feel
    verts = seat.vertices.copy()
    for i in range(3):
        verts[:, i] += np.random.uniform(-0.003, 0.003, len(verts))
    seat.vertices = verts
    seat.apply_translation([0, 0, seat_height])
    apply_color_variation(seat, (*color, 255), 0.1)
    meshes.append(seat)

    # --- Front legs (straight, slightly splayed) ---
    front_leg_h = seat_height
    for side in [-1, 1]:
        leg = _turned_leg(front_leg_h, base_radius=leg_radius, style="taper")
        lx = side * (seat_w / 2 - leg_inset)
        ly = seat_d / 2 - leg_inset
        leg.apply_translation([lx, ly, 0])
        apply_color_variation(leg, (*color, 255), 0.08)
        meshes.append(leg)

    # --- Back legs (extend up to support backrest) ---
    back_leg_h = seat_height + backrest_height
    for side in [-1, 1]:
        leg = _turned_leg(back_leg_h, base_radius=leg_radius, style="taper")
        lx = side * (seat_w / 2 - leg_inset)
        ly = -seat_d / 2 + leg_inset
        leg.apply_translation([lx, ly, 0])
        apply_color_variation(leg, (*color, 255), 0.08)
        meshes.append(leg)

    # --- Backrest slats (3 horizontal bars) ---
    slat_w = seat_w - 2 * leg_inset - 0.02
    slat_thickness = 0.015
    slat_height = 0.025
    num_slats = 3
    back_y = -seat_d / 2 + leg_inset

    for si in range(num_slats):
        t = (si + 1) / (num_slats + 1)
        slat_z = seat_height + seat_thickness + t * (backrest_height - seat_thickness)
        slat = _wobble_box([slat_w, slat_thickness, slat_height], wobble=0.006)
        slat.apply_translation([0, back_y, slat_z])
        apply_color_variation(slat, (*color, 255), 0.1)
        meshes.append(slat)

    # --- Top rail (curved top bar of backrest) ---
    top_rail = _wobble_box([slat_w + 0.02, slat_thickness * 1.5, 0.03], wobble=0.008)
    top_rail.apply_translation([0, back_y, seat_height + backrest_height - 0.02])
    apply_color_variation(top_rail, (*color, 255), 0.08)
    meshes.append(top_rail)

    # --- Stretchers (cross bars between legs for stability) ---
    stretcher_z = seat_height * 0.3
    stretcher_r = 0.008
    # Front stretcher
    front_str = trimesh.creation.cylinder(
        radius=stretcher_r, height=seat_w - 2 * leg_inset, sections=8)
    front_str.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    front_str.apply_translation([0, seat_d / 2 - leg_inset, stretcher_z])
    apply_color_variation(front_str, (*color, 255), 0.08)
    meshes.append(front_str)

    # Back stretcher
    back_str = trimesh.creation.cylinder(
        radius=stretcher_r, height=seat_w - 2 * leg_inset, sections=8)
    back_str.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    back_str.apply_translation([0, back_y, stretcher_z])
    apply_color_variation(back_str, (*color, 255), 0.08)
    meshes.append(back_str)

    # Side stretchers
    for side in [-1, 1]:
        side_str = trimesh.creation.cylinder(
            radius=stretcher_r, height=seat_d - 2 * leg_inset, sections=8)
        side_str.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        side_str.apply_translation([side * (seat_w / 2 - leg_inset), 0, stretcher_z])
        apply_color_variation(side_str, (*color, 255), 0.08)
        meshes.append(side_str)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_wardrobe(
    width: float = 1.2,
    height: float = 2.0,
    depth: float = 0.55,
    color: Tuple[int, int, int] = (107, 58, 42),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique wardrobe with paneled doors and crown molding."""
    set_seed(seed)
    meshes = []

    wall_thickness = 0.025
    base_h = 0.08
    crown_h = 0.05
    body_h = height - base_h - crown_h

    # --- Main body (back, sides, top, bottom) ---
    # Back panel
    back = _wobble_box([width, wall_thickness, body_h], wobble=0.01)
    back.apply_translation([0, -depth / 2 + wall_thickness / 2, base_h + body_h / 2])
    apply_color_variation(back, (*color, 255), 0.08)
    meshes.append(back)

    # Side panels
    for side in [-1, 1]:
        panel = _wobble_box([wall_thickness, depth, body_h], wobble=0.01)
        panel.apply_translation([side * (width / 2 - wall_thickness / 2), 0,
                                 base_h + body_h / 2])
        apply_color_variation(panel, (*color, 255), 0.08)
        meshes.append(panel)

    # Top panel
    top = _wobble_box([width, depth, wall_thickness], wobble=0.01)
    top.apply_translation([0, 0, base_h + body_h])
    apply_color_variation(top, (*color, 255), 0.08)
    meshes.append(top)

    # Bottom panel
    bottom = _wobble_box([width, depth, wall_thickness], wobble=0.01)
    bottom.apply_translation([0, 0, base_h])
    apply_color_variation(bottom, (*color, 255), 0.08)
    meshes.append(bottom)

    # --- Front door panels (2 doors) ---
    door_gap = 0.01
    door_w = (width - 3 * door_gap - 2 * wall_thickness) / 2
    door_h = body_h - 0.04
    door_y = depth / 2 - wall_thickness / 2

    for side in [-1, 1]:
        dx = side * (door_w / 2 + door_gap / 2)

        # Door panel
        door = _wobble_box([door_w, wall_thickness * 0.8, door_h], wobble=0.008)
        door.apply_translation([dx, door_y, base_h + body_h / 2])
        apply_color_variation(door, (*color, 255), 0.1)
        meshes.append(door)

        # Raised inner panel (decorative inset)
        panel_w = door_w * 0.7
        panel_h = door_h * 0.75
        panel = _wobble_box([panel_w, 0.008, panel_h], wobble=0.005)
        panel.apply_translation([dx, door_y + wall_thickness * 0.5, base_h + body_h / 2])
        # Slightly lighter color for raised panel
        lighter = tuple(min(255, c + 15) for c in color)
        apply_color_variation(panel, (*lighter, 255), 0.08)
        meshes.append(panel)

        # Door knob
        knob = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
        knob.apply_translation([dx - side * door_w * 0.35, door_y + 0.02,
                                base_h + body_h / 2])
        apply_color_variation(knob, (180, 160, 80, 255), 0.05)
        meshes.append(knob)

    # --- Base molding (wider, with feet) ---
    base_mold = _wobble_box([width + 0.03, depth + 0.03, base_h * 0.6], wobble=0.01)
    base_mold.apply_translation([0, 0, base_h * 0.3])
    apply_color_variation(base_mold, (*color, 255), 0.08)
    meshes.append(base_mold)

    # Feet (4 small blocks)
    foot_size = 0.04
    for fx in [-1, 1]:
        for fy in [-1, 1]:
            foot = _wobble_box([foot_size, foot_size, 0.02])
            foot.apply_translation([fx * (width / 2 - 0.02),
                                    fy * (depth / 2 - 0.02), 0.01])
            apply_color_variation(foot, (*color, 255), 0.06)
            meshes.append(foot)

    # --- Crown molding (wider top, decorative) ---
    crown = _wobble_box([width + 0.04, depth + 0.02, crown_h], wobble=0.01)
    crown.apply_translation([0, 0, height - crown_h / 2])
    apply_color_variation(crown, (*color, 255), 0.08)
    meshes.append(crown)

    # Crown lip (extra overhang)
    lip = _wobble_box([width + 0.06, depth + 0.04, crown_h * 0.3], wobble=0.008)
    lip.apply_translation([0, 0, height - crown_h * 0.15])
    apply_color_variation(lip, (*color, 255), 0.06)
    meshes.append(lip)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_bed(
    width: float = 1.4,
    length: float = 2.0,
    color: Tuple[int, int, int] = (122, 64, 40),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique bed with turned posts, headboard, and mattress."""
    set_seed(seed)
    meshes = []

    frame_h = 0.35       # height of side rails
    rail_thickness = 0.04
    post_radius = 0.03
    headboard_h = 0.7
    footboard_h = 0.45
    mattress_h = 0.15

    # --- Corner posts (4 turned posts) ---
    post_positions = [
        (-width / 2, -length / 2, headboard_h + frame_h),    # back-left
        (width / 2, -length / 2, headboard_h + frame_h),     # back-right
        (-width / 2, length / 2, footboard_h + frame_h),     # front-left
        (width / 2, length / 2, footboard_h + frame_h),      # front-right
    ]
    for px, py, post_h in post_positions:
        post = _turned_leg(post_h, base_radius=post_radius, style="ornate")
        post.apply_translation([px, py, 0])
        apply_color_variation(post, (*color, 255), 0.08)
        meshes.append(post)

        # Finial (decorative ball on top of post)
        finial = trimesh.creation.icosphere(subdivisions=2, radius=post_radius * 1.5)
        finial.vertices[:, 2] *= 1.3  # slightly elongate
        finial.apply_translation([px, py, post_h + post_radius])
        apply_color_variation(finial, (*color, 255), 0.06)
        meshes.append(finial)

    # --- Side rails ---
    rail_z = frame_h + rail_thickness / 2
    # Left and right rails
    for side in [-1, 1]:
        rail = _wobble_box([rail_thickness, length, rail_thickness * 2], wobble=0.008)
        rail.apply_translation([side * width / 2, 0, rail_z])
        apply_color_variation(rail, (*color, 255), 0.08)
        meshes.append(rail)

    # --- Headboard (back, taller) ---
    hb_panel_w = width - 0.02
    hb_panel_h = headboard_h - 0.05
    hb_thickness = 0.025
    headboard = _wobble_box([hb_panel_w, hb_thickness, hb_panel_h], wobble=0.012)
    headboard.apply_translation([0, -length / 2, frame_h + hb_panel_h / 2])
    apply_color_variation(headboard, (*color, 255), 0.1)
    meshes.append(headboard)

    # Headboard decorative top arch
    arch_parts = _decorative_arch(hb_panel_w * 0.8, headboard_h * 0.35,
                                   hb_thickness * 0.8, segments=16)
    for part in arch_parts:
        part.apply_translation([0, -length / 2, frame_h + hb_panel_h * 0.7])
        apply_color_variation(part, (*color, 255), 0.08)
        meshes.append(part)

    # --- Footboard (front, shorter) ---
    fb_panel_h = footboard_h - 0.05
    footboard = _wobble_box([hb_panel_w, hb_thickness, fb_panel_h], wobble=0.012)
    footboard.apply_translation([0, length / 2, frame_h + fb_panel_h / 2])
    apply_color_variation(footboard, (*color, 255), 0.1)
    meshes.append(footboard)

    # --- Slat support (cross bars under mattress) ---
    num_slats = 5
    for i in range(num_slats):
        t = (i + 0.5) / num_slats
        sy = -length / 2 + t * length
        slat = _wobble_box([width - 0.04, 0.06, 0.02], wobble=0.005)
        slat.apply_translation([0, sy, frame_h])
        apply_color_variation(slat, (*color, 255), 0.06)
        meshes.append(slat)

    # --- Mattress ---
    matt_w = width - 0.06
    matt_l = length - 0.06
    mattress = _wobble_box([matt_w, matt_l, mattress_h], wobble=0.02)
    mattress.apply_translation([0, 0, frame_h + rail_thickness + mattress_h / 2])
    # Mattress color: off-white/cream
    apply_color_variation(mattress, (235, 225, 200, 255), 0.05)
    meshes.append(mattress)

    # --- Pillow (at headboard end) ---
    pillow = _wobble_box([matt_w * 0.4, 0.25, 0.08], wobble=0.02)
    pillow.apply_translation([0, -length / 2 + 0.2,
                              frame_h + rail_thickness + mattress_h + 0.04])
    apply_color_variation(pillow, (240, 235, 220, 255), 0.04)
    meshes.append(pillow)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_clothes_rack(
    height: float = 1.7,
    color: Tuple[int, int, int] = (90, 56, 32),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique standing clothes rack / coat stand."""
    set_seed(seed)
    meshes = []

    pole_radius = 0.025
    num_feet = 4
    num_hooks = 6
    foot_length = 0.35
    hook_length = 0.12

    # --- Central pole (turned) ---
    pole = _turned_leg(height, base_radius=pole_radius, style="ornate", sections=14)
    apply_color_variation(pole, (*color, 255), 0.08)
    meshes.append(pole)

    # --- Decorative center piece (where hooks attach) ---
    hub = trimesh.creation.icosphere(subdivisions=2, radius=pole_radius * 2.5)
    hub.vertices[:, 2] *= 0.5  # flatten
    hub.apply_translation([0, 0, height - 0.02])
    apply_color_variation(hub, (*color, 255), 0.06)
    meshes.append(hub)

    # --- Top finial ---
    finial = trimesh.creation.icosphere(subdivisions=2, radius=pole_radius * 1.8)
    finial.vertices[:, 2] *= 1.5  # elongate
    finial.apply_translation([0, 0, height + pole_radius * 2])
    apply_color_variation(finial, (*color, 255), 0.06)
    meshes.append(finial)

    # --- Hooks (radiating from top) ---
    for i in range(num_hooks):
        angle = (i / num_hooks) * 2 * np.pi
        # Main arm (slightly angled upward)
        arm_r = 0.008
        arm = trimesh.creation.cylinder(radius=arm_r, height=hook_length, sections=8)
        # Tilt outward and slightly up
        arm.apply_transform(trimesh.transformations.rotation_matrix(0.4, [0, 1, 0]))
        arm.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
        arm.apply_translation([
            np.cos(angle) * hook_length * 0.4,
            np.sin(angle) * hook_length * 0.4,
            height - 0.02
        ])
        apply_color_variation(arm, (*color, 255), 0.06)
        meshes.append(arm)

        # Hook tip (small upward curl)
        tip = trimesh.creation.icosphere(subdivisions=1, radius=arm_r * 1.8)
        tip.apply_translation([
            np.cos(angle) * hook_length * 0.75,
            np.sin(angle) * hook_length * 0.75,
            height + 0.01
        ])
        apply_color_variation(tip, (*color, 255), 0.06)
        meshes.append(tip)

    # --- Base feet (radiating outward from bottom) ---
    for i in range(num_feet):
        angle = (i / num_feet) * 2 * np.pi + np.pi / num_feet  # offset from hooks

        # Foot: curved piece going outward and down
        foot_segments = 6
        foot_r = 0.015
        prev_pos = [0, 0, 0.03]

        for fi in range(foot_segments):
            ft = (fi + 1) / foot_segments
            fx = np.cos(angle) * foot_length * ft
            fy = np.sin(angle) * foot_length * ft
            # Curve down then flatten at the end
            fz = 0.03 * (1 - ft) + 0.005

            seg = trimesh.creation.cylinder(
                radius=foot_r * (1.2 - 0.3 * ft), height=foot_length / foot_segments * 1.1,
                sections=8)

            # Orient segment
            curr_pos = [fx, fy, fz]
            diff = np.array(curr_pos) - np.array(prev_pos)
            seg_len = np.linalg.norm(diff)
            if seg_len > 1e-6:
                direction = diff / seg_len
                z_axis = np.array([0, 0, 1])
                if abs(np.dot(direction, z_axis)) < 0.999:
                    axis = np.cross(z_axis, direction)
                    axis = axis / np.linalg.norm(axis)
                    rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                    seg.apply_transform(
                        trimesh.transformations.rotation_matrix(rot_angle, axis))

            mid = [(prev_pos[j] + curr_pos[j]) / 2 for j in range(3)]
            seg.apply_translation(mid)
            apply_color_variation(seg, (*color, 255), 0.06)
            meshes.append(seg)

            prev_pos = curr_pos

        # Foot pad (small flat disc at the end)
        pad = trimesh.creation.cylinder(radius=foot_r * 1.5, height=0.005, sections=8)
        pad.apply_translation([
            np.cos(angle) * foot_length,
            np.sin(angle) * foot_length,
            0.003
        ])
        apply_color_variation(pad, (*color, 255), 0.05)
        meshes.append(pad)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


# =============================================================================
# REGISTRY
# =============================================================================

register(
    name="table", label="Table", category="Furniture",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.6, max=2.5),
        Param("depth", "Depth", "float", default=0.7, min=0.4, max=1.5),
        Param("height", "Height", "float", default=0.78, min=0.5, max=1.0),
        Param("color", "Wood Color", "color", default="#8B6914"),
    ],
)(generate_table)

register(
    name="chair", label="Chair", category="Furniture",
    params=[
        Param("seat_height", "Seat Height", "float", default=0.45, min=0.3, max=0.6),
        Param("backrest_height", "Backrest Height", "float", default=0.45, min=0.25, max=0.7),
        Param("color", "Wood Color", "color", default="#5C3A1E"),
    ],
)(generate_chair)

register(
    name="wardrobe", label="Wardrobe", category="Furniture",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.8, max=2.0),
        Param("height", "Height", "float", default=2.0, min=1.5, max=2.5),
        Param("depth", "Depth", "float", default=0.55, min=0.4, max=0.8),
        Param("color", "Wood Color", "color", default="#6B3A2A"),
    ],
)(generate_wardrobe)

register(
    name="double_bed", label="Double Bed", category="Furniture", export_name="double_bed",
    params=[
        Param("width", "Width", "float", default=1.4, min=0.9, max=2.0),
        Param("length", "Length", "float", default=2.0, min=1.5, max=2.5),
        Param("color", "Wood Color", "color", default="#7A4028"),
    ],
)(generate_bed)

register(
    name="clothes_rack", label="Clothes Rack", category="Furniture",
    params=[
        Param("height", "Height", "float", default=1.7, min=1.2, max=2.2),
        Param("color", "Wood Color", "color", default="#5A3820"),
    ],
)(generate_clothes_rack)


# =============================================================================
# ADDITIONAL GENERATORS
# =============================================================================

def generate_single_bed(
    width: float = 0.9,
    length: float = 2.0,
    color: Tuple[int, int, int] = (122, 64, 40),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique single bed — narrower, simpler headboard."""
    set_seed(seed)
    meshes = []

    frame_h = 0.30
    rail_thickness = 0.035
    post_radius = 0.025
    headboard_h = 0.55
    footboard_h = 0.35
    mattress_h = 0.12

    # Corner posts
    post_positions = [
        (-width / 2, -length / 2, headboard_h + frame_h),
        (width / 2, -length / 2, headboard_h + frame_h),
        (-width / 2, length / 2, footboard_h + frame_h),
        (width / 2, length / 2, footboard_h + frame_h),
    ]
    for px, py, post_h in post_positions:
        post = _turned_leg(post_h, base_radius=post_radius, style="bulge")
        post.apply_translation([px, py, 0])
        apply_color_variation(post, (*color, 255), 0.08)
        meshes.append(post)
        finial = trimesh.creation.icosphere(subdivisions=2, radius=post_radius * 1.3)
        finial.vertices[:, 2] *= 1.2
        finial.apply_translation([px, py, post_h + post_radius])
        apply_color_variation(finial, (*color, 255), 0.06)
        meshes.append(finial)

    # Side rails
    rail_z = frame_h + rail_thickness / 2
    for side in [-1, 1]:
        rail = _wobble_box([rail_thickness, length, rail_thickness * 2], wobble=0.008)
        rail.apply_translation([side * width / 2, 0, rail_z])
        apply_color_variation(rail, (*color, 255), 0.08)
        meshes.append(rail)

    # Headboard
    hb_w = width - 0.02
    hb_h = headboard_h - 0.04
    headboard = _wobble_box([hb_w, 0.022, hb_h], wobble=0.01)
    headboard.apply_translation([0, -length / 2, frame_h + hb_h / 2])
    apply_color_variation(headboard, (*color, 255), 0.1)
    meshes.append(headboard)

    # Footboard
    fb_h = footboard_h - 0.04
    footboard = _wobble_box([hb_w, 0.022, fb_h], wobble=0.01)
    footboard.apply_translation([0, length / 2, frame_h + fb_h / 2])
    apply_color_variation(footboard, (*color, 255), 0.1)
    meshes.append(footboard)

    # Slats
    for i in range(4):
        t = (i + 0.5) / 4
        sy = -length / 2 + t * length
        slat = _wobble_box([width - 0.04, 0.06, 0.018], wobble=0.005)
        slat.apply_translation([0, sy, frame_h])
        apply_color_variation(slat, (*color, 255), 0.06)
        meshes.append(slat)

    # Mattress
    mattress = _wobble_box([width - 0.05, length - 0.05, mattress_h], wobble=0.015)
    mattress.apply_translation([0, 0, frame_h + rail_thickness + mattress_h / 2])
    apply_color_variation(mattress, (235, 225, 200, 255), 0.05)
    meshes.append(mattress)

    # Pillow
    pillow = _wobble_box([width * 0.6, 0.22, 0.07], wobble=0.015)
    pillow.apply_translation([0, -length / 2 + 0.18,
                              frame_h + rail_thickness + mattress_h + 0.035])
    apply_color_variation(pillow, (240, 235, 220, 255), 0.04)
    meshes.append(pillow)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_rocking_chair(
    seat_height: float = 0.40,
    backrest_height: float = 0.50,
    color: Tuple[int, int, int] = (85, 50, 25),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique rocking chair with curved rockers and armrests."""
    set_seed(seed)
    meshes = []

    seat_w = 0.44
    seat_d = 0.42
    seat_thickness = 0.03
    leg_radius = 0.018
    leg_inset = 0.04
    rocker_radius = 0.012

    # Rounded seat (same as chair)
    r = 0.025
    seat_pts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                cx = sx * (seat_w / 2 - r)
                cy = sy * (seat_d / 2 - r)
                cz = sz * (seat_thickness / 2 - r * 0.5)
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
                seat_pts.append(sphere.vertices + [cx, cy, cz])
    seat = trimesh.convex.convex_hull(np.vstack(seat_pts))
    verts = seat.vertices.copy()
    for i in range(3):
        verts[:, i] += np.random.uniform(-0.003, 0.003, len(verts))
    seat.vertices = verts
    seat.apply_translation([0, 0, seat_height])
    apply_color_variation(seat, (*color, 255), 0.1)
    meshes.append(seat)

    # Front legs
    front_leg_h = seat_height
    for side in [-1, 1]:
        leg = _turned_leg(front_leg_h, base_radius=leg_radius, style="taper")
        lx = side * (seat_w / 2 - leg_inset)
        ly = seat_d / 2 - leg_inset
        leg.apply_translation([lx, ly, 0])
        apply_color_variation(leg, (*color, 255), 0.08)
        meshes.append(leg)

    # Back legs (extend up for backrest)
    back_leg_h = seat_height + backrest_height
    for side in [-1, 1]:
        leg = _turned_leg(back_leg_h, base_radius=leg_radius, style="taper")
        lx = side * (seat_w / 2 - leg_inset)
        ly = -seat_d / 2 + leg_inset
        leg.apply_translation([lx, ly, 0])
        apply_color_variation(leg, (*color, 255), 0.08)
        meshes.append(leg)

    # Backrest slats
    slat_w = seat_w - 2 * leg_inset - 0.02
    back_y = -seat_d / 2 + leg_inset
    for si in range(4):
        t = (si + 1) / 5
        slat_z = seat_height + seat_thickness + t * (backrest_height - seat_thickness)
        slat = _wobble_box([slat_w, 0.015, 0.022], wobble=0.006)
        slat.apply_translation([0, back_y, slat_z])
        apply_color_variation(slat, (*color, 255), 0.1)
        meshes.append(slat)

    # Top rail
    top_rail = _wobble_box([slat_w + 0.02, 0.02, 0.03], wobble=0.008)
    top_rail.apply_translation([0, back_y, seat_height + backrest_height - 0.02])
    apply_color_variation(top_rail, (*color, 255), 0.08)
    meshes.append(top_rail)

    # Armrests
    arm_h = seat_height + 0.22
    arm_length = seat_d - 2 * leg_inset
    for side in [-1, 1]:
        # Arm support post
        arm_post = _turned_leg(arm_h - seat_height, base_radius=0.012, style="taper",
                               sections=6)
        arm_post.apply_translation([side * (seat_w / 2 - leg_inset),
                                     seat_d / 2 - leg_inset, seat_height])
        apply_color_variation(arm_post, (*color, 255), 0.08)
        meshes.append(arm_post)

        # Arm rest (horizontal bar)
        arm = _wobble_box([0.04, arm_length, 0.025], wobble=0.006)
        arm.apply_translation([side * (seat_w / 2 - leg_inset), 0, arm_h])
        apply_color_variation(arm, (*color, 255), 0.08)
        meshes.append(arm)

    # Rockers (curved pieces under legs)
    rocker_len = seat_d + 0.25
    rocker_curve_r = 1.2  # large radius for gentle curve
    n_seg = 20
    for side in [-1, 1]:
        rx = side * (seat_w / 2 - leg_inset)
        for i in range(n_seg):
            t0 = i / n_seg
            t1 = (i + 1) / n_seg
            a0 = (t0 - 0.5) * (rocker_len / rocker_curve_r)
            a1 = (t1 - 0.5) * (rocker_len / rocker_curve_r)

            y0 = rocker_curve_r * np.sin(a0)
            z0 = -rocker_curve_r * (1 - np.cos(a0))
            y1 = rocker_curve_r * np.sin(a1)
            z1 = -rocker_curve_r * (1 - np.cos(a1))

            seg_len = np.sqrt((y1 - y0)**2 + (z1 - z0)**2)
            seg = trimesh.creation.cylinder(radius=rocker_radius, height=seg_len, sections=8)

            mid_y = (y0 + y1) / 2
            mid_z = (z0 + z1) / 2
            seg_angle = np.arctan2(z1 - z0, y1 - y0)
            seg.apply_transform(trimesh.transformations.rotation_matrix(
                -seg_angle, [1, 0, 0]))
            seg.apply_translation([rx, mid_y, mid_z - 0.01])
            apply_color_variation(seg, (*color, 255), 0.06)
            meshes.append(seg)

    # Stretchers
    stretcher_z = seat_height * 0.3
    for y_pos in [seat_d / 2 - leg_inset, -seat_d / 2 + leg_inset]:
        s = trimesh.creation.cylinder(radius=0.007, height=seat_w - 2 * leg_inset, sections=8)
        s.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        s.apply_translation([0, y_pos, stretcher_z])
        apply_color_variation(s, (*color, 255), 0.08)
        meshes.append(s)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_cupboard(
    width: float = 0.9,
    height: float = 1.4,
    depth: float = 0.45,
    color: Tuple[int, int, int] = (100, 60, 35),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique cupboard with upper glass section and lower cabinet."""
    set_seed(seed)
    meshes = []

    wall = 0.022
    base_h = 0.06
    crown_h = 0.04
    body_h = height - base_h - crown_h
    mid_shelf_z = base_h + body_h * 0.55  # divider between upper and lower

    # Back panel
    back = _wobble_box([width, wall, body_h], wobble=0.01)
    back.apply_translation([0, -depth / 2 + wall / 2, base_h + body_h / 2])
    apply_color_variation(back, (*color, 255), 0.08)
    meshes.append(back)

    # Side panels
    for side in [-1, 1]:
        panel = _wobble_box([wall, depth, body_h], wobble=0.01)
        panel.apply_translation([side * (width / 2 - wall / 2), 0, base_h + body_h / 2])
        apply_color_variation(panel, (*color, 255), 0.08)
        meshes.append(panel)

    # Top, bottom, and middle shelf
    for sz in [base_h, mid_shelf_z, base_h + body_h]:
        shelf = _wobble_box([width, depth, wall], wobble=0.008)
        shelf.apply_translation([0, 0, sz])
        apply_color_variation(shelf, (*color, 255), 0.08)
        meshes.append(shelf)

    # Internal shelves in upper section (2 shelves)
    upper_h = base_h + body_h - mid_shelf_z
    for i in range(2):
        t = (i + 1) / 3
        sz = mid_shelf_z + t * upper_h
        shelf = _wobble_box([width - 2 * wall, depth - wall, wall * 0.7], wobble=0.005)
        shelf.apply_translation([0, wall / 2, sz])
        apply_color_variation(shelf, (*color, 255), 0.06)
        meshes.append(shelf)

    # Lower door panels (2 doors)
    door_gap = 0.008
    lower_h = mid_shelf_z - base_h - wall
    door_w = (width - 3 * door_gap - 2 * wall) / 2
    door_y = depth / 2 - wall / 2

    for side in [-1, 1]:
        dx = side * (door_w / 2 + door_gap / 2)
        door = _wobble_box([door_w, wall * 0.8, lower_h - 0.02], wobble=0.008)
        door.apply_translation([dx, door_y, base_h + lower_h / 2])
        apply_color_variation(door, (*color, 255), 0.1)
        meshes.append(door)

        # Raised panel
        rp = _wobble_box([door_w * 0.7, 0.006, lower_h * 0.7], wobble=0.004)
        rp.apply_translation([dx, door_y + wall * 0.5, base_h + lower_h / 2])
        lighter = tuple(min(255, c + 12) for c in color)
        apply_color_variation(rp, (*lighter, 255), 0.08)
        meshes.append(rp)

        # Knob
        knob = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        knob.apply_translation([dx - side * door_w * 0.35, door_y + 0.018,
                                base_h + lower_h / 2])
        apply_color_variation(knob, (180, 160, 80, 255), 0.05)
        meshes.append(knob)

    # Upper section - open frame (no glass, just frame suggesting glass doors)
    upper_door_h = upper_h - wall - 0.02
    for side in [-1, 1]:
        dx = side * (door_w / 2 + door_gap / 2)
        # Outer frame (4 bars)
        frame_t = 0.02
        # Top bar
        fb = _wobble_box([door_w, frame_t, frame_t], wobble=0.004)
        fb.apply_translation([dx, door_y, mid_shelf_z + upper_door_h])
        apply_color_variation(fb, (*color, 255), 0.08)
        meshes.append(fb)
        # Bottom bar
        fb2 = _wobble_box([door_w, frame_t, frame_t], wobble=0.004)
        fb2.apply_translation([dx, door_y, mid_shelf_z + 0.01])
        apply_color_variation(fb2, (*color, 255), 0.08)
        meshes.append(fb2)
        # Side bars
        for s2 in [-1, 1]:
            sb = _wobble_box([frame_t, frame_t, upper_door_h], wobble=0.004)
            sb.apply_translation([dx + s2 * door_w / 2, door_y,
                                  mid_shelf_z + upper_door_h / 2])
            apply_color_variation(sb, (*color, 255), 0.08)
            meshes.append(sb)
        # Cross divider (vertical)
        cross = _wobble_box([frame_t * 0.7, frame_t * 0.7, upper_door_h], wobble=0.004)
        cross.apply_translation([dx, door_y, mid_shelf_z + upper_door_h / 2])
        apply_color_variation(cross, (*color, 255), 0.08)
        meshes.append(cross)

        # Upper knob
        knob = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
        knob.apply_translation([dx - side * door_w * 0.35, door_y + 0.015,
                                mid_shelf_z + upper_door_h / 2])
        apply_color_variation(knob, (180, 160, 80, 255), 0.05)
        meshes.append(knob)

    # Base molding
    base = _wobble_box([width + 0.02, depth + 0.02, base_h * 0.6], wobble=0.008)
    base.apply_translation([0, 0, base_h * 0.3])
    apply_color_variation(base, (*color, 255), 0.08)
    meshes.append(base)

    # Crown molding
    crown = _wobble_box([width + 0.03, depth + 0.015, crown_h], wobble=0.008)
    crown.apply_translation([0, 0, height - crown_h / 2])
    apply_color_variation(crown, (*color, 255), 0.08)
    meshes.append(crown)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_settle_bed(
    width: float = 1.2,
    height: float = 0.85,
    depth: float = 0.45,
    color: Tuple[int, int, int] = (95, 55, 30),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a settle bed — an antique indoor bench with storage underneath."""
    set_seed(seed)
    meshes = []

    wall = 0.025
    seat_h = height * 0.47
    seat_thickness = 0.03
    arm_h = height * 0.6
    back_h = height

    # Storage box (body)
    # Front panel
    front = _wobble_box([width, wall, seat_h], wobble=0.01)
    front.apply_translation([0, depth / 2 - wall / 2, seat_h / 2])
    apply_color_variation(front, (*color, 255), 0.1)
    meshes.append(front)

    # Back panel
    back = _wobble_box([width, wall, back_h], wobble=0.01)
    back.apply_translation([0, -depth / 2 + wall / 2, back_h / 2])
    apply_color_variation(back, (*color, 255), 0.1)
    meshes.append(back)

    # Side panels
    for side in [-1, 1]:
        sp = _wobble_box([wall, depth, arm_h], wobble=0.01)
        sp.apply_translation([side * (width / 2 - wall / 2), 0, arm_h / 2])
        apply_color_variation(sp, (*color, 255), 0.1)
        meshes.append(sp)

    # Bottom
    bottom = _wobble_box([width - 2 * wall, depth - 2 * wall, wall], wobble=0.008)
    bottom.apply_translation([0, 0, wall / 2])
    apply_color_variation(bottom, (*color, 255), 0.06)
    meshes.append(bottom)

    # Seat/lid (slightly overhanging)
    seat = _wobble_box([width + 0.01, depth + 0.01, seat_thickness], wobble=0.01)
    seat.apply_translation([0, 0, seat_h])
    apply_color_variation(seat, (*color, 255), 0.1)
    meshes.append(seat)

    # Decorative panel on front (raised rectangle)
    panel_w = width * 0.7
    panel_h = seat_h * 0.6
    panel = _wobble_box([panel_w, 0.008, panel_h], wobble=0.005)
    panel.apply_translation([0, depth / 2 + 0.003, seat_h * 0.45])
    lighter = tuple(min(255, c + 12) for c in color)
    apply_color_variation(panel, (*lighter, 255), 0.08)
    meshes.append(panel)

    # Back panel above seat (tall back for leaning)
    back_panel_h = back_h - seat_h - seat_thickness
    if back_panel_h > 0.05:
        # Vertical slats for the back (3 panels)
        slat_w = (width - 4 * wall) / 3
        for i in range(3):
            sx = -width / 2 + wall * 2 + slat_w * (i + 0.5)
            slat = _wobble_box([slat_w - 0.01, wall * 0.7, back_panel_h - 0.02], wobble=0.006)
            slat.apply_translation([sx, -depth / 2 + wall, seat_h + seat_thickness + back_panel_h / 2])
            apply_color_variation(slat, (*color, 255), 0.1)
            meshes.append(slat)

    # Arm caps (rounded tops on side panels)
    for side in [-1, 1]:
        cap = _wobble_box([wall + 0.01, depth + 0.01, 0.02], wobble=0.005)
        cap.apply_translation([side * (width / 2 - wall / 2), 0, arm_h + 0.01])
        apply_color_variation(cap, (*color, 255), 0.08)
        meshes.append(cap)

    # Small feet
    for fx in [-1, 1]:
        for fy in [-1, 1]:
            foot = _wobble_box([0.04, 0.04, 0.02])
            foot.apply_translation([fx * (width / 2 - 0.04),
                                    fy * (depth / 2 - 0.04), -0.01])
            apply_color_variation(foot, (*color, 255), 0.06)
            meshes.append(foot)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_chest(
    width: float = 0.8,
    height: float = 0.5,
    depth: float = 0.45,
    color: Tuple[int, int, int] = (110, 65, 35),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique storage chest with arched lid and metal fittings."""
    set_seed(seed)
    meshes = []

    wall = 0.022
    body_h = height * 0.7
    lid_h = height * 0.3

    # Body box (front, back, sides, bottom)
    front = _wobble_box([width, wall, body_h], wobble=0.01)
    front.apply_translation([0, depth / 2 - wall / 2, body_h / 2])
    apply_color_variation(front, (*color, 255), 0.1)
    meshes.append(front)

    back_panel = _wobble_box([width, wall, body_h], wobble=0.01)
    back_panel.apply_translation([0, -depth / 2 + wall / 2, body_h / 2])
    apply_color_variation(back_panel, (*color, 255), 0.1)
    meshes.append(back_panel)

    for side in [-1, 1]:
        sp = _wobble_box([wall, depth, body_h], wobble=0.01)
        sp.apply_translation([side * (width / 2 - wall / 2), 0, body_h / 2])
        apply_color_variation(sp, (*color, 255), 0.1)
        meshes.append(sp)

    bottom = _wobble_box([width, depth, wall], wobble=0.008)
    bottom.apply_translation([0, 0, wall / 2])
    apply_color_variation(bottom, (*color, 255), 0.06)
    meshes.append(bottom)

    # Arched lid (half-cylinder shape)
    lid_segments = 16
    lid_r = depth / 2
    lid_verts = []
    lid_faces = []
    for i in range(lid_segments + 1):
        t = i / lid_segments
        angle = t * np.pi
        y = -lid_r * np.cos(angle)
        z = body_h + lid_h * np.sin(angle)
        for xi in range(2):
            x = (-width / 2 + wall * 0.5) if xi == 0 else (width / 2 - wall * 0.5)
            lid_verts.append([x + np.random.uniform(-0.003, 0.003),
                              y + np.random.uniform(-0.003, 0.003),
                              z + np.random.uniform(-0.003, 0.003)])
    lid_verts = np.array(lid_verts)
    for i in range(lid_segments):
        v0 = i * 2
        v1 = i * 2 + 1
        v2 = (i + 1) * 2 + 1
        v3 = (i + 1) * 2
        lid_faces.append([v0, v1, v2])
        lid_faces.append([v0, v2, v3])
    lid_mesh = trimesh.Trimesh(vertices=lid_verts, faces=np.array(lid_faces))
    lid_mesh.fix_normals()
    apply_color_variation(lid_mesh, (*color, 255), 0.1)
    meshes.append(lid_mesh)

    # End caps for the lid (triangular fan)
    for xi, x_pos in enumerate([-width / 2 + wall * 0.5, width / 2 - wall * 0.5]):
        cap_verts = [[x_pos, 0, body_h]]
        for i in range(lid_segments + 1):
            t = i / lid_segments
            angle = t * np.pi
            y = -lid_r * np.cos(angle)
            z = body_h + lid_h * np.sin(angle)
            cap_verts.append([x_pos, y, z])
        cap_verts = np.array(cap_verts)
        cap_faces = []
        for i in range(lid_segments):
            # Both winding orders for double-sided visibility
            cap_faces.append([0, i + 1, i + 2])
            cap_faces.append([0, i + 2, i + 1])
        cap_mesh = trimesh.Trimesh(vertices=cap_verts, faces=np.array(cap_faces))
        cap_mesh.fix_normals()
        apply_color_variation(cap_mesh, (*color, 255), 0.1)
        meshes.append(cap_mesh)

    # Metal bands (2 horizontal straps across front)
    metal_color = (60, 55, 50, 255)
    for bz in [body_h * 0.3, body_h * 0.7]:
        band = _wobble_box([width + 0.006, 0.008, 0.025], wobble=0.003)
        band.apply_translation([0, depth / 2, bz])
        apply_color_variation(band, metal_color, 0.05)
        meshes.append(band)
        # Same on back
        band2 = _wobble_box([width + 0.006, 0.008, 0.025], wobble=0.003)
        band2.apply_translation([0, -depth / 2, bz])
        apply_color_variation(band2, metal_color, 0.05)
        meshes.append(band2)

    # Metal latch/lock on front
    lock = _wobble_box([0.04, 0.01, 0.035], wobble=0.002)
    lock.apply_translation([0, depth / 2 + 0.005, body_h])
    apply_color_variation(lock, metal_color, 0.04)
    meshes.append(lock)

    # Keyhole (small cylinder)
    keyhole = trimesh.creation.cylinder(radius=0.005, height=0.012, sections=8)
    keyhole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    keyhole.apply_translation([0, depth / 2 + 0.01, body_h - 0.01])
    apply_color_variation(keyhole, (40, 35, 30, 255), 0.03)
    meshes.append(keyhole)

    # Corner brackets (small L-shapes on corners)
    bracket_size = 0.03
    for fx in [-1, 1]:
        for fz_pos in [0, body_h]:
            bracket = _wobble_box([bracket_size, 0.008, bracket_size], wobble=0.002)
            bracket.apply_translation([fx * width / 2, depth / 2,
                                       fz_pos + (bracket_size / 2 if fz_pos == 0 else -bracket_size / 2)])
            apply_color_variation(bracket, metal_color, 0.04)
            meshes.append(bracket)

    # Small feet
    for fx in [-1, 1]:
        for fy in [-1, 1]:
            foot = _wobble_box([0.035, 0.035, 0.015])
            foot.apply_translation([fx * (width / 2 - 0.03),
                                    fy * (depth / 2 - 0.03), -0.005])
            apply_color_variation(foot, (*color, 255), 0.06)
            meshes.append(foot)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


# --- Register new generators ---

register(
    name="single_bed", label="Single Bed", category="Furniture",
    params=[
        Param("width", "Width", "float", default=0.9, min=0.7, max=1.2),
        Param("length", "Length", "float", default=2.0, min=1.5, max=2.2),
        Param("color", "Wood Color", "color", default="#7A4028"),
    ],
)(generate_single_bed)

register(
    name="rocking_chair", label="Rocking Chair", category="Furniture",
    params=[
        Param("seat_height", "Seat Height", "float", default=0.40, min=0.3, max=0.55),
        Param("backrest_height", "Backrest Height", "float", default=0.50, min=0.3, max=0.7),
        Param("color", "Wood Color", "color", default="#55321A"),
    ],
)(generate_rocking_chair)

register(
    name="cupboard", label="Cupboard", category="Furniture",
    params=[
        Param("width", "Width", "float", default=0.9, min=0.6, max=1.5),
        Param("height", "Height", "float", default=1.4, min=1.0, max=2.0),
        Param("depth", "Depth", "float", default=0.45, min=0.3, max=0.6),
        Param("color", "Wood Color", "color", default="#643C23"),
    ],
)(generate_cupboard)

register(
    name="settle_bed", label="Settle Bed", category="Furniture",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.8, max=1.8),
        Param("height", "Height", "float", default=0.85, min=0.6, max=1.2),
        Param("depth", "Depth", "float", default=0.45, min=0.3, max=0.6),
        Param("color", "Wood Color", "color", default="#5F371E"),
    ],
)(generate_settle_bed)

register(
    name="chest", label="Chest", category="Furniture",
    params=[
        Param("width", "Width", "float", default=0.8, min=0.5, max=1.2),
        Param("height", "Height", "float", default=0.5, min=0.3, max=0.7),
        Param("depth", "Depth", "float", default=0.45, min=0.3, max=0.6),
        Param("color", "Wood Color", "color", default="#6E4123"),
    ],
)(generate_chest)


# =============================================================================
# WALL-MOUNTED & ADDITIONAL FURNITURE
# =============================================================================

def generate_wall_coat_rack(
    width: float = 0.80,
    color: Tuple[int, int, int] = (100, 60, 32),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a wall-mounted coat rack with pegs/hooks on a backboard."""
    set_seed(seed)
    meshes = []

    board_h = 0.12
    board_t = 0.02
    num_hooks = max(3, int(width / 0.15))

    # Backboard (long horizontal plank)
    board = _wobble_box([width, board_t, board_h], wobble=0.008)
    board.apply_translation([0, 0, 0])
    apply_color_variation(board, (*color, 255), 0.1)
    meshes.append(board)

    # Top decorative edge (slightly wider, shaped)
    top_edge = _wobble_box([width + 0.01, board_t + 0.002, 0.015], wobble=0.005)
    top_edge.apply_translation([0, 0, board_h / 2 + 0.007])
    apply_color_variation(top_edge, (*color, 255), 0.08)
    meshes.append(top_edge)

    # Bottom decorative edge
    bot_edge = _wobble_box([width + 0.01, board_t + 0.002, 0.012], wobble=0.005)
    bot_edge.apply_translation([0, 0, -board_h / 2 - 0.006])
    apply_color_variation(bot_edge, (*color, 255), 0.08)
    meshes.append(bot_edge)

    # Hooks/pegs
    for i in range(num_hooks):
        hx = -width / 2 + (i + 0.5) * width / num_hooks

        # Peg base (turned knob on backboard)
        base = trimesh.creation.cylinder(radius=0.012, height=0.015, sections=12)
        base.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        base.apply_translation([hx, board_t / 2 + 0.007, -0.01])
        apply_color_variation(base, (*color, 255), 0.06)
        meshes.append(base)

        # Peg shaft (sticking outward)
        shaft_len = 0.06
        shaft = trimesh.creation.cylinder(radius=0.008, height=shaft_len, sections=10)
        shaft.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        shaft.apply_translation([hx, board_t / 2 + 0.015 + shaft_len / 2, -0.01])
        apply_color_variation(shaft, (*color, 255), 0.08)
        meshes.append(shaft)

        # Peg tip (rounded knob)
        tip = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        tip.apply_translation([hx, board_t / 2 + 0.015 + shaft_len + 0.005, -0.01])
        apply_color_variation(tip, (*color, 255), 0.06)
        meshes.append(tip)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_plate_rack(
    width: float = 0.80,
    height: float = 0.60,
    depth: float = 0.18,
    color: Tuple[int, int, int] = (110, 70, 38),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a wall-mounted plate rack with grooved shelves for dinner plates."""
    set_seed(seed)
    meshes = []

    wall = 0.015
    num_shelves = 2
    shelf_spacing = height / (num_shelves + 1)

    # Side panels
    for side in [-1, 1]:
        panel = _wobble_box([wall, depth, height], wobble=0.008)
        panel.apply_translation([side * (width / 2 - wall / 2), 0, height / 2])
        apply_color_variation(panel, (*color, 255), 0.1)
        meshes.append(panel)

    # Back panel (thin)
    back = _wobble_box([width, wall * 0.6, height], wobble=0.008)
    back.apply_translation([0, -depth / 2 + wall * 0.3, height / 2])
    apply_color_variation(back, (*color, 255), 0.08)
    meshes.append(back)

    # Top rail
    top_rail = _wobble_box([width, wall, wall * 1.5], wobble=0.005)
    top_rail.apply_translation([0, -depth / 2 + wall / 2, height - wall])
    apply_color_variation(top_rail, (*color, 255), 0.08)
    meshes.append(top_rail)

    # Shelves with front lip (to hold plates)
    plate_color = (230, 225, 210, 255)
    for si in range(num_shelves):
        sz = (si + 1) * shelf_spacing

        # Shelf surface (angled slightly back so plates lean)
        shelf = _wobble_box([width - 2 * wall, depth * 0.6, wall], wobble=0.006)
        shelf.apply_translation([0, depth * 0.1, sz])
        apply_color_variation(shelf, (*color, 255), 0.08)
        meshes.append(shelf)

        # Front lip (prevents plates sliding off)
        lip = _wobble_box([width - 2 * wall, wall, 0.02], wobble=0.004)
        lip.apply_translation([0, depth * 0.1 + depth * 0.3, sz + 0.01])
        apply_color_variation(lip, (*color, 255), 0.08)
        meshes.append(lip)

        # Groove dowels (thin horizontal rods above shelf for plates to lean against)
        dowel_z = sz + 0.12
        if dowel_z < height - 0.05:
            dowel = trimesh.creation.cylinder(
                radius=0.004, height=width - 2 * wall, sections=8)
            dowel.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            dowel.apply_translation([0, -depth * 0.15, dowel_z])
            apply_color_variation(dowel, (*color, 255), 0.06)
            meshes.append(dowel)

        # Sample plates on each shelf (3 plates)
        for pi in range(3):
            px = -width / 3 + pi * width / 3
            # Plate: thin disc leaning back
            plate = trimesh.creation.cylinder(radius=0.065, height=0.004, sections=16)
            # Tilt back ~75 degrees
            plate.apply_transform(
                trimesh.transformations.rotation_matrix(1.3, [1, 0, 0]))
            plate.apply_translation([px, -depth * 0.05, sz + 0.07])
            apply_color_variation(plate, plate_color, 0.03)
            meshes.append(plate)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def generate_grandfather_clock(
    height: float = 2.1,
    color: Tuple[int, int, int] = (95, 55, 30),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate an antique grandfather clock with pendulum case and clock face."""
    set_seed(seed)
    meshes = []

    total_h = height
    body_w = 0.45
    body_d = 0.28
    wall = 0.022

    # Proportions
    base_h = total_h * 0.12
    trunk_h = total_h * 0.45  # pendulum case
    hood_h = total_h * 0.35  # clock face area
    crown_h = total_h * 0.08

    base_w = body_w + 0.06
    base_d = body_d + 0.04
    hood_w = body_w + 0.03
    hood_d = body_d + 0.02

    # --- Base (wider, with molding) ---
    base = _wobble_box([base_w, base_d, base_h], wobble=0.01)
    base.apply_translation([0, 0, base_h / 2])
    apply_color_variation(base, (*color, 255), 0.1)
    meshes.append(base)

    # Base feet
    for fx in [-1, 1]:
        for fy in [-1, 1]:
            foot = _wobble_box([0.05, 0.05, 0.02])
            foot.apply_translation([fx * (base_w / 2 - 0.04),
                                    fy * (base_d / 2 - 0.04), 0.01])
            apply_color_variation(foot, (*color, 255), 0.06)
            meshes.append(foot)

    # --- Trunk (pendulum case, narrower) ---
    trunk_z = base_h
    # Back
    trunk_back = _wobble_box([body_w, wall, trunk_h], wobble=0.01)
    trunk_back.apply_translation([0, -body_d / 2 + wall / 2, trunk_z + trunk_h / 2])
    apply_color_variation(trunk_back, (*color, 255), 0.08)
    meshes.append(trunk_back)
    # Sides
    for side in [-1, 1]:
        sp = _wobble_box([wall, body_d, trunk_h], wobble=0.01)
        sp.apply_translation([side * (body_w / 2 - wall / 2), 0, trunk_z + trunk_h / 2])
        apply_color_variation(sp, (*color, 255), 0.08)
        meshes.append(sp)

    # Trunk front door (glass window to see pendulum)
    door_w = body_w - 2 * wall - 0.02
    door_h = trunk_h - 0.04
    # Door frame
    frame_t = 0.02
    for part_name, dims, pos in [
        ("top", [door_w, frame_t, frame_t],
         [0, body_d / 2 - wall / 2, trunk_z + trunk_h - 0.02]),
        ("bottom", [door_w, frame_t, frame_t],
         [0, body_d / 2 - wall / 2, trunk_z + 0.02]),
        ("left", [frame_t, frame_t, door_h],
         [-door_w / 2, body_d / 2 - wall / 2, trunk_z + trunk_h / 2]),
        ("right", [frame_t, frame_t, door_h],
         [door_w / 2, body_d / 2 - wall / 2, trunk_z + trunk_h / 2]),
    ]:
        bar = _wobble_box(dims, wobble=0.004)
        bar.apply_translation(pos)
        apply_color_variation(bar, (*color, 255), 0.08)
        meshes.append(bar)

    # Pendulum rod
    pend_len = trunk_h * 0.65
    pend_rod = trimesh.creation.cylinder(radius=0.003, height=pend_len, sections=8)
    pend_rod.apply_translation([0, 0, trunk_z + trunk_h * 0.6 - pend_len / 2])
    apply_color_variation(pend_rod, (180, 160, 80, 255), 0.05)
    meshes.append(pend_rod)

    # Pendulum bob (disc)
    bob = trimesh.creation.cylinder(radius=0.04, height=0.008, sections=16)
    bob.apply_translation([0, 0, trunk_z + trunk_h * 0.6 - pend_len + 0.02])
    apply_color_variation(bob, (190, 170, 80, 255), 0.06)
    meshes.append(bob)

    # --- Hood (clock face area, slightly wider) ---
    hood_z = trunk_z + trunk_h
    # Back
    hood_back = _wobble_box([hood_w, wall, hood_h], wobble=0.01)
    hood_back.apply_translation([0, -hood_d / 2 + wall / 2, hood_z + hood_h / 2])
    apply_color_variation(hood_back, (*color, 255), 0.08)
    meshes.append(hood_back)
    # Sides
    for side in [-1, 1]:
        sp = _wobble_box([wall, hood_d, hood_h], wobble=0.01)
        sp.apply_translation([side * (hood_w / 2 - wall / 2), 0, hood_z + hood_h / 2])
        apply_color_variation(sp, (*color, 255), 0.08)
        meshes.append(sp)
    # Top
    hood_top = _wobble_box([hood_w, hood_d, wall], wobble=0.008)
    hood_top.apply_translation([0, 0, hood_z + hood_h])
    apply_color_variation(hood_top, (*color, 255), 0.08)
    meshes.append(hood_top)

    # Clock face (cream circle on front)
    face_r = min(hood_w, hood_h) * 0.35
    face_z = hood_z + hood_h * 0.5
    face = trimesh.creation.cylinder(radius=face_r, height=0.005, sections=24)
    face.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    face.apply_translation([0, hood_d / 2 + 0.002, face_z])
    apply_color_variation(face, (235, 228, 205, 255), 0.03)
    meshes.append(face)

    # Clock face rim (brass ring)
    rim_pts = []
    for i in range(32):
        a = (i / 32) * 2 * np.pi
        for j in range(8):
            b = (j / 8) * 2 * np.pi
            rx = (face_r + 0.005 * np.cos(b)) * np.cos(a)
            rz = (face_r + 0.005 * np.cos(b)) * np.sin(a)
            ry = 0.005 * np.sin(b)
            rim_pts.append([rx, ry, rz])
    rim = trimesh.convex.convex_hull(np.array(rim_pts))
    rim.apply_translation([0, hood_d / 2 + 0.002, face_z])
    apply_color_variation(rim, (185, 165, 75, 255), 0.05)
    meshes.append(rim)

    # Hour markers (12 small dots around face)
    for i in range(12):
        a = (i / 12) * 2 * np.pi
        mx = face_r * 0.8 * np.sin(a)
        mz = face_z + face_r * 0.8 * np.cos(a)
        marker = trimesh.creation.icosphere(subdivisions=1, radius=0.006)
        marker.apply_translation([mx, hood_d / 2 + 0.006, mz])
        apply_color_variation(marker, (40, 35, 30, 255), 0.03)
        meshes.append(marker)

    # Clock hands (hour and minute)
    for hand_len, hand_w, angle in [
        (face_r * 0.5, 0.004, 0.8),   # hour hand
        (face_r * 0.7, 0.003, 2.5),   # minute hand
    ]:
        hand = _wobble_box([hand_w, 0.003, hand_len], wobble=0.002)
        hand.apply_transform(
            trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
        hand.apply_translation([0, hood_d / 2 + 0.008, face_z])
        apply_color_variation(hand, (30, 25, 20, 255), 0.03)
        meshes.append(hand)

    # Center pin
    pin = trimesh.creation.icosphere(subdivisions=1, radius=0.005)
    pin.apply_translation([0, hood_d / 2 + 0.008, face_z])
    apply_color_variation(pin, (185, 165, 75, 255), 0.04)
    meshes.append(pin)

    # --- Crown (decorative top) ---
    crown_z = hood_z + hood_h
    crown = _wobble_box([hood_w + 0.02, hood_d + 0.01, crown_h * 0.4], wobble=0.008)
    crown.apply_translation([0, 0, crown_z + crown_h * 0.2])
    apply_color_variation(crown, (*color, 255), 0.08)
    meshes.append(crown)

    # Arch/pediment on top
    arch_parts = _decorative_arch(hood_w * 0.7, crown_h * 0.7, wall * 0.8, segments=12)
    for part in arch_parts:
        part.apply_translation([0, 0, crown_z + crown_h * 0.3])
        apply_color_variation(part, (*color, 255), 0.08)
        meshes.append(part)

    # Finial on top center
    finial = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
    finial.vertices[:, 2] *= 1.4
    finial.apply_translation([0, 0, total_h + 0.01])
    apply_color_variation(finial, (185, 165, 75, 255), 0.06)
    meshes.append(finial)

    # Transition molding between sections
    for mz in [trunk_z, hood_z]:
        mold_w = body_w + 0.02 if mz == trunk_z else hood_w + 0.01
        mold_d = body_d + 0.01 if mz == trunk_z else hood_d + 0.005
        mold = _wobble_box([mold_w, mold_d, 0.02], wobble=0.005)
        mold.apply_translation([0, 0, mz])
        apply_color_variation(mold, (*color, 255), 0.08)
        meshes.append(mold)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


register(
    name="wall_coat_rack", label="Wall Coat Rack", category="Furniture",
    params=[
        Param("width", "Width", "float", default=0.80, min=0.40, max=1.2),
        Param("color", "Wood Color", "color", default="#643C20"),
    ],
)(generate_wall_coat_rack)

register(
    name="plate_rack", label="Plate Rack", category="Furniture",
    params=[
        Param("width", "Width", "float", default=0.80, min=0.50, max=1.2),
        Param("height", "Height", "float", default=0.60, min=0.40, max=1.0),
        Param("depth", "Depth", "float", default=0.18, min=0.12, max=0.30),
        Param("color", "Wood Color", "color", default="#6E4626"),
    ],
)(generate_plate_rack)

register(
    name="grandfather_clock", label="Grandfather Clock", category="Furniture",
    params=[
        Param("height", "Height", "float", default=2.1, min=1.6, max=2.5),
        Param("color", "Wood Color", "color", default="#5F371E"),
    ],
)(generate_grandfather_clock)
