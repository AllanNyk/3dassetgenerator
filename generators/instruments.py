"""
Musical instrument generators with vertex color support.
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


def _make_tube(radius, points, sections=12):
    """Create a tube (open cylinder) following a list of 3D points."""
    meshes = []
    for i in range(len(points) - 1):
        p0 = np.array(points[i])
        p1 = np.array(points[i + 1])
        diff = p1 - p0
        length = np.linalg.norm(diff)
        if length < 1e-6:
            continue
        mid = (p0 + p1) / 2

        seg = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

        # Align cylinder (default Z-axis) to the segment direction
        direction = diff / length
        z_axis = np.array([0, 0, 1])
        if abs(np.dot(direction, z_axis)) < 0.999:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot = trimesh.transformations.rotation_matrix(angle, axis)
            seg.apply_transform(rot)
        elif np.dot(direction, z_axis) < 0:
            seg.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

        seg.apply_translation(mid)
        meshes.append(seg)
    return meshes


def _violin_body_profile(n_points=60):
    """Return the right-half outline of a violin body as (x, z) pairs.

    Z runs from 0 (bottom) to 1.0 (top/neck joint).
    X is the half-width at each Z position.
    Based on classical violin proportions.
    """
    # Key control points along Z axis (normalized 0-1):
    #   0.00 = bottom tip
    #   0.18 = max lower bout width
    #   0.42 = waist (C-bout, narrowest)
    #   0.68 = max upper bout width
    #   1.00 = top (neck joint)
    #
    # Half-widths at key points (normalized, will be scaled):
    #   bottom tip:      0.00
    #   lower bout max:  0.29  (full width 0.58)
    #   waist:           0.155 (full width 0.31)
    #   upper bout max:  0.235 (full width 0.47)
    #   top:             0.00

    # Violin proportions (more slender than viola):
    # - Narrower waist relative to bouts
    # - More pronounced C-bout curves
    # - Upper bout noticeably smaller than lower bout
    lower_bout_w = 0.27    # half-width of lower bout
    upper_bout_w = 0.21    # half-width of upper bout
    waist_w = 0.125        # half-width at waist (narrow!)
    neck_joint_w = 0.028   # half-width where neck meets body

    points = []
    for i in range(n_points):
        t = i / (n_points - 1)  # 0 to 1

        if t < 0.05:
            # Bottom tip curve
            u = t / 0.05
            x = lower_bout_w * np.sin(u * np.pi / 2)
        elif t < 0.12:
            # Lower bout rising to max
            u = (t - 0.05) / 0.07
            x = lower_bout_w * (1.0 - 0.02 * np.cos(u * np.pi))
        elif t < 0.22:
            # Lower bout at max, gentle curve
            u = (t - 0.12) / 0.10
            x = lower_bout_w * (1.0 - 0.01 * (1 - np.cos(u * np.pi)))
        elif t < 0.35:
            # C-bout: sharp descent to waist
            u = (t - 0.22) / 0.13
            x = lower_bout_w - (lower_bout_w - waist_w) * (0.5 - 0.5 * np.cos(u * np.pi))
        elif t < 0.45:
            # Waist (narrowest, with slight concavity)
            u = (t - 0.35) / 0.10
            x = waist_w - 0.008 * np.sin(u * np.pi)
        elif t < 0.58:
            # C-bout: sharp ascent to upper bout
            u = (t - 0.45) / 0.13
            x = waist_w + (upper_bout_w - waist_w) * (0.5 - 0.5 * np.cos(u * np.pi))
        elif t < 0.68:
            # Upper bout at max
            u = (t - 0.58) / 0.10
            x = upper_bout_w * (1.0 - 0.01 * (1 - np.cos(u * np.pi)))
        elif t < 0.78:
            # Upper bout descending
            u = (t - 0.68) / 0.10
            x = upper_bout_w - (upper_bout_w - 0.06) * (0.5 - 0.5 * np.cos(u * np.pi))
        elif t < 0.90:
            # Narrowing toward neck joint
            u = (t - 0.78) / 0.12
            x = 0.06 - (0.06 - neck_joint_w) * (0.5 - 0.5 * np.cos(u * np.pi))
        else:
            # Top closing to neck width
            u = (t - 0.90) / 0.10
            x = neck_joint_w * np.cos(u * np.pi / 2)

        points.append((x, t))

    return points


def _build_violin_body(body_length, body_depth, scale):
    """Build violin body mesh from profile with arched top and back plates."""
    profile = _violin_body_profile(n_points=60)

    # Arch layers: y-position and how much to inset the profile
    # Creates convex arching on front and back
    depth = body_depth * scale
    layers = [
        (0.0,           0.92),   # back edge
        (depth * 0.15,  0.97),   # back arch rise
        (depth * 0.35,  1.00),   # back arch peak
        (depth * 0.50,  1.00),   # middle (max width)
        (depth * 0.65,  1.00),   # front arch peak
        (depth * 0.85,  0.97),   # front arch rise
        (depth * 1.0,   0.92),   # front edge
    ]

    n_profile = len(profile)
    n_layers = len(layers)

    # Build vertices: for each layer, place the profile points
    vertices = []
    for y_pos, width_factor in layers:
        for x_half, z_norm in profile:
            x = x_half * body_length * scale * width_factor
            z = z_norm * body_length * scale
            vertices.append([x, y_pos - depth / 2, z])
        # Mirror: negative X side (reverse order for consistent winding)
        for x_half, z_norm in profile:
            x = -x_half * body_length * scale * width_factor
            z = z_norm * body_length * scale
            vertices.append([x, y_pos - depth / 2, z])

    vertices = np.array(vertices)
    faces = []
    stride = n_profile * 2  # vertices per layer (right side + left side)

    # Connect adjacent layers with quads (as triangle pairs)
    for layer_i in range(n_layers - 1):
        base0 = layer_i * stride
        base1 = (layer_i + 1) * stride

        # Right side
        for i in range(n_profile - 1):
            v0 = base0 + i
            v1 = base0 + i + 1
            v2 = base1 + i + 1
            v3 = base1 + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

        # Left side
        for i in range(n_profile - 1):
            v0 = base0 + n_profile + i
            v1 = base0 + n_profile + i + 1
            v2 = base1 + n_profile + i + 1
            v3 = base1 + n_profile + i
            faces.append([v0, v2, v1])
            faces.append([v0, v3, v2])

        # Connect right side to left side at the edges (top and bottom of profile)
        # Bottom edge (index 0 on both sides)
        r0_0 = base0 + 0
        l0_0 = base0 + n_profile + 0
        r1_0 = base1 + 0
        l1_0 = base1 + n_profile + 0
        faces.append([r0_0, l0_0, l1_0])
        faces.append([r0_0, l1_0, r1_0])

        # Top edge (last index on both sides)
        r0_e = base0 + n_profile - 1
        l0_e = base0 + n_profile + n_profile - 1
        r1_e = base1 + n_profile - 1
        l1_e = base1 + n_profile + n_profile - 1
        faces.append([r0_e, l1_e, l0_e])
        faces.append([r0_e, r1_e, l1_e])

    # Cap the front and back plates with triangle fans
    # Back plate (layer 0): right side verts 0..n_profile-1, left side n_profile..2*n_profile-1
    # Front plate (last layer): same offsets + (n_layers-1)*stride
    for layer_idx in [0, n_layers - 1]:
        base = layer_idx * stride
        # Collect all outline vertices for this layer in order:
        # right side forward (0..n-1), then left side backward (2n-1..n)
        # This traces the full outline clockwise or counterclockwise
        outline = []
        for i in range(n_profile):
            outline.append(base + i)
        for i in range(n_profile - 1, -1, -1):
            outline.append(base + n_profile + i)

        # Add center vertex for this plate
        center_idx = len(vertices)
        # Compute center position from the layer's vertices
        layer_verts = vertices[base:base + stride]
        center_pos = layer_verts.mean(axis=0)
        vertices = np.vstack([vertices, [center_pos]])

        # Create triangle fan
        n_outline = len(outline)
        for i in range(n_outline):
            i_next = (i + 1) % n_outline
            if layer_idx == 0:
                # Back plate: normals face backward (negative Y)
                faces.append([center_idx, outline[i_next], outline[i]])
            else:
                # Front plate: normals face forward (positive Y)
                faces.append([center_idx, outline[i], outline[i_next]])

    faces = np.array(faces)
    body = trimesh.Trimesh(vertices=vertices, faces=faces)
    body.fix_normals()
    return body


def generate_violin(
    body_length: float = 1.0,
    neck_length: float = 0.6,
    body_color: Tuple[int, int, int] = (160, 80, 20),
    detail_color: Tuple[int, int, int] = (40, 30, 20),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a violin with profiled body, neck, scroll, strings, bridge, and tailpiece."""
    set_seed(seed)

    meshes = []
    s = body_length  # scale factor
    body_depth = 0.09  # depth relative to body_length

    # --- Body (profile-based with arched plates) ---
    body = _build_violin_body(body_length, body_depth, 1.0)
    apply_color_variation(body, (*body_color, 255), 0.06)
    meshes.append(body)

    # --- Neck ---
    neck_w = 0.055 * s
    neck_d = 0.038 * s
    neck_start_z = 0.87 * s  # where body profile ends
    neck = trimesh.creation.box(extents=[neck_w, neck_d, neck_length * s])
    neck.apply_translation([0, 0, neck_start_z + (neck_length * s) / 2])
    apply_color_variation(neck, (*body_color, 255), 0.05)
    meshes.append(neck)

    # --- Fingerboard (on top of neck, slightly wider, dark) ---
    fb_w = 0.06 * s
    fb_d = 0.012 * s
    fb_len = (neck_length * 0.95 + 0.15) * s  # extends slightly over body
    fb = trimesh.creation.box(extents=[fb_w, fb_d, fb_len])
    fb.apply_translation([0, neck_d / 2 + fb_d / 2,
                          neck_start_z + (neck_length * s) / 2 - 0.07 * s])
    apply_color_variation(fb, (20, 15, 10, 255), 0.03)
    meshes.append(fb)

    # --- Pegbox ---
    scroll_z_start = neck_start_z + neck_length * s
    pegbox_len = 0.08 * s
    pegbox = trimesh.creation.box(extents=[0.045 * s, 0.035 * s, pegbox_len])
    pegbox.apply_translation([0, 0, scroll_z_start + pegbox_len / 2])
    apply_color_variation(pegbox, (*detail_color, 255), 0.05)
    meshes.append(pegbox)

    # --- Scroll (spiral of connected cylinders) ---
    scroll_z = scroll_z_start + pegbox_len + 0.005 * s
    num_spiral = 28
    scroll_meshes = []
    prev_pos = None
    for i in range(num_spiral):
        t = i / (num_spiral - 1)
        angle = t * 4.0 * np.pi  # ~2 full turns
        r = (0.022 - t * 0.016) * s
        sx = 0
        sy = r * np.cos(angle)
        sz = scroll_z + r * np.sin(angle)
        pos = [sx, sy, sz]

        if prev_pos is not None:
            seg_r = (0.009 - t * 0.005) * s
            seg_r = max(seg_r, 0.002 * s)
            segs = _make_tube(seg_r, [prev_pos, pos], sections=8)
            scroll_meshes.extend(segs)

        prev_pos = pos

    for sm in scroll_meshes:
        apply_color_variation(sm, (*detail_color, 255), 0.05)
    meshes.extend(scroll_meshes)

    # Scroll end cap
    end_cap = trimesh.creation.icosphere(subdivisions=1, radius=0.006 * s)
    end_cap.apply_translation(prev_pos)
    apply_color_variation(end_cap, (*detail_color, 255), 0.05)
    meshes.append(end_cap)

    # --- Tuning pegs (4 pegs through pegbox, alternating sides) ---
    # Each peg: shaft through pegbox + handle on the outside + end cap
    # Cylinders default along Z axis; rotate around Y to align with X axis
    rot_to_x = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    peg_shaft_r = 0.004 * s
    peg_handle_r = 0.006 * s
    pegbox_half_w = 0.0225 * s
    peg_handle_len = 0.03 * s
    peg_z_positions = [
        scroll_z_start + 0.015 * s,
        scroll_z_start + 0.03 * s,
        scroll_z_start + 0.05 * s,
        scroll_z_start + 0.065 * s,
    ]
    for i, pz in enumerate(peg_z_positions):
        side = 1 if i % 2 == 0 else -1

        # Shaft (goes through the pegbox, along X axis)
        shaft = trimesh.creation.cylinder(
            radius=peg_shaft_r, height=pegbox_half_w * 2, sections=8)
        shaft.apply_transform(rot_to_x)
        shaft.apply_translation([0, 0, pz])
        apply_color_variation(shaft, (*detail_color, 255), 0.05)
        meshes.append(shaft)

        # Handle (thicker, on the outside, along X axis)
        handle = trimesh.creation.cylinder(
            radius=peg_handle_r, height=peg_handle_len, sections=8)
        handle.apply_transform(rot_to_x)
        handle.apply_translation([side * (pegbox_half_w + peg_handle_len / 2), 0, pz])
        apply_color_variation(handle, (*detail_color, 255), 0.05)
        meshes.append(handle)

        # Rounded end cap on handle
        peg_head = trimesh.creation.icosphere(subdivisions=1, radius=peg_handle_r * 1.3)
        peg_head.apply_translation([side * (pegbox_half_w + peg_handle_len), 0, pz])
        apply_color_variation(peg_head, (*detail_color, 255), 0.05)
        meshes.append(peg_head)

    # --- Bridge ---
    bridge_w = 0.07 * s
    bridge_h = 0.022 * s
    bridge_d = 0.006 * s
    bridge_z = 0.35 * s  # on the lower bout area
    bridge = trimesh.creation.box(extents=[bridge_w, bridge_d, bridge_h])
    bridge.apply_translation([0, body_depth * s / 2 + bridge_d / 2, bridge_z])
    apply_color_variation(bridge, (200, 180, 140, 255), 0.05)
    meshes.append(bridge)

    # --- Tailpiece ---
    tp_w = 0.04 * s
    tp_len = 0.1 * s
    tp_d = 0.01 * s
    tp_z = 0.15 * s
    tailpiece = trimesh.creation.box(extents=[tp_w, tp_d, tp_len])
    tailpiece.apply_translation([0, body_depth * s / 2 + tp_d / 2, tp_z])
    apply_color_variation(tailpiece, (*detail_color, 255), 0.05)
    meshes.append(tailpiece)

    # Tailgut (thin connector from tailpiece to endpin)
    tg = trimesh.creation.cylinder(radius=0.002 * s, height=0.12 * s, sections=6)
    tg.apply_translation([0, body_depth * s * 0.3, 0.05 * s])
    apply_color_variation(tg, (60, 60, 60, 255), 0.03)
    meshes.append(tg)

    # --- Chinrest (to the left of tailpiece when viewed from front) ---
    chinrest = trimesh.creation.icosphere(subdivisions=2, radius=0.04 * s)
    chinrest.vertices[:, 1] *= 0.3   # flatten (thin)
    chinrest.vertices[:, 0] *= 0.8   # slightly narrow
    chinrest.vertices[:, 2] *= 1.8   # elongate along body length
    chinrest.apply_translation([0.1 * s, body_depth * s / 2 + 0.008 * s, 0.1 * s])
    apply_color_variation(chinrest, (*detail_color, 255), 0.05)
    meshes.append(chinrest)

    # --- Strings (4 strings from tailpiece over bridge to pegbox) ---
    string_spacing = 0.012 * s
    string_z_start = tp_z + tp_len / 2  # top of tailpiece
    string_z_end = scroll_z_start + 0.015 * s  # into pegbox
    string_y = body_depth * s / 2 + bridge_d + 0.003 * s  # above front plate

    for i in range(4):
        x_off = (i - 1.5) * string_spacing
        sr = 0.0015 * s if i < 2 else 0.001 * s
        string = trimesh.creation.cylinder(
            radius=sr,
            height=string_z_end - string_z_start,
            sections=6
        )
        string.apply_translation([x_off, string_y, (string_z_start + string_z_end) / 2])
        string_color = (220, 210, 180, 255) if i < 2 else (190, 190, 200, 255)
        apply_color_variation(string, string_color, 0.03)
        meshes.append(string)

    # --- F-holes (dark elongated shapes on front plate) ---
    for side in [-1, 1]:
        # Each f-hole: elongated narrow shape
        fh = trimesh.creation.cylinder(radius=0.005 * s, height=0.1 * s, sections=6)
        # Slight rotation for the italic f-shape
        fh.apply_transform(trimesh.transformations.rotation_matrix(
            side * 0.15, [0, 1, 0]))
        fh.apply_translation([side * 0.09 * s, body_depth * s / 2 + 0.002 * s, 0.38 * s])
        apply_color_variation(fh, (15, 10, 5, 255), 0.02)
        meshes.append(fh)

        # Small circles at top and bottom of f-hole
        for z_off in [-0.05 * s, 0.05 * s]:
            dot = trimesh.creation.icosphere(subdivisions=1, radius=0.006 * s)
            dot.vertices[:, 1] *= 0.3
            dot.apply_translation([side * 0.09 * s, body_depth * s / 2 + 0.002 * s,
                                   0.38 * s + z_off])
            apply_color_variation(dot, (15, 10, 5, 255), 0.02)
            meshes.append(dot)

    # --- End pin (bottom of body) ---
    endpin = trimesh.creation.cylinder(radius=0.003 * s, height=0.03 * s, sections=6)
    endpin.apply_translation([0, 0, -0.015 * s])
    apply_color_variation(endpin, (80, 80, 80, 255), 0.05)
    meshes.append(endpin)

    # --- Saddle (small piece at neck/body joint, strings pass over) ---
    saddle = trimesh.creation.box(extents=[0.05 * s, 0.006 * s, 0.008 * s])
    saddle.apply_translation([0, body_depth * s / 2 + 0.003 * s, 0.83 * s])
    apply_color_variation(saddle, (20, 15, 10, 255), 0.03)
    meshes.append(saddle)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


def _make_bell(radius_start, radius_end, length, sections=24, length_segments=12):
    """Create a flared bell shape (like a trumpet bell) along the Z axis."""
    vertices = []
    faces = []

    for j in range(length_segments + 1):
        t = j / length_segments
        z = t * length
        # Exponential flare toward the end
        r = radius_start + (radius_end - radius_start) * (t ** 2.5)
        for i in range(sections):
            angle = (i / sections) * 2 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for j in range(length_segments):
        for i in range(sections):
            i_next = (i + 1) % sections
            v0 = j * sections + i
            v1 = j * sections + i_next
            v2 = (j + 1) * sections + i_next
            v3 = (j + 1) * sections + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # End rim (ring at the bell opening)
    # No cap - bell is open

    faces = np.array(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def generate_trumpet(
    length: float = 1.0,
    bell_flare: float = 0.5,
    body_color: Tuple[int, int, int] = (220, 190, 80),
    detail_color: Tuple[int, int, int] = (180, 160, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a trumpet with bell, main tube, valves, slides, and mouthpiece."""
    set_seed(seed)

    meshes = []
    s = length  # scale factor
    tube_r = 0.012 * s  # main tube radius

    # Trumpet is oriented along Z axis (bell at top, mouthpiece at bottom)

    # --- Bell ---
    bell_len = 0.3 * s
    bell_r_end = (0.08 + bell_flare * 0.08) * s  # flare controlled by param
    bell = _make_bell(tube_r, bell_r_end, bell_len, sections=24, length_segments=16)
    bell.apply_translation([0, 0, 0.7 * s])  # at the top
    bell.fix_normals()
    apply_color_variation(bell, (*body_color, 255), 0.05)
    meshes.append(bell)

    # Bell rim (thickened edge)
    rim = trimesh.creation.cylinder(radius=bell_r_end + 0.003 * s, height=0.005 * s, sections=24)
    # Make it a torus-like ring by subtracting... just use a thin cylinder ring
    rim_inner = trimesh.creation.cylinder(radius=bell_r_end - 0.002 * s, height=0.006 * s, sections=24)
    rim.apply_translation([0, 0, 1.0 * s + 0.002 * s])
    apply_color_variation(rim, (*body_color, 255), 0.04)
    meshes.append(rim)

    # --- Main tube (leadpipe to bell) ---
    # Straight section from valve block to bell
    main_tube_len = 0.35 * s
    main_tube = trimesh.creation.cylinder(radius=tube_r, height=main_tube_len, sections=12)
    main_tube.apply_translation([0, 0, 0.7 * s - main_tube_len / 2 + 0.02 * s])
    apply_color_variation(main_tube, (*body_color, 255), 0.05)
    meshes.append(main_tube)

    # --- Valve section (3 valves) ---
    valve_z = 0.42 * s  # center of valve block
    valve_r = 0.02 * s
    valve_h = 0.08 * s
    valve_spacing = 0.05 * s
    valve_block_y = 0.0  # centered

    for vi in range(3):
        vx = (vi - 1) * valve_spacing

        # Valve casing (vertical cylinder)
        casing = trimesh.creation.cylinder(radius=valve_r, height=valve_h, sections=12)
        casing.apply_translation([vx, valve_block_y, valve_z])
        apply_color_variation(casing, (*body_color, 255), 0.04)
        meshes.append(casing)

        # Valve cap (top)
        cap = trimesh.creation.cylinder(radius=valve_r * 1.15, height=0.008 * s, sections=12)
        cap.apply_translation([vx, valve_block_y, valve_z + valve_h / 2 + 0.004 * s])
        apply_color_variation(cap, (*body_color, 255), 0.04)
        meshes.append(cap)

        # Finger button (on top of cap)
        button = trimesh.creation.icosphere(subdivisions=1, radius=0.008 * s)
        button.vertices[:, 2] *= 0.5  # flatten
        button.apply_translation([vx, valve_block_y, valve_z + valve_h / 2 + 0.012 * s])
        apply_color_variation(button, (200, 200, 200, 255), 0.03)
        meshes.append(button)

        # Valve bottom cap
        bot_cap = trimesh.creation.cylinder(radius=valve_r * 1.1, height=0.006 * s, sections=12)
        bot_cap.apply_translation([vx, valve_block_y, valve_z - valve_h / 2 - 0.003 * s])
        apply_color_variation(bot_cap, (*body_color, 255), 0.04)
        meshes.append(bot_cap)

    # --- Tube connecting valves (horizontal tubes between valve casings) ---
    for vi in range(2):
        vx_start = (vi - 1) * valve_spacing + valve_r
        vx_end = (vi) * valve_spacing - valve_r
        conn_len = vx_end - vx_start
        conn = trimesh.creation.cylinder(radius=tube_r * 0.9, height=conn_len, sections=8)
        conn.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        conn.apply_translation([(vx_start + vx_end) / 2, valve_block_y, valve_z + valve_h * 0.3])
        apply_color_variation(conn, (*body_color, 255), 0.05)
        meshes.append(conn)

        # Bottom connector too
        conn_b = trimesh.creation.cylinder(radius=tube_r * 0.9, height=conn_len, sections=8)
        conn_b.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        conn_b.apply_translation([(vx_start + vx_end) / 2, valve_block_y, valve_z - valve_h * 0.3])
        apply_color_variation(conn_b, (*body_color, 255), 0.05)
        meshes.append(conn_b)

    # --- Tuning slides (U-shaped tubes behind the valves) ---
    slide_y = -0.04 * s  # behind the valve block
    for vi in range(3):
        vx = (vi - 1) * valve_spacing
        slide_depth = (0.06 + vi * 0.025) * s  # 3rd slide is longest

        # Two vertical tubes going down
        for side_x in [-0.008 * s, 0.008 * s]:
            tube_v = trimesh.creation.cylinder(radius=tube_r * 0.8, height=slide_depth, sections=8)
            tube_v.apply_translation([vx + side_x, slide_y, valve_z - valve_h / 2 - slide_depth / 2])
            apply_color_variation(tube_v, (*detail_color, 255), 0.05)
            meshes.append(tube_v)

        # U-bend at bottom
        u_bend = trimesh.creation.cylinder(
            radius=tube_r * 0.8, height=0.016 * s, sections=8)
        u_bend.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        u_bend.apply_translation([vx, slide_y, valve_z - valve_h / 2 - slide_depth])
        apply_color_variation(u_bend, (*detail_color, 255), 0.05)
        meshes.append(u_bend)

    # --- Main tuning slide (large U-shaped slide) ---
    main_slide_y = 0.04 * s  # in front
    ms_depth = 0.12 * s
    ms_width = 0.04 * s
    for side_x in [-ms_width / 2, ms_width / 2]:
        ms_tube = trimesh.creation.cylinder(radius=tube_r, height=ms_depth, sections=8)
        ms_tube.apply_translation([side_x, main_slide_y, 0.25 * s - ms_depth / 2])
        apply_color_variation(ms_tube, (*body_color, 255), 0.05)
        meshes.append(ms_tube)

    # U-bend
    ms_bend = trimesh.creation.cylinder(radius=tube_r, height=ms_width, sections=8)
    ms_bend.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    ms_bend.apply_translation([0, main_slide_y, 0.25 * s - ms_depth])
    apply_color_variation(ms_bend, (*body_color, 255), 0.05)
    meshes.append(ms_bend)

    # --- Leadpipe (from mouthpiece receiver to valve block) ---
    lp_len = 0.2 * s
    leadpipe = trimesh.creation.cylinder(radius=tube_r * 0.9, height=lp_len, sections=10)
    leadpipe.apply_translation([valve_spacing, 0, 0.25 * s - lp_len / 2 + 0.04 * s])
    apply_color_variation(leadpipe, (*body_color, 255), 0.05)
    meshes.append(leadpipe)

    # --- Mouthpiece receiver ---
    mp_recv_r = tube_r * 1.3
    mp_recv = trimesh.creation.cylinder(radius=mp_recv_r, height=0.03 * s, sections=10)
    mp_recv.apply_translation([valve_spacing, 0, 0.12 * s])
    apply_color_variation(mp_recv, (*body_color, 255), 0.04)
    meshes.append(mp_recv)

    # --- Mouthpiece ---
    # Cup
    mp_cup = trimesh.creation.cylinder(radius=0.015 * s, height=0.02 * s, sections=12)
    mp_cup.apply_translation([valve_spacing, 0, 0.095 * s])
    apply_color_variation(mp_cup, (210, 200, 180, 255), 0.03)
    meshes.append(mp_cup)

    # Rim
    mp_rim = trimesh.creation.cylinder(radius=0.017 * s, height=0.005 * s, sections=12)
    mp_rim.apply_translation([valve_spacing, 0, 0.084 * s])
    apply_color_variation(mp_rim, (220, 210, 190, 255), 0.03)
    meshes.append(mp_rim)

    # Shank (narrow tube into receiver)
    mp_shank = trimesh.creation.cylinder(radius=0.006 * s, height=0.025 * s, sections=8)
    mp_shank.apply_translation([valve_spacing, 0, 0.107 * s + 0.012 * s])
    apply_color_variation(mp_shank, (210, 200, 180, 255), 0.03)
    meshes.append(mp_shank)

    # --- Water key (small lever on main slide) ---
    wk = trimesh.creation.box(extents=[0.004 * s, 0.004 * s, 0.02 * s])
    wk.apply_translation([ms_width / 2 + 0.005 * s, main_slide_y, 0.18 * s])
    apply_color_variation(wk, (*body_color, 255), 0.05)
    meshes.append(wk)

    # --- Finger hook (on leadpipe) ---
    hook = trimesh.creation.cylinder(radius=0.003 * s, height=0.025 * s, sections=6)
    hook.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    hook.apply_translation([valve_spacing + tube_r + 0.005 * s, -0.012 * s, 0.22 * s])
    apply_color_variation(hook, (*body_color, 255), 0.05)
    meshes.append(hook)

    result = trimesh.util.concatenate(meshes)
    result.fix_normals()
    return result


# --- Registry ---

register(
    name="violin", label="Violin", category="Instruments",
    params=[
        Param("body_length", "Body Length", "float", default=1.0, min=0.5, max=2.0),
        Param("neck_length", "Neck Length", "float", default=0.6, min=0.3, max=1.2),
        Param("body_color", "Body Color", "color", default="#A05014"),
        Param("detail_color", "Detail Color", "color", default="#281E14"),
    ],
)(generate_violin)

register(
    name="trumpet", label="Trumpet", category="Instruments",
    params=[
        Param("length", "Length", "float", default=1.0, min=0.5, max=2.0),
        Param("bell_flare", "Bell Flare", "float", default=0.5, min=0.0, max=1.0),
        Param("body_color", "Body Color", "color", default="#DCBE50"),
        Param("detail_color", "Detail Color", "color", default="#B4A03C"),
    ],
)(generate_trumpet)
