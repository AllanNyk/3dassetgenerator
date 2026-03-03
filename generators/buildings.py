"""
Building and structure generators with vertex color support.
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


def generate_building(
    width: float = 4.0,
    depth: float = 4.0,
    height: float = 6.0,
    num_floors: int = 2,
    has_roof: bool = True,
    roof_style: str = "pointed",
    wall_color: Tuple[int, int, int] = (180, 170, 150),
    roof_color: Tuple[int, int, int] = (120, 80, 60),
    color_variation: float = 0.1,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate a simple procedural building with colors.

    Args:
        width: Building width (X)
        depth: Building depth (Y)
        height: Total building height
        num_floors: Number of floors
        has_roof: Whether to add a roof
        roof_style: "flat", "pointed", or "sloped"
        wall_color: RGB color for walls
        roof_color: RGB color for roof
        color_variation: Amount of color variation (0-1)
        seed: Random seed

    Returns:
        trimesh.Trimesh object with vertex colors
    """
    set_seed(seed)

    wall_rgba = (*wall_color, 255)
    roof_rgba = (*roof_color, 255)

    meshes = []

    # Main body
    body = trimesh.creation.box(extents=[width, depth, height])
    body.apply_translation([0, 0, height / 2])
    apply_color_variation(body, wall_rgba, color_variation)
    meshes.append(body)

    if has_roof:
        roof_height = height * 0.2

        if roof_style == "pointed":
            roof = trimesh.creation.cone(
                radius=max(width, depth) * 0.7,
                height=roof_height,
                sections=4
            )
            roof.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi/4, [0, 0, 1]
            ))
            roof.apply_translation([0, 0, height + roof_height/2])
            apply_color_variation(roof, roof_rgba, color_variation)
            meshes.append(roof)

        elif roof_style == "sloped":
            roof_base = trimesh.creation.box(
                extents=[width * 1.1, depth * 1.1, 0.1]
            )
            roof_base.apply_translation([0, 0, height])
            apply_color_variation(roof_base, roof_rgba, color_variation)
            meshes.append(roof_base)

            vertices = np.array([
                [-width/2 * 1.1, -depth/2 * 1.1, 0],
                [width/2 * 1.1, -depth/2 * 1.1, 0],
                [width/2 * 1.1, depth/2 * 1.1, 0],
                [-width/2 * 1.1, depth/2 * 1.1, 0],
                [0, -depth/2 * 1.1, roof_height],
                [0, depth/2 * 1.1, roof_height],
            ])
            vertices[:, 2] += height

            faces = np.array([
                [0, 1, 4],
                [1, 2, 5], [1, 5, 4],
                [2, 3, 5],
                [3, 0, 4], [3, 4, 5],
                [0, 3, 2], [0, 2, 1],
            ])

            roof = trimesh.Trimesh(vertices=vertices, faces=faces)
            roof.fix_normals()
            apply_color_variation(roof, roof_rgba, color_variation)
            meshes.append(roof)

    combined = trimesh.util.concatenate(meshes)
    combined.fix_normals()

    return combined


def generate_village(
    num_buildings: int = 5,
    spread: float = 20.0,
    wall_color: Tuple[int, int, int] = (180, 170, 150),
    roof_color: Tuple[int, int, int] = (120, 80, 60),
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a small village of buildings with colors."""
    set_seed(seed)

    roof_styles = ["flat", "pointed", "sloped"]
    meshes = []

    for i in range(num_buildings):
        width = np.random.uniform(3, 6)
        depth = np.random.uniform(3, 6)
        height = np.random.uniform(4, 10)

        # Vary colors slightly for each building
        varied_wall = (
            int(np.clip(wall_color[0] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(wall_color[1] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(wall_color[2] + np.random.randint(-15, 15), 0, 255))
        )
        varied_roof = (
            int(np.clip(roof_color[0] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(roof_color[1] + np.random.randint(-15, 15), 0, 255)),
            int(np.clip(roof_color[2] + np.random.randint(-15, 15), 0, 255))
        )

        building = generate_building(
            width=width,
            depth=depth,
            height=height,
            roof_style=random.choice(roof_styles),
            wall_color=varied_wall,
            roof_color=varied_roof
        )

        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        angle = np.random.uniform(0, 2 * np.pi)

        building.apply_transform(trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1]
        ))
        building.apply_translation([x, y, 0])
        meshes.append(building)

    return trimesh.util.concatenate(meshes)


def generate_wall(
    length: float = 5.0,
    height: float = 2.0,
    thickness: float = 0.3,
    has_crenellations: bool = False,
    color: Tuple[int, int, int] = (140, 140, 140),
    color_variation: float = 0.1,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """
    Generate a wall segment with color.

    Args:
        length: Wall length
        height: Wall height
        thickness: Wall thickness
        has_crenellations: Add castle-style crenellations
        color: RGB color for the wall
        color_variation: Amount of color variation (0-1)
        seed: Random seed

    Returns:
        trimesh.Trimesh object with vertex colors
    """
    set_seed(seed)

    color_rgba = (*color, 255)
    meshes = []

    # Main wall
    wall = trimesh.creation.box(extents=[length, thickness, height])
    wall.apply_translation([0, 0, height / 2])
    apply_color_variation(wall, color_rgba, color_variation)
    meshes.append(wall)

    if has_crenellations:
        cren_height = height * 0.2
        cren_width = length / 10
        cren_spacing = cren_width * 2

        num_crens = int(length / cren_spacing)
        start_x = -length/2 + cren_width/2

        for i in range(num_crens):
            if i % 2 == 0:
                cren = trimesh.creation.box(
                    extents=[cren_width, thickness, cren_height]
                )
                x = start_x + i * cren_spacing
                cren.apply_translation([x, 0, height + cren_height/2])
                apply_color_variation(cren, color_rgba, color_variation)
                meshes.append(cren)

    return trimesh.util.concatenate(meshes)


def _wobble_box(extents, wobble=0.015):
    """Create a box with slight vertex displacement for handcrafted look."""
    box = trimesh.creation.box(extents=extents)
    verts = box.vertices.copy()
    max_dim = max(extents)
    for i in range(3):
        verts[:, i] += np.random.uniform(-wobble, wobble, len(verts)) * max_dim
    box.vertices = verts
    return box


def _wobble_cylinder(radius, height, sections=8, wobble=0.015):
    """Create a cylinder with slight vertex displacement for handcrafted look."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    verts = cyl.vertices.copy()
    for i in range(3):
        verts[:, i] += np.random.uniform(-wobble, wobble, len(verts)) * radius
    cyl.vertices = verts
    return cyl


def generate_fence(
    length: float = 5.0,
    height: float = 1.0,
    post_spacing: float = 1.0,
    color: Tuple[int, int, int] = (101, 67, 33),
    color_variation: float = 0.15,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a wobbly hand-built wooden fence (1870s style)."""
    set_seed(seed)

    color_rgba = (*color, 255)
    meshes = []
    post_w = 0.08
    rail_height = 0.05

    num_posts = int(length / post_spacing) + 1

    # Posts (square timber, wobbly, slightly varied heights)
    for i in range(num_posts):
        h_var = height * np.random.uniform(0.96, 1.04)
        post = _wobble_box([post_w, post_w, h_var], wobble=0.02)
        x = -length / 2 + i * post_spacing
        # Slight random lean
        lean_x = np.random.uniform(-0.01, 0.01)
        lean_y = np.random.uniform(-0.01, 0.01)
        post.apply_translation([x + lean_x, lean_y, h_var / 2])
        apply_color_variation(post, color_rgba, color_variation)
        meshes.append(post)

        # Pointed top cap
        cap = _wobble_box([post_w * 0.8, post_w * 0.8, post_w * 0.6], wobble=0.015)
        # Taper the top vertices upward
        cap_v = cap.vertices.copy()
        for vi in range(len(cap_v)):
            if cap_v[vi, 2] > 0:
                cap_v[vi, 0] *= 0.3
                cap_v[vi, 1] *= 0.3
        cap.vertices = cap_v
        cap.apply_translation([x + lean_x, lean_y, h_var + post_w * 0.2])
        apply_color_variation(cap, color_rgba, color_variation)
        meshes.append(cap)

    # Horizontal rails (wobbly planks)
    rail_positions = [height * 0.3, height * 0.7]
    for rail_z in rail_positions:
        rail = _wobble_box([length, post_w * 0.6, rail_height], wobble=0.02)
        rail.apply_translation([0, 0, rail_z + np.random.uniform(-0.01, 0.01)])
        apply_color_variation(rail, color_rgba, color_variation)
        meshes.append(rail)

    return trimesh.util.concatenate(meshes)


def generate_fence_gate(
    width: float = 1.2,
    height: float = 1.0,
    color: Tuple[int, int, int] = (101, 67, 33),
    color_variation: float = 0.15,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a wobbly hand-built wooden fence gate (1870s style)."""
    set_seed(seed)

    color_rgba = (*color, 255)
    meshes = []
    post_w = 0.10  # gate posts slightly thicker
    plank_w = 0.06
    plank_t = 0.025

    # Two gate posts (thicker than fence posts)
    for side in [-1, 1]:
        h_var = height * np.random.uniform(1.05, 1.12)  # taller than fence
        post = _wobble_box([post_w, post_w, h_var], wobble=0.02)
        post.apply_translation([side * width / 2, 0, h_var / 2])
        apply_color_variation(post, color_rgba, color_variation)
        meshes.append(post)

        # Pointed cap
        cap = _wobble_box([post_w * 0.85, post_w * 0.85, post_w * 0.7], wobble=0.015)
        cap_v = cap.vertices.copy()
        for vi in range(len(cap_v)):
            if cap_v[vi, 2] > 0:
                cap_v[vi, 0] *= 0.25
                cap_v[vi, 1] *= 0.25
        cap.vertices = cap_v
        cap.apply_translation([side * width / 2, 0, h_var + post_w * 0.25])
        apply_color_variation(cap, color_rgba, color_variation)
        meshes.append(cap)

    # Gate frame
    gate_w = width - post_w  # inner width
    gate_h = height * 0.85

    # Top rail
    top_rail = _wobble_box([gate_w, plank_t, plank_w], wobble=0.02)
    top_rail.apply_translation([0, 0, gate_h])
    apply_color_variation(top_rail, color_rgba, color_variation)
    meshes.append(top_rail)

    # Bottom rail
    bot_rail = _wobble_box([gate_w, plank_t, plank_w], wobble=0.02)
    bot_rail.apply_translation([0, 0, height * 0.15])
    apply_color_variation(bot_rail, color_rgba, color_variation)
    meshes.append(bot_rail)

    # Middle rail
    mid_rail = _wobble_box([gate_w, plank_t, plank_w], wobble=0.02)
    mid_rail.apply_translation([0, 0, gate_h * 0.5])
    apply_color_variation(mid_rail, color_rgba, color_variation)
    meshes.append(mid_rail)

    # Vertical planks (fill the gate)
    num_planks = max(4, int(gate_w / 0.12))
    plank_spacing = gate_w / num_planks
    for i in range(num_planks):
        px = -gate_w / 2 + plank_spacing * (i + 0.5)
        p_h = gate_h - height * 0.15 - plank_w  # between top and bottom rails
        plank = _wobble_box([plank_spacing * 0.85, plank_t * 0.8, p_h], wobble=0.018)
        plank.apply_translation([px, 0.001, height * 0.15 + plank_w / 2 + p_h / 2])
        apply_color_variation(plank, color_rgba, color_variation)
        meshes.append(plank)

    # Diagonal brace (Z-brace for strength)
    brace_len = np.sqrt(gate_w ** 2 + (gate_h - height * 0.15) ** 2)
    brace = _wobble_box([brace_len, plank_t * 0.7, plank_w * 0.7], wobble=0.015)
    brace_angle = np.arctan2(gate_h - height * 0.15, gate_w)
    brace.apply_transform(trimesh.transformations.rotation_matrix(brace_angle, [0, 1, 0]))
    brace.apply_translation([0, -0.005, (gate_h + height * 0.15) / 2])
    apply_color_variation(brace, color_rgba, color_variation)
    meshes.append(brace)

    # Hinges (2 metal straps on one side)
    metal_color = (55, 50, 45, 255)
    for hz in [height * 0.25, gate_h - 0.05]:
        hinge = _wobble_box([0.15, 0.008, 0.03], wobble=0.005)
        hinge.apply_translation([-gate_w / 2 + 0.05, plank_t / 2 + 0.005, hz])
        apply_color_variation(hinge, metal_color, 0.05)
        meshes.append(hinge)
        # Hinge pin
        pin = trimesh.creation.cylinder(radius=0.008, height=0.02, sections=8)
        pin.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        pin.apply_translation([-gate_w / 2, plank_t / 2 + 0.005, hz])
        apply_color_variation(pin, metal_color, 0.04)
        meshes.append(pin)

    # Latch on the other side
    latch = _wobble_box([0.06, 0.01, 0.025], wobble=0.004)
    latch.apply_translation([gate_w / 2 - 0.06, plank_t / 2 + 0.006, gate_h * 0.5])
    apply_color_variation(latch, metal_color, 0.05)
    meshes.append(latch)

    return trimesh.util.concatenate(meshes)


def generate_outhouse(
    width: float = 1.1,
    height: float = 2.2,
    depth: float = 1.1,
    color: Tuple[int, int, int] = (110, 75, 40),
    color_variation: float = 0.15,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a 1870s-style outhouse with carved heart on the door."""
    set_seed(seed)

    color_rgba = (*color, 255)
    meshes = []
    wall_t = 0.03
    roof_overhang = 0.08

    # Roof is sloped: higher at back, lower at front
    roof_h_back = height
    roof_h_front = height * 0.88

    # --- Walls (planked look using multiple wobbly boards) ---
    plank_h = 0.18
    plank_gap = 0.003

    def _build_plank_wall(w, h_left, h_right, thickness, offset_pos, axis='x'):
        """Build a wall from horizontal planks."""
        wall_meshes = []
        max_h = max(h_left, h_right)
        num_planks = int(max_h / (plank_h + plank_gap)) + 1
        for pi in range(num_planks):
            pz = pi * (plank_h + plank_gap) + plank_h / 2
            # Interpolate height for sloped walls
            if pz + plank_h / 2 > max_h:
                continue
            plank = _wobble_box([w, thickness, plank_h], wobble=0.012)
            plank.apply_translation([offset_pos[0], offset_pos[1], pz])
            apply_color_variation(plank, color_rgba, color_variation)
            wall_meshes.append(plank)
        return wall_meshes

    # Back wall
    meshes.extend(_build_plank_wall(
        width, roof_h_back, roof_h_back, wall_t,
        [0, -depth / 2 + wall_t / 2, 0]))

    # Left wall
    for pi in range(int(roof_h_back / (plank_h + plank_gap)) + 1):
        pz = pi * (plank_h + plank_gap) + plank_h / 2
        # Height at this z: interpolate front-to-back slope
        if pz + plank_h / 2 > roof_h_back:
            continue
        plank = _wobble_box([wall_t, depth, plank_h], wobble=0.012)
        plank.apply_translation([-width / 2 + wall_t / 2, 0, pz])
        apply_color_variation(plank, color_rgba, color_variation)
        meshes.append(plank)

    # Right wall
    for pi in range(int(roof_h_back / (plank_h + plank_gap)) + 1):
        pz = pi * (plank_h + plank_gap) + plank_h / 2
        if pz + plank_h / 2 > roof_h_back:
            continue
        plank = _wobble_box([wall_t, depth, plank_h], wobble=0.012)
        plank.apply_translation([width / 2 - wall_t / 2, 0, pz])
        apply_color_variation(plank, color_rgba, color_variation)
        meshes.append(plank)

    # Front wall (with door opening)
    door_w = width * 0.55
    door_h = height * 0.75
    for pi in range(int(roof_h_front / (plank_h + plank_gap)) + 1):
        pz = pi * (plank_h + plank_gap) + plank_h / 2
        if pz + plank_h / 2 > roof_h_front:
            continue
        # Skip planks in the door opening
        if pz - plank_h / 2 < door_h:
            # Left of door
            side_w = (width - door_w) / 2
            if side_w > 0.05:
                for sx, sign in [(-1, -1), (1, 1)]:
                    lp = _wobble_box([side_w, wall_t, plank_h], wobble=0.012)
                    lp.apply_translation([sign * (width / 2 - side_w / 2),
                                          depth / 2 - wall_t / 2, pz])
                    apply_color_variation(lp, color_rgba, color_variation)
                    meshes.append(lp)
        else:
            # Full width plank above door
            plank = _wobble_box([width, wall_t, plank_h], wobble=0.012)
            plank.apply_translation([0, depth / 2 - wall_t / 2, pz])
            apply_color_variation(plank, color_rgba, color_variation)
            meshes.append(plank)

    # --- Door ---
    door_t = 0.025
    # Door planks (vertical)
    num_door_planks = 5
    dp_w = door_w / num_door_planks
    for di in range(num_door_planks):
        dx = -door_w / 2 + dp_w * (di + 0.5)
        dp = _wobble_box([dp_w * 0.92, door_t, door_h], wobble=0.015)
        dp.apply_translation([dx, depth / 2 + door_t / 2, door_h / 2])
        apply_color_variation(dp, color_rgba, color_variation * 0.8)
        meshes.append(dp)

    # Door cross braces (2 horizontal)
    for bz in [door_h * 0.25, door_h * 0.7]:
        brace = _wobble_box([door_w * 0.9, 0.008, 0.04], wobble=0.01)
        brace.apply_translation([0, depth / 2 + door_t + 0.004, bz])
        apply_color_variation(brace, color_rgba, color_variation)
        meshes.append(brace)

    # --- Heart cutout on the door (carved through, double-sided) ---
    # Heart shape built from two circles + a triangle, positioned at top of door
    heart_z = door_h * 0.85
    heart_size = 0.06
    heart_verts = []
    # Heart outline: upper two bumps + bottom point
    n_pts = 24
    for i in range(n_pts):
        t = i / n_pts
        a = t * 2 * np.pi
        # Heart parametric: x = 16*sin^3(t), y = 13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t)
        hx = heart_size * 0.06 * (16 * np.sin(a) ** 3)
        hz = heart_size * 0.06 * (13 * np.cos(a) - 5 * np.cos(2 * a)
                                    - 2 * np.cos(3 * a) - np.cos(4 * a))
        heart_verts.append([hx, 0, hz])

    # Build heart as triangle fan (both sides)
    heart_center = [0, 0, 0]
    h_verts = [heart_center] + heart_verts
    h_faces = []
    for i in range(n_pts):
        ni = (i + 1) % n_pts
        # Front face
        h_faces.append([0, i + 1, ni + 1])
        # Back face
        h_faces.append([0, ni + 1, i + 1])

    h_verts = np.array(h_verts)
    # Position on door front
    h_verts[:, 1] = depth / 2 + door_t + 0.005
    h_verts[:, 0] += 0
    h_verts[:, 2] += heart_z

    heart_mesh = trimesh.Trimesh(vertices=h_verts, faces=np.array(h_faces))
    # Dark color to simulate a hole (very dark brown / black)
    apply_color_variation(heart_mesh, (25, 20, 15, 255), 0.03)
    meshes.append(heart_mesh)

    # Door handle (simple knob)
    handle = trimesh.creation.icosphere(subdivisions=2, radius=0.018)
    handle.apply_translation([door_w * 0.3, depth / 2 + door_t + 0.015, door_h * 0.5])
    apply_color_variation(handle, (55, 50, 45, 255), 0.05)
    meshes.append(handle)

    # Door hinges
    for hz in [door_h * 0.2, door_h * 0.7]:
        hinge = _wobble_box([0.01, 0.008, 0.06], wobble=0.003)
        hinge.apply_translation([-door_w / 2, depth / 2 + door_t / 2, hz])
        apply_color_variation(hinge, (55, 50, 45, 255), 0.04)
        meshes.append(hinge)

    # --- Roof (sloped, overhanging) ---
    roof_depth = depth + 2 * roof_overhang
    roof_width = width + 2 * roof_overhang
    roof_t = 0.03

    # Roof slope angle
    roof_angle = np.arctan2(roof_h_back - roof_h_front, depth)
    roof_len = depth / np.cos(roof_angle) + 2 * roof_overhang

    roof = _wobble_box([roof_width, roof_len, roof_t], wobble=0.015)
    # Tilt to slope front-to-back
    roof.apply_transform(trimesh.transformations.rotation_matrix(-roof_angle, [1, 0, 0]))
    roof_mid_h = (roof_h_back + roof_h_front) / 2 + roof_t
    roof.apply_translation([0, 0, roof_mid_h])
    # Darker roof color
    roof_color = tuple(max(0, c - 20) for c in color)
    apply_color_variation(roof, (*roof_color, 255), color_variation)
    meshes.append(roof)

    # --- Floor ---
    floor = _wobble_box([width, depth, wall_t], wobble=0.01)
    floor.apply_translation([0, 0, wall_t / 2])
    apply_color_variation(floor, color_rgba, color_variation)
    meshes.append(floor)

    # --- Seat/bench inside (the toilet seat) ---
    seat_h = 0.45
    seat_d = depth * 0.45
    seat_w = width - 2 * wall_t - 0.05

    # Seat box
    seat_front = _wobble_box([seat_w, wall_t, seat_h], wobble=0.01)
    seat_front.apply_translation([0, -depth / 2 + seat_d, seat_h / 2])
    apply_color_variation(seat_front, color_rgba, color_variation)
    meshes.append(seat_front)

    # Seat top (with hole implied by darker circle)
    seat_top = _wobble_box([seat_w, seat_d, wall_t], wobble=0.01)
    seat_top.apply_translation([0, -depth / 2 + seat_d / 2, seat_h])
    apply_color_variation(seat_top, color_rgba, color_variation)
    meshes.append(seat_top)

    # Seat hole (dark circle on top)
    hole = trimesh.creation.cylinder(radius=0.12, height=0.005, sections=16)
    hole.apply_translation([0, -depth / 2 + seat_d * 0.5, seat_h + wall_t / 2 + 0.002])
    apply_color_variation(hole, (30, 25, 20, 255), 0.03)
    meshes.append(hole)

    return trimesh.util.concatenate(meshes)


def generate_tower(
    radius: float = 2.0,
    height: float = 10.0,
    has_roof: bool = True,
    wall_color: Tuple[int, int, int] = (160, 160, 160),
    roof_color: Tuple[int, int, int] = (120, 80, 60),
    color_variation: float = 0.1,
    seed: Optional[int] = None
) -> trimesh.Trimesh:
    """Generate a cylindrical tower with colors."""
    set_seed(seed)

    wall_rgba = (*wall_color, 255)
    roof_rgba = (*roof_color, 255)
    meshes = []

    # Main tower
    tower = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=16
    )
    tower.apply_translation([0, 0, height / 2])
    apply_color_variation(tower, wall_rgba, color_variation)
    meshes.append(tower)

    if has_roof:
        roof = trimesh.creation.cone(
            radius=radius * 1.2,
            height=radius * 1.5
        )
        roof.apply_translation([0, 0, height + radius * 0.75])
        apply_color_variation(roof, roof_rgba, color_variation)
        meshes.append(roof)

    return trimesh.util.concatenate(meshes)


# --- Registry ---

register(
    name="building", label="Building", category="Buildings",
    params=[
        Param("width", "Width", "float", default=4.0, min=2.0, max=15.0),
        Param("depth", "Depth", "float", default=4.0, min=2.0, max=15.0),
        Param("height", "Height", "float", default=6.0, min=3.0, max=20.0),
        Param("roof_style", "Roof", "str", default="Pointed",
              choices=["Flat", "Pointed", "Sloped"]),
        Param("wall_color", "Walls", "color", default="#B4AA96"),
        Param("roof_color", "Roof Color", "color", default="#78503C"),
    ],
)(generate_building)

register(
    name="village", label="Village", category="Buildings",
    params=[
        Param("num_buildings", "Buildings", "int", default=5, min=2, max=15, step=1),
        Param("spread", "Spread", "float", default=20.0, min=10.0, max=50.0),
        Param("wall_color", "Walls", "color", default="#B4AA96"),
        Param("roof_color", "Roof Color", "color", default="#78503C"),
    ],
)(generate_village)

register(
    name="tower", label="Tower", category="Buildings",
    params=[
        Param("radius", "Radius", "float", default=2.0, min=1.0, max=5.0),
        Param("height", "Height", "float", default=10.0, min=5.0, max=20.0),
        Param("has_roof", "Has Roof", "bool", default=True),
        Param("wall_color", "Walls", "color", default="#A0A0A0"),
        Param("roof_color", "Roof Color", "color", default="#78503C"),
    ],
)(generate_tower)

register(
    name="wall", label="Wall", category="Buildings",
    params=[
        Param("length", "Length", "float", default=5.0, min=2.0, max=20.0),
        Param("height", "Height", "float", default=2.0, min=1.0, max=5.0),
        Param("thickness", "Thickness", "float", default=0.3, min=0.1, max=1.0),
        Param("has_crenellations", "Crenellations", "bool", default=False),
        Param("color", "Wall Color", "color", default="#8C8C8C"),
    ],
)(generate_wall)

register(
    name="fence", label="Fence", category="Buildings",
    params=[
        Param("length", "Length", "float", default=5.0, min=2.0, max=15.0),
        Param("height", "Height", "float", default=1.0, min=0.5, max=2.0),
        Param("post_spacing", "Post Spacing", "float", default=1.0, min=0.5, max=2.0),
        Param("color", "Fence Color", "color", default="#654321"),
    ],
)(generate_fence)

register(
    name="fence_gate", label="Fence Gate", category="Buildings",
    params=[
        Param("width", "Width", "float", default=1.2, min=0.8, max=2.0),
        Param("height", "Height", "float", default=1.0, min=0.5, max=2.0),
        Param("color", "Wood Color", "color", default="#654321"),
    ],
)(generate_fence_gate)

register(
    name="outhouse", label="Outhouse", category="Buildings",
    params=[
        Param("width", "Width", "float", default=1.1, min=0.8, max=1.5),
        Param("height", "Height", "float", default=2.2, min=1.8, max=2.8),
        Param("depth", "Depth", "float", default=1.1, min=0.8, max=1.5),
        Param("color", "Wood Color", "color", default="#6E4B28"),
    ],
)(generate_outhouse)


# ---------------------------------------------------------------------------
# Danish 1870s Farmhouse Generator
# ---------------------------------------------------------------------------

def _build_wall_with_opening(
    wall_w: float, wall_h: float, wall_t: float,
    open_cx: float, open_sill_z: float,
    open_w: float, open_h: float,
    outer_color: Tuple[int, int, int, int],
    inner_color: Tuple[int, int, int, int],
    variation: float = 0.04,
    wobble: float = 0.012,
) -> list:
    """Build a wall section with a rectangular opening (window or door) as 4 panels.

    Returns list of meshes. Each panel is a dual-layer slab (outer + inner).
    open_cx is the opening center X relative to the wall center.
    """
    panels = []
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
        return [outer, inner]

    # Left panel
    left_w = open_left - (-hw)
    if left_w > 0.01:
        cx = -hw + left_w / 2
        panels.extend(_dual_slab(
            [left_w, wall_t, wall_h],
            [cx, 0, wall_h / 2],
        ))

    # Right panel
    right_w = hw - open_right
    if right_w > 0.01:
        cx = open_right + right_w / 2
        panels.extend(_dual_slab(
            [right_w, wall_t, wall_h],
            [cx, 0, wall_h / 2],
        ))

    # Below opening
    below_h = open_sill_z
    if below_h > 0.01:
        panels.extend(_dual_slab(
            [open_w, wall_t, below_h],
            [open_cx, 0, below_h / 2],
        ))

    # Above opening
    above_h = wall_h - open_top
    if above_h > 0.01:
        panels.extend(_dual_slab(
            [open_w, wall_t, above_h],
            [open_cx, 0, open_top + above_h / 2],
        ))

    return panels


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
    wobble: float = 0.008,
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
            p = _wobble_box([actual_w * 0.995, depth, thickness], wobble=wobble)
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
            p = _wobble_box([width, actual_w * 0.995, thickness], wobble=wobble)
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


def generate_danish_farmhouse(
    size: float = 1.0,
    wall_color: Tuple[int, int, int] = (245, 240, 230),
    interior_color: Tuple[int, int, int] = (235, 195, 140),
    roof_color: Tuple[int, int, int] = (178, 155, 110),
    floor_color: Tuple[int, int, int] = (140, 95, 50),
    glass_alpha: int = 80,
    seed: Optional[int] = None,
) -> trimesh.Trimesh:
    """Generate a Danish 1870s farmhouse with 5 rooms, thatched roof, and traversable interior."""
    set_seed(seed)

    meshes = []

    # --- Base dimensions ---
    L = 15.0   # length along X
    D = 7.0    # depth along Y
    WALL_H = 2.5
    RIDGE_H = 5.0
    WALL_T = 0.4   # exterior wall thickness
    PART_T = 0.25  # partition wall thickness

    # Door opening dimensions for interior connections
    DOOR_W = 0.9
    DOOR_H = 2.1
    FRAME_T = 0.06  # door/window frame thickness

    # Colors
    ext_rgba = (*wall_color, 255)
    int_rgba = (*interior_color, 255)
    roof_rgba = (*roof_color, 255)
    floor_rgba = (*floor_color, 255)
    tar_rgba = (25, 22, 18, 255)
    stone_rgba = (120, 115, 105, 255)
    frame_rgba = (90, 65, 35, 255)
    chimney_rgba = (*wall_color, 255)  # white chimneys
    glass_rgba = (180, 210, 230, int(glass_alpha))  # semi-transparent pale blue

    # --- Room layout ---
    # Rooms: 0=Kitchen, 1=Entré, 2=Main room, 3=Bedroom1, 4=Bedroom2
    # The entré is room index 1, positioned between kitchen and main room
    room_widths = [3.0, 1.8, 3.2, 2.8, 2.4]
    total_interior = L - 2 * WALL_T
    raw_total = sum(room_widths)
    room_widths = [w * total_interior / raw_total for w in room_widths]
    room_x_starts = []
    x_cursor = -L / 2 + WALL_T
    for rw in room_widths:
        room_x_starts.append(x_cursor)
        x_cursor += rw
    room_centers_x = [xs + rw / 2 for xs, rw in zip(room_x_starts, room_widths)]
    # Partition wall X positions (4 walls between 5 rooms)
    partition_xs = [room_x_starts[i + 1] for i in range(4)]

    # --- 1. Foundation (no wobble, split around entrance, inset corners) ---
    found_h = 0.15
    found_t = 0.08
    entre_cx_f = -L / 2 + WALL_T + sum(room_widths[:1]) + room_widths[1] / 2
    entre_half_f = (1.0 + 2 * FRAME_T) / 2

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

    for ri in range(5):
        rx_start = room_x_starts[ri]
        rw = room_widths[ri]
        rx_center = room_centers_x[ri]

        # Entré (room 1) has a front door opening instead of a window
        if ri == 1:
            # Front wall with door opening for entré (widened + taller to clear frame)
            door_open_w = 1.0 + 2 * FRAME_T
            door_open_h = 2.1 + FRAME_T
            wall_panels = _build_wall_with_opening(
                rw, WALL_H, WALL_T,
                0, 0, door_open_w, door_open_h,
                ext_rgba, int_rgba, wobble=0,
            )
            for p in wall_panels:
                p.apply_translation([rx_center, front_y, 0])
            meshes.extend(wall_panels)
        else:
            # Normal front wall segment with window
            wall_panels = _build_wall_with_opening(
                rw, WALL_H, WALL_T,
                0, WIN_OPEN_SILL, WIN_OPEN_W, WIN_OPEN_H,
                ext_rgba, int_rgba, wobble=0,
            )
            for p in wall_panels:
                p.apply_translation([rx_center, front_y, 0])
            meshes.extend(wall_panels)

        # Back wall segment with window (all rooms)
        wall_panels = _build_wall_with_opening(
            rw, WALL_H, WALL_T,
            0, WIN_OPEN_SILL, WIN_OPEN_W, WIN_OPEN_H,
            ext_rgba, int_rgba, wobble=0,
        )
        for p in wall_panels:
            p.apply_translation([rx_center, back_y, 0])
        meshes.extend(wall_panels)

    # End walls (left and right, rectangular + gable)
    for side_x, sign in [(-L / 2 + WALL_T / 2, -1), (L / 2 - WALL_T / 2, 1)]:
        # Rectangular portion (full depth D to seal corners, no wobble)
        outer = trimesh.creation.box(extents=[WALL_T * 0.55, D, WALL_H])
        outer.apply_translation([side_x + sign * WALL_T * 0.45 / 2, 0, WALL_H / 2])
        apply_color_variation(outer, ext_rgba, 0.04)
        meshes.append(outer)
        inner = trimesh.creation.box(extents=[WALL_T * 0.45, D, WALL_H])
        inner.apply_translation([side_x - sign * WALL_T * 0.55 / 2, 0, WALL_H / 2])
        apply_color_variation(inner, int_rgba, 0.04)
        meshes.append(inner)

        # Gable (stacked boxes tapering to ridge, full depth, no wobble)
        gable_h = RIDGE_H - WALL_H
        inner_d = D
        n_gable = 8
        for gi in range(n_gable):
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

    # Tar base strip on exterior walls
    tar_h = WALL_H * 0.05
    tar_t = 0.03
    entre_cx = room_centers_x[1]
    entre_half = (1.0 + 2 * FRAME_T) / 2  # half of door opening width

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
    # Each partition gets a door opening so rooms are connected.
    # Door frame placed at center of partition (Y=0).
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
        frame_meshes = _build_door_frame(DOOR_W, DOOR_H, PART_T, frame_rgba)
        for fm in frame_meshes:
            fm.apply_translation([px, door_y, 0])
        meshes.extend(frame_meshes)

    # --- 4. Floor (plank strips per room) ---
    for ri in range(5):
        if ri == 0:
            floor_x_start = -L / 2 + WALL_T
            floor_x_end = partition_xs[0] - PART_T / 2
        elif ri == 4:
            floor_x_start = partition_xs[3] + PART_T / 2
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
    entre_plank = trimesh.creation.box(extents=[1.0 + 2 * FRAME_T, WALL_T, 0.04])
    entre_plank.apply_translation([room_centers_x[1], front_y, 0.02])
    apply_color_variation(entre_plank, floor_rgba, 0.06)
    meshes.append(entre_plank)

    # --- 4c. Solid base plane under floor to seal gaps ---
    floor_base = trimesh.creation.box(extents=[L - 2 * WALL_T, D - 2 * WALL_T, 0.01])
    floor_base.apply_translation([0, 0, -0.01])
    apply_color_variation(floor_base, floor_rgba, 0.02)
    meshes.append(floor_base)

    # --- 5. Ceiling (plank strips) ---
    ceil_color = tuple(max(0, c - 15) for c in floor_color) + (255,)
    for ri in range(5):
        if ri == 0:
            ceil_x_start = -L / 2 + WALL_T
            ceil_x_end = partition_xs[0] - PART_T / 2
        elif ri == 4:
            ceil_x_start = partition_xs[3] + PART_T / 2
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
    for ri in range(5):
        rx_center = room_centers_x[ri]

        # Back window (all rooms)
        win_meshes = _build_arched_window(
            WIN_W, WIN_H, WIN_ARCH, WIN_SILL_Z, WALL_T,
            frame_rgba, stone_rgba, glass_color=glass_rgba,
        )
        for wm in win_meshes:
            wm.apply_translation([rx_center, back_y, 0])
        meshes.extend(win_meshes)

        # Front window (skip entré room — it has the front door)
        if ri != 1:
            win_meshes = _build_arched_window(
                WIN_W, WIN_H, WIN_ARCH, WIN_SILL_Z, WALL_T,
                frame_rgba, stone_rgba, glass_color=glass_rgba,
            )
            for wm in win_meshes:
                wm.apply_translation([rx_center, front_y, 0])
            meshes.extend(win_meshes)

    # Front door frame on entré
    entre_door_frame = _build_front_door_frame(1.0, 2.1, WALL_T, frame_rgba)
    for fm in entre_door_frame:
        fm.apply_translation([room_centers_x[1], front_y, 0])
    meshes.extend(entre_door_frame)

    # --- 7. Simple gabled roof (single slab per slope, light brown for texturing) ---
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
    ridge_cap_w = 2 * ridge_gap / np.cos(roof_pitch_angle) + 0.25
    ridge = trimesh.creation.box(extents=[roof_length, ridge_cap_w, 0.1])
    ridge.apply_translation([0, 0, RIDGE_H + 0.05])
    apply_color_variation(ridge, roof_rgba, 0.04)
    meshes.append(ridge)

    # --- 8. Chimneys (2, white, 10% shorter) ---
    chimney_h = 1.35
    # Kitchen end
    chim_meshes = _build_farmhouse_chimney(0.5, 0.5, chimney_h, chimney_rgba)
    for cm in chim_meshes:
        cm.apply_translation([room_centers_x[0], 0, RIDGE_H - 0.1])
    meshes.extend(chim_meshes)

    # Last room end
    chim_meshes = _build_farmhouse_chimney(0.45, 0.45, chimney_h, chimney_rgba)
    for cm in chim_meshes:
        cm.apply_translation([room_centers_x[4], 0, RIDGE_H - 0.1])
    meshes.extend(chim_meshes)

    # --- Combine and scale ---
    combined = trimesh.util.concatenate(meshes)
    combined.fix_normals()

    if size != 1.0:
        combined.apply_scale(size)

    return combined


register(
    name="danish_farmhouse", label="Danish Farmhouse", category="Buildings",
    params=[
        Param("size", "Size", "float", default=1.0, min=0.3, max=3.0),
        Param("wall_color", "Ext. Walls", "color", default="#F5F0E6"),
        Param("interior_color", "Int. Walls", "color", default="#EBC38C"),
        Param("roof_color", "Roof Color", "color", default="#B29B6E"),
        Param("floor_color", "Floor Color", "color", default="#8C5F32"),
        Param("glass_alpha", "Glass Opacity", "int", default=80, min=0, max=255, step=5),
    ],
)(generate_danish_farmhouse)
