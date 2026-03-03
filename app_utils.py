"""
Shared utilities for the 3D Asset Generator UI.

Contains mesh saving, color parsing, and grid/axes creation.
"""

import tempfile
import re
import numpy as np
import trimesh


def _export_glb(mesh, path):
    """Export mesh to GLB, splitting into opaque/transparent parts if needed.

    If the mesh has any vertices with alpha < 255, faces are split into two
    sub-meshes: opaque (alphaMode=OPAQUE) and transparent (alphaMode=BLEND),
    exported as a Scene so each gets its own material.
    """
    from trimesh.visual.material import PBRMaterial

    # Scene already has materials set — export directly
    if isinstance(mesh, trimesh.Scene):
        mesh.export(path, file_type='glb')
        return

    has_transparency = False
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors
        if vc.shape[1] >= 4:
            has_transparency = (vc[:, 3] < 255).any()

    if not has_transparency:
        mesh.export(path, file_type='glb')
        return

    # Split faces by whether any vertex has alpha < 255
    vc = mesh.visual.vertex_colors
    face_alphas = vc[mesh.faces, 3]
    face_is_trans = np.any(face_alphas < 255, axis=1)

    opaque_idx = np.where(~face_is_trans)[0]
    trans_idx = np.where(face_is_trans)[0]

    scene = trimesh.Scene()

    if len(opaque_idx) > 0:
        sub = mesh.submesh([opaque_idx], append=True)
        sub.visual.material = PBRMaterial(alphaMode='OPAQUE')
        scene.add_geometry(sub, node_name='opaque')

    if len(trans_idx) > 0:
        sub = mesh.submesh([trans_idx], append=True)
        sub.visual.material = PBRMaterial(alphaMode='BLEND', doubleSided=True)
        scene.add_geometry(sub, node_name='transparent')

    scene.export(path, file_type='glb')


def parse_color(color_value) -> tuple:
    """Convert color value (hex or rgb string) to RGB tuple."""
    if isinstance(color_value, str):
        color_value = color_value.strip()

        # Handle hex format: #RRGGBB
        if color_value.startswith('#'):
            hex_color = color_value.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Handle rgb format: rgb(R, G, B) or rgba(R, G, B, A)
        if color_value.startswith('rgb'):
            numbers = re.findall(r'[\d.]+', color_value)
            if len(numbers) >= 3:
                return (int(float(numbers[0])), int(float(numbers[1])), int(float(numbers[2])))

    # Default gray if parsing fails
    return (128, 128, 128)


def create_grid_and_axes(grid_size: float = 5.0, grid_divisions: int = 10, unit_size: float = 1.0):
    """
    Create a ground grid with XYZ axes indicator.

    Args:
        grid_size: Total size of the grid (e.g., 5.0 = 5 units in each direction from center)
        grid_divisions: Number of divisions in the grid
        unit_size: Size of one unit (for scaling the axes)

    Returns:
        Combined mesh with grid and axes
    """
    meshes = []

    line_radius = grid_size * 0.003
    axis_radius = grid_size * 0.008

    grid_color = [80, 80, 80, 255]
    grid_color_major = [120, 120, 120, 255]

    step = (grid_size * 2) / grid_divisions

    # Grid lines parallel to X axis
    for i in range(grid_divisions + 1):
        y_pos = -grid_size + i * step
        is_center = abs(y_pos) < step * 0.1
        is_major = i % 5 == 0

        if is_center:
            continue

        line = trimesh.creation.cylinder(
            radius=line_radius * (1.5 if is_major else 1.0),
            height=grid_size * 2,
            sections=6
        )
        line.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        line.apply_translation([0, y_pos, 0])
        color = grid_color_major if is_major else grid_color
        line.visual.vertex_colors = np.tile(color, (len(line.vertices), 1))
        meshes.append(line)

    # Grid lines parallel to Y axis
    for i in range(grid_divisions + 1):
        x_pos = -grid_size + i * step
        is_center = abs(x_pos) < step * 0.1
        is_major = i % 5 == 0

        if is_center:
            continue

        line = trimesh.creation.cylinder(
            radius=line_radius * (1.5 if is_major else 1.0),
            height=grid_size * 2,
            sections=6
        )
        line.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        line.apply_translation([x_pos, 0, 0])
        color = grid_color_major if is_major else grid_color
        line.visual.vertex_colors = np.tile(color, (len(line.vertices), 1))
        meshes.append(line)

    # Measurement markers at unit intervals
    marker_positions = []
    for i in range(1, int(grid_size) + 1):
        marker_positions.extend([i, -i])

    for pos in marker_positions:
        marker = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        marker.apply_translation([pos, 0, 0])
        marker.visual.vertex_colors = np.tile([255, 100, 100, 255], (len(marker.vertices), 1))
        meshes.append(marker)

        marker = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        marker.apply_translation([0, pos, 0])
        marker.visual.vertex_colors = np.tile([100, 255, 100, 255], (len(marker.vertices), 1))
        meshes.append(marker)

    # XYZ Axes
    axis_length = grid_size * 0.6

    # X axis - Red
    x_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length, sections=8)
    x_axis.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    x_axis.apply_translation([axis_length/2, 0, 0])
    x_axis.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(x_axis.vertices), 1))
    meshes.append(x_axis)

    x_arrow = trimesh.creation.cone(radius=axis_radius*3, height=axis_radius*10, sections=8)
    x_arrow.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    x_arrow.apply_translation([axis_length, 0, 0])
    x_arrow.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(x_arrow.vertices), 1))
    meshes.append(x_arrow)

    x_label = trimesh.creation.icosphere(radius=axis_radius*4, subdivisions=1)
    x_label.apply_translation([axis_length + axis_radius*12, 0, 0])
    x_label.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(x_label.vertices), 1))
    meshes.append(x_label)

    # Y axis - Green
    y_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length, sections=8)
    y_axis.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    y_axis.apply_translation([0, axis_length/2, 0])
    y_axis.visual.vertex_colors = np.tile([0, 255, 0, 255], (len(y_axis.vertices), 1))
    meshes.append(y_axis)

    y_arrow = trimesh.creation.cone(radius=axis_radius*3, height=axis_radius*10, sections=8)
    y_arrow.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    y_arrow.apply_translation([0, axis_length, 0])
    y_arrow.visual.vertex_colors = np.tile([0, 255, 0, 255], (len(y_arrow.vertices), 1))
    meshes.append(y_arrow)

    y_label = trimesh.creation.icosphere(radius=axis_radius*4, subdivisions=1)
    y_label.apply_translation([0, axis_length + axis_radius*12, 0])
    y_label.visual.vertex_colors = np.tile([0, 255, 0, 255], (len(y_label.vertices), 1))
    meshes.append(y_label)

    # Z axis - Blue
    z_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length, sections=8)
    z_axis.apply_translation([0, 0, axis_length/2])
    z_axis.visual.vertex_colors = np.tile([0, 100, 255, 255], (len(z_axis.vertices), 1))
    meshes.append(z_axis)

    z_arrow = trimesh.creation.cone(radius=axis_radius*3, height=axis_radius*10, sections=8)
    z_arrow.apply_translation([0, 0, axis_length])
    z_arrow.visual.vertex_colors = np.tile([0, 100, 255, 255], (len(z_arrow.vertices), 1))
    meshes.append(z_arrow)

    z_label = trimesh.creation.icosphere(radius=axis_radius*4, subdivisions=1)
    z_label.apply_translation([0, 0, axis_length + axis_radius*12])
    z_label.visual.vertex_colors = np.tile([0, 100, 255, 255], (len(z_label.vertices), 1))
    meshes.append(z_label)

    # Origin marker (white sphere)
    origin = trimesh.creation.icosphere(radius=axis_radius*2, subdivisions=1)
    origin.visual.vertex_colors = np.tile([255, 255, 255, 255], (len(origin.vertices), 1))
    meshes.append(origin)

    return trimesh.util.concatenate(meshes)


def _apply_transform_to_mesh_or_scene(mesh, transform):
    """Apply a transform matrix to a Trimesh or every geometry in a Scene."""
    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            geom.apply_transform(transform)
    else:
        mesh.apply_transform(transform)


def save_mesh(mesh, name: str, show_grid: bool = False, rotation: tuple = (0, 0, 0), y_up: bool = True):
    """
    Save mesh to temp files and return paths for preview and download.

    Args:
        mesh: The trimesh object (Trimesh or Scene)
        name: Filename prefix
        show_grid: Whether to show grid and axes in preview
        rotation: (rx, ry, rz) rotation in degrees
        y_up: If True, convert from Z-up to Y-up coordinate system

    Returns:
        Tuple of (preview_path, download_path)
    """
    is_scene = isinstance(mesh, trimesh.Scene)

    # Convert from Z-up to Y-up coordinate system
    if y_up:
        _apply_transform_to_mesh_or_scene(
            mesh, trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

    # Apply user rotation if any
    rx, ry, rz = rotation
    if rx != 0 or ry != 0 or rz != 0:
        if rx != 0:
            _apply_transform_to_mesh_or_scene(
                mesh, trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))
        if ry != 0:
            _apply_transform_to_mesh_or_scene(
                mesh, trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]))
        if rz != 0:
            _apply_transform_to_mesh_or_scene(
                mesh, trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]))

    # Save clean mesh for download (without grid)
    download_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False, prefix=f"{name}_export_")
    download_path = download_file.name
    download_file.close()
    _export_glb(mesh, download_path)

    # Create preview mesh (with grid if enabled)
    if show_grid:
        bounds = mesh.bounds if not is_scene else np.array(mesh.bounds)
        if bounds is not None:
            max_dim = max(bounds[1] - bounds[0])
            grid_size = max(2.0, float(int(max_dim) + 2))
        else:
            grid_size = 5.0

        divisions = int(grid_size * 2)
        grid_and_axes = create_grid_and_axes(grid_size=grid_size, grid_divisions=divisions)
        if y_up:
            grid_and_axes.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

        if is_scene:
            mesh.add_geometry(grid_and_axes, node_name='grid')
            preview_mesh = mesh
        else:
            preview_mesh = trimesh.util.concatenate([mesh, grid_and_axes])
    else:
        preview_mesh = mesh

    # Save preview mesh
    preview_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False, prefix=f"{name}_preview_")
    preview_path = preview_file.name
    preview_file.close()
    _export_glb(preview_mesh, preview_path)

    return preview_path, download_path
