"""
Mesh utilities for creating, manipulating, and exporting 3D meshes.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple, Optional


class MeshBuilder:
    """Helper class for building meshes programmatically."""

    def __init__(self):
        self.vertices: List[np.ndarray] = []
        self.faces: List[np.ndarray] = []
        self.vertex_colors: Optional[np.ndarray] = None
        self.uv_coords: Optional[np.ndarray] = None

    def add_vertex(self, x: float, y: float, z: float) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(np.array([x, y, z]))
        return len(self.vertices) - 1

    def add_face(self, v0: int, v1: int, v2: int):
        """Add a triangular face using vertex indices."""
        self.faces.append(np.array([v0, v1, v2]))

    def add_quad(self, v0: int, v1: int, v2: int, v3: int):
        """Add a quad as two triangles."""
        self.add_face(v0, v1, v2)
        self.add_face(v0, v2, v3)

    def build(self) -> trimesh.Trimesh:
        """Build and return the trimesh object."""
        if not self.vertices or not self.faces:
            raise ValueError("Mesh has no vertices or faces")

        vertices = np.array(self.vertices)
        faces = np.array(self.faces)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if self.vertex_colors is not None:
            mesh.visual.vertex_colors = self.vertex_colors

        mesh.fix_normals()
        return mesh


def export_mesh(
    mesh: trimesh.Trimesh,
    filepath: str,
    file_format: Optional[str] = None
) -> str:
    """
    Export mesh to file.

    Args:
        mesh: The trimesh object to export
        filepath: Output file path
        file_format: Force format or auto-detect from extension

    Returns:
        The filepath that was written
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format:
        mesh.export(filepath, file_type=file_format)
    else:
        mesh.export(filepath)

    return str(path)


def combine_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Combine multiple meshes into one."""
    if not meshes:
        raise ValueError("No meshes to combine")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def apply_vertex_colors(
    mesh: trimesh.Trimesh,
    color: Tuple[int, int, int, int]
) -> trimesh.Trimesh:
    """Apply a solid color to all vertices."""
    colors = np.tile(color, (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = colors
    return mesh


def generate_lod(
    mesh: trimesh.Trimesh,
    target_faces: int
) -> trimesh.Trimesh:
    """
    Generate a lower level-of-detail version of the mesh.

    Args:
        mesh: Source mesh
        target_faces: Target number of faces

    Returns:
        Simplified mesh
    """
    # Use trimesh's built-in simplification if available
    if hasattr(mesh, 'simplify_quadric_decimation'):
        return mesh.simplify_quadric_decimation(target_faces)

    # Fallback: just return the original
    # In the future, implement custom decimation
    return mesh.copy()


def calculate_bounds(mesh: trimesh.Trimesh) -> dict:
    """Calculate mesh bounding box info."""
    bounds = mesh.bounds
    return {
        'min': bounds[0].tolist(),
        'max': bounds[1].tolist(),
        'size': (bounds[1] - bounds[0]).tolist(),
        'center': mesh.centroid.tolist()
    }


def unweld_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Duplicate vertices so each triangle has its own 3 vertices.

    Needed for per-face UV mapping (triplanar). Preserves vertex colors.
    """
    faces = mesh.faces
    verts = mesh.vertices[faces.flatten()]
    new_faces = np.arange(len(verts)).reshape(-1, 3)

    # Preserve vertex colors if present
    colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if len(vc) == len(mesh.vertices):
            colors = vc[faces.flatten()]

    new_mesh = trimesh.Trimesh(vertices=verts, faces=new_faces, process=False)
    if colors is not None:
        new_mesh.visual.vertex_colors = colors
    new_mesh.fix_normals()
    return new_mesh


def compute_triplanar_uvs(mesh: trimesh.Trimesh, scale: float = 1.0) -> np.ndarray:
    """Compute world-space UV coordinates based on face normals (triplanar).

    For each face, picks the dominant normal axis and projects the other two
    axes as UV. Requires an unwelded mesh (each vertex belongs to one face).

    Returns:
        (N, 2) array of UV coordinates, one per vertex.
    """
    face_normals = mesh.face_normals
    abs_normals = np.abs(face_normals)
    dominant = np.argmax(abs_normals, axis=1)  # 0=X, 1=Y, 2=Z

    uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
    verts = mesh.vertices

    for fi, face in enumerate(mesh.faces):
        fv = verts[face]  # (3, 3)
        d = dominant[fi]
        if d == 0:    # X-dominant -> UV = (y, z)
            uvs[face, 0] = fv[:, 1] * scale
            uvs[face, 1] = fv[:, 2] * scale
        elif d == 1:  # Y-dominant -> UV = (x, z)
            uvs[face, 0] = fv[:, 0] * scale
            uvs[face, 1] = fv[:, 2] * scale
        else:         # Z-dominant -> UV = (x, y)
            uvs[face, 0] = fv[:, 0] * scale
            uvs[face, 1] = fv[:, 1] * scale

    return uvs


def center_mesh(mesh: trimesh.Trimesh, center_z: bool = False) -> trimesh.Trimesh:
    """Center mesh at origin. Optionally keep Z base at 0."""
    centroid = mesh.centroid.copy()
    if not center_z:
        centroid[2] = mesh.bounds[0][2]  # Keep bottom at 0
    mesh.apply_translation(-centroid)
    return mesh
