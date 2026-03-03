"""
Procedural 3D Asset Generator for Games
Generates .obj and .glb files locally without internet
"""

import numpy as np
import trimesh
from pathlib import Path
import argparse
import random


def set_seed(seed: int | None = None):
    """Set random seed for reproducible generation."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


# =============================================================================
# ROCK GENERATOR
# =============================================================================

def generate_rock(
    base_size: float = 1.0,
    irregularity: float = 0.3,
    subdivisions: int = 2,
    seed: int | None = None
) -> trimesh.Trimesh:
    """
    Generate a procedural rock mesh.

    Args:
        base_size: Base size of the rock
        irregularity: How bumpy/irregular (0.0 to 1.0)
        subdivisions: Detail level (1-4 recommended)
        seed: Random seed for reproducibility

    Returns:
        trimesh.Trimesh object
    """
    set_seed(seed)

    # Start with an icosphere
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=base_size)

    # Displace vertices randomly for rocky appearance
    vertices = mesh.vertices.copy()

    # Apply noise displacement
    for i in range(len(vertices)):
        # Direction from center
        direction = vertices[i] / np.linalg.norm(vertices[i])
        # Random displacement along the direction
        displacement = np.random.uniform(-irregularity, irregularity) * base_size
        vertices[i] += direction * displacement

    # Squash slightly to make it less spherical
    vertices[:, 2] *= np.random.uniform(0.6, 0.9)

    mesh.vertices = vertices
    mesh.fix_normals()

    return mesh


def generate_rock_pile(
    num_rocks: int = 5,
    spread: float = 2.0,
    size_range: tuple = (0.3, 1.0),
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a pile of rocks."""
    set_seed(seed)

    meshes = []
    for i in range(num_rocks):
        size = np.random.uniform(*size_range)
        rock = generate_rock(base_size=size, irregularity=0.35, seed=None)

        # Random position
        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        z = size * 0.3  # Slightly above ground

        rock.apply_translation([x, y, z])
        meshes.append(rock)

    return trimesh.util.concatenate(meshes)


# =============================================================================
# TREE GENERATOR
# =============================================================================

def generate_tree(
    trunk_height: float = 2.0,
    trunk_radius: float = 0.15,
    canopy_radius: float = 1.2,
    canopy_style: str = "spherical",  # "spherical", "conical", "layered"
    seed: int | None = None
) -> trimesh.Trimesh:
    """
    Generate a procedural tree mesh.

    Args:
        trunk_height: Height of the trunk
        trunk_radius: Radius of the trunk
        canopy_radius: Radius of the foliage
        canopy_style: Shape of foliage - "spherical", "conical", or "layered"
        seed: Random seed for reproducibility

    Returns:
        trimesh.Trimesh object
    """
    set_seed(seed)

    meshes = []

    # Create trunk (cylinder with slight taper)
    trunk = trimesh.creation.cylinder(
        radius=trunk_radius,
        height=trunk_height,
        sections=8
    )
    # Move trunk so base is at origin
    trunk.apply_translation([0, 0, trunk_height / 2])
    meshes.append(trunk)

    # Create canopy based on style
    canopy_base_z = trunk_height * 0.7

    if canopy_style == "spherical":
        # Simple spherical canopy
        canopy = trimesh.creation.icosphere(subdivisions=2, radius=canopy_radius)
        # Squash and position
        canopy.vertices[:, 2] *= 0.7
        canopy.apply_translation([0, 0, canopy_base_z + canopy_radius * 0.5])
        meshes.append(canopy)

    elif canopy_style == "conical":
        # Pine tree style cone
        canopy = trimesh.creation.cone(
            radius=canopy_radius,
            height=canopy_radius * 2
        )
        canopy.apply_translation([0, 0, canopy_base_z + canopy_radius])
        meshes.append(canopy)

    elif canopy_style == "layered":
        # Multiple layers like a cartoon tree
        for i in range(3):
            layer_radius = canopy_radius * (1 - i * 0.2)
            layer = trimesh.creation.cone(
                radius=layer_radius,
                height=layer_radius * 0.8
            )
            layer_z = canopy_base_z + i * (canopy_radius * 0.5)
            layer.apply_translation([0, 0, layer_z])
            meshes.append(layer)

    return trimesh.util.concatenate(meshes)


def generate_forest_patch(
    num_trees: int = 10,
    spread: float = 10.0,
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a patch of varied trees."""
    set_seed(seed)

    styles = ["spherical", "conical", "layered"]
    meshes = []

    for i in range(num_trees):
        style = random.choice(styles)
        height = np.random.uniform(1.5, 3.0)

        tree = generate_tree(
            trunk_height=height,
            canopy_radius=height * 0.5,
            canopy_style=style
        )

        # Random position
        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        tree.apply_translation([x, y, 0])
        meshes.append(tree)

    return trimesh.util.concatenate(meshes)


# =============================================================================
# BUILDING GENERATOR
# =============================================================================

def generate_building(
    width: float = 4.0,
    depth: float = 4.0,
    height: float = 6.0,
    num_floors: int = 2,
    has_roof: bool = True,
    roof_style: str = "flat",  # "flat", "pointed", "sloped"
    seed: int | None = None
) -> trimesh.Trimesh:
    """
    Generate a simple procedural building.

    Args:
        width: Building width (X)
        depth: Building depth (Y)
        height: Total building height
        num_floors: Number of floors (for window placement reference)
        has_roof: Whether to add a roof structure
        roof_style: "flat", "pointed", or "sloped"
        seed: Random seed for reproducibility

    Returns:
        trimesh.Trimesh object
    """
    set_seed(seed)

    meshes = []

    # Main building body
    body = trimesh.creation.box(extents=[width, depth, height])
    body.apply_translation([0, 0, height / 2])
    meshes.append(body)

    # Add roof
    if has_roof:
        roof_height = height * 0.2

        if roof_style == "pointed":
            # Pyramid roof
            roof = trimesh.creation.cone(
                radius=max(width, depth) * 0.7,
                height=roof_height,
                sections=4
            )
            # Rotate to align with building
            roof.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi/4, [0, 0, 1]
            ))
            roof.apply_translation([0, 0, height + roof_height/2])
            meshes.append(roof)

        elif roof_style == "sloped":
            # Simple sloped roof (triangular prism)
            # Create a box and we'll use it as base
            roof_base = trimesh.creation.box(
                extents=[width * 1.1, depth * 1.1, 0.1]
            )
            roof_base.apply_translation([0, 0, height])
            meshes.append(roof_base)

            # Add a triangular prism for the slope
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
                [0, 1, 4],  # Front slope
                [1, 2, 5], [1, 5, 4],  # Right side
                [2, 3, 5],  # Back slope
                [3, 0, 4], [3, 4, 5],  # Left side
                [0, 3, 2], [0, 2, 1],  # Bottom
            ])

            roof = trimesh.Trimesh(vertices=vertices, faces=faces)
            roof.fix_normals()
            meshes.append(roof)

    combined = trimesh.util.concatenate(meshes)
    combined.fix_normals()

    return combined


def generate_village(
    num_buildings: int = 5,
    spread: float = 20.0,
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a small village of buildings."""
    set_seed(seed)

    roof_styles = ["flat", "pointed", "sloped"]
    meshes = []

    for i in range(num_buildings):
        width = np.random.uniform(3, 6)
        depth = np.random.uniform(3, 6)
        height = np.random.uniform(4, 10)

        building = generate_building(
            width=width,
            depth=depth,
            height=height,
            roof_style=random.choice(roof_styles)
        )

        # Random position with spacing
        x = np.random.uniform(-spread/2, spread/2)
        y = np.random.uniform(-spread/2, spread/2)
        # Random rotation
        angle = np.random.uniform(0, 2 * np.pi)

        building.apply_transform(trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1]
        ))
        building.apply_translation([x, y, 0])
        meshes.append(building)

    return trimesh.util.concatenate(meshes)


# =============================================================================
# PRIMITIVE SHAPES (for custom combinations)
# =============================================================================

def generate_crystal(
    height: float = 2.0,
    radius: float = 0.5,
    points: int = 6,
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a crystal/gem shape."""
    set_seed(seed)

    # Create a double-ended cone (octahedron-like)
    top = trimesh.creation.cone(radius=radius, height=height * 0.6, sections=points)
    bottom = trimesh.creation.cone(radius=radius, height=height * 0.4, sections=points)

    # Flip bottom cone
    bottom.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    top.apply_translation([0, 0, height * 0.2])
    bottom.apply_translation([0, 0, height * 0.2])

    crystal = trimesh.util.concatenate([top, bottom])
    crystal.fix_normals()

    return crystal


def generate_crate(
    size: float = 1.0,
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a simple crate/box with slight variation."""
    set_seed(seed)

    # Slightly varied dimensions
    w = size * np.random.uniform(0.9, 1.1)
    h = size * np.random.uniform(0.9, 1.1)
    d = size * np.random.uniform(0.9, 1.1)

    crate = trimesh.creation.box(extents=[w, d, h])
    crate.apply_translation([0, 0, h/2])

    return crate


def generate_barrel(
    height: float = 1.5,
    radius: float = 0.5,
    bulge: float = 0.1,
    seed: int | None = None
) -> trimesh.Trimesh:
    """Generate a barrel shape."""
    set_seed(seed)

    # Create cylinder with bulge in middle
    sections = 16
    height_segments = 8

    vertices = []
    faces = []

    for j in range(height_segments + 1):
        z = (j / height_segments) * height
        # Bulge factor - maximum at middle
        t = j / height_segments
        bulge_factor = 1 + bulge * np.sin(t * np.pi)
        r = radius * bulge_factor

        for i in range(sections):
            angle = (i / sections) * 2 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create faces
    for j in range(height_segments):
        for i in range(sections):
            i_next = (i + 1) % sections
            v0 = j * sections + i
            v1 = j * sections + i_next
            v2 = (j + 1) * sections + i_next
            v3 = (j + 1) * sections + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Add top and bottom caps
    bottom_center = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])
    top_center = len(vertices)
    vertices = np.vstack([vertices, [0, 0, height]])

    for i in range(sections):
        i_next = (i + 1) % sections
        # Bottom cap
        faces.append([bottom_center, i_next, i])
        # Top cap
        top_ring_start = height_segments * sections
        faces.append([top_center, top_ring_start + i, top_ring_start + i_next])

    faces = np.array(faces)

    barrel = trimesh.Trimesh(vertices=vertices, faces=faces)
    barrel.fix_normals()

    return barrel


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_mesh(mesh: trimesh.Trimesh, filepath: str, file_format: str | None = None):
    """
    Export mesh to file.

    Args:
        mesh: The trimesh object to export
        filepath: Output file path
        file_format: Force format ("obj", "glb", "stl", etc.) or auto-detect from extension
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format:
        mesh.export(filepath, file_type=file_format)
    else:
        mesh.export(filepath)

    print(f"Exported: {filepath}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Procedural 3D Asset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asset_generator.py rock -o rock.obj
  python asset_generator.py rock --irregularity 0.5 -o bumpy_rock.glb
  python asset_generator.py tree --style conical -o pine.obj
  python asset_generator.py building --roof sloped -o house.glb
  python asset_generator.py rock_pile --count 8 -o rocks.obj
  python asset_generator.py forest --count 15 -o forest.glb
  python asset_generator.py crystal -o gem.obj
  python asset_generator.py barrel -o barrel.glb
        """
    )

    parser.add_argument(
        "asset_type",
        choices=["rock", "rock_pile", "tree", "forest", "building", "village",
                 "crystal", "crate", "barrel"],
        help="Type of asset to generate"
    )

    parser.add_argument("-o", "--output", required=True, help="Output file path (.obj, .glb, .stl)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--size", type=float, default=1.0, help="Base size multiplier")

    # Rock options
    parser.add_argument("--irregularity", type=float, default=0.3,
                        help="Rock irregularity (0.0-1.0)")

    # Tree options
    parser.add_argument("--style", choices=["spherical", "conical", "layered"],
                        default="spherical", help="Tree canopy or roof style")

    # Building options
    parser.add_argument("--roof", choices=["flat", "pointed", "sloped"],
                        default="pointed", help="Roof style")
    parser.add_argument("--width", type=float, default=4.0, help="Building width")
    parser.add_argument("--height", type=float, default=6.0, help="Building/tree height")

    # Group options
    parser.add_argument("--count", type=int, default=5, help="Number of items for groups")
    parser.add_argument("--spread", type=float, default=10.0, help="Spread distance for groups")

    args = parser.parse_args()

    # Generate based on type
    if args.asset_type == "rock":
        mesh = generate_rock(
            base_size=args.size,
            irregularity=args.irregularity,
            seed=args.seed
        )
    elif args.asset_type == "rock_pile":
        mesh = generate_rock_pile(
            num_rocks=args.count,
            spread=args.spread,
            seed=args.seed
        )
    elif args.asset_type == "tree":
        mesh = generate_tree(
            trunk_height=args.height,
            canopy_radius=args.height * 0.5,
            canopy_style=args.style,
            seed=args.seed
        )
    elif args.asset_type == "forest":
        mesh = generate_forest_patch(
            num_trees=args.count,
            spread=args.spread,
            seed=args.seed
        )
    elif args.asset_type == "building":
        mesh = generate_building(
            width=args.width,
            height=args.height,
            roof_style=args.roof,
            seed=args.seed
        )
    elif args.asset_type == "village":
        mesh = generate_village(
            num_buildings=args.count,
            spread=args.spread,
            seed=args.seed
        )
    elif args.asset_type == "crystal":
        mesh = generate_crystal(height=args.size * 2, seed=args.seed)
    elif args.asset_type == "crate":
        mesh = generate_crate(size=args.size, seed=args.seed)
    elif args.asset_type == "barrel":
        mesh = generate_barrel(height=args.size * 1.5, seed=args.seed)

    export_mesh(mesh, args.output)
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")


if __name__ == "__main__":
    main()
