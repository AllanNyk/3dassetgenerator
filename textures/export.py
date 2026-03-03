"""
Texture export utilities.
Supports PNG, TGA, and other common formats.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from .generator import TextureSet

# Try to import PIL for image saving
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def save_texture(
    texture: np.ndarray,
    filepath: str,
    file_format: Optional[str] = None
) -> str:
    """
    Save a texture to file.

    Args:
        texture: Numpy array (H, W) for grayscale or (H, W, 3) for RGB
        filepath: Output path
        file_format: Force format (png, tga, etc.) or auto-detect

    Returns:
        Path to saved file
    """
    if not HAS_PIL:
        raise ImportError("PIL/Pillow is required for texture export. Install with: pip install Pillow")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect image mode from array shape
    if len(texture.shape) == 2:
        image = Image.fromarray(texture, mode='L')
    elif texture.shape[2] == 4:
        image = Image.fromarray(texture, mode='RGBA')
    else:
        image = Image.fromarray(texture, mode='RGB')

    # Determine format
    if file_format:
        fmt = file_format.upper()
    else:
        ext = path.suffix.lower()
        fmt = {
            '.png': 'PNG',
            '.tga': 'TGA',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.bmp': 'BMP',
            '.tiff': 'TIFF',
            '.tif': 'TIFF'
        }.get(ext, 'PNG')

    image.save(filepath, format=fmt)
    return str(path)


def save_texture_set(
    texture_set: TextureSet,
    output_dir: str,
    prefix: Optional[str] = None,
    file_format: str = "png"
) -> dict:
    """
    Save a complete texture set to files.

    Args:
        texture_set: TextureSet to save
        output_dir: Output directory
        prefix: Filename prefix (defaults to texture_set.name)
        file_format: Output format (png, tga, etc.)

    Returns:
        Dictionary of saved filepaths
    """
    if not HAS_PIL:
        raise ImportError("PIL/Pillow is required for texture export. Install with: pip install Pillow")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = prefix or texture_set.name
    ext = f".{file_format.lower()}"

    saved = {}

    # Save diffuse
    diffuse_path = output_path / f"{prefix}_diffuse{ext}"
    save_texture(texture_set.diffuse, str(diffuse_path), file_format)
    saved['diffuse'] = str(diffuse_path)

    # Save normal
    normal_path = output_path / f"{prefix}_normal{ext}"
    save_texture(texture_set.normal, str(normal_path), file_format)
    saved['normal'] = str(normal_path)

    # Save roughness
    roughness_path = output_path / f"{prefix}_roughness{ext}"
    save_texture(texture_set.roughness, str(roughness_path), file_format)
    saved['roughness'] = str(roughness_path)

    return saved


def create_material_file(
    texture_set: TextureSet,
    output_dir: str,
    prefix: Optional[str] = None
) -> str:
    """
    Create a simple .mtl material file for OBJ export.

    Args:
        texture_set: TextureSet to reference
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Path to .mtl file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = prefix or texture_set.name
    mtl_path = output_path / f"{prefix}.mtl"

    mtl_content = f"""# Material file for {prefix}
newmtl {prefix}_material
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.1 0.1 0.1
Ns 10.0
d 1.0
illum 2
map_Kd {prefix}_diffuse.png
map_Bump {prefix}_normal.png
"""

    with open(mtl_path, 'w') as f:
        f.write(mtl_content)

    return str(mtl_path)
