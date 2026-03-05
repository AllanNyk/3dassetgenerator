"""
Procedural texture generation.
Creates diffuse, normal, and roughness maps.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from core.noise import noise_2d_grid, PerlinNoise


@dataclass
class TextureSet:
    """Container for a complete texture set."""
    diffuse: np.ndarray      # RGB diffuse/albedo map
    normal: np.ndarray       # RGB normal map
    roughness: np.ndarray    # Grayscale roughness map
    width: int
    height: int
    name: str = "texture"


def generate_diffuse(
    width: int = 512,
    height: int = 512,
    base_color: Tuple[int, int, int] = (128, 128, 128),
    variation: float = 0.2,
    noise_scale: float = 4.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a diffuse/albedo texture.

    Args:
        width, height: Texture dimensions
        base_color: RGB base color (0-255)
        variation: Color variation amount (0-1)
        noise_scale: Scale of noise pattern
        seed: Random seed

    Returns:
        RGB numpy array (height, width, 3) with uint8 values
    """
    # Generate noise
    noise = noise_2d_grid(width, height, scale=noise_scale, octaves=4, seed=seed)

    # Create RGB image
    texture = np.zeros((height, width, 3), dtype=np.uint8)

    for c in range(3):
        base = base_color[c]
        # Apply noise variation
        channel = base + (noise - 0.5) * 2 * variation * base
        channel = np.clip(channel, 0, 255)
        texture[:, :, c] = channel.astype(np.uint8)

    return texture


def generate_normal(
    height_map: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Generate a normal map from a height map.

    Args:
        height_map: 2D grayscale array (0-1 range)
        strength: Normal map strength

    Returns:
        RGB numpy array (height, width, 3) with uint8 values
    """
    h, w = height_map.shape

    # Calculate gradients
    dx = np.zeros_like(height_map)
    dy = np.zeros_like(height_map)

    # Sobel-like gradient
    dx[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) * strength
    dy[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) * strength

    # Create normal vectors
    normal = np.zeros((h, w, 3), dtype=np.float32)
    normal[:, :, 0] = -dx  # X component (red)
    normal[:, :, 1] = -dy  # Y component (green)
    normal[:, :, 2] = 1.0  # Z component (blue)

    # Normalize
    length = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    normal = normal / (length + 1e-8)

    # Convert from [-1, 1] to [0, 255]
    normal = ((normal + 1) * 0.5 * 255).astype(np.uint8)

    return normal


def generate_roughness(
    width: int = 512,
    height: int = 512,
    base_roughness: float = 0.5,
    variation: float = 0.3,
    noise_scale: float = 4.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a roughness map.

    Args:
        width, height: Texture dimensions
        base_roughness: Base roughness value (0-1)
        variation: Roughness variation amount
        noise_scale: Scale of noise pattern
        seed: Random seed

    Returns:
        Grayscale numpy array (height, width) with uint8 values
    """
    noise = noise_2d_grid(width, height, scale=noise_scale, octaves=3, seed=seed)

    roughness = base_roughness + (noise - 0.5) * 2 * variation
    roughness = np.clip(roughness, 0, 1)

    return (roughness * 255).astype(np.uint8)


def generate_texture_set(
    width: int = 512,
    height: int = 512,
    base_color: Tuple[int, int, int] = (128, 128, 128),
    color_variation: float = 0.2,
    base_roughness: float = 0.5,
    roughness_variation: float = 0.3,
    normal_strength: float = 1.0,
    noise_scale: float = 4.0,
    seed: Optional[int] = None,
    name: str = "texture"
) -> TextureSet:
    """
    Generate a complete texture set (diffuse, normal, roughness).

    Args:
        width, height: Texture dimensions
        base_color: RGB base color
        color_variation: Diffuse color variation
        base_roughness: Base roughness value
        roughness_variation: Roughness variation
        normal_strength: Normal map strength
        noise_scale: Scale of noise patterns
        seed: Random seed
        name: Texture set name

    Returns:
        TextureSet with all maps
    """
    # Generate height map for normal calculation
    height_map = noise_2d_grid(width, height, scale=noise_scale, octaves=4, seed=seed)

    diffuse = generate_diffuse(
        width, height, base_color, color_variation, noise_scale, seed
    )

    normal = generate_normal(height_map, normal_strength)

    roughness = generate_roughness(
        width, height, base_roughness, roughness_variation, noise_scale,
        seed=seed + 1 if seed else None
    )

    return TextureSet(
        diffuse=diffuse,
        normal=normal,
        roughness=roughness,
        width=width,
        height=height,
        name=name
    )


# Preset texture generators for common materials

def stone_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a stone/rock texture set."""
    return generate_texture_set(
        width, height,
        base_color=(120, 115, 105),
        color_variation=0.25,
        base_roughness=0.7,
        roughness_variation=0.2,
        normal_strength=1.5,
        noise_scale=6.0,
        seed=seed,
        name="stone"
    )


def wood_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a wood texture set."""
    return generate_texture_set(
        width, height,
        base_color=(139, 90, 43),
        color_variation=0.15,
        base_roughness=0.6,
        roughness_variation=0.15,
        normal_strength=0.8,
        noise_scale=8.0,
        seed=seed,
        name="wood"
    )


def metal_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a metal texture set."""
    return generate_texture_set(
        width, height,
        base_color=(180, 180, 190),
        color_variation=0.08,
        base_roughness=0.3,
        roughness_variation=0.15,
        normal_strength=0.5,
        noise_scale=10.0,
        seed=seed,
        name="metal"
    )


def grass_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a grass/foliage texture set."""
    return generate_texture_set(
        width, height,
        base_color=(76, 135, 45),
        color_variation=0.3,
        base_roughness=0.8,
        roughness_variation=0.1,
        normal_strength=0.6,
        noise_scale=5.0,
        seed=seed,
        name="grass"
    )


def dirt_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a dirt/soil texture set."""
    return generate_texture_set(
        width, height,
        base_color=(101, 67, 33),
        color_variation=0.2,
        base_roughness=0.9,
        roughness_variation=0.1,
        normal_strength=1.2,
        noise_scale=4.0,
        seed=seed,
        name="dirt"
    )


def thatch_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a thatched roof texture with horizontal straw streaking."""
    # Coarse noise for bundle-level color variation
    coarse = noise_2d_grid(width, height, scale=4.0, octaves=3, seed=seed)
    # Fine noise for individual straw detail
    fine = noise_2d_grid(width, height, scale=12.0, octaves=4,
                         seed=seed + 1 if seed else None)

    # Horizontal running-average on fine noise to create streak lines
    kernel_w = max(1, width // 16)
    streaked = uniform_filter1d(fine, size=kernel_w, axis=1, mode='constant', cval=0.0)

    # Blend coarse and streaked layers
    blended = 0.4 * coarse + 0.6 * streaked

    # Build diffuse from warm straw base color
    base_color = (195, 170, 110)
    variation = 0.25
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + (blended - 0.5) * 2 * variation * base_color[c]
        texture[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    normal = generate_normal(blended, strength=1.0)
    roughness = generate_roughness(width, height, base_roughness=0.85,
                                   variation=0.1, noise_scale=6.0,
                                   seed=seed + 2 if seed else None)

    return TextureSet(
        diffuse=texture, normal=normal, roughness=roughness,
        width=width, height=height, name="thatch"
    )


# =============================================================================
# VEGETATION SPRITE HELPERS
# =============================================================================

def _paint_ellipse(canvas: np.ndarray, cx: float, cy: float,
                   rx: float, ry: float, angle: float,
                   color: Tuple[int, int, int, int]) -> None:
    """Paint a rotated filled ellipse onto an RGBA canvas in-place."""
    h, w = canvas.shape[:2]
    # Bounding box for the rotated ellipse
    max_r = max(rx, ry) + 1
    y_min = max(0, int(cy - max_r))
    y_max = min(h, int(cy + max_r) + 1)
    x_min = max(0, int(cx - max_r))
    x_max = min(w, int(cx + max_r) + 1)
    if y_min >= y_max or x_min >= x_max:
        return

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    # Translate to ellipse center
    dx = xx - cx
    dy = yy - cy
    # Rotate into ellipse-local coordinates
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Ellipse equation
    mask = (lx / max(rx, 0.5))**2 + (ly / max(ry, 0.5))**2 <= 1.0
    canvas[yy[mask], xx[mask]] = color


def _paint_stem(canvas: np.ndarray, x0: float, y0: float,
                x1: float, y1: float, thickness: float,
                color: Tuple[int, int, int, int]) -> None:
    """Paint a thick line/stem between two points onto an RGBA canvas."""
    h, w = canvas.shape[:2]
    length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    if length < 1:
        return
    # Direction and normal vectors
    dx_dir = (x1 - x0) / length
    dy_dir = (y1 - y0) / length
    nx = -dy_dir * thickness * 0.5
    ny = dx_dir * thickness * 0.5

    # Bounding box
    corners_x = [x0 + nx, x0 - nx, x1 + nx, x1 - nx]
    corners_y = [y0 + ny, y0 - ny, y1 + ny, y1 - ny]
    bx_min = max(0, int(min(corners_x)) - 1)
    bx_max = min(w, int(max(corners_x)) + 2)
    by_min = max(0, int(min(corners_y)) - 1)
    by_max = min(h, int(max(corners_y)) + 2)
    if bx_min >= bx_max or by_min >= by_max:
        return

    yy, xx = np.mgrid[by_min:by_max, bx_min:bx_max]
    # Project each pixel onto the line segment
    px = xx - x0
    py = yy - y0
    t = (px * dx_dir + py * dy_dir) / length
    t = np.clip(t, 0, 1)
    # Closest point on the segment
    closest_x = x0 + t * (x1 - x0)
    closest_y = y0 + t * (y1 - y0)
    dist_sq = (xx - closest_x)**2 + (yy - closest_y)**2
    mask = dist_sq <= (thickness * 0.5)**2
    canvas[yy[mask], xx[mask]] = color


def _generate_vegetation_sprite(width: int, height: int, seed: Optional[int],
                                 paint_fn, name: str,
                                 base_roughness: float = 0.7,
                                 normal_strength: float = 0.6) -> TextureSet:
    """Orchestrator for vegetation sprite texture generation."""
    rng = np.random.RandomState(seed)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    paint_fn(canvas, rng, width, height)

    # Build height map from alpha for normal generation
    alpha = canvas[:, :, 3].astype(np.float32) / 255.0
    normal = generate_normal(alpha, normal_strength)

    # Roughness: high where plant is, zero in transparent area
    roughness = np.zeros((height, width), dtype=np.uint8)
    roughness[canvas[:, :, 3] > 0] = int(base_roughness * 255)

    return TextureSet(
        diffuse=canvas,
        normal=normal,
        roughness=roughness,
        width=width,
        height=height,
        name=name
    )


# =============================================================================
# VEGETATION SPRITE PRESETS
# =============================================================================

def wildflower_sprite(width: int = 512, height: int = 512,
                      seed: Optional[int] = None) -> TextureSet:
    """Generate a wildflower sprite texture (diffuse+alpha, normal, roughness)."""

    def _paint(canvas, rng, w, h):
        # Stem
        stem_color = (50 + rng.randint(30), 120 + rng.randint(40), 30 + rng.randint(20), 255)
        stem_top_y = int(h * (0.25 + rng.uniform(0, 0.15)))
        stem_base_x = w // 2 + rng.randint(-5, 6)
        stem_top_x = stem_base_x + rng.randint(-8, 9)
        thickness = max(2, w * 0.012)
        _paint_stem(canvas, stem_base_x, h - 1, stem_top_x, stem_top_y, thickness, stem_color)

        # 1-2 small leaves on stem
        n_leaves = rng.randint(1, 3)
        for i in range(n_leaves):
            t = 0.4 + i * 0.25 + rng.uniform(-0.05, 0.05)
            lx = stem_base_x + t * (stem_top_x - stem_base_x)
            ly = h - 1 + t * (stem_top_y - (h - 1))
            side = 1 if rng.random() > 0.5 else -1
            leaf_angle = side * (0.5 + rng.uniform(0, 0.6))
            leaf_rx = w * 0.04 + rng.uniform(0, w * 0.02)
            leaf_ry = w * 0.015
            leaf_color = (40 + rng.randint(30), 110 + rng.randint(50), 25 + rng.randint(20), 255)
            _paint_ellipse(canvas, lx + side * leaf_rx * 0.6, ly,
                           leaf_rx, leaf_ry, leaf_angle, leaf_color)

        # Petals
        petal_colors = [
            (220 + rng.randint(35), 100 + rng.randint(80), 140 + rng.randint(60)),  # pink
            (140 + rng.randint(50), 80 + rng.randint(50), 200 + rng.randint(55)),   # purple
            (230 + rng.randint(25), 210 + rng.randint(40), 50 + rng.randint(40)),   # yellow
            (220 + rng.randint(35), 220 + rng.randint(35), 220 + rng.randint(35)),  # white
            (80 + rng.randint(50), 120 + rng.randint(60), 210 + rng.randint(45)),   # blue
        ]
        petal_base = petal_colors[rng.randint(len(petal_colors))]
        n_petals = rng.randint(5, 9)
        petal_rx = w * (0.05 + rng.uniform(0, 0.03))
        petal_ry = w * (0.025 + rng.uniform(0, 0.015))
        petal_dist = w * (0.04 + rng.uniform(0, 0.02))

        for i in range(n_petals):
            angle = 2 * np.pi * i / n_petals + rng.uniform(-0.15, 0.15)
            px = stem_top_x + np.cos(angle) * petal_dist
            py = stem_top_y + np.sin(angle) * petal_dist
            # Per-petal color variation
            pc = tuple(min(255, max(0, c + rng.randint(-15, 16))) for c in petal_base) + (255,)
            _paint_ellipse(canvas, px, py, petal_rx, petal_ry, angle, pc)

        # Flower center
        center_r = w * 0.02
        center_color = (200 + rng.randint(55), 160 + rng.randint(60), 30 + rng.randint(40), 255)
        _paint_ellipse(canvas, stem_top_x, stem_top_y, center_r, center_r, 0, center_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "wildflower")


def bush_sprite(width: int = 512, height: int = 512,
                seed: Optional[int] = None) -> TextureSet:
    """Generate a bush sprite texture (diffuse+alpha, normal, roughness)."""

    def _paint(canvas, rng, w, h):
        # Short brown stem at bottom
        stem_color = (90 + rng.randint(30), 60 + rng.randint(20), 30 + rng.randint(15), 255)
        stem_x = w // 2
        stem_top = int(h * 0.7)
        thickness = max(3, w * 0.02)
        _paint_stem(canvas, stem_x, h - 1, stem_x, stem_top, thickness, stem_color)

        # Dense dome of overlapping leaf ellipses
        dome_cx = w * 0.5
        dome_cy = h * 0.45
        dome_rx = w * 0.35
        dome_ry = h * 0.3
        n_leaves = 60 + rng.randint(40)

        for _ in range(n_leaves):
            # Random position within the dome
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, 1) ** 0.5  # sqrt for uniform distribution in disk
            lx = dome_cx + np.cos(a) * r * dome_rx * 0.85
            ly = dome_cy + np.sin(a) * r * dome_ry * 0.85
            leaf_rx = w * (0.03 + rng.uniform(0, 0.03))
            leaf_ry = w * (0.015 + rng.uniform(0, 0.015))
            leaf_angle = rng.uniform(0, np.pi)
            # Dark-to-medium green
            g = 80 + rng.randint(80)
            leaf_color = (30 + rng.randint(30), g, 20 + rng.randint(25), 255)
            _paint_ellipse(canvas, lx, ly, leaf_rx, leaf_ry, leaf_angle, leaf_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "bush")


def fern_sprite(width: int = 512, height: int = 512,
                seed: Optional[int] = None) -> TextureSet:
    """Generate a fern sprite texture (diffuse+alpha, normal, roughness)."""

    def _paint(canvas, rng, w, h):
        # Central stem with slight curve
        stem_color = (50 + rng.randint(25), 100 + rng.randint(40), 30 + rng.randint(20), 255)
        stem_base_x = w * 0.5
        stem_top_y = int(h * 0.1)
        curve = rng.uniform(-w * 0.04, w * 0.04)
        n_segments = 12
        thickness = max(2, w * 0.01)

        # Build stem points
        stem_points = []
        for i in range(n_segments + 1):
            t = i / n_segments
            sx = stem_base_x + curve * np.sin(t * np.pi)
            sy = h - 1 + t * (stem_top_y - (h - 1))
            stem_points.append((sx, sy))

        # Draw stem segments
        for i in range(n_segments):
            _paint_stem(canvas, stem_points[i][0], stem_points[i][1],
                        stem_points[i+1][0], stem_points[i+1][1],
                        thickness, stem_color)

        # Alternating leaflet pairs along the stem
        n_pairs = 8 + rng.randint(5)
        for i in range(n_pairs):
            t = 0.15 + 0.75 * (i / n_pairs)
            seg_idx = int(t * n_segments)
            seg_idx = min(seg_idx, n_segments - 1)
            frac = t * n_segments - seg_idx
            sx = stem_points[seg_idx][0] + frac * (stem_points[seg_idx+1][0] - stem_points[seg_idx][0])
            sy = stem_points[seg_idx][1] + frac * (stem_points[seg_idx+1][1] - stem_points[seg_idx][1])

            # Taper: leaflets get smaller toward tip
            taper = 1.0 - t * 0.7
            leaflet_rx = w * 0.06 * taper + rng.uniform(0, w * 0.015)
            leaflet_ry = w * 0.018 * taper

            for side in [-1, 1]:
                angle = side * (0.4 + rng.uniform(0, 0.3))
                lx = sx + side * leaflet_rx * 0.5
                # Color: lighter toward tips
                g = int(100 + 60 * (1 - taper) + rng.randint(-15, 16))
                g = min(255, max(0, g))
                leaflet_color = (30 + rng.randint(25), g, 25 + rng.randint(20), 255)
                _paint_ellipse(canvas, lx, sy, leaflet_rx, leaflet_ry, angle, leaflet_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "fern")


def grass_tuft_sprite(width: int = 512, height: int = 512,
                      seed: Optional[int] = None) -> TextureSet:
    """Generate a grass tuft sprite texture (diffuse+alpha, normal, roughness)."""

    def _paint(canvas, rng, w, h):
        n_blades = rng.randint(5, 10)
        base_x = w * 0.5
        base_y = h - 1

        for _ in range(n_blades):
            # Fan angle from center
            fan_angle = rng.uniform(-0.6, 0.6)
            blade_height = h * (0.4 + rng.uniform(0, 0.45))
            # Tip position
            tip_x = base_x + np.sin(fan_angle) * blade_height + rng.uniform(-w * 0.03, w * 0.03)
            tip_y = base_y - blade_height

            # Blade as tapered stem (thicker at base, thinner at tip)
            # Draw multiple segments for slight curve
            n_seg = 6
            curve = rng.uniform(-w * 0.03, w * 0.03)
            # Color variety
            g = 90 + rng.randint(100)
            r = 40 + rng.randint(50)
            blade_color = (r, min(255, g), 20 + rng.randint(30), 255)

            prev_x, prev_y = base_x + rng.uniform(-3, 3), base_y
            for s in range(n_seg):
                t = (s + 1) / n_seg
                sx = prev_x + t * (tip_x - base_x) / n_seg + curve * np.sin(t * np.pi) / n_seg
                sy = base_y + t * (tip_y - base_y)
                thickness = max(1, w * 0.012 * (1 - t * 0.7))
                _paint_stem(canvas, prev_x, prev_y, sx, sy, thickness, blade_color)
                prev_x, prev_y = sx, sy

    return _generate_vegetation_sprite(width, height, seed, _paint, "grass_tuft")


def roof_tile_texture(width: int = 256, height: int = 256,
                      seed: Optional[int] = None) -> TextureSet:
    """Generate a terracotta roof tile texture set."""
    base = np.array([178, 75, 48], dtype=np.float32)

    tile_h = max(4, height // 16)
    tile_w = max(8, width // 8)

    yy, xx = np.mgrid[0:height, 0:width]
    row = yy // tile_h
    offset = np.where(row % 2 == 1, tile_w // 2, 0)
    tile_col = (xx + offset) // tile_w

    # Per-tile color variation via integer hash
    tile_id = (row * 997 + tile_col).astype(np.int64)
    h1 = ((tile_id * 2654435761) % (2**31)).astype(np.float32) / (2**31)
    h2 = ((tile_id * 2246822507 + 12345) % (2**31)).astype(np.float32) / (2**31)
    h3 = ((tile_id * 3266489917 + 67890) % (2**31)).astype(np.float32) / (2**31)

    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    diffuse[:, :, 0] = base[0] + (h1 * 30 - 15)
    diffuse[:, :, 1] = base[1] + (h2 * 20 - 10)
    diffuse[:, :, 2] = base[2] + (h3 * 20 - 10)

    # Noise overlay for extra grit
    noise = noise_2d_grid(width, height, scale=8.0, octaves=2, seed=seed)
    for i in range(3):
        diffuse[:, :, i] += noise * 10

    # Darken tile edges
    row_pos = (yy % tile_h).astype(np.float32) / tile_h
    col_pos = ((xx + offset) % tile_w).astype(np.float32) / tile_w
    edge = ((row_pos < 0.08) | (row_pos > 0.92) |
            (col_pos < 0.04) | (col_pos > 0.96)).astype(np.float32) * 0.18
    for i in range(3):
        diffuse[:, :, i] *= (1.0 - edge)

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # Height map: curved tile profile per row
    height_map = np.sin(row_pos * np.pi) * 0.5

    normal = generate_normal(height_map, strength=1.5)
    roughness = np.clip(165 + noise * 20, 0, 255).astype(np.uint8)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="roof_tile")


def oak_leaf_sprite(width: int = 256, height: int = 256,
                    seed: Optional[int] = None) -> TextureSet:
    """Generate an oak leaf cluster sprite for billboard tree canopies."""

    def _paint(canvas, rng, w, h):
        n_leaves = 15 + rng.randint(12)
        for _ in range(n_leaves):
            cx = w * (0.15 + rng.uniform(0, 0.7))
            cy = h * (0.15 + rng.uniform(0, 0.7))
            # Oak leaves: elongated with lobed shape (approximated by overlapping ellipses)
            rx = w * (0.06 + rng.uniform(0, 0.05))
            ry = w * (0.03 + rng.uniform(0, 0.02))
            angle = rng.uniform(0, np.pi)
            g = 70 + rng.randint(90)
            leaf_color = (25 + rng.randint(30), g, 15 + rng.randint(20), 255)
            _paint_ellipse(canvas, cx, cy, rx, ry, angle, leaf_color)
            # Add lobe bumps along the leaf
            for lobe in range(rng.randint(2, 5)):
                lt = rng.uniform(-0.8, 0.8)
                lx = cx + lt * rx * np.cos(angle)
                ly = cy + lt * rx * np.sin(angle)
                lobe_rx = rx * rng.uniform(0.3, 0.6)
                lobe_ry = ry * rng.uniform(0.5, 1.0)
                lobe_color = (leaf_color[0] + rng.randint(-8, 9),
                              min(255, leaf_color[1] + rng.randint(-10, 11)),
                              leaf_color[2] + rng.randint(-5, 6), 255)
                _paint_ellipse(canvas, lx, ly, lobe_rx, lobe_ry,
                               angle + rng.uniform(-0.4, 0.4), lobe_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "oak_leaf")


def pine_needle_sprite(width: int = 256, height: int = 256,
                       seed: Optional[int] = None) -> TextureSet:
    """Generate a pine needle cluster sprite for billboard tree canopies."""

    def _paint(canvas, rng, w, h):
        n_clusters = 12 + rng.randint(9)  # 12-20 clusters
        for _ in range(n_clusters):
            cx = w * (0.10 + rng.uniform(0, 0.80))
            cy = h * (0.10 + rng.uniform(0, 0.80))
            n_needles = 6 + rng.randint(7)  # 6-12 needles per cluster
            needle_len = w * (0.04 + rng.uniform(0, 0.03))

            for _ in range(n_needles):
                angle = rng.uniform(0, 2 * np.pi)
                tip_x = cx + np.cos(angle) * needle_len
                tip_y = cy + np.sin(angle) * needle_len
                # Dark green needle color
                r = 15 + rng.randint(26)   # 15-40
                g = 80 + rng.randint(81)   # 80-160
                b = 10 + rng.randint(21)   # 10-30
                needle_color = (r, g, b, 255)
                thickness = max(1, w * 0.006)
                _paint_stem(canvas, cx, cy, tip_x, tip_y, thickness, needle_color)

            # Small dark bud ellipse at cluster center
            bud_r = w * 0.008 + rng.uniform(0, w * 0.004)
            bud_color = (30 + rng.randint(20), 50 + rng.randint(30),
                         15 + rng.randint(15), 255)
            _paint_ellipse(canvas, cx, cy, bud_r, bud_r, 0, bud_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "pine_needle")


def apple_leaf_sprite(width: int = 256, height: int = 256,
                      seed: Optional[int] = None) -> TextureSet:
    """Generate an apple leaf cluster sprite for billboard tree canopies."""

    def _paint(canvas, rng, w, h):
        n_leaves = 20 + rng.randint(11)  # 20-30 leaves
        for _ in range(n_leaves):
            cx = w * (0.12 + rng.uniform(0, 0.76))
            cy = h * (0.12 + rng.uniform(0, 0.76))
            # Apple leaves: smooth ovals, rounder than oak (rx/ry ~2:1)
            rx = w * (0.05 + rng.uniform(0, 0.04))
            ry = rx * rng.uniform(0.4, 0.6)  # ~2:1 ratio
            angle = rng.uniform(0, np.pi)
            # Lighter/yellower green than oak
            r = 30 + rng.randint(31)   # 30-60
            g = 100 + rng.randint(81)  # 100-180
            b = 15 + rng.randint(25)   # 15-39
            leaf_color = (r, g, b, 255)
            _paint_ellipse(canvas, cx, cy, rx, ry, angle, leaf_color)

        # A few thin stems connecting some leaves
        n_stems = 3 + rng.randint(3)
        for _ in range(n_stems):
            x0 = w * (0.3 + rng.uniform(0, 0.4))
            y0 = h * (0.5 + rng.uniform(0, 0.3))
            x1 = x0 + rng.uniform(-w * 0.12, w * 0.12)
            y1 = y0 - rng.uniform(w * 0.05, w * 0.15)
            stem_color = (50 + rng.randint(25), 90 + rng.randint(40),
                          25 + rng.randint(15), 255)
            thickness = max(1, w * 0.008)
            _paint_stem(canvas, x0, y0, x1, y1, thickness, stem_color)

    return _generate_vegetation_sprite(width, height, seed, _paint, "apple_leaf")


def white_stone_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a white/cream stone texture set."""
    return generate_texture_set(
        width, height,
        base_color=(235, 230, 220),
        color_variation=0.25,
        base_roughness=0.7,
        roughness_variation=0.2,
        normal_strength=1.5,
        noise_scale=6.0,
        seed=seed,
        name="white_stone"
    )


def dark_thatch_texture(width: int = 512, height: int = 512, seed: Optional[int] = None) -> TextureSet:
    """Generate a dark thatched roof texture with horizontal straw streaking."""
    coarse = noise_2d_grid(width, height, scale=4.0, octaves=3, seed=seed)
    fine = noise_2d_grid(width, height, scale=12.0, octaves=4,
                         seed=seed + 1 if seed else None)

    kernel_w = max(1, width // 16)
    streaked = uniform_filter1d(fine, size=kernel_w, axis=1, mode='constant', cval=0.0)

    blended = 0.4 * coarse + 0.6 * streaked

    base_color = (140, 120, 75)
    variation = 0.25
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + (blended - 0.5) * 2 * variation * base_color[c]
        texture[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    normal = generate_normal(blended, strength=1.0)
    roughness = generate_roughness(width, height, base_roughness=0.85,
                                   variation=0.1, noise_scale=6.0,
                                   seed=seed + 2 if seed else None)

    return TextureSet(
        diffuse=texture, normal=normal, roughness=roughness,
        width=width, height=height, name="dark_thatch"
    )


def brick_texture(width: int = 512, height: int = 512,
                   seed: Optional[int] = None) -> TextureSet:
    """Generate a running-bond brick texture set."""
    tile_h = max(4, height // 12)
    tile_w = max(8, width // 6)

    yy, xx = np.mgrid[0:height, 0:width]
    row = yy // tile_h
    offset = np.where(row % 2 == 1, tile_w // 2, 0)
    tile_col = (xx + offset) // tile_w

    # Per-brick color variation via integer hash
    tile_id = (row * 997 + tile_col).astype(np.int64)
    h1 = ((tile_id * 2654435761) % (2**31)).astype(np.float32) / (2**31)
    h2 = ((tile_id * 2246822507 + 12345) % (2**31)).astype(np.float32) / (2**31)
    h3 = ((tile_id * 3266489917 + 67890) % (2**31)).astype(np.float32) / (2**31)

    base = np.array([165, 70, 50], dtype=np.float32)
    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    diffuse[:, :, 0] = base[0] + (h1 * 30 - 15)
    diffuse[:, :, 1] = base[1] + (h2 * 20 - 10)
    diffuse[:, :, 2] = base[2] + (h3 * 16 - 8)

    # Noise overlay for grit/weathering
    noise = noise_2d_grid(width, height, scale=8.0, octaves=2, seed=seed)
    for i in range(3):
        diffuse[:, :, i] += noise * 10

    # Mortar joints: darken brick edges, blend in lighter mortar color
    row_pos = (yy % tile_h).astype(np.float32) / tile_h
    col_pos = ((xx + offset) % tile_w).astype(np.float32) / tile_w
    edge = ((row_pos < 0.08) | (row_pos > 0.92) |
            (col_pos < 0.04) | (col_pos > 0.96)).astype(np.float32)
    mortar_color = np.array([180, 175, 165], dtype=np.float32)
    for i in range(3):
        diffuse[:, :, i] = diffuse[:, :, i] * (1.0 - edge) + mortar_color[i] * edge

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # Height map: flat brick face with recessed mortar grooves
    height_map = np.ones((height, width), dtype=np.float32) * 0.6
    height_map[edge > 0] = 0.2
    height_map += noise * 0.1

    normal = generate_normal(height_map, strength=1.5)
    roughness_base = np.full((height, width), 190, dtype=np.float32)
    roughness_base[edge > 0] = 210  # rougher mortar
    roughness_base += noise * 15
    roughness = np.clip(roughness_base, 0, 255).astype(np.uint8)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="brick")


def white_brick_texture(width: int = 512, height: int = 512,
                        seed: Optional[int] = None) -> TextureSet:
    """Generate a white/cream masonry brick texture set."""
    tile_h = max(4, height // 12)
    tile_w = max(8, width // 6)

    yy, xx = np.mgrid[0:height, 0:width]
    row = yy // tile_h
    offset = np.where(row % 2 == 1, tile_w // 2, 0)
    tile_col = (xx + offset) // tile_w

    # Per-brick color variation via integer hash
    tile_id = (row * 997 + tile_col).astype(np.int64)
    h1 = ((tile_id * 2654435761) % (2**31)).astype(np.float32) / (2**31)
    h2 = ((tile_id * 2246822507 + 12345) % (2**31)).astype(np.float32) / (2**31)
    h3 = ((tile_id * 3266489917 + 67890) % (2**31)).astype(np.float32) / (2**31)

    base = np.array([235, 228, 215], dtype=np.float32)
    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    diffuse[:, :, 0] = base[0] + (h1 * 16 - 8)
    diffuse[:, :, 1] = base[1] + (h2 * 14 - 7)
    diffuse[:, :, 2] = base[2] + (h3 * 12 - 6)

    # Noise overlay for grit/weathering
    noise = noise_2d_grid(width, height, scale=8.0, octaves=2, seed=seed)
    for i in range(3):
        diffuse[:, :, i] += noise * 8

    # Mortar joints
    row_pos = (yy % tile_h).astype(np.float32) / tile_h
    col_pos = ((xx + offset) % tile_w).astype(np.float32) / tile_w
    edge = ((row_pos < 0.08) | (row_pos > 0.92) |
            (col_pos < 0.04) | (col_pos > 0.96)).astype(np.float32)
    mortar_color = np.array([200, 195, 185], dtype=np.float32)
    for i in range(3):
        diffuse[:, :, i] = diffuse[:, :, i] * (1.0 - edge) + mortar_color[i] * edge

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # Height map: flat brick face with recessed mortar grooves
    height_map = np.ones((height, width), dtype=np.float32) * 0.6
    height_map[edge > 0] = 0.2
    height_map += noise * 0.1

    normal = generate_normal(height_map, strength=1.2)
    roughness_base = np.full((height, width), 180, dtype=np.float32)
    roughness_base[edge > 0] = 200
    roughness_base += noise * 12
    roughness = np.clip(roughness_base, 0, 255).astype(np.uint8)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="white_brick")


def bark_texture(width: int = 512, height: int = 512,
                 seed: Optional[int] = None) -> TextureSet:
    """Generate a tree bark texture with vertical grain."""
    # Coarse noise for large ridges
    coarse = noise_2d_grid(width, height, scale=3.0, octaves=3, seed=seed)
    # Fine noise for detail
    fine = noise_2d_grid(width, height, scale=14.0, octaves=4,
                         seed=seed + 1 if seed else None)

    # Vertical running-average on fine noise to create vertical streaks
    kernel_h = max(1, height // 10)
    streaked = uniform_filter1d(fine, size=kernel_h, axis=0, mode='constant', cval=0.0)

    # Blend: more coarse than thatch for chunky bark ridges
    blended = 0.5 * coarse + 0.5 * streaked

    # Build diffuse from dark brown base color
    base_color = (85, 60, 35)
    variation = 0.30
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + (blended - 0.5) * 2 * variation * base_color[c]
        texture[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    normal = generate_normal(blended, strength=2.0)
    roughness = generate_roughness(width, height, base_roughness=0.9,
                                   variation=0.08, noise_scale=6.0,
                                   seed=seed + 2 if seed else None)

    return TextureSet(
        diffuse=texture, normal=normal, roughness=roughness,
        width=width, height=height, name="bark"
    )


def cobblestone_texture(width: int = 512, height: int = 512,
                        seed: Optional[int] = None) -> TextureSet:
    """Generate an organic cobblestone pavement texture set.

    Uses jittered Voronoi cells for irregular stone placement with
    per-stone color variation, domed height profiles, and variable-width
    mortar gaps.
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)

    # --- Voronoi cell setup (jittered grid, tileable) ---
    cell_size = max(8, min(width, height) // 16)
    n_cx = width // cell_size
    n_cy = height // cell_size

    # Jittered cell centers with hex offset for organic layout
    jx = rng.uniform(-0.4, 0.4, (n_cy, n_cx))
    jy = rng.uniform(-0.4, 0.4, (n_cy, n_cx))
    base_points = []
    for j in range(n_cy):
        hex_off = 0.5 if j % 2 else 0.0
        for i in range(n_cx):
            px = (i + hex_off + 0.5 + jx[j, i]) * cell_size
            py = (j + 0.5 + jy[j, i]) * cell_size
            base_points.append((px, py))
    n_base = len(base_points)
    base_points = np.array(base_points, dtype=np.float64)

    # Toroidal copies: replicate base points in 3x3 grid of tiles
    # so KDTree finds correct nearest neighbors across edges
    all_points = []
    for dy in (-height, 0, height):
        for dx in (-width, 0, width):
            shifted = base_points + np.array([dx, dy], dtype=np.float64)
            all_points.append(shifted)
    all_points = np.concatenate(all_points, axis=0)

    # KDTree for exact nearest / second-nearest (no grid artifacts)
    tree = KDTree(all_points)
    yy, xx = np.mgrid[0:height, 0:width]
    pixel_coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    dists, indices = tree.query(pixel_coords, k=2)

    d1 = dists[:, 0].reshape(height, width).astype(np.float32)
    d2 = dists[:, 1].reshape(height, width).astype(np.float32)
    # Map all toroidal copies back to the same base cell ID
    cell_id = (indices[:, 0] % n_base).reshape(height, width)

    # --- Gap/mortar detection ---
    edge_factor = d2 - d1
    gap_noise = noise_2d_grid(width, height, scale=6.0, octaves=2,
                              seed=seed + 3 if seed else None)
    gap_threshold = 1.5 + gap_noise * 1.5
    is_gap = edge_factor < gap_threshold
    gap_blend = np.clip(1.0 - edge_factor / (gap_threshold + 1e-6), 0, 1)
    gap_blend = gap_blend ** 0.5  # soften transition

    # --- Normalized distance within each stone (0=center, 1=edge) ---
    stone_radius = d2 * 0.5
    norm_dist = np.clip(d1 / (stone_radius + 1e-6), 0, 1)

    # --- Per-stone hashes for color/height variation ---
    cid64 = cell_id.astype(np.int64)
    h1 = ((cid64 * 2654435761) % (2**31)).astype(np.float32) / (2**31)
    h2 = ((cid64 * 2246822507 + 12345) % (2**31)).astype(np.float32) / (2**31)
    h3 = ((cid64 * 3266489917 + 67890) % (2**31)).astype(np.float32) / (2**31)
    h4 = ((cid64 * 1640531527 + 999983) % (2**31)).astype(np.float32) / (2**31)

    # --- Diffuse map ---
    base = np.array([140, 135, 125], dtype=np.float32)
    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    # Per-stone brightness shift (correlated across channels for natural gray variation)
    brightness = h1 * 30 - 15  # shared lightness shift
    diffuse[:, :, 0] = base[0] + brightness + (h2 * 10 - 5)  # slight warm/cool
    diffuse[:, :, 1] = base[1] + brightness + (h3 * 8 - 4)
    diffuse[:, :, 2] = base[2] + brightness - (h2 * 6 - 3)   # inverse for warm bias

    # Fine noise for surface grit
    grit_noise = noise_2d_grid(width, height, scale=10.0, octaves=3,
                               seed=seed)
    for c in range(3):
        diffuse[:, :, c] += grit_noise * 10

    # Broad weathering patches (subtle)
    broad_noise = noise_2d_grid(width, height, scale=2.0, octaves=2,
                                seed=seed + 5 if seed else None)
    diffuse[:, :, 0] += (broad_noise - 0.5) * 8
    diffuse[:, :, 2] -= (broad_noise - 0.5) * 5

    # Edge darkening (grime collects at stone edges)
    edge_darken = np.clip(norm_dist ** 2 * 0.15, 0, 0.15)
    for c in range(3):
        diffuse[:, :, c] *= (1.0 - edge_darken)

    # Mortar/gap color blend
    mortar_color = np.array([95, 85, 70], dtype=np.float32)
    for c in range(3):
        diffuse[:, :, c] = diffuse[:, :, c] * (1.0 - gap_blend) + mortar_color[c] * gap_blend

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # --- Height map (domed stones + recessed mortar) ---
    dome = np.cos(norm_dist * np.pi * 0.5)
    dome = np.clip(dome, 0, 1)

    # Per-stone height offset
    stone_h_offset = h4 * 0.3 - 0.15
    height_map = dome * 0.6 + stone_h_offset + 0.3
    height_map = height_map * (1.0 - gap_blend) + 0.1 * gap_blend

    # Fine surface bumps
    fine_noise = noise_2d_grid(width, height, scale=16.0, octaves=2,
                               seed=seed + 2 if seed else None)
    height_map += fine_noise * 0.08
    height_map = np.clip(height_map, 0, 1).astype(np.float32)

    normal = generate_normal(height_map, strength=2.0)

    # --- Roughness map ---
    h5 = ((cid64 * 2996863034 + 54321) % (2**31)).astype(np.float32) / (2**31)
    rough_base = np.full((height, width), 175, dtype=np.float32)
    rough_base += h5 * 30 - 15              # per-stone variation
    rough_base += norm_dist * 20            # rougher toward edges
    rough_base = rough_base * (1.0 - gap_blend) + 220 * gap_blend  # mortar roughest
    rough_base += fine_noise * 15
    roughness = np.clip(rough_base, 0, 255).astype(np.uint8)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="cobblestone")


def lawn_grass_texture(width: int = 512, height: int = 512,
                       seed: Optional[int] = None) -> TextureSet:
    """Generate a natural lawn grass texture set.

    Uses multi-layer directional noise for blade-like streaks with
    clump variation and subtle dead-patch coloring. Seamlessly tileable.
    """
    # --- Blade streaks (vertical directional noise) ---
    # Fine noise for individual blade detail
    blade_noise = noise_2d_grid(width, height, scale=14.0, octaves=4, seed=seed)
    # Vertical streak: blur along Y axis to simulate blade direction
    blade_kernel = max(1, height // 12)
    streaked = uniform_filter1d(blade_noise, size=blade_kernel, axis=0,
                                mode='wrap')

    # Second blade layer at different frequency for depth
    blade2 = noise_2d_grid(width, height, scale=20.0, octaves=3,
                           seed=seed + 1 if seed else None)
    streaked2 = uniform_filter1d(blade2, size=max(1, height // 18), axis=0,
                                 mode='wrap')

    # Coarse noise for clump-level variation (patches of thicker/thinner grass)
    clump = noise_2d_grid(width, height, scale=3.0, octaves=2,
                          seed=seed + 2 if seed else None)

    # Blend layers: fine blades dominate, clump modulates
    blended = 0.35 * streaked + 0.35 * streaked2 + 0.30 * clump

    # --- Diffuse map ---
    # Base green with warm/cool variation per clump
    base_g = np.array([55, 120, 35], dtype=np.float32)   # healthy green
    brown_g = np.array([95, 105, 45], dtype=np.float32)   # dried/yellow patches

    # Clump noise drives green-to-brown blend
    brown_blend = np.clip((clump - 0.55) * 3.0, 0, 1)  # patches go brownish

    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        base_val = base_g[c] * (1.0 - brown_blend) + brown_g[c] * brown_blend
        diffuse[:, :, c] = base_val + (blended - 0.5) * 2 * 0.25 * base_val

    # Fine brightness variation for individual blade contrast
    diffuse[:, :, 1] += (blade_noise - 0.5) * 20  # green channel extra variation

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # --- Height map (blade tips create subtle relief) ---
    height_map = 0.5 * streaked + 0.3 * streaked2 + 0.2 * blade_noise
    normal = generate_normal(height_map, strength=0.8)

    # --- Roughness (grass is fairly rough/matte) ---
    roughness = generate_roughness(width, height, base_roughness=0.85,
                                   variation=0.08, noise_scale=6.0,
                                   seed=seed + 3 if seed else None)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="lawn_grass")


def gravel_texture(width: int = 512, height: int = 512,
                   seed: Optional[int] = None) -> TextureSet:
    """Generate a natural gravel/pebble texture set.

    Uses the same tileable Voronoi approach as cobblestone but with
    much smaller, more numerous stones and a tighter color range.
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)

    # --- Voronoi cell setup (small cells for pebbles, tileable) ---
    cell_size = max(4, min(width, height) // 32)  # half cobblestone size
    n_cx = width // cell_size
    n_cy = height // cell_size

    # Jittered cell centers with hex offset
    jx = rng.uniform(-0.4, 0.4, (n_cy, n_cx))
    jy = rng.uniform(-0.4, 0.4, (n_cy, n_cx))
    base_points = []
    for j in range(n_cy):
        hex_off = 0.5 if j % 2 else 0.0
        for i in range(n_cx):
            px = (i + hex_off + 0.5 + jx[j, i]) * cell_size
            py = (j + 0.5 + jy[j, i]) * cell_size
            base_points.append((px, py))
    n_base = len(base_points)
    base_points = np.array(base_points, dtype=np.float64)

    # Toroidal copies for seamless tiling
    all_points = []
    for dy in (-height, 0, height):
        for dx in (-width, 0, width):
            all_points.append(base_points + np.array([dx, dy], dtype=np.float64))
    all_points = np.concatenate(all_points, axis=0)

    tree = KDTree(all_points)
    yy, xx = np.mgrid[0:height, 0:width]
    pixel_coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    dists, indices = tree.query(pixel_coords, k=2)

    d1 = dists[:, 0].reshape(height, width).astype(np.float32)
    d2 = dists[:, 1].reshape(height, width).astype(np.float32)
    cell_id = (indices[:, 0] % n_base).reshape(height, width)

    # --- Gap detection (narrower gaps than cobblestone) ---
    edge_factor = d2 - d1
    gap_threshold = 1.0 + 0.5 * noise_2d_grid(width, height, scale=8.0,
                                                octaves=2,
                                                seed=seed + 3 if seed else None)
    gap_blend = np.clip(1.0 - edge_factor / (gap_threshold + 1e-6), 0, 1)
    gap_blend = gap_blend ** 0.6

    # Normalized distance within each pebble
    stone_radius = d2 * 0.5
    norm_dist = np.clip(d1 / (stone_radius + 1e-6), 0, 1)

    # --- Per-pebble hashes ---
    cid64 = cell_id.astype(np.int64)
    h1 = ((cid64 * 2654435761) % (2**31)).astype(np.float32) / (2**31)
    h2 = ((cid64 * 2246822507 + 12345) % (2**31)).astype(np.float32) / (2**31)
    h3 = ((cid64 * 3266489917 + 67890) % (2**31)).astype(np.float32) / (2**31)
    h4 = ((cid64 * 1640531527 + 999983) % (2**31)).astype(np.float32) / (2**31)

    # --- Diffuse map ---
    # Gravel is lighter and more varied than cobblestone
    base = np.array([155, 148, 135], dtype=np.float32)
    diffuse = np.zeros((height, width, 3), dtype=np.float32)
    brightness = h1 * 40 - 20  # wider brightness range for variety
    diffuse[:, :, 0] = base[0] + brightness + (h2 * 12 - 6)
    diffuse[:, :, 1] = base[1] + brightness + (h3 * 10 - 5)
    diffuse[:, :, 2] = base[2] + brightness - (h2 * 8 - 4)

    # Surface grit
    grit = noise_2d_grid(width, height, scale=12.0, octaves=3, seed=seed)
    for c in range(3):
        diffuse[:, :, c] += grit * 8

    # Subtle edge darkening
    edge_darken = np.clip(norm_dist ** 2 * 0.12, 0, 0.12)
    for c in range(3):
        diffuse[:, :, c] *= (1.0 - edge_darken)

    # Gap fill (sandy/dirt color between pebbles)
    gap_color = np.array([115, 105, 85], dtype=np.float32)
    for c in range(3):
        diffuse[:, :, c] = diffuse[:, :, c] * (1.0 - gap_blend) + gap_color[c] * gap_blend

    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)

    # --- Height map (gentle domes, less pronounced than cobblestone) ---
    dome = np.cos(norm_dist * np.pi * 0.5)
    dome = np.clip(dome, 0, 1)
    stone_h_offset = h4 * 0.2 - 0.1  # less height variation than cobblestone
    height_map = dome * 0.4 + stone_h_offset + 0.35
    height_map = height_map * (1.0 - gap_blend) + 0.15 * gap_blend
    fine = noise_2d_grid(width, height, scale=18.0, octaves=2,
                         seed=seed + 2 if seed else None)
    height_map += fine * 0.06
    height_map = np.clip(height_map, 0, 1).astype(np.float32)

    normal = generate_normal(height_map, strength=1.5)

    # --- Roughness ---
    h5 = ((cid64 * 2996863034 + 54321) % (2**31)).astype(np.float32) / (2**31)
    rough_base = np.full((height, width), 185, dtype=np.float32)
    rough_base += h5 * 25 - 12
    rough_base += norm_dist * 15
    rough_base = rough_base * (1.0 - gap_blend) + 210 * gap_blend
    rough_base += fine * 10
    roughness = np.clip(rough_base, 0, 255).astype(np.uint8)

    return TextureSet(diffuse=diffuse, normal=normal, roughness=roughness,
                      width=width, height=height, name="gravel")


def farmhouse_heightmap(seed: Optional[int] = None) -> np.ndarray:
    """Generate a 385x385 grayscale heightmap for the Danish farmhouse terrain.

    Brightness mapping (for height_scale = 5.0 in Godot):
        0   (black)  = 0.0 units (lake floor / lowest)
        128 (50%)    = 2.5 units (ground level — house sits here)
        255 (white)  = 5.0 units (highest hills)

    Returns:
        385x385 uint8 grayscale ndarray
    """
    S = 385
    hmap = np.full((S, S), 128.0)

    # Coordinate grids normalised to [0, 1]
    ys, xs = np.mgrid[0:S, 0:S] / float(S - 1)
    cx, cy = xs - 0.5, ys - 0.5

    # 1. Perlin noise for gentle rolling hills
    noise = noise_2d_grid(S, S, scale=3.0, octaves=3, seed=seed)
    hmap += (noise - 0.5) * 40  # ±20 gray levels

    # 2. Center plateau (flat area for the house)
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    plateau_radius = 0.12
    plateau_falloff = 0.08
    plateau_mask = np.clip(1.0 - (dist_center - plateau_radius) / plateau_falloff, 0, 1)
    plateau_mask = plateau_mask ** 2
    hmap = hmap * (1 - plateau_mask) + 128.0 * plateau_mask

    # 3. Lake depression in the north (top of image = low y index)
    lake_cx, lake_cy = 0.5, 0.2
    lake_dist = np.sqrt((xs - lake_cx) ** 2 + (ys - lake_cy) ** 2)
    lake_radius = 0.10
    lake_falloff = 0.06
    lake_mask = np.clip(1.0 - (lake_dist - lake_radius) / lake_falloff, 0, 1)
    lake_mask = lake_mask ** 2
    hmap = hmap * (1 - lake_mask) + 40.0 * lake_mask

    # 4. Edge ramp (border wall — higher at edges)
    edge_dist = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
    edge_width = 0.06
    edge_mask = np.clip(1.0 - edge_dist / edge_width, 0, 1)
    edge_mask = edge_mask ** 1.5
    hmap = hmap * (1 - edge_mask) + 180.0 * edge_mask

    return np.clip(hmap, 0, 255).astype(np.uint8)
