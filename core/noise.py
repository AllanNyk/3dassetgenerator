"""
Noise functions for procedural generation.
Provides Perlin noise, fractal noise, and other useful noise patterns.
"""

import numpy as np
from typing import Optional


def _fade(t):
    """Fade function for smooth interpolation. Works on scalars and arrays."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    """Linear interpolation. Works on scalars and arrays."""
    return a + t * (b - a)


def _gradient(h, x, y, z):
    """Calculate gradient vector (scalar version)."""
    h = h & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _gradient_vec(h, x, y, z):
    """Vectorized gradient calculation for NumPy arrays."""
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1 == 0, u, -u) + np.where(h & 2 == 0, v, -v)


class PerlinNoise:
    """Perlin noise generator."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        # Generate permutation table
        self.p = np.arange(256, dtype=int)
        np.random.shuffle(self.p)
        self.p = np.tile(self.p, 2)

    def noise(self, x: float, y: float = 0, z: float = 0) -> float:
        """
        Calculate 3D Perlin noise at given coordinates.

        Args:
            x, y, z: Coordinates

        Returns:
            Noise value in range [-1, 1]
        """
        # Find unit cube
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        Z = int(np.floor(z)) & 255

        # Relative position in cube
        x -= np.floor(x)
        y -= np.floor(y)
        z -= np.floor(z)

        # Fade curves
        u = _fade(x)
        v = _fade(y)
        w = _fade(z)

        # Hash coordinates of cube corners
        p = self.p
        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z

        # Blend results from corners
        return _lerp(
            _lerp(
                _lerp(_gradient(p[AA], x, y, z), _gradient(p[BA], x - 1, y, z), u),
                _lerp(_gradient(p[AB], x, y - 1, z), _gradient(p[BB], x - 1, y - 1, z), u),
                v
            ),
            _lerp(
                _lerp(_gradient(p[AA + 1], x, y, z - 1), _gradient(p[BA + 1], x - 1, y, z - 1), u),
                _lerp(_gradient(p[AB + 1], x, y - 1, z - 1), _gradient(p[BB + 1], x - 1, y - 1, z - 1), u),
                v
            ),
            w
        )

    def noise_2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Vectorized 2D Perlin noise for entire coordinate arrays.
        Since z=0, fade(0)=0, so the outer lerp collapses and we only
        need 4 gradient evaluations instead of 8.

        Args:
            x, y: NumPy arrays of coordinates (same shape)

        Returns:
            Array of noise values in [-1, 1], same shape as inputs
        """
        p = self.p

        # Integer grid coordinates (wrapped to 0-255)
        X = np.floor(x).astype(int) & 255
        Y = np.floor(y).astype(int) & 255

        # Fractional position within cell
        xf = x - np.floor(x)
        yf = y - np.floor(y)

        # Fade curves
        u = _fade(xf)
        v = _fade(yf)

        # Hash coordinates of the 4 corners (z=0 throughout)
        A  = p[X] + Y
        AA = p[A]
        AB = p[A + 1]
        B  = p[X + 1] + Y
        BA = p[B]
        BB = p[B + 1]

        z0 = np.zeros_like(xf)

        # Bilinear interpolation of 4 corner gradients
        return _lerp(
            _lerp(
                _gradient_vec(p[AA], xf, yf, z0),
                _gradient_vec(p[BA], xf - 1, yf, z0),
                u
            ),
            _lerp(
                _gradient_vec(p[AB], xf, yf - 1, z0),
                _gradient_vec(p[BB], xf - 1, yf - 1, z0),
                u
            ),
            v
        )


def perlin_noise(
    x: float,
    y: float = 0,
    z: float = 0,
    seed: Optional[int] = None
) -> float:
    """
    Simple Perlin noise function.

    Args:
        x, y, z: Coordinates
        seed: Random seed

    Returns:
        Noise value in range [-1, 1]
    """
    noise_gen = PerlinNoise(seed)
    return noise_gen.noise(x, y, z)


def fractal_noise(
    x: float,
    y: float = 0,
    z: float = 0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: Optional[int] = None
) -> float:
    """
    Fractal/fBm noise (multiple octaves of Perlin noise).

    Args:
        x, y, z: Coordinates
        octaves: Number of noise layers
        persistence: Amplitude multiplier per octave
        lacunarity: Frequency multiplier per octave
        seed: Random seed

    Returns:
        Noise value (roughly in range [-1, 1])
    """
    noise_gen = PerlinNoise(seed)

    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += noise_gen.noise(x * frequency, y * frequency, z * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value


def noise_2d_grid(
    width: int,
    height: int,
    scale: float = 1.0,
    octaves: int = 4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a 2D grid of fractal noise values.

    Args:
        width, height: Grid dimensions
        scale: Noise scale (higher = more zoomed out)
        octaves: Number of noise octaves
        seed: Random seed

    Returns:
        2D numpy array with values in [0, 1]
    """
    noise_gen = PerlinNoise(seed)

    # Build coordinate grids once
    x_coords = np.arange(width, dtype=np.float64) / width * scale
    y_coords = np.arange(height, dtype=np.float64) / height * scale
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    grid = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0

    for _ in range(octaves):
        grid += noise_gen.noise_2d(x_grid * frequency, y_grid * frequency) * amplitude
        max_val += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    # Normalize to [0, 1]
    return (grid / max_val + 1) / 2


def voronoi_noise(
    x: float,
    y: float,
    num_points: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    Simple Voronoi/cellular noise.

    Args:
        x, y: Coordinates (should be in [0, 1] range)
        num_points: Number of cell points
        seed: Random seed

    Returns:
        Distance to nearest point (normalized)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random points
    points = np.random.rand(num_points, 2)

    # Find minimum distance
    min_dist = float('inf')
    for px, py in points:
        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        min_dist = min(min_dist, dist)

    # Normalize (approximate max distance is ~0.7 for uniform distribution)
    return min(min_dist / 0.5, 1.0)
