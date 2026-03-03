# 3D Asset Generator

A Python tool for generating game-ready 3D assets (.glb) and textures procedurally. Runs locally without internet access.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```
Opens at http://localhost:7860

Or use `run.bat` for easy start/stop (Ctrl+C to stop).

## Architecture

The app uses a **registry-based modular architecture**. The UI builds itself dynamically from generator metadata -- no hardcoded tabs.

### How it works
1. Each generator module (`generators/*.py`) contains generator functions + `register()` calls
2. `register()` declares the function's UI metadata (param names, types, ranges, defaults)
3. `app.py` reads the registry and auto-builds Gradio tabs for every registered generator
4. Adding a new generator requires **zero changes to app.py** (unless it's a new file, then one import line)

### Key files
- **`app.py`** (~230 lines) - Dynamic Gradio UI builder. Reads registry, builds tabs.
- **`app_utils.py`** - `save_mesh()` (Z→Y-up conversion + GLB export), `parse_color()`, `create_grid_and_axes()`
- **`core/registry.py`** - `Param`, `GeneratorInfo`, `register()`, `get_by_category()`
- **`core/mesh.py`** - `MeshBuilder`, `combine_meshes()`, `apply_vertex_colors()`, `unweld_mesh()`, `compute_triplanar_uvs()`
- **`core/noise.py`** - `PerlinNoise` (scalar + vectorized 2D), `noise_2d_grid()`, `fractal_noise()`
- **`generators/*.py`** - Generator functions + registration metadata

## Current Features

### Asset Types
- **Rocks**: Single rocks, rock piles, boulders, pebbles
- **Vegetation**: Trees (spherical/conical/layered), forests, bushes, stumps
- **Buildings**: Buildings with roofs, villages, towers, walls, fences, outhouse, Danish farmhouse
- **Terrain**: Heightmap terrain, islands, plateaus
- **Props**: Crystals, crates, barrels, lamps, signs, benches, flower pots, candlesticks, books, clogs, brooms, kitchen stove, fireplace, chimney, washing basin
- **Furniture**: Tables, chairs, wardrobes, beds, single beds, rocking chairs, cupboards, settle beds, chests, clothes racks, wall coat racks, plate racks, grandfather clocks
- **Instruments**: Violin, trumpet
- **Buildings (special)**: Childhood home

### UI Features
- **Live preview**: Parameters update the 3D preview in real-time via `.change()` handlers
- **Vertex colors**: Color customization via color pickers
- **Rotation controls**: X/Y/Z rotation sliders (baked into exported mesh)
- **Grid & axes**: Toggle grid overlay with measurement markers and XYZ axes
- **Separate download**: Preview shows grid, download file is clean (no grid)
- **Seed control**: Set seed for reproducible generation

### Export Pipeline (`app_utils.save_mesh()`)
1. Z-up → Y-up conversion: rotates mesh -90° around X axis
2. Applies user rotation (X, Y, Z sliders)
3. Exports clean GLB for download (no grid)
4. If grid enabled: creates grid mesh, converts it to Y-up, concatenates with asset, exports preview GLB
- **Format**: GLB (binary glTF), Y-up (Unity/Godot compatible)
- **Colors**: Vertex colors embedded in mesh

### Textures
- Procedural texture generation with presets: Stone, White Stone, Wood, Grass, Thatch, Dark Thatch
- Vegetation sprite textures (RGBA with alpha cutout): Wildflower, Bush, Fern, Grass Tuft
- Farmhouse heightmap (385x385 grayscale)
- Outputs: Diffuse, normal, and roughness maps
- Texture tab uses a "Generate" button (not live preview)

## Project Structure

```
├── app.py                 # Dynamic Gradio UI builder
├── app_utils.py           # save_mesh, parse_color, create_grid_and_axes
├── run.bat                # Easy launcher script
├── asset_generator.py     # Legacy CLI (still works)
├── core/
│   ├── registry.py       # Generator registry system
│   ├── mesh.py           # MeshBuilder, export, UVs, LOD
│   ├── noise.py          # Perlin noise (scalar + vectorized), fractal noise
│   └── presets.py        # Save/load presets (not yet in UI)
├── generators/
│   ├── __init__.py       # Imports all modules to trigger registration
│   ├── rocks.py          # Rocks, boulders, pebbles
│   ├── trees.py          # Trees, bushes, stumps
│   ├── buildings.py      # Buildings, walls, fences, towers, outhouse, Danish farmhouse
│   ├── terrain.py        # Terrain, islands, plateaus
│   ├── props.py          # Props (crystals, barrels, lamps, stove, fireplace, etc.)
│   ├── furniture.py      # Tables, chairs, beds, wardrobes, clocks, etc.
│   ├── instruments.py    # Violin, trumpet
│   └── childhood_home.py # Childhood home (special building)
├── textures/
│   ├── generator.py      # Diffuse, normal, roughness maps + vegetation sprites
│   └── export.py         # PNG/TGA export (handles RGBA for sprites)
├── presets/               # Saved preset JSON files
└── output/                # Generated assets
```

## Adding New Generators

### Adding to an existing module (zero changes to app.py)

1. Write the generator function in the appropriate `generators/*.py` file
2. Add a `register()` call at the bottom:

```python
from core.registry import register, Param

def generate_flower(petal_count: int = 6, radius: float = 0.5,
                    color: Tuple[int, int, int] = (255, 100, 150),
                    seed: Optional[int] = None) -> trimesh.Trimesh:
    set_seed(seed)
    # ... create mesh ...
    return mesh

register(
    name="flower", label="Flower", category="Vegetation",
    params=[
        Param("petal_count", "Petals", "int", default=6, min=3, max=12, step=1),
        Param("radius", "Radius", "float", default=0.5, min=0.1, max=2.0),
        Param("color", "Petal Color", "color", default="#FF6496"),
    ],
)(generate_flower)
```

3. Done. The UI tab appears automatically.

### Adding a new module file (one import line in app.py)

1. Create `generators/newtype.py` with functions + register() calls
2. Add `import generators.newtype` in `app.py` (for PyInstaller compatibility)
3. Done.

### Supported parameter types

| Type | Gradio Widget | Registry Declaration |
|------|--------------|---------------------|
| `float` | Slider | `Param("name", "Label", "float", default=1.0, min=0, max=10)` |
| `int` | Slider (step=1) | `Param("name", "Label", "int", default=5, min=1, max=20, step=1)` |
| `bool` | Checkbox | `Param("name", "Label", "bool", default=True)` |
| `str` | Dropdown | `Param("name", "Label", "str", default="A", choices=["A", "B"])` |
| `color` | ColorPicker | `Param("name", "Label", "color", default="#FF0000")` |
| `range` | Two Sliders | `Param("name", "Label", "range", range_default=(0.3, 1.0), range_min=0.1, range_max=2.0)` |
| `image` | Image Upload | `Param("name", "Label", "image")` |

Every tab automatically gets: rotation sliders, grid toggle, seed input, 3D preview, and download file.

### Color pipeline

`ColorPicker` sends hex strings (e.g. `"#FF6496"`) → `app_utils.parse_color()` converts to `(R, G, B)` tuple → generator receives the tuple. Also handles `rgb(R, G, B)` format. Falls back to `(128, 128, 128)` on parse failure.

## Code Conventions

- `set_seed(seed)` and `apply_color_variation(mesh, rgba, variation)` are defined locally in each generator module (not imported from a shared location)
- All generators accept optional `seed` parameter; call `set_seed(seed)` at function start
- Return `trimesh.Trimesh` objects with vertex colors
- Generators use Z-up internally, converted to Y-up on export by `save_mesh()`
- Use `trimesh.util.concatenate()` to combine meshes
- Call `mesh.fix_normals()` after vertex manipulation

## Available Utilities

### `core/mesh.py`
- `MeshBuilder` — Programmatic mesh building: `add_vertex()`, `add_face()`, `add_quad()`, `build()`
- `combine_meshes(meshes)` — Wrapper around `trimesh.util.concatenate()`
- `apply_vertex_colors(mesh, rgba)` — Solid color to all vertices
- `unweld_mesh(mesh)` — Duplicate vertices so each triangle owns its 3 verts (needed for triplanar UVs)
- `compute_triplanar_uvs(mesh, scale)` — World-space UVs based on face normal dominant axis
- `center_mesh(mesh, center_z=False)` — Center at origin, optionally keep bottom at Z=0
- `generate_lod(mesh, target_faces)` — Simplify mesh (uses trimesh quadric decimation if available)

### `generators/buildings.py` — Reusable helpers
- `_wobble_box(extents, wobble=0.015)` — Box with random vertex displacement for handcrafted look
- `_wobble_cylinder(radius, height, sections=8, wobble=0.015)` — Same for cylinders
- `_build_wall_with_opening(wall_w, wall_h, wall_t, open_cx, open_sill_z, open_w, open_h, outer_color, inner_color)` — Dual-layer wall with rectangular opening (window or door)
- `_build_arched_window(win_w, win_h, arch_rise, sill_z, wall_t, frame_color, sill_color)` — Window frame with arch top
- `_build_door_frame(door_w, door_h, wall_t, frame_color)` — Door frame for partition walls (run along Y)
- `_build_front_door_frame(door_w, door_h, wall_t, frame_color)` — Door frame for front/back walls (run along X)
- `_build_plank_surface(width, depth, z, plank_w, axis, color)` — Flat surface of planks at given height
- `_build_farmhouse_chimney(base_w, base_d, height, color)` — Stacked wobble boxes with taper + crown

### `textures/generator.py` — Sprite helpers
- `_paint_ellipse(canvas, cx, cy, rx, ry, angle, color)` — Rotated filled ellipse on RGBA canvas
- `_paint_stem(canvas, x0, y0, x1, y1, thickness, color)` — Thick line segment
- `_generate_vegetation_sprite(width, height, seed, paint_fn, name)` — Orchestrator: creates transparent RGBA canvas, calls paint_fn, generates normal/roughness from alpha

## Performance

Texture generation was optimized from ~60s to ~0.6s via two changes:

- **Noise vectorization** (`core/noise.py`): `noise_2d_grid()` uses `PerlinNoise.noise_2d()` — a fully vectorized NumPy implementation that operates on entire coordinate arrays instead of triple-nested Python loops. The 2D case exploits z=0 → `fade(0)=0`, collapsing the outer lerp so only 4 gradient evaluations are needed per point instead of 8. **130x speedup**.
- **Convolution** (`textures/generator.py`): Thatch streak effect uses `scipy.ndimage.uniform_filter1d()` instead of `np.apply_along_axis(lambda: np.convolve(...))`. The old approach was a hidden Python loop over rows; `uniform_filter1d` runs entirely in C. **8x speedup**.

When adding new noise-based features, prefer `PerlinNoise.noise_2d()` for grid operations over the scalar `noise()` method.

## Dependencies

- **numpy** - Math operations
- **trimesh** - 3D mesh creation and export
- **scipy** - Required by trimesh; also used for `uniform_filter1d` in texture generation
- **gradio** - Web UI with 3D preview
- **Pillow** - Texture image export

## Future Improvements

### Easy
- Granular shape controls (e.g., trunk taper, canopy squash)
- Add colors to terrain and remaining props
- OBJ export option
- Preset save/load in UI

### Medium
- Batch export (multiple variants)
- Height-based terrain coloring
- More asset types (bridges, stairs, wells, gates)

### Complex (would require different framework)
- Interactive 3D editing (drag vertices/corners)
- Real-time mesh manipulation in browser

## Packaging as .exe

The architecture uses explicit imports (not dynamic discovery) specifically for PyInstaller compatibility:

```bash
pip install pyinstaller
pyinstaller --onefile --add-data "generators;generators" --add-data "textures;textures" --add-data "core;core" app.py
```

## Web Deployment

The registry is pure data. A future Flask/FastAPI backend could read `get_by_category()` and emit a JSON schema of parameters instead of Gradio widgets. The `make_wrapper()` pattern works with any framework that passes keyword arguments.
