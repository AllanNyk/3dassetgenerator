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
- **`app.py`** (~200 lines) - Dynamic Gradio UI builder. Reads registry, builds tabs.
- **`app_utils.py`** - Shared utilities: `save_mesh()`, `parse_color()`, `create_grid_and_axes()`
- **`core/registry.py`** - `Param`, `GeneratorInfo`, `register()`, `get_by_category()`
- **`generators/*.py`** - Generator functions + registration metadata

## Current Features

### Asset Types (24 generators)
- **Rocks**: Single rocks, rock piles, boulders, pebbles
- **Vegetation**: Trees (spherical/conical/layered), forests, bushes, stumps
- **Buildings**: Buildings with roofs, villages, towers, walls, fences
- **Terrain**: Heightmap terrain, islands, plateaus
- **Props**: Crystals, crates, barrels, lamps, signs, benches, flower pots

### UI Features
- **Live preview**: Parameters update the 3D preview in real-time
- **Vertex colors**: Color customization via color pickers
- **Rotation controls**: X/Y/Z rotation sliders (baked into exported mesh)
- **Grid & axes**: Toggle grid overlay with measurement markers and XYZ axes
- **Separate download**: Preview shows grid, download file is clean (no grid)
- **Seed control**: Set seed for reproducible generation

### Export
- **Format**: GLB (binary glTF)
- **Coordinate system**: Y-up (Unity/Godot compatible). Generators use Z-up internally, converted on export.
- **Colors**: Vertex colors embedded in mesh

### Textures
- Procedural texture generation (stone, wood, grass)
- Outputs: Diffuse, normal, and roughness maps

## Project Structure

```
├── app.py                 # Dynamic Gradio UI builder (~200 lines)
├── app_utils.py           # save_mesh, parse_color, create_grid_and_axes
├── run.bat                # Easy launcher script
├── asset_generator.py     # Legacy CLI (still works)
├── core/
│   ├── registry.py       # Generator registry system
│   ├── mesh.py           # Mesh utilities, export
│   ├── noise.py          # Perlin/fractal noise
│   └── presets.py        # Save/load presets (not yet in UI)
├── generators/
│   ├── __init__.py       # Imports all modules to trigger registration
│   ├── rocks.py          # Rocks, boulders, pebbles + register() calls
│   ├── trees.py          # Trees, bushes, stumps + register() calls
│   ├── buildings.py      # Buildings, walls, fences, towers + register() calls
│   ├── terrain.py        # Terrain, islands, plateaus + register() calls
│   └── props.py          # Props (crystals, barrels, lamps, etc.) + register() calls
├── textures/
│   ├── generator.py      # Diffuse, normal, roughness maps
│   └── export.py         # PNG/TGA export
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

Every tab automatically gets: rotation sliders, grid toggle, seed input, 3D preview, and download file.

## Code Conventions

- All generators accept optional `seed` parameter
- Use `set_seed(seed)` at function start
- Return `trimesh.Trimesh` objects with vertex colors
- Generators use Z-up internally, converted to Y-up on export
- Use `trimesh.util.concatenate()` to combine meshes
- Call `mesh.fix_normals()` after vertex manipulation
- Use `apply_color_variation(mesh, rgba, variation)` for natural color variation

## Dependencies

- **numpy** - Math operations
- **trimesh** - 3D mesh creation and export
- **scipy** - Required by trimesh
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
