# 3D Asset Generator - Roadmap

## Current State (v0.1 - Prototype)
- Basic procedural generators (rock, tree, building, etc.)
- Gradio web UI with live preview
- Export to .obj/.glb

---

## Phase 1: Project Structure & Textures
- [ ] Restructure into modular architecture
- [ ] Texture generation module (diffuse, normal, roughness)
- [ ] TGA/PNG export for textures
- [ ] Vertex color support
- [ ] Basic UV mapping

## Phase 2: More Asset Types
- [ ] Terrain/heightmap generator
- [ ] Walls and fences
- [ ] Stairs and platforms
- [ ] Props (lamps, signs, crates, barrels - improve existing)
- [ ] Vegetation (bushes, grass clumps)
- [ ] Dungeon/interior pieces

## Phase 3: Batch & Presets
- [ ] Save/load parameter presets (JSON)
- [ ] Batch generation (N variations)
- [ ] Asset library browser
- [ ] Export multiple assets at once
- [ ] Naming conventions and organization

## Phase 4: Advanced Geometry
- [ ] LOD generation (high/medium/low poly)
- [ ] Mesh optimization (reduce triangles)
- [ ] Boolean operations (combine/subtract)
- [ ] Modular snap-together pieces
- [ ] Collision mesh generation (simplified)

## Phase 5: Polish & Distribution
- [ ] PyInstaller packaging (.exe)
- [ ] Settings/preferences persistence
- [ ] Undo/redo for parameters
- [ ] Keyboard shortcuts
- [ ] Documentation and tutorials

---

## Architecture

```
asset_generator/
├── app.py                 # Gradio UI entry point
├── core/
│   ├── mesh.py           # Mesh utilities, export, UV mapping
│   ├── noise.py          # Noise functions (perlin, simplex)
│   └── presets.py        # Preset save/load system
├── generators/
│   ├── rocks.py          # Rock variants
│   ├── trees.py          # Tree variants
│   ├── buildings.py      # Buildings, walls, structures
│   ├── terrain.py        # Terrain, heightmaps
│   └── props.py          # Small props, decorations
├── textures/
│   ├── generator.py      # Procedural texture generation
│   ├── materials.py      # Material definitions
│   └── export.py         # TGA/PNG export
├── presets/              # Saved preset JSON files
└── output/               # Generated assets
```
