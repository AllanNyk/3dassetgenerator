# Danish 1870s Farmhouse Generator — Reference

This is a major gameplay feature. The farmhouse is the player's home and hub. It's a traversable interior building with 5 rooms, no furniture (user populates rooms in Godot).

## File Location

All code lives in `generators/buildings.py`, starting after the outhouse generator (~line 742).

## Main Function

```python
def generate_danish_farmhouse(
    size: float = 1.0,           # uniform scale multiplier
    wall_color: (245,240,230),   # white exterior masonry
    interior_color: (235,195,140),# light orange interior walls
    roof_color: (178,155,110),   # light brown (texture does the thatch look)
    floor_color: (140,95,50),    # wood plank color
    seed: Optional[int] = None,
) -> trimesh.Trimesh
```

Registered as `danish_farmhouse` in the "Buildings" category. UI params: Size, Ext. Walls, Int. Walls, Roof Color, Floor Color.

## Coordinate System

- Generators work in **Z-up** internally. Converted to Y-up on export by `app_utils.save_mesh()`.
- House is **centered at origin**. Length along X, depth along Y, height along Z.
- Front wall faces **+Y**, back wall faces **-Y**.

## Base Dimensions (at size=1.0)

| Dimension | Value |
|-----------|-------|
| Length (X) | 15.0m |
| Depth (Y) | 7.0m |
| Wall height | 2.5m |
| Ridge height | 5.0m (~50 degree pitch) |
| Exterior wall thickness | 0.4m (dual-layer: 0.55 outer + 0.45 inner) |
| Partition wall thickness | 0.25m |
| Interior door openings | 0.9m wide x 2.1m tall |
| Front door opening | 1.0m wide x 2.1m tall |
| Windows | 0.8m wide x 1.1m tall, sill at 0.8m, 0.1m arch rise |

## Room Layout (left to right along X)

Rooms are defined by `room_widths = [3.0, 1.8, 3.2, 2.8, 2.4]` (proportional, scaled to fit interior).

| Index | Room | Approx Width | Notes |
|-------|------|-------------|-------|
| 0 | Kitchen | ~3.0m | Has chimney above, front+back windows |
| 1 | Entré | ~1.8m | Front door (no front window), back window |
| 2 | Main room | ~3.2m | Largest room, front+back windows |
| 3 | Bedroom 1 | ~2.8m | Front+back windows |
| 4 | Bedroom 2 | ~2.4m | Has chimney above, front+back windows |

All rooms are connected by door openings (centered at Y=0) in each partition wall. The entré serves as the interior hallway connecting kitchen to main room.

## Key Computed Values

These are calculated at runtime — important to understand when modifying:

- `room_x_starts[]` — X coordinate where each room begins (after left exterior wall)
- `room_centers_x[]` — X center of each room
- `partition_xs[]` — X positions of the 4 partition walls (between rooms)
- `front_y = D/2 - WALL_T/2` — Y center of front wall thickness
- `back_y = -D/2 + WALL_T/2` — Y center of back wall thickness
- `interior_d = D - 2*WALL_T` — interior depth available for partitions/floors

## Construction Steps (in order within `generate_danish_farmhouse`)

1. **Foundation** — Stone-colored perimeter strips (4 wobble boxes)
2. **Exterior walls** — Per-room front+back segments via `_build_wall_with_opening()`. Entré gets a door opening instead of window. Left+right end walls are solid rectangles + stacked gable boxes tapering to ridge. Tar base strips on all 4 sides.
3. **Interior partition walls** — 4 walls with door openings. Each partition is 3 pieces (left of door, right of door, above door) + a door frame via `_build_door_frame()`.
4. **Floor** — Per-room plank strips via `_build_plank_surface()`, 0.25m wide planks along Y axis.
5. **Ceiling** — Same as floor but at WALL_H (2.5m), slightly darker color.
6. **Windows** — 9 arched windows via `_build_arched_window()` (all rooms front+back, except entré front). Front door gets a frame via `_build_front_door_frame()`.
7. **Roof** — Two simple wobble-box slabs (one per slope), tilted by `-side * roof_pitch_angle` around X axis. Ridge cap on top. Light brown color — thatch appearance is handled by textures.
8. **Chimneys** — 2 white chimneys (1.35m tall) via `_build_farmhouse_chimney()`, positioned above kitchen (room 0) and bedroom 2 (room 4).

## Helper Functions

### `_build_wall_with_opening(wall_w, wall_h, wall_t, open_cx, open_sill_z, open_w, open_h, outer_color, inner_color, ...)`
Builds a wall section with a rectangular opening as 4 dual-layer panels (left, right, below, above the opening). Each panel has an outer slab (exterior color) and inner slab (interior color). Used for all front/back wall segments.

- `open_cx` is relative to wall center (usually 0)
- For a door: `open_sill_z=0`; for a window: `open_sill_z=WIN_SILL_Z`

### `_build_arched_window(win_w, win_h, arch_rise, sill_z, wall_t, frame_color, sill_color)`
Builds a window frame: 2 side jambs + bottom sill + protruding outer sill + 8-segment arched top. Returns meshes centered at X=0; caller translates to room center + wall Y position.

### `_build_door_frame(door_w, door_h, wall_t, frame_color)`
Interior door frame for **partition walls** (which run along Y). Jambs spread along Y, depth along X. 2 jambs + 1 lintel.

### `_build_front_door_frame(door_w, door_h, wall_t, frame_color)`
Door frame for **front/back walls** (which run along X). Jambs spread along X, depth along Y. Same structure, different orientation.

**Important**: These two door frame functions exist because partition walls and exterior walls run perpendicular to each other. Using the wrong one produces rotated frames.

### `_build_plank_surface(width, depth, z, plank_w, axis, color, ...)`
Builds a flat surface of planks at height `z`. `axis='y'` means planks run along Y (used for floors/ceilings). Each plank is 99.5% width for tight fit with subtle seams. Per-plank color variation for natural wood look.

### `_build_farmhouse_chimney(base_w, base_d, height, color)`
Stacked wobble boxes with slight taper + wider crown cap. Base at Z=0; caller translates to ridge height.

## Shared Utilities (from buildings.py, used by all building generators)

- `_wobble_box(extents, wobble=0.015)` — Box with random vertex displacement for handcrafted look
- `_wobble_cylinder(radius, height, sections=8, wobble=0.015)` — Same for cylinders
- `apply_color_variation(mesh, base_color_rgba, variation)` — Per-vertex color noise
- `set_seed(seed)` — Sets both numpy and random seeds

## Design Decisions and Gotchas

- **Walls are dual-layer**: exterior walls have an outer slab (white) and inner slab (orange) so both sides have the correct color. This is why `_build_wall_with_opening` takes two color parameters.
- **Roof rotation sign**: The roof uses `-side * roof_pitch_angle` (note the negative). Getting this wrong inverts the roof.
- **Door frame orientation**: Two separate functions because partition walls (along Y) and front/back walls (along X) are perpendicular. The jamb/depth axes must match the wall orientation.
- **Room widths are proportional**: The raw widths are scaled to exactly fill `L - 2*WALL_T`. Changing one room's width affects all others proportionally.
- **No furniture**: The interior is intentionally empty. The user will place furniture in Godot.
- **Textures over geometry**: Masonry pattern, thatch detail, and moss are NOT modeled as geometry. The mesh provides plain surfaces with vertex colors; textures will add visual detail. This keeps vertex count low (~2,800).
- **No external entryway**: The entré is purely an interior room. No protruding porch or foyer on the outside.

## Vertex Budget

~2,800 vertices at size=1.0. Well within game-ready range.

## Future Work Ideas

- Furniture generators (separate assets placed in Godot, not part of the house mesh)
- Window glass panes (translucent material)
- Fireplace geometry inside kitchen / bedroom
- Attic space above the ceiling (currently just roof shell)
- Weathering / age variation parameter
- Multiple door placement options (not just centered at Y=0)
