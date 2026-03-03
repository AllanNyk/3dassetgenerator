"""
3D Asset Generator - Gradio Interface
Run with: python app.py
Opens in browser at http://localhost:7860
"""

import gradio as gr
import tempfile

# Core systems
from core.registry import get_by_category, Param, GeneratorInfo
from app_utils import save_mesh, parse_color

# Import generator modules to trigger registration.
# Explicit imports (not dynamic) for PyInstaller compatibility.
import generators.rocks
import generators.trees
import generators.buildings
import generators.terrain
import generators.props
import generators.furniture
import generators.instruments
import generators.childhood_home

# Texture imports (texture tab stays hardcoded - different UI pattern)
from textures.generator import (generate_texture_set, stone_texture, wood_texture, grass_texture,
                                thatch_texture, white_stone_texture, dark_thatch_texture,
                                wildflower_sprite, bush_sprite, fern_sprite, grass_tuft_sprite,
                                farmhouse_heightmap)
from textures.export import save_texture_set


# =============================================================================
# DYNAMIC UI BUILDER
# =============================================================================

def build_param_inputs(params):
    """Create Gradio input components from parameter metadata."""
    components = []
    for p in params:
        if p.type == "float":
            components.append(gr.Slider(p.min, p.max, value=p.default,
                                        step=p.step, label=p.label))
        elif p.type == "int":
            components.append(gr.Slider(p.min, p.max, value=p.default,
                                        step=p.step or 1, label=p.label))
        elif p.type == "bool":
            components.append(gr.Checkbox(value=p.default, label=p.label))
        elif p.type == "str":
            components.append(gr.Dropdown(p.choices, value=p.default, label=p.label))
        elif p.type == "color":
            components.append(gr.ColorPicker(label=p.label, value=p.default))
        elif p.type == "image":
            components.append(gr.Image(label=p.label, type="filepath",
                                       sources=["upload"], height=80))
        elif p.type == "range":
            lo, hi = p.range_default
            components.append(gr.Slider(p.range_min, p.range_max, value=lo,
                                        label=f"{p.label} Min"))
            components.append(gr.Slider(p.range_min, p.range_max, value=hi,
                                        label=f"{p.label} Max"))
    return components


def make_wrapper(info):
    """Build a generic UI callback for any registered generator.

    Handles param type conversion, range reassembly, and mesh export.
    """
    # Pre-compute which params are range type (they produce 2 Gradio inputs)
    range_flags = [p.type == "range" for p in info.params]
    n_gradio_inputs = sum(2 if r else 1 for r in range_flags)

    def wrapper(*args):
        # Split args: [param widgets...] + [rot_x, rot_y, rot_z, show_grid, seed]
        param_args = list(args[:n_gradio_inputs])
        rot_x, rot_y, rot_z, show_grid, seed = args[n_gradio_inputs:]

        # Build kwargs from param values
        kwargs = {}
        arg_idx = 0
        for param, is_range in zip(info.params, range_flags):
            if is_range:
                kwargs[param.name] = (param_args[arg_idx], param_args[arg_idx + 1])
                arg_idx += 2
            elif param.type == "color":
                kwargs[param.name] = parse_color(param_args[arg_idx])
                arg_idx += 1
            elif param.type == "int":
                kwargs[param.name] = int(param_args[arg_idx])
                arg_idx += 1
            elif param.type == "image":
                kwargs[param.name] = param_args[arg_idx]
                arg_idx += 1
            elif param.type == "str":
                kwargs[param.name] = param_args[arg_idx].lower()
                arg_idx += 1
            else:
                kwargs[param.name] = param_args[arg_idx]
                arg_idx += 1

        kwargs["seed"] = int(seed) if seed else None

        mesh = info.func(**kwargs)
        return save_mesh(mesh, info.export_name,
                         show_grid=show_grid,
                         rotation=(rot_x, rot_y, rot_z))

    return wrapper


def build_tab(info):
    """Build one generator sub-tab from its metadata."""
    with gr.TabItem(info.label):
        with gr.Row():
            with gr.Column(scale=1):
                param_inputs = build_param_inputs(info.params)
                gr.Markdown("**Rotation (degrees)**")
                with gr.Row():
                    rot_x = gr.Slider(-180, 180, value=0, step=5, label="X")
                    rot_y = gr.Slider(-180, 180, value=0, step=5, label="Y")
                    rot_z = gr.Slider(-180, 180, value=0, step=5, label="Z")
                show_grid = gr.Checkbox(label="Show Grid & Axes", value=False)
                seed = gr.Number(label="Seed", precision=0)
            with gr.Column(scale=2):
                preview = gr.Model3D(label="Preview")
                download = gr.File(label="Download (clean, no grid)")

        all_inputs = param_inputs + [rot_x, rot_y, rot_z, show_grid, seed]
        outputs = [preview, download]
        callback = make_wrapper(info)

        for inp in all_inputs:
            inp.change(callback, inputs=all_inputs, outputs=outputs)


# =============================================================================
# TEXTURE TAB (hardcoded - different UI pattern)
# =============================================================================

def ui_generate_texture(tex_type, size, seed):
    seed = int(seed) if seed else None
    size = int(size)

    # Heightmap is a single grayscale image (fixed 385x385), not a texture set
    if tex_type == "Farmhouse Heightmap":
        from textures.export import save_texture
        hmap = farmhouse_heightmap(seed)
        temp_dir = tempfile.mkdtemp()
        path = save_texture(hmap, f"{temp_dir}/farmhouse_heightmap.png")
        return path, None, None

    if tex_type == "Stone":
        tex = stone_texture(size, size, seed)
    elif tex_type == "White Stone":
        tex = white_stone_texture(size, size, seed)
    elif tex_type == "Wood":
        tex = wood_texture(size, size, seed)
    elif tex_type == "Grass":
        tex = grass_texture(size, size, seed)
    elif tex_type == "Thatch":
        tex = thatch_texture(size, size, seed)
    elif tex_type == "Dark Thatch":
        tex = dark_thatch_texture(size, size, seed)
    elif tex_type == "Wildflower Sprite":
        tex = wildflower_sprite(size, size, seed)
    elif tex_type == "Bush Sprite":
        tex = bush_sprite(size, size, seed)
    elif tex_type == "Fern Sprite":
        tex = fern_sprite(size, size, seed)
    elif tex_type == "Grass Tuft Sprite":
        tex = grass_tuft_sprite(size, size, seed)
    else:
        tex = generate_texture_set(size, size, seed=seed)

    temp_dir = tempfile.mkdtemp()
    paths = save_texture_set(tex, temp_dir, prefix=tex_type.lower())

    return paths['diffuse'], paths['normal'], paths['roughness']


# =============================================================================
# BUILD APP
# =============================================================================

def create_app():
    with gr.Blocks(title="3D Asset Generator") as app:
        gr.Markdown("# 3D Asset Generator")
        gr.Markdown("Generate procedural 3D assets and textures. Parameters update live!")
        gr.Markdown("**Export:** Y-up (Unity/Godot) | **Grid:** 1 unit = 1 meter | **Axes:** X=Red, Y=Green, Z=Blue")

        with gr.Tabs():
            # Dynamic generator tabs from registry
            for category, generators in get_by_category().items():
                with gr.TabItem(category):
                    with gr.Tabs():
                        for gen_info in generators:
                            build_tab(gen_info)

            # Textures tab (hardcoded - different UI pattern)
            with gr.TabItem("Textures"):
                gr.Markdown("Generate procedural textures (diffuse, normal, roughness maps)")
                with gr.Row():
                    with gr.Column(scale=1):
                        tex_type = gr.Dropdown(["Stone", "White Stone", "Wood", "Grass",
                                                "Thatch", "Dark Thatch",
                                                "Wildflower Sprite", "Bush Sprite",
                                                "Fern Sprite", "Grass Tuft Sprite",
                                                "Farmhouse Heightmap",
                                                "Custom"],
                                               value="Stone", label="Material")
                        tex_size = gr.Dropdown(["256", "512", "1024"],
                                               value="512", label="Size")
                        tex_seed = gr.Number(label="Seed", precision=0)
                        tex_btn = gr.Button("Generate Textures", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            tex_diffuse = gr.Image(label="Diffuse", type="filepath")
                            tex_normal = gr.Image(label="Normal", type="filepath")
                            tex_rough = gr.Image(label="Roughness", type="filepath")

                tex_btn.click(
                    ui_generate_texture,
                    inputs=[tex_type, tex_size, tex_seed],
                    outputs=[tex_diffuse, tex_normal, tex_rough]
                )

        gr.Markdown("---")
        gr.Markdown("*Download: Use the File component below the preview to download clean exports.*")

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
