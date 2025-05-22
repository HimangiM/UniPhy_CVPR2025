import gradio as gr
import numpy as np
from PIL import Image
import os
import open3d as o3d
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infer_material_latent import main as infer_material
from omegaconf import OmegaConf
import time
import hydra
from hydra import initialize, compose

MATERIALS = {
    "Elastic1": [
        {"label": "Blobby"},
    ],
    "Elastic2": [
        {"label": "Toy"},
    ],
    "Newtonian1": [
        {"label": "Sphere"},
    ],
    "Newtonian2": [
        {"label": "Pawn"},
    ],
    "Plasticine1": [
        {"label": "Octahedron"},
    ],
    "Plasticine2": [
        {"label": "Ellipsoid"},
    ],
    "Sand1": [
        {"label": "Blobby"},
    ],
    "Sand2": [
        {"label": "Ellipsoid"},
    ],
    "Non-Newtonian1": [
        {"label": "Toy"},
    ],
    "Non-Newtonian2": [
        {"label": "Sphere"},
    ],
}

def get_geometries_for_material(material):
    return [entry["label"] for entry in MATERIALS[material]]

def recursive_update(cfg, overrides):
    """Recursively update OmegaConf config with a dictionary of overrides."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg:
            recursive_update(cfg[k], v)
        else:
            cfg[k] = v
    return cfg

def simulate_point_cloud(config_overrides=None):
    if config_overrides is not None:
        cfg = config_overrides
    else:
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name="default")
            if config_overrides:
                recursive_update(cfg, config_overrides)
    print (cfg)
    infer_material(cfg)

def render_point_cloud_simulation(material, geometry_label):
    idx = [entry["label"] for entry in MATERIALS[material]].index(geometry_label)
    entry = MATERIALS[material][idx]
    geometry_path = material[:-1] + '_' + entry["label"]
    current_folder = os.getcwd()
    config_overrides = {
        "train_cfg": {
            # "traj_data_dir": os.path.join('/home/hmittal/diff-sim/houdini_webpage/demo/git_data', geometry_path),
            "traj_data_dir": os.path.join(current_folder, 'gradio_demo/gradio_demo_data', geometry_path),
            # "load_model": '/data18/hmittal/azs_221_166/trained_models/latentlogs/pretrained_stress_fproj_models_latent_space/ckpt.pth',
            "load_model": os.path.join(current_folder, 'ckpt/pretrained_stress_fproj_models_latent_space/ckpt.pth'),
            # "traj_latent_path": '/data18/hmittal/azs_221_166/trained_models/latentlogs/pretrained_stress_fproj_models_latent_space/traj_latent.pth',
            "traj_latent_path": os.path.join(current_folder, 'ckpt/pretrained_stress_fproj_models_latent_space/traj_latent.pth')
        }
    }
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="default")
        recursive_update(cfg, config_overrides)
        simulate_point_cloud(config_overrides=cfg)

    sample_name = geometry_label.lower().replace(' ', '_') if geometry_label else 'sample'
    gif_path = os.path.join(current_folder, 'gradio_demo/gradio_demo_data', geometry_path, f"{material[:-1]}_{geometry_label}.gif")
    return gif_path

def update_geometries(material):
    choices = get_geometries_for_material(material)
    return gr.update(choices=choices, value=choices[0])

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            material_dd = gr.Dropdown(choices=list(MATERIALS.keys()), value="Elastic1", label="Select Material")
            geometry_dd = gr.Dropdown(choices=get_geometries_for_material("Elastic1"), value=get_geometries_for_material("Elastic1")[0], label="Select Geometry")
            btn = gr.Button("Simulate & Render GIF")
        with gr.Column(scale=2):
            out_gif = gr.Image(type="filepath", label="Simulation GIF")
    material_dd.change(fn=update_geometries, inputs=material_dd, outputs=geometry_dd)
    
    btn.click(fn=render_point_cloud_simulation, inputs=[material_dd, geometry_dd], outputs=out_gif)

demo.launch()
