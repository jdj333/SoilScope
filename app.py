
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import panel as pn
import plotly.graph_objects as go

pn.extension('plotly')

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

def load_crop(crop_name:str):
    path = os.path.join(DATA, f"{crop_name}.json")
    with open(path, "r") as f:
        d = json.load(f)
    layers = pd.DataFrame(d["sample_profile_layers"])
    targets = d["ideal_targets"]
    texture_tgt = d.get("texture_targets_pct", {})
    bacteria_tgt = d.get("bacteria_targets", [])
    return d["crop"], layers, targets, texture_tgt, bacteria_tgt

CROPS = {"corn":"corn", "alfalfa":"alfalfa"}

# properties available
PLOT_PROPS = ["pH","NO3_N_ppm","P_ppm_Olsen","K_ppm","Ca_ppm","Mg_ppm","S_ppm","Zn_ppm","Mn_ppm","B_ppm"]

def layers_to_depth_df(layers: pd.DataFrame, step: int = 1):
    """Expand layer table into a regular 0..48 inch grid with per-inch values (forward-fill within each layer)."""
    max_depth = int(layers["depth_bottom_in"].max())
    grid = pd.DataFrame({"depth_in": np.arange(0, max_depth+1, step)})
    # For each property, fill values by layer membership
    out = grid.copy()
    for col in [c for c in layers.columns if c not in ["layer_id","depth_top_in","depth_bottom_in","sand_pct","silt_pct","clay_pct"]]:
        vals = np.zeros_like(grid["depth_in"], dtype=float)
        for _, row in layers.iterrows():
            mask = (grid["depth_in"] >= row["depth_top_in"]) & (grid["depth_in"] <= row["depth_bottom_in"])
            vals[mask] = row[col]
        out[col] = vals
    # texture as well (optional)
    for col in ["sand_pct","silt_pct","clay_pct"]:
        vals = np.zeros_like(grid["depth_in"], dtype=float)
        for _, row in layers.iterrows():
            mask = (grid["depth_in"] >= row["depth_top_in"]) & (grid["depth_in"] <= row["depth_bottom_in"])
            vals[mask] = row[col]
        out[col] = vals
    return out

def plot_profile(df: pd.DataFrame, prop: str, targets: dict):
    """Matplotlib depth profile with ideal band."""
    import io
    fig, ax = plt.subplots(figsize=(5,4), dpi=120)
    ax.plot(df[prop], df["depth_in"])
    ax.invert_yaxis()
    ax.set_xlabel(prop)
    ax.set_ylabel("Depth (in)")
    # Ideal band if exists
    if prop in targets:
        t = targets[prop]
        ax.fill_betweenx(df["depth_in"], t["min"], t["max"], alpha=0.2)
    ax.grid(True, linestyle=":")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return pn.pane.PNG(buf.getvalue(), height=350)

def make_volume(df: pd.DataFrame, prop: str, nx=20, ny=20):
    """Create a simple 3D volume (x,y repeated) with z=depth using Plotly Volume."""
    z = df["depth_in"].values
    v = df[prop].values
    # replicate across x,y
    V = np.tile(v[:,None,None], (1, nx, ny))
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:len(z)]
    # Plotly expects 3D array; we'll use isomin/isomax to bound
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=V.flatten(),
        opacity=0.15,
        surface_count=12
    ))
    fig.update_layout(height=450, margin=dict(l=10,r=10,t=30,b=10), scene=dict(
        zaxis_title="Depth index (~inches)",
        xaxis_title="X",
        yaxis_title="Y"
    ), title=f"3D Volume: {prop}")
    return pn.pane.Plotly(fig)

def make_app():
    crop_select = pn.widgets.Select(name="Crop", options=list(CROPS.keys()), value="corn")
    prop_select = pn.widgets.Select(name="Property", options=PLOT_PROPS, value="K_ppm")

    @pn.depends(crop_select, watch=True)
    def _update_props(crop):
        # no-op, props static in this demo
        return

    @pn.depends(crop_select, prop_select)
    def views(crop, prop):
        _, layers, targets, texture_tgt, bacteria_tgt = load_crop(CROPS[crop])
        df = layers_to_depth_df(layers, step=1)
        profile = plot_profile(df, prop, targets)
        volume = make_volume(df, prop)
        # Simple layer table preview
        table = pn.widgets.DataFrame(layers, name="Layer Table", height=200)
        return pn.Column(
            pn.Row(profile, volume),
            table
        )

    header = pn.pane.Markdown("# Soil Profile Viewer (0–48 in)\nSelect a crop and property to view 2D & 3D.")
    return pn.Column(header, pn.Row(crop_select, prop_select), views)

if __name__ == "__main__":
    app = make_app()
    pn.serve(app, show=True, port=5006)
