import json, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import panel as pn

# stretch all layouts/widgets to the page width
pn.extension('plotly', sizing_mode='stretch_width')

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

CROPS = {"corn": "corn", "alfalfa": "alfalfa"}
PLOT_PROPS = [
    "pH", "NO3_N_ppm", "P_ppm_Olsen", "K_ppm", "Ca_ppm",
    "Mg_ppm", "S_ppm", "Zn_ppm", "Mn_ppm", "B_ppm"
]

def load_crop(crop_key: str):
    path = os.path.join(DATA, f"{crop_key}.json")
    with open(path, "r") as f:
        d = json.load(f)
    layers = pd.DataFrame(d["sample_profile_layers"])
    return d, layers

def layers_to_depth_df(layers: pd.DataFrame, step: int = 1):
    max_depth = int(layers["depth_bottom_in"].max())
    grid = pd.DataFrame({"depth_in": np.arange(0, max_depth + 1, step)})
    out = grid.copy()
    numeric_cols = [c for c in layers.columns if c not in ["layer_id", "depth_top_in", "depth_bottom_in"]]
    for col in numeric_cols:
        vals = np.zeros_like(grid["depth_in"], dtype=float)
        for _, row in layers.iterrows():
            mask = (grid["depth_in"] >= row["depth_top_in"]) & (grid["depth_in"] <= row["depth_bottom_in"])
            vals[mask] = row[col]
        out[col] = vals
    return out

def depth_conditioned_lines(df: pd.DataFrame, prop: str, ideal_targets_depth: dict, fallback: dict):
    """
    Build min/max lines per depth from ideal_targets_depth if present;
    otherwise use the global fallback ideal_targets.
    """
    depth = df["depth_in"].values
    min_line = np.full_like(depth, np.nan, dtype=float)
    max_line = np.full_like(depth, np.nan, dtype=float)

    segments = (ideal_targets_depth or {}).get(prop, [])
    if segments:
        for seg in segments:
            mask = (depth >= seg["depth_top_in"]) & (depth <= seg["depth_bottom_in"])
            min_line[mask] = seg["min"]
            max_line[mask] = seg["max"]
        # forward/back fill any gaps
        for arr in (min_line, max_line):
            last = np.nan
            for i in range(len(arr)):
                if not np.isnan(arr[i]):
                    last = arr[i]
                else:
                    arr[i] = last
            last = np.nan
            for i in range(len(arr) - 1, -1, -1):
                if np.isnan(arr[i]):
                    arr[i] = last
                else:
                    last = arr[i]
    else:
        if prop in fallback:
            min_line[:] = fallback[prop]["min"]
            max_line[:] = fallback[prop]["max"]
    return depth, min_line, max_line

def profile_png(df: pd.DataFrame, prop: str, meta: dict):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.plot(df[prop], df["depth_in"], linewidth=2, label=f"Measured {prop}")
    ax.invert_yaxis()
    ax.set_xlabel(prop)
    ax.set_ylabel("Depth (in)")

    # draw depth-conditioned band + red min/max lines
    depth, min_line, max_line = depth_conditioned_lines(
        df, prop, meta.get("ideal_targets_depth"), meta.get("ideal_targets", {})
    )
    if np.isfinite(min_line).any() and np.isfinite(max_line).any():
        ax.fill_betweenx(depth, min_line, max_line, alpha=0.12, label="Ideal band")
        ax.plot(min_line, depth, color="red", linestyle="--", linewidth=1.5, label="Min (by depth)")
        ax.plot(max_line, depth, color="red", linestyle="--", linewidth=1.5, label="Max (by depth)")

    ax.grid(True, linestyle=":")
    ax.legend(loc="best", frameon=False)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return pn.pane.PNG(buf.getvalue(), height=350)

def volume_plot(df: pd.DataFrame, prop: str, nx=20, ny=20):
    z = df["depth_in"].values
    v = df[prop].values
    V = np.tile(v[:, None, None], (1, nx, ny))  # replicate along X/Y

    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:len(z)]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=V.flatten(),
        opacity=0.15, surface_count=12,
        colorbar=dict(title=prop)
    ))
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            zaxis_title="Depth index (~inches)",
            xaxis_title="X",
            yaxis_title="Y"
        ),
        title=f"3D Volume: {prop}"
    )
    return pn.pane.Plotly(fig)

def make_app():
    crop = pn.widgets.Select(name="Crop", options=list(CROPS.keys()), value="corn")
    prop = pn.widgets.Select(name="Property", options=PLOT_PROPS, value="K_ppm")

    @pn.depends(crop, prop)
    def view(crop, prop):
        meta, layers = load_crop(crop)
        df = layers_to_depth_df(layers, step=1)
        prof = profile_png(df, prop, meta)
        vol  = volume_plot(df, prop)

        # Table stretched to full width; fit columns to viewport if supported
        table = pn.widgets.DataFrame(
            layers,
            name="Layer Table",
            height=240,
            sizing_mode="stretch_width",
            autosize_mode="fit_viewport"  # remove if your Panel/Bokeh version doesn't support it
        )

        header = pn.pane.Markdown("### 2D Profile with **Depth-Conditioned** Min/Max Lines")
        return pn.Column(
            header,
            pn.Row(prof, vol, sizing_mode="stretch_width"),
            pn.Row(table, sizing_mode="stretch_width"),
            sizing_mode="stretch_width"
        )

    title = pn.pane.Markdown("# Soil Profile Viewer v3 (depth-conditioned min/max)")
    return pn.Column(title, pn.Row(crop, prop), view, sizing_mode="stretch_width")

if __name__ == "__main__":
    app = make_app()
    pn.serve(app, show=True, port=5008, websocket_origin=["localhost:5008", "127.0.0.1:5008"])
