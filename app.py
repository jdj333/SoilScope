
import json, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import panel as pn

pn.extension('plotly', sizing_mode='stretch_width')

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

CROPS = {"corn":"corn", "alfalfa":"alfalfa"}
PLOT_PROPS = ["pH","NO3_N_ppm","P_ppm_Olsen","K_ppm","Ca_ppm","Mg_ppm","S_ppm","Zn_ppm","Mn_ppm","B_ppm"]

TYPICAL_RULES = {
    "pH":            ("pH_decline",   {"delta24": -0.1, "delta48": -0.2}),
    "NO3_N_ppm":     ("surface_decay",{"factor0": 1.0, "factor12": 0.75, "factor24": 0.55, "factor36": 0.45, "factor48": 0.35}),
    "P_ppm_Olsen":   ("surface_decay",{"factor0": 1.0, "factor12": 0.85, "factor24": 0.7,  "factor36": 0.55, "factor48": 0.45}),
    "K_ppm":         ("slight_decay", {"factor0": 1.0, "factor12": 0.97, "factor24": 0.94, "factor36": 0.92, "factor48": 0.90}),
    "Ca_ppm":        ("flat",         {}),
    "Mg_ppm":        ("slight_decay", {"factor0": 1.0, "factor12": 0.96, "factor24": 0.94, "factor36": 0.92, "factor48": 0.90}),
    "S_ppm":         ("slight_decay", {"factor0": 1.0, "factor12": 0.95, "factor24": 0.90, "factor36": 0.86, "factor48": 0.82}),
    "Zn_ppm":        ("surface_decay",{"factor0": 1.0, "factor12": 0.8,  "factor24": 0.65, "factor36": 0.55, "factor48": 0.45}),
    "Mn_ppm":        ("surface_decay",{"factor0": 1.0, "factor12": 0.85, "factor24": 0.72, "factor36": 0.60, "factor48": 0.50}),
    "B_ppm":         ("slight_decay", {"factor0": 1.0, "factor12": 0.95, "factor24": 0.92, "factor36": 0.9,  "factor48": 0.88})
}

def load_crop(crop_key:str):
    path = os.path.join(DATA, f"{crop_key}.json")
    with open(path, "r") as f:
        d = json.load(f)
    layers = pd.DataFrame(d["sample_profile_layers"])
    return d, layers

def layers_to_depth_df(layers: pd.DataFrame, step: int = 1):
    max_depth = int(layers["depth_bottom_in"].max())
    grid = pd.DataFrame({"depth_in": np.arange(0, max_depth+1, step)})
    out = grid.copy()
    numeric_cols = [c for c in layers.columns if c not in ["layer_id","depth_top_in","depth_bottom_in"]]
    for col in numeric_cols:
        vals = np.zeros_like(grid["depth_in"], dtype=float)
        for _, row in layers.iterrows():
            mask = (grid["depth_in"] >= row["depth_top_in"]) & (grid["depth_in"] <= row["depth_bottom_in"])
            vals[mask] = row[col]
        out[col] = vals
    return out

def ideal_mid(meta: dict, prop: str, default: float = None):
    it = (meta.get("ideal_targets") or {}).get(prop)
    if it:
        return 0.5*(it["min"] + it["max"])
    return default if default is not None else 1.0

def modeled_typical_series(depth_in: np.ndarray, meta: dict, prop: str, fallback_series: np.ndarray):
    rule = TYPICAL_RULES.get(prop)
    if not rule:
        return fallback_series
    pattern, params = rule
    z = depth_in.astype(float)
    zmax = float(depth_in[-1]) if len(depth_in)>0 else 48.0

    # Baseline: measured at surface if present, else ideal midpoint, else mean of fallback
    surface_val = fallback_series[0] if np.isfinite(fallback_series[0]) else ideal_mid(meta, prop, default=np.nan)
    if not np.isfinite(surface_val):
        m = np.nanmean(fallback_series)
        surface_val = m if np.isfinite(m) else 1.0

    if pattern == "pH_decline":
        d24 = params.get("delta24", -0.1); d48 = params.get("delta48", -0.2)
        def interp_delta(zz):
            if zz <= 24: return d24*(zz/24.0)
            else: return d24 + (d48-d24)*((zz-24.0)/24.0)
        return np.array([surface_val + interp_delta(zz) for zz in z], dtype=float)

    if pattern in ("surface_decay","slight_decay"):
        f0  = params.get("factor0", 1.0)
        f12 = params.get("factor12", 0.9)
        f24 = params.get("factor24", 0.8)
        f36 = params.get("factor36", 0.7)
        f48 = params.get("factor48", 0.6)
        pts = np.array([0,12,24,36,48], dtype=float)
        fac = np.interp(z, pts, np.array([f0,f12,f24,f36,f48], dtype=float), left=f0, right=f48)
        return surface_val * fac

    if pattern == "increase":
        pct = params.get("pct48", 0.10)
        fac = 1.0 + (z/zmax)*pct
        return surface_val * fac

    if pattern == "flat":
        return np.full_like(z, surface_val, dtype=float)

    return fallback_series

def depth_conditioned_lines(df: pd.DataFrame, prop: str, ideal_targets_depth: dict, fallback: dict):
    depth = df["depth_in"].values
    min_line = np.full_like(depth, np.nan, dtype=float)
    max_line = np.full_like(depth, np.nan, dtype=float)
    segments = (ideal_targets_depth or {}).get(prop, [])
    if segments:
        for seg in segments:
            mask = (depth >= seg["depth_top_in"]) & (depth <= seg["depth_bottom_in"])
            min_line[mask] = seg["min"]; max_line[mask] = seg["max"]
        for arr in (min_line, max_line):
            last = np.nan
            for i in range(len(arr)):
                if not np.isnan(arr[i]): last = arr[i]
                else: arr[i] = last
            last = np.nan
            for i in range(len(arr)-1, -1, -1):
                if np.isnan(arr[i]): arr[i] = last
                else: last = arr[i]
    else:
        if prop in fallback:
            min_line[:] = fallback[prop]["min"]; max_line[:] = fallback[prop]["max"]
    return depth, min_line, max_line

def profile_png(df: pd.DataFrame, prop: str, meta: dict, mode: str):
    fig, ax = plt.subplots(figsize=(5,4), dpi=120)
    series = df[prop].values.copy()
    if mode == "Typical (Modeled)":
        series = modeled_typical_series(df["depth_in"].values, meta, prop, series)
    ax.plot(series, df["depth_in"], linewidth=2, label=f"{'Typical' if mode=='Typical (Modeled)' else 'Measured'} {prop}")
    ax.invert_yaxis(); ax.set_xlabel(prop); ax.set_ylabel("Depth (in)")
    depth, min_line, max_line = depth_conditioned_lines(df, prop, meta.get("ideal_targets_depth"), meta.get("ideal_targets", {}))
    if np.isfinite(min_line).any() and np.isfinite(max_line).any():
        ax.fill_betweenx(depth, min_line, max_line, alpha=0.12, label="Ideal band")
        ax.plot(min_line, depth, color="red", linestyle="--", linewidth=1.5, label="Min (by depth)")
        ax.plot(max_line, depth, color="red", linestyle="--", linewidth=1.5, label="Max (by depth)")
    ax.grid(True, linestyle=":"); ax.legend(loc="best", frameon=False)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return pn.pane.PNG(buf.getvalue(), height=350)

def volume_plot(df: pd.DataFrame, prop: str, meta: dict, mode: str, nx=20, ny=20):
    z = df["depth_in"].values
    v = df[prop].values.copy()
    if mode == "Typical (Modeled)":
        v = modeled_typical_series(z, meta, prop, v)
    V = np.tile(v[:,None,None], (1, nx, ny))
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:len(z)]
    fig = go.Figure(data=go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                   value=V.flatten(), opacity=0.15, surface_count=12,
                                   colorbar=dict(title=prop)))
    fig.update_layout(height=450, margin=dict(l=10,r=10,t=30,b=10),
                      scene=dict(zaxis_title="Depth index (~inches)", xaxis_title="X", yaxis_title="Y"),
                      title=f"3D Volume: {prop}")
    return pn.pane.Plotly(fig)

def make_app():
    crop = pn.widgets.Select(name="Crop", options=list(CROPS.keys()), value="corn")
    prop = pn.widgets.Select(name="Property", options=PLOT_PROPS, value="K_ppm")
    mode = pn.widgets.RadioButtonGroup(name="Data Source", options=["Measured (JSON)","Typical (Modeled)"], value="Typical (Modeled)")

    @pn.depends(crop, prop, mode)
    def view(crop, prop, mode):
        meta, layers = load_crop(crop)
        df = layers_to_depth_df(layers, step=1)
        prof = profile_png(df, prop, meta, mode)
        vol  = volume_plot(df, prop, meta, mode)
        table = pn.widgets.DataFrame(layers, name="Layer Table", height=240,
                                     sizing_mode="stretch_width", autosize_mode="fit_viewport")
        header = pn.pane.Markdown("### 2D Profile with **Depth-Conditioned** Min/Max Lines")
        return pn.Column(pn.Row(pn.Spacer(width=10), sizing_mode="stretch_width"),
                         pn.Row(pn.Column(header, prof), vol, sizing_mode="stretch_width"),
                         pn.Row(table, sizing_mode="stretch_width"),
                         sizing_mode="stretch_width")

    title = pn.pane.Markdown("# Soil Profile Viewer v4 (Measured vs Typical Profiles)")
    controls = pn.Row(crop, prop, mode, sizing_mode="stretch_width")
    return pn.Column(title, controls, view, sizing_mode="stretch_width")

if __name__ == "__main__":
    app = make_app()
    pn.serve(app, show=True, port=5009, websocket_origin=["localhost:5009","127.0.0.1:5009"])
