
# Soil Profile 2D/3D Viewer (Dent Corn & Alfalfa)

An example Python app to load **layered 0–48 inch soil profiles** (with macros, micros, texture, pH)
and visualize them in **2D** (matplotlib depth profiles with ideal bands) and **3D** (Plotly volume).


![Soil profile 2D chart](data/example_chart.png)


## Contents

- `app.py` — Panel app with a crop selector and property selector; shows 2D depth profile + 3D volume.
- `data/corn.json` — Sample dent corn dataset with **ideal_targets** and **sample_profile_layers**.
- `data/alfalfa.json` — Sample alfalfa dataset with **ideal_targets** and **sample_profile_layers**.
- `requirements.txt` — Minimal dependencies.
- This `README.md`.

> Notes: Targets and values are **illustrative** and should be replaced with your local agronomic guidelines.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

### Run the app

```bash
python3 app.py
# or with Panel explicitly (optional)
panel serve app.py --show --port 5006
```

Then open your browser to the URL displayed (defaults to http://localhost:5006).

## Data format

Each JSON file contains:
- `crop`: name
- `ideal_targets`: dict of property -> {min, max}
- `texture_targets_pct`: target texture percent
- `bacteria_targets`: list of beneficial genera (illustrative)
- `sample_profile_layers`: **list of layer dicts** with keys:
  - `layer_id`, `depth_top_in`, `depth_bottom_in`
  - `pH`, `OM_pct`, `CEC_cmolc_kg`
  - Macro/micro nutrients (e.g., `NO3_N_ppm`, `P_ppm_Olsen`, `K_ppm`, `Ca_ppm`, `Mg_ppm`, `S_ppm`, `Zn_ppm`, `Mn_ppm`, `B_ppm`)
  - Texture (`sand_pct`, `silt_pct`, `clay_pct`)

## Customize

- Add your own layers to `data/*.json`.
- Add more properties to `PLOT_PROPS` in `app.py`.
- Replace ideal target bands in the JSON to match your region & lab method (e.g., Olsen vs Bray).

## 3D notes

The demo builds a simple rectangular volume by **replicating each depth's scalar across X and Y**. If you have spatial grids,
replace `make_volume` with your `nx, ny` raster per depth for a true 3D block model.

## References

1. **Corn nutrient sufficiency ranges** — Iowa State University Extension and Outreach. *Soil Fertility Interpretations for Corn.* PM 1688 (updated 2021). Available at: https://store.extension.iastate.edu/Product/pm1688-pdf  
2. **Corn & alfalfa micronutrient guidelines** — University of Minnesota Extension. *Nutrient Management for Commercial Corn Production.* https://extension.umn.edu/crop-specific-needs/nutrient-management-corn  
3. **Alfalfa nutrient recommendations** — Penn State Extension. *Nutrient Management for Forage Crops: Alfalfa.* https://extension.psu.edu/nutrient-management-for-forage-crops-alfalfa  
4. **Soil test interpretation for phosphorus and potassium** — University of Wisconsin–Madison Division of Extension. *Nutrient Application Guidelines for Field, Vegetable, and Fruit Crops in Wisconsin (A2809).* https://extension.soils.wisc.edu/a2809  
5. **Boron and micronutrients for alfalfa** — University of California Agriculture & Natural Resources. *Alfalfa Fertility Management.* https://alfalfa.ucdavis.edu/+fertility.html  
6. **CEC, pH, and OM guidelines** — USDA NRCS. *Soil Quality Indicators: Cation Exchange Capacity.* https://www.nrcs.usda.gov/resources/education-and-outreach/soil-quality-indicators/cation-exchange-capacity  
7. **Soil texture recommendations for irrigated corn and alfalfa** — USDA NRCS Soil Health Technical Note No. 430–3. *Soil Texture and Water.* https://www.nrcs.usda.gov/sites/default/files/2022-10/Soil%20Texture%20and%20Water.pdf  

These sources outline the commonly accepted target ranges for macronutrients, micronutrients, pH, cation exchange capacity, organic matter, and soil texture for dent corn and alfalfa. The `corn.json` and `alfalfa.json` files in this app were defined using values synthesized from these recommendations for **irrigated production systems in temperate climates**.

## License

MIT (example code).

