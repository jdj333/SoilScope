# Soil Profile 2D/3D Viewer — v4 (Measured vs Typical Profiles)

Adds a **Data Source** toggle to switch the blue line between:
- **Measured (JSON)** values from your layered files.
- **Typical (Modeled)** values using agronomic depth patterns (surface stratification, mild declines, pH slight decline).

Tweak the patterns in `TYPICAL_RULES` inside `app_v4.py`.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_v4.py   # http://localhost:5009
```
