# NLR ESIF Streamlit App

Run from this folder:

```powershell
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

The app reuses `data/esif_dc_power_metrics.parquet` when present. If the file is missing, keep "Download data if missing" enabled in the sidebar and the app will download the Parquet file from the NLR submission page.

The app keeps column provenance explicit:

- PUE plots from `pue`.
- ERE plots from `ere`.
- `energy_reuse` remains available as a separate imported series.
- ERF is calculated as `(1 - ere / pue) * 100`.
