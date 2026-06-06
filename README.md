# Samsung health record analyzer

A Streamlit dashboard for Samsung Health CSV exports from Android. It supports the two-line Samsung CSV format used by the bundled `sample_dataset` files, where the first row contains metadata and the second row contains real column headers.

## Dependencies

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0.3-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?style=flat-square&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.8.0-3F4F75?style=flat-square&logo=plotly&logoColor=white)

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

The app can load the bundled sample files or uploaded Samsung Health CSV files. Current visualizations focus on:

- `com.samsung.health.sleep_stage`: sleep-stage timelines, session totals, and stage distribution.
- `com.samsung.health.sleep`: sleep-session duration summaries when stage rows are not available.
- `com.samsung.health.weight`: weight and body-composition trends, including BMI when height is available.

Other Samsung Health CSV files still appear in the data explorer so their normalized rows can be inspected.
