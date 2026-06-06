# Samsung health record analyzer

A Streamlit dashboard for Samsung Health CSV exports from Android. It supports the two-line Samsung CSV format used by the bundled `sample_dataset` files, where the first row contains metadata and the second row contains real column headers.

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
