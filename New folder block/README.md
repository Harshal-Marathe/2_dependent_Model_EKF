# Rainbrain 2 v8 — EKF Marketing Mix Modeling Platform

Modularized version of the original single-file Streamlit app.

## Structure

```
rainbrain_app/
├── app.py                       # Entry point — run this with `streamlit run app.py`
├── requirements.txt
└── modules/
    ├── dependencies.py          # Optional-package detection (Prophet, nevergrad)
    ├── styles.py                # Global CSS
    ├── state.py                 # st.session_state defaults
    ├── sidebar.py                # Sidebar (branding, step checklist)
    ├── ui_helpers.py             # section()/info()/safe_multiselect() etc.
    ├── transforms.py             # Hill saturation + adstock functions
    ├── params.py                 # _make_globals() / unpack_theta()
    ├── kalman.py                  # EKF forward filter + RTS smoother
    ├── bounds.py                  # theta0 + per-channel bounds builder
    ├── optimizer.py               # Nevergrad multi-objective optimizer
    ├── pipeline.py                 # run_full_ekf_pipeline() — ties it all together
    └── tabs/
        ├── tab1_data_upload.py
        ├── tab2_prophet.py
        ├── tab3_correlation.py
        ├── tab4_configuration.py
        ├── tab5_run_model.py
        └── tab6_results.py
```

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Prophet and Nevergrad are optional — the app detects their availability
at import time (`modules/dependencies.py`) and degrades gracefully
(Tab 2 / the Nevergrad optimizer option are disabled with a clear
message if not installed).

## Key fix preserved from the original file

`modules/ui_helpers.py::safe_multiselect()` is a drop-in replacement for
`st.multiselect` that sanitizes any stored selection against the current
`options=` list on every render, so Streamlit can never raise
`StreamlitAPIException` on a stale value (e.g. a prophet column that
was merged into the dataset out-of-band in Tab 2). It also supports a
`require=` argument to permanently re-inject specific values (like new
prophet columns) into a selection on every rerun, regardless of what
else the user has clicked.
