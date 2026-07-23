"""
Session-state initialisation for the Rainbrain 2 app.
"""

import streamlit as st

_defaults = {
    "df": None,
    "prophet_results": None,
    "prophet_cols_added": [],   # list of "prophet_*" column names merged into df
    "config": None,
    "model_results": None,
    "model_fitted": False,
    "model_results_2": None,   # second dependent variable (same equation, e.g. ToM/Consideration)
    "model_fitted_2": False,
    "refit_config": None,      # working config for Tab 8 · Refine & Refit (starts as a copy of config)
    "refit_result": None,      # latest refit result dict (starts as the Tab 6 fit)
    "refit_history": [],       # log of refit steps: [{Step, Action, Variable, MAPE, R2}, ...]
    "refit_last_message": None,  # stashed success toast, shown once after the forced rerun in Tab 8
    "_last_uploaded_file_id": None,  # (name, size) of last processed upload — see tab1
}


def init_session_state():
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
