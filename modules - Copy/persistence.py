"""
Save / Load the full model workspace to a single downloadable file, so
the user can come back — later in the same session, or in a brand-new
browser session on a different day — and restore exactly where they left
off (uploaded data, Tab 5 configuration, fitted model(s), refit history,
and the Tab 2 Sales Modeling Basis / ROI settings) without redoing Tabs
1-8 from scratch.

Format: a single pickled dict (".rbe" — just a renamed .pkl so it's less
likely to be double-clicked/opened by something else). Pickle is used
rather than JSON because the saved state includes a pandas DataFrame and
numpy arrays nested inside the fitted-model result dict.
"""

import io
import pickle
from datetime import datetime

import streamlit as st

_FORMAT_VERSION = 1

# Every session-state key captured in a saved workspace. Deliberately
# excludes purely-transient UI/widget state (e.g. slider positions) —
# those just fall back to their defaults on reload, which is harmless.
_WORKSPACE_KEYS = [
    "df",
    "config",
    "model_results", "model_fitted",
    "model_results_2", "model_fitted_2",
    "prophet_cols_added",
    "refit_config", "refit_result", "refit_history",
    "sales_modeling_basis", "sales_price_col", "sales_avg_price",
    "sales_volume_unit", "price_conversion_factor",
]


def _prophet_results_for_save():
    """Prophet's fitted model object wraps a Stan backend that isn't
    reliably picklable across machines/versions, and isn't needed to
    restore the app anyway — the useful part is its forecast/components
    output (and by the time Prophet columns are merged into `df`, that's
    already saved there too). Keep everything except the raw model."""
    pr = st.session_state.get("prophet_results")
    if not pr:
        return None
    return {k: v for k, v in pr.items() if k != "model"}


def build_workspace_bytes() -> bytes:
    """Pickle the current app state into a single portable file."""
    bundle = {
        "__format_version__": _FORMAT_VERSION,
        "__saved_at__": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k in _WORKSPACE_KEYS:
        bundle[k] = st.session_state.get(k)
    bundle["prophet_results"] = _prophet_results_for_save()

    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def restore_workspace(file_bytes: bytes):
    """Unpickle a saved workspace and write it back into session_state.

    Returns (ok: bool, message: str).
    """
    try:
        bundle = pickle.loads(file_bytes)
    except Exception as e:
        return False, f"Couldn't read this file — it doesn't look like a saved model ({e})."

    if not isinstance(bundle, dict) or "__format_version__" not in bundle:
        return False, "This file doesn't look like a saved model workspace."

    for k in _WORKSPACE_KEYS:
        if k in bundle:
            st.session_state[k] = bundle[k]
    if "prophet_results" in bundle:
        st.session_state["prophet_results"] = bundle["prophet_results"]

    saved_at = bundle.get("__saved_at__", "an earlier session")
    return True, f"✅ Model workspace restored (saved on {saved_at})."
