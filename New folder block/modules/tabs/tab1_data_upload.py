"""
Tab 1 — Data Upload.
"""

import pandas as pd
import streamlit as st

from modules.ui_helpers import section, info


def render_tab1():
    section("01", "Data Upload")
    info("Upload a <b>CSV</b> or <b>Excel</b> file. Rows = time periods, columns = variables.")
    uploaded = st.file_uploader("Choose file", type=["csv", "xlsx"])
    if uploaded:
        # ── BUG FIX ────────────────────────────────────────────────
        # `with tab1:` code runs on EVERY script rerun, not just when
        # Tab 1 is visible — that's how Streamlit tabs work. Since the
        # file stays inside the uploader widget across reruns, `if
        # uploaded:` was True on every single interaction anywhere in
        # the app (e.g. picking a media channel in Tab 4), which kept
        # re-deleting cfg_media / corr_vars / etc. below — resetting
        # the user's selection instantly, every time.
        #
        # Fix: only reload the dataframe and clear stale widget keys
        # the first time we see THIS particular file (tracked by name
        # + size), not on every rerun where it's merely still present
        # in the uploader.
        file_id = (uploaded.name, uploaded.size)
        is_new_file = st.session_state.get("_last_uploaded_file_id") != file_id

        if is_new_file:
            df_raw = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                      else pd.read_excel(uploaded))
            st.session_state.df = df_raw
            st.session_state.prophet_cols_added = []
            # Clear any stale widget states so Tab 3/4 rebuild cleanly with new data.
            # (safe_multiselect would handle this gracefully even without the clear,
            # but clearing avoids confusing leftover selections from a previous file.)
            for key in ["corr_vars", "cfg_nonmedia", "cfg_media", "cfg_price",
                        "cfg_comp_media", "cfg_comp_nonmedia", "media_cand",
                        "positive_beta_cols", "intercept_eff"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["_last_uploaded_file_id"] = file_id
        else:
            df_raw = st.session_state.df

        st.success(f"✅ Loaded **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Date col", "Found ✓" if any("date" in c.lower() for c in df_raw.columns) else "Not found")
        st.subheader("Preview")
        st.dataframe(df_raw.head(8), use_container_width=True)
        dtype_df = pd.DataFrame({
            "Column":   df_raw.columns,
            "Type":     df_raw.dtypes.astype(str).values,
            "Non-null": df_raw.notnull().sum().values,
            "Nulls":    df_raw.isnull().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
