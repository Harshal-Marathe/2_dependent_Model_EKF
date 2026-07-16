"""
EKF MMM Platform · Rainbrain 2 v8 — Main entry point.

Run with:  streamlit run app.py

This file only wires the page together: page config, global styles,
session-state init, sidebar, and the six tabs. All business logic lives
under modules/.
"""

import streamlit as st

from modules.dependencies import PROPHET_AVAILABLE, HOLIDAYS_AVAILABLE, NEVERGRAD_AVAILABLE
from modules.styles import apply_styles
from modules.state import init_session_state
from modules.sidebar import render_sidebar

from modules.tabs.tab1_data_upload import render_tab1
from modules.tabs.tab2_prophet import render_tab2
from modules.tabs.tab3_correlation import render_tab3
from modules.tabs.tab4_configuration import render_tab4
from modules.tabs.tab5_run_model import render_tab5
from modules.tabs.tab6_results import render_tab6
from modules.tabs.tab7_refine import render_tab7


# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the first Streamlit call)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="2 Dependent Model", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------- LOGIN ----------------
USERNAME = "harshal"
PASSWORD = "123456"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()
# ---------------------------------------

apply_styles()
init_session_state()
render_sidebar(NEVERGRAD_AVAILABLE)


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1 · Data Upload", "2 · Prophet Decomp", "3 · Correlation & Synergy",
    "4 · Configuration", "5 · Run Model", "6 · Results & ROI",
    "7 · Refine & Refit",
])

with tab1:
    render_tab1()

with tab2:
    render_tab2(PROPHET_AVAILABLE, HOLIDAYS_AVAILABLE)

with tab3:
    render_tab3()

with tab4:
    render_tab4()

with tab5:
    render_tab5(NEVERGRAD_AVAILABLE)

with tab6:
    render_tab6()

with tab7:
    render_tab7()
