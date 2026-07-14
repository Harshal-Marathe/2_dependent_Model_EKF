"""
Sidebar: branding, dependency status, step checklist, prophet column log.
"""

import streamlit as st


def render_sidebar(nevergrad_available: bool):
    with st.sidebar:
        st.markdown("## 📡 2 dependent model")
        st.markdown("**Extended Kalman Filter**  \nMarketing Mix Modeling")
        
        st.divider()
        steps = {
            "1 · Data Upload":    st.session_state.df is not None,
            "2 · Prophet Decomp": st.session_state.prophet_results is not None,
            "3 · Correlation":    st.session_state.df is not None,
            "4 · Configuration":  st.session_state.config is not None,
            "5 · Run Model":      st.session_state.model_fitted,
            "6 · Results":        st.session_state.model_fitted,
        }
        for label, done in steps.items():
            st.markdown(f"`{'✅' if done else '○'}` {label}")

        if st.session_state.prophet_cols_added:
            st.divider()
            st.caption("📌 Prophet cols in dataset:")
            for pc in st.session_state.prophet_cols_added:
                st.caption(f"  • {pc}")
        st.divider()
        st.caption("Complete steps 1 → 5 in order.")
