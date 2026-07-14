"""
Shared per-channel hyperparameter bounds widget — used by Tab 4
(Configuration, Dependent 1 & 2) and Tab 7 (Refine & Refit, for the bounds
of a newly added or re-opened variable). Extracted so both tabs render the
exact same widgets/keys/defaults instead of drifting apart.
"""

import pandas as pd
import streamlit as st


def render_per_channel_bounds(channel_cols, comp_cols, key_prefix, df,
                               use_hill, use_weibull,
                               price_cols=None, nonmedia_cols=None):
    """
    Renders per-channel bound widgets for media/comp-media columns (with
    adstock + transformation bounds) and optionally simplified Ls+delta
    bounds for price and non-media/control columns.

    Returns a dict {col: {param: (lo, hi), ...}}.
    """
    price_cols     = price_cols or []
    nonmedia_cols  = nonmedia_cols or []
    bounds: dict   = {}

    # ── Media / competitor-media channels ─────────────────────────
    all_adstock_cols = list(channel_cols)
    if all_adstock_cols:
        st.markdown("#### Own Media & Competitor Media")
        for col in all_adstock_cols:
            is_comp  = col in comp_cols
            col_label = f"📉 {col}  *(competitor)*" if is_comp else f"📺 {col}"
            vals    = df[col][df[col] > 0] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_p90 = float(vals.quantile(0.90)) if len(vals) > 0 else 1.0
            col_max = float(vals.max())    if len(vals) > 0 else 1.0
            def_s_lo = max(1e-6, col_med * 0.01)
            def_s_hi = col_max * 5.0

            with st.expander(col_label, expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · P90: **{col_p90:,.2f}** · "
                           f"max: **{col_max:,.2f}**")
                bounds[col] = {}

                # Beta persistence
                st.markdown("**Beta Persistence (Ls)**")
                lsc1, lsc2 = st.columns(2)
                with lsc1:
                    ls_lo = st.number_input("Ls min", 0.0, 1.0,
                                             0.8 if is_comp else 0.5, 0.01,
                                             key=f"{key_prefix}ls_lo_{col}")
                with lsc2:
                    ls_hi = st.number_input("Ls max", 0.0, 1.0, 1.0, 0.01,
                                             key=f"{key_prefix}ls_hi_{col}")
                bounds[col]["ls"] = (ls_lo, ls_hi)
                st.divider()

                # Transformation parameters
                if use_hill:
                    st.markdown("**Hill Slope (n)** — controls steepness of S-curve")
                    hnc1, hnc2 = st.columns(2)
                    with hnc1:
                        hn_lo = st.number_input("n min", 0.01, 15.0, 1.0, 0.1,
                                                 key=f"{key_prefix}hn_lo_{col}")
                    with hnc2:
                        hn_hi = st.number_input("n max", 0.01, 15.0, 15.0, 0.5,
                                                 key=f"{key_prefix}hn_hi_{col}")
                    bounds[col]["hill_n"] = (hn_lo, hn_hi)
                    st.divider()

                    st.markdown(f"**Hill Half-Saturation (S)** — data range: 0 – {col_max:,.0f}")
                    hsc1, hsc2 = st.columns(2)
                    with hsc1:
                        hs_lo = st.number_input("S min", 1e-9, 1e12,
                                                 float(f"{def_s_lo:.4g}"),
                                                 key=f"{key_prefix}hs_lo_{col}",
                                                 format="%.4g")
                    with hsc2:
                        hs_hi = st.number_input("S max", 1e-9, 1e12,
                                                 float(f"{def_s_hi:.4g}"),
                                                 key=f"{key_prefix}hs_hi_{col}",
                                                 format="%.4g")
                    bounds[col]["hill_s"] = (hs_lo, hs_hi)
                else:
                    st.markdown("**Power Exponent (n)** — n ∈ (0,1] for diminishing returns")
                    pnc1, pnc2 = st.columns(2)
                    with pnc1:
                        pn_lo = st.number_input("n min", 0.001, 1.0, 0.01, 0.01,
                                                 key=f"{key_prefix}pn_lo_{col}")
                    with pnc2:
                        pn_hi = st.number_input("n max", 0.001, 1.0, 1.0, 0.01,
                                                 key=f"{key_prefix}pn_hi_{col}")
                    bounds[col]["transform_n"] = (pn_lo, pn_hi)
                st.divider()

                # Adstock parameters
                if use_weibull:
                    st.markdown("**Weibull Adstock — Shape (k) and Scale (λ)**")
                    st.caption(
                        "Shape k controls the weight distribution across lags "
                        "(k < 1: front-loaded, k = 1: exponential, k > 1: bell-shaped). "
                        "Scale λ controls how quickly weights decay."
                    )
                    wa1, wa2, wa3, wa4 = st.columns(4)
                    with wa1: wa_klo = st.number_input("k min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"{key_prefix}wa_klo_{col}")
                    with wa2: wa_khi = st.number_input("k max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"{key_prefix}wa_khi_{col}")
                    with wa3: wl_lo  = st.number_input("λ min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"{key_prefix}wl_lo_{col}")
                    with wa4: wl_hi  = st.number_input("λ max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"{key_prefix}wl_hi_{col}")
                    bounds[col]["adstock_shape"] = (wa_klo, wa_khi)
                    bounds[col]["adstock_scale"] = (wl_lo,  wl_hi)
                else:
                    st.caption(
                        "ℹ️ Instant adstock has no separate λ decay parameter — "
                        "carryover is carried entirely by this channel's **Beta "
                        "Persistence (Ls)** above. A separate adstock λ on top of "
                        "Ls would double-count the same carryover."
                    )

                st.caption(f"✅ Bounds set for: {', '.join(bounds[col].keys())}")
    else:
        st.info("Select media channels in Section A first.")

    # ── Price columns ──────────────────────────────────────────────
    if price_cols:
        st.markdown("#### Price Variables")
        for col in price_cols:
            vals    = df[col][df[col] > 0] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_max = float(vals.max())    if len(vals) > 0 else 1.0
            with st.expander(f"💲 {col}", expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · max: **{col_max:,.2f}**")
                bounds[col] = {}
                st.markdown("**Beta Persistence (Ls_price)**")
                pc1, pc2 = st.columns(2)
                with pc1:
                    p_ls_lo = st.number_input("Ls min", 0.0, 1.0, 0.8, 0.01,
                                               key=f"{key_prefix}p_ls_lo_{col}")
                with pc2:
                    p_ls_hi = st.number_input("Ls max", 0.0, 1.0, 1.0, 0.01,
                                               key=f"{key_prefix}p_ls_hi_{col}")
                bounds[col]["ls"] = (p_ls_lo, p_ls_hi)
                st.caption(f"✅ Bounds set for: {', '.join(bounds[col].keys())}")

    # ── Non-media / control columns ────────────────────────────────
    if nonmedia_cols:
        st.markdown("#### Non-Media / Control Variables")
        for col in nonmedia_cols:
            vals    = df[col] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_max = float(abs(vals).max()) if len(vals) > 0 else 1.0
            with st.expander(f"🗂️ {col}", expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · max: **{col_max:,.2f}**")
                bounds[col] = {}
                st.markdown("**Beta Persistence (Ls_nonmedia)**")
                nm1, nm2 = st.columns(2)
                with nm1:
                    nm_ls_lo = st.number_input("Ls min", 0.0, 1.0, 0.8, 0.01,
                                                key=f"{key_prefix}nm_ls_lo_{col}")
                with nm2:
                    nm_ls_hi = st.number_input("Ls max", 0.0, 1.0, 1.0, 0.01,
                                                key=f"{key_prefix}nm_ls_hi_{col}")
                bounds[col]["ls"] = (nm_ls_lo, nm_ls_hi)
                st.caption(f"✅ Bounds set for: {', '.join(bounds[col].keys())}")

    return bounds
