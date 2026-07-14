"""
Tab 4 — Model Configuration: variable roles, positive-beta constraints,
cross-media learning, adstock type, transformation type, lag count,
per-variable hyperparameter bounds, train/test split, and saving config.
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_helpers import (
    section, info, positive_info, per_channel_info, prophet_info,
    weibull_placeholder, need_data, safe_multiselect,
)
from modules.bounds_ui import render_per_channel_bounds


def render_tab4():
    section("04", "Model Configuration")
    if st.session_state.df is None: need_data()

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if st.session_state.prophet_cols_added:
        missing_now = [c for c in st.session_state.prophet_cols_added if c not in num_cols]
        if missing_now:
            st.error(
                f"⚠️ Prophet column(s) were added but are no longer numeric / present in the "
                f"dataset: `{'`, `'.join(missing_now)}`. Re-run the prophet merge in Tab 2."
            )
        prophet_info(
            f"📌 Prophet columns available: "
            f"<b>{', '.join(st.session_state.prophet_cols_added)}</b>. "
            "They are pre-selected in <b>Non-media / organic</b> below — "
            "just save the configuration to include them in the model."
        )

    # ── A. Variable Selection ─────────────────────────────────────────
    st.markdown("### A · Variable Selection")
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Dependent variable (KPI)", num_cols, key="cfg_target")
    with col2:
        remaining = [c for c in num_cols if c != target]
        media = safe_multiselect("📺 Media / paid channels", options=remaining,
                                  default=[], key="cfg_media")

    remaining2 = [c for c in remaining if c not in media]
    col3, col4 = st.columns(2)
    with col3:
        prophet_in_scope = [c for c in st.session_state.prophet_cols_added
                             if c in remaining2]
        non_media = safe_multiselect(
            "🗂️ Non-media / organic (incl. prophet columns)",
            options=remaining2,
            require=prophet_in_scope,
            key="cfg_nonmedia",
        )
    with col4:
        remaining3 = [c for c in remaining2 if c not in non_media]
        price_vars = safe_multiselect("💲 Price variables", options=remaining3,
                                       default=[], key="cfg_price")

    remaining4 = [c for c in remaining3 if c not in price_vars]
    col5, col6 = st.columns(2)
    with col5:
        comp_media = safe_multiselect("📉 Competitor media", options=remaining4,
                                       default=[], key="cfg_comp_media")
    with col6:
        remaining5 = [c for c in remaining4 if c not in comp_media]
        comp_nonmedia = safe_multiselect("📉 Competitor non-media", options=remaining5,
                                          default=[], key="cfg_comp_nonmedia")

    st.divider()

    # ── A2. Second Dependent Variable (optional, multi-dependent MMM) ──
    st.markdown("### A2 · Second Dependent Variable (optional)")
    info(
        "Common in MMM when you want two KPIs explained by the <b>same media "
        "mix</b> — e.g. <b>Sales Volume</b> as Dependent 1 and "
        "<b>Top-of-Mind / Consideration</b> as Dependent 2. Both dependents share "
        "the same regressors x_t (media, non-media, price, competitor) and the "
        "same adstock/transformation family, but each keeps its own betas. "
        "Unlike two separate models, Dependent 1 and Dependent 2 are fitted "
        "<b>jointly</b> with a single bivariate Kalman filter: one optimizer run "
        "estimates both equations' parameters together with the correlation "
        "between their errors, so a surprise in one KPI at time t also informs "
        "the state update of the other KPI at that same t."
    )
    enable_second_dependent = st.checkbox(
        "➕ Enable a second dependent variable (jointly fitted, bivariate Kalman filter)",
        key="cfg_enable_target2",
    )
    target2 = None
    if enable_second_dependent:
        used_cols = {target, *media, *non_media, *price_vars, *comp_media, *comp_nonmedia}
        target2_options = [c for c in num_cols if c not in used_cols]
        if target2_options:
            target2 = st.selectbox(
                "🎯 Dependent variable 2 (KPI) — e.g. Top-of-Mind / Consideration",
                target2_options, key="cfg_target2",
            )
            st.caption(
                f"Dependent 2 will be modeled with the **same predictors** as "
                f"Dependent 1 (`{target}`): {len(media)} media · {len(non_media)} "
                f"non-media · {len(price_vars)} price · {len(comp_media)} comp-media · "
                f"{len(comp_nonmedia)} comp-non-media, and the same adstock/transform "
                f"choices set in Section D below. Dependent 1 and Dependent 2 will be "
                f"fitted **jointly** in Tab 5 with a bivariate Kalman filter "
                f"(shared time index, correlated errors) — not as two separate models."
            )
        else:
            st.warning(
                "No remaining numeric columns are available to use as a second "
                "dependent variable (everything is already used as a predictor)."
            )
            enable_second_dependent = False

    st.divider()

    # ── A3. Predictor Variables — Dependent 2 (optional independent set) ──
    media_2 = list(media); non_media_2 = list(non_media)
    comp_media_2 = list(comp_media); comp_nonmedia_2 = list(comp_nonmedia)
    price_vars_2 = list(price_vars)
    use_price_2 = False
    different_predictors_2 = False
    if enable_second_dependent and target2:
        st.markdown("### A3 · Predictor Variables — Dependent 2 (optional)")
        info(
            "By default Dependent 2 reuses the exact same media / non-media / price / "
            "competitor variables as Dependent 1 (x_t is shared). If Dependent 2 is "
            "actually driven by a <b>different — but possibly overlapping — set of "
            "variables</b> (e.g. only TV and Digital feed Consideration, while Sales "
            "also responds to Price and a promo flag) enable independent selection "
            "below. Any variable may appear in <b>both</b> Dependent 1's and "
            "Dependent 2's predictor sets at once — the two equations still share the "
            "same underlying time index and are fitted jointly, but each equation's "
            "own observation matrix only includes the variables assigned to it."
        )
        different_predictors_2 = st.checkbox(
            "🔀 Use a different predictor set for Dependent 2",
            key="cfg_diff_predictors_2",
        )
        if different_predictors_2:
            options2 = [c for c in num_cols if c not in {target, target2}]
            dc1, dc2 = st.columns(2)
            with dc1:
                media_2 = safe_multiselect(
                    "📺 Media / paid channels — Dep 2", options=options2,
                    default=[c for c in media if c in options2], key="cfg_media_2")
            with dc2:
                opts_nm2 = [c for c in options2 if c not in media_2]
                non_media_2 = safe_multiselect(
                    "🗂️ Non-media / organic — Dep 2", options=opts_nm2,
                    default=[c for c in non_media if c in opts_nm2], key="cfg_nonmedia_2")

            dc3, dc4 = st.columns(2)
            with dc3:
                opts_price2 = [c for c in options2 if c not in media_2 and c not in non_media_2]
                price_vars_2 = safe_multiselect(
                    "💲 Price variables — Dep 2", options=opts_price2,
                    default=[c for c in price_vars if c in opts_price2], key="cfg_price_2")
            with dc4:
                opts_cm2 = [c for c in options2
                            if c not in media_2 and c not in non_media_2 and c not in price_vars_2]
                comp_media_2 = safe_multiselect(
                    "📉 Competitor media — Dep 2", options=opts_cm2,
                    default=[c for c in comp_media if c in opts_cm2], key="cfg_comp_media_2")

            opts_cnm2 = [c for c in options2
                         if c not in media_2 and c not in non_media_2
                         and c not in price_vars_2 and c not in comp_media_2]
            comp_nonmedia_2 = safe_multiselect(
                "📉 Competitor non-media — Dep 2", options=opts_cnm2,
                default=[c for c in comp_nonmedia if c in opts_cnm2], key="cfg_comp_nonmedia_2")

            use_price_2 = st.checkbox(
                "Include price effects — Dep 2", value=bool(price_vars_2), key="cfg_use_price_2")

            if not media_2:
                st.warning("Dependent 2 needs at least one media channel — falling back to Dependent 1's media list.")
                media_2 = list(media)

            shared = (
                (set(media) | set(non_media) | set(price_vars) | set(comp_media) | set(comp_nonmedia)) &
                (set(media_2) | set(non_media_2) | set(price_vars_2) | set(comp_media_2) | set(comp_nonmedia_2))
            )
            st.caption(
                f"Dep 2 predictors: **{len(media_2)}** media · **{len(non_media_2)}** non-media · "
                f"**{len(price_vars_2)}** price · **{len(comp_media_2)}** comp-media · "
                f"**{len(comp_nonmedia_2)}** comp-non-media. "
                f"**{len(shared)}** variable(s) shared with Dependent 1: "
                f"{', '.join(sorted(shared)) if shared else 'none'}."
            )
        else:
            use_price_2 = None  # resolved after Section F, once use_price (Dep 1) is known

    st.divider()

    # ── B. Beta Sign Constraints ─────────────────────────────────────
    st.markdown("### B · Beta Sign Constraints")

    all_own   = list(media) + list(non_media)
    all_comp  = list(comp_media) + list(comp_nonmedia)
    all_price = list(price_vars)
    all_sign_cols = all_own + all_comp + all_price

    bcol1, bcol2 = st.columns(2)

    with bcol1:
        positive_info(
            "📈 <b>Positive Beta Enforcement</b> — Variables selected here must "
            "contribute <b>positively</b> to the KPI. The optimizer enforces "
            "non-negative <code>delta</code> bounds and floors filtered betas at zero."
        )
        if all_own:
            positive_beta_cols = safe_multiselect(
                "📈 Variables that must have POSITIVE betas",
                options=all_own, default=list(media), key="positive_beta_cols")
        else:
            positive_beta_cols = []
            st.info("Select media or non-media channels in Section A first.")

    with bcol2:
        st.markdown(
            '<div style="background:#1e293b;border-left:4px solid #ef4444;'
            'padding:8px 12px;border-radius:4px;margin-bottom:8px;">'
            '📉 <b>Negative Beta Enforcement</b> — Variables selected here must '
            'contribute <b>negatively</b> to the KPI. The optimizer enforces '
            'non-positive <code>delta</code> bounds and caps filtered betas at zero.'
            '</div>', unsafe_allow_html=True)
        neg_candidates = all_comp + all_price + all_own
        # default: competitor media + price are naturally negative
        neg_defaults = list(comp_media) + list(comp_nonmedia) + list(price_vars)
        neg_defaults = [c for c in neg_defaults if c in neg_candidates]
        if neg_candidates:
            negative_beta_cols = safe_multiselect(
                "📉 Variables that must have NEGATIVE betas",
                options=neg_candidates, default=neg_defaults, key="negative_beta_cols")
            # Remove overlap — positive takes precedence
            negative_beta_cols = [c for c in negative_beta_cols
                                   if c not in positive_beta_cols]
        else:
            negative_beta_cols = []
            st.info("Select channels in Section A first.")

    if positive_beta_cols or negative_beta_cols:
        st.caption(
            f"🔒 Positive: **{', '.join(positive_beta_cols) or 'none'}**  |  "
            f"📉 Negative: **{', '.join(negative_beta_cols) or 'none'}**"
        )

    # ── B2. Beta Sign Constraints — Dependent 2 (only if predictors differ) ──
    positive_beta_cols_2 = list(positive_beta_cols)
    negative_beta_cols_2 = list(negative_beta_cols)
    if enable_second_dependent and target2 and different_predictors_2:
        st.divider()
        st.markdown("### B2 · Beta Sign Constraints — Dependent 2")
        all_own_2  = list(media_2) + list(non_media_2)
        all_comp_2 = list(comp_media_2) + list(comp_nonmedia_2)
        bcol1b, bcol2b = st.columns(2)
        with bcol1b:
            if all_own_2:
                positive_beta_cols_2 = safe_multiselect(
                    "📈 Dep 2 — variables that must have POSITIVE betas",
                    options=all_own_2, default=[c for c in media_2 if c in all_own_2],
                    key="positive_beta_cols_2")
            else:
                positive_beta_cols_2 = []
                st.info("Select Dep 2 media or non-media channels in Section A3 first.")
        with bcol2b:
            neg_candidates_2 = all_comp_2 + list(price_vars_2) + all_own_2
            neg_defaults_2 = [c for c in (list(comp_media_2) + list(comp_nonmedia_2) + list(price_vars_2))
                               if c in neg_candidates_2]
            if neg_candidates_2:
                negative_beta_cols_2 = safe_multiselect(
                    "📉 Dep 2 — variables that must have NEGATIVE betas",
                    options=neg_candidates_2, default=neg_defaults_2,
                    key="negative_beta_cols_2")
                negative_beta_cols_2 = [c for c in negative_beta_cols_2 if c not in positive_beta_cols_2]
            else:
                negative_beta_cols_2 = []
                st.info("Select Dep 2 channels in Section A3 first.")
        st.caption(
            f"🔒 Dep 2 Positive: **{', '.join(positive_beta_cols_2) or 'none'}**  |  "
            f"📉 Dep 2 Negative: **{', '.join(negative_beta_cols_2) or 'none'}**"
        )

    st.divider()

    # ── C. Cross-media Learning ───────────────────────────────────────
    st.markdown("### C · Cross-media Learning")
    info("Define which channels learn from each other.")
    cross_map = {}
    if media:
        for tgt in media:
            sources = safe_multiselect(
                f"Channels that influence **{tgt}**",
                options=[m for m in media if m != tgt], default=[], key=f"cross_{tgt}")
            if sources:
                cross_map[tgt] = set(sources)

    cross_map_2 = dict(cross_map)
    if enable_second_dependent and target2 and different_predictors_2 and media_2:
        st.markdown("#### Cross-media Learning — Dependent 2")
        cross_map_2 = {}
        for tgt in media_2:
            sources = safe_multiselect(
                f"Dep 2 — channels that influence **{tgt}**",
                options=[m for m in media_2 if m != tgt], default=[], key=f"cross2_{tgt}")
            if sources:
                cross_map_2[tgt] = set(sources)

    st.divider()

    # ── D. Adstock & Transformation ───────────────────────────────────
    st.markdown("### D · Adstock Function & Media Transformation")

    info(
        "Choose <b>Adstock</b> (how past spend carries over) and <b>Transformation</b> "
        "(how raw spend maps to marketing effectiveness). These two choices define the "
        "state equation used for all media beta coefficients."
    )

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("#### Adstock (Carry-over)")
        adstock_choice = st.radio(
            "Adstock type",
            ["Instant (Nerlove-Arrow / geometric)", "Delayed (Weibull distributed lag)"],
            horizontal=False,
            key="adstock_type_radio",
            help=(
                "**Instant**: β_t = λ·β_{t-1} + δ·f(x_t). "
                "Fast geometric decay — parameter λ ∈ (0,1).\n\n"
                "**Delayed (Weibull)**: β_t = Σ_{l=0}^{L} w_l·x_{t-l} + δ·f(x_t). "
                "Weighted lag distribution — parameters shape k, scale λ, "
                "and number of lags L (0–8)."
            ),
        )
        use_weibull = adstock_choice.startswith("Delayed")

        if use_weibull:
            n_lags = st.number_input(
                "Number of lags to consider (L)",
                min_value=0, max_value=8, value=8, step=1,
                key="adstock_n_lags",
                help=(
                    "Weibull weights are computed for lag = 0, 1, …, L "
                    "(L+1 weights total) and normalised to sum to 1. "
                    "L = 0 means only the current period is used (no carry-over)."
                ),
            )
            st.caption(
                f"📐 Weibull PDF: w_lag = (k/λ) · ((lag+1)/λ)^(k−1) · exp(−((lag+1)/λ)^k), "
                f"normalised over lag = 0…{n_lags} ({n_lags + 1} weights). "
                "Parameters **shape k** and **scale λ** are fitted per channel "
                "(set bounds below)."
            )
        else:
            n_lags = 8  # default, unused for instant

    with dcol2:
        st.markdown("#### Transformation (Response Curve)")
        transform_choice = st.radio(
            "Transformation type",
            ["Hill (S-curve saturation)", "Power (diminishing returns)"],
            horizontal=False,
            key="transform_type_radio",
            help=(
                "**Hill**: f(x) = x^n / (x^n + S^n). "
                "S-shaped saturation curve — parameters n ∈ [1,15], S > 0.\n\n"
                "**Power**: f(x) = x^n. "
                "Pure diminishing returns — parameter n ∈ (0,1]."
            ),
        )
        use_hill = transform_choice.startswith("Hill")

    # Summary box showing the active state equation
    adstock_label = "Delayed (Weibull)" if use_weibull else "Instant (λ)"
    transform_label = "Hill(x; n, S)" if use_hill else "x^n"
    transform_type_str = "hill" if use_hill else "power"
    adstock_type_str   = "weibull" if use_weibull else "instant"

    if use_weibull and use_hill:
        eq_text = (
            "β_{i,t} = **Σ w_l · x_{i,t-l}** + δ_i · **Hill(x_{i,t})** "
            "+ Σ_j δ_{ij} · Hill(x_{j,t})"
        )
        params_text = "Parameters: shape k, scale λ (per lag group), n, S per channel"
    elif use_weibull and not use_hill:
        eq_text = (
            "β_{i,t} = **Σ w_l · x_{i,t-l}** + δ_i · **x_{i,t}^n** "
            "+ Σ_j δ_{ij} · x_{j,t}^n"
        )
        params_text = "Parameters: shape k, scale λ, n per channel  (n ∈ (0,1))"
    elif not use_weibull and use_hill:
        eq_text = (
            "β_{i,t} = **λ_i · β_{i,t-1}** + δ_i · **Hill(x_{i,t})** "
            "+ Σ_j δ_{ij} · Hill(x_{j,t})"
        )
        params_text = "Parameters: λ (decay), n, S per channel"
    else:
        eq_text = (
            "β_{i,t} = **λ_i · β_{i,t-1}** + δ_i · **x_{i,t}^n** "
            "+ Σ_j δ_{ij} · x_{j,t}^n"
        )
        params_text = "Parameters: λ (decay), n per channel  (n ∈ (0,1))"

    st.info(f"**Active state equation:** {eq_text}\n\n*{params_text}*")

    st.markdown(
        "**Intercept state (all modes):** "
        "I_t = G₀ · I_{t-1} + Σ_k γ_k · media_k,t^{n_k}"
    )

    st.divider()

    # ── F. Additional Options ─────────────────────────────────────────
    # (placed here so use_price / intercept_effectors are available for D2 / D3 below)
    st.markdown("### F · Additional Options")
    c1, c2 = st.columns(2)
    with c1: use_organic = st.checkbox("Organic drift (μ) in intercept state", key="use_organic")
    with c2: use_price   = st.checkbox("Include price effects", value=bool(price_vars), key="use_price")
    if use_price_2 is None:
        use_price_2 = use_price  # Dep 2 mirrors Dep 1 unless an independent predictor set was configured
    # Any numeric column from the uploaded file can boost the intercept —
    # not just variables already assigned a role (media/non-media) in the
    # sales equation above. Only the dependent variable(s) are excluded.
    _excluded_targets = {target, target2} if target2 else {target}
    intercept_effector_options = [c for c in num_cols if c not in _excluded_targets]
    intercept_effectors = safe_multiselect(
        "Intercept effectors — Dep 1 (any variable from your data boosting baseline)",
        options=intercept_effector_options, default=list(media), key="intercept_eff")
    if any(c in non_media for c in intercept_effectors):
        info(
            "Non-media effectors boost the intercept using their <b>raw value</b> each "
            "period (no transformation applied)."
        )

    # Intercept effectors for Dep 2 — independent selection
    intercept_effectors_2 = list(media_2)  # default: Dep 2's own media list
    if enable_second_dependent and target2:
        info(
            "🎯 <b>Dep 2 Intercept Effectors</b> — because Dep 2 (e.g. Top-of-Mind / "
            "Consideration) is also driven by media spend boosting the baseline, you can "
            "choose which media (and non-media) channels feed into the Dep 2 intercept "
            "state. Defaults to Dep 2's own media channels — deselect any that should not "
            "influence Dep 2's baseline."
        )
        intercept_effectors_2 = safe_multiselect(
            f"Intercept effectors — Dep 2 · {target2}",
            options=intercept_effector_options,
            default=list(media_2),
            key="intercept_eff_2",
        )
        if any(c in non_media_2 for c in intercept_effectors_2):
            info(
                "Non-media effectors for Dep 2 boost its intercept using their "
                "<b>raw value</b> each period (no transformation applied)."
            )

    st.divider()

    # Per-variable hyperparameter bounds widget (shared with Tab 7 · Refine & Refit)
    _render_per_channel_bounds = render_per_channel_bounds

    # ── D2. Per-Variable Hyperparameter Bounds — Dependent 1 ─────────
    st.markdown("### D2 · Per-Variable Hyperparameter Bounds — Dependent 1")
    per_channel_info(
        "🎛️ <b>Hyperparameter bounds for Dependent 1.</b> "
        "Own-media, competitor-media, price, and non-media/control variables each have "
        "separate bounds — so media betas, competition betas, price betas, and control "
        "betas are constrained independently. "
        "Defaults are derived from each channel's data distribution. "
        "Expand a channel to customise its bounds."
    )

    all_channel_cols = list(media) + list(comp_media)
    per_channel_bounds: dict = _render_per_channel_bounds(
        channel_cols=all_channel_cols,
        comp_cols=comp_media,
        key_prefix="d1_",
        df=df,
        use_hill=use_hill,
        use_weibull=use_weibull,
        price_cols=price_vars if use_price else [],
        nonmedia_cols=non_media,
    )

    if per_channel_bounds:
        rows = [
            {"Channel": col,
             "Type": (
                 "Competitor Media" if col in comp_media else
                 "Price" if col in price_vars else
                 "Non-Media / Control" if col in non_media else
                 "Own Media"
             ),
             "Parameter": param,
             "Min": f"{v[0]:.4g}",
             "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
            for col, bdict in per_channel_bounds.items()
            for param, v in bdict.items()
        ]
        if rows:
            with st.expander("📋 Dep 1 — all per-variable bounds summary", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── D3. Per-Variable Hyperparameter Bounds — Dependent 2 ─────────
    per_channel_bounds_2: dict = {}
    if enable_second_dependent and target2:
        st.divider()
        st.markdown("### D3 · Per-Variable Hyperparameter Bounds — Dependent 2")
        per_channel_info(
            f"🎛️ <b>Independent hyperparameter bounds for Dependent 2 "
            f"(<code>{target2}</code>).</b> "
            "Because Dep 2 may have a very different scale and dynamics from Dep 1, "
            "you can set separate bounds here for own-media betas, competition betas, "
            "price betas, and non-media / control betas. "
            "The model structure (state equation, adstock, transformation type) is "
            "shared with Dep 1; only the fitted betas and these bounds differ."
        )
        per_channel_bounds_2 = _render_per_channel_bounds(
            channel_cols=list(media_2) + list(comp_media_2),
            comp_cols=comp_media_2,
            key_prefix="d2_",
            df=df,
            use_hill=use_hill,
            use_weibull=use_weibull,
            price_cols=price_vars_2 if use_price_2 else [],
            nonmedia_cols=non_media_2,
        )
        if per_channel_bounds_2:
            rows2 = [
                {"Channel": col,
                 "Type": (
                     "Competitor Media" if col in comp_media_2 else
                     "Price" if col in price_vars_2 else
                     "Non-Media / Control" if col in non_media_2 else
                     "Own Media"
                 ),
                 "Parameter": param,
                 "Min": f"{v[0]:.4g}",
                 "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
                for col, bdict in per_channel_bounds_2.items()
                for param, v in bdict.items()
            ]
            if rows2:
                with st.expander("📋 Dep 2 — all per-variable bounds summary", expanded=False):
                    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    st.divider()

    # ── E. Train / Test Split ─────────────────────────────────────────
    st.markdown("### E · Train / Test Split")
    train_ratio = st.slider("Training proportion", 0.50, 0.95, 0.80, 0.05,
                             format="%.0f%%", key="train_ratio")
    n_total = len(df); n_train = int(n_total * train_ratio); n_test = n_total - n_train
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", n_total); c2.metric("Train", n_train); c3.metric("Test", n_test)

    st.divider()

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        if not media:
            st.error("Select at least one media channel.")
        else:
            n_bounds_set = sum(len(v) for v in per_channel_bounds.values())
            prophet_in_model = [c for c in non_media if c.startswith("prophet_")]
            st.session_state.config = {
                "target": target,
                "target2": target2 if (enable_second_dependent and target2) else None,
                "enable_second_dependent": bool(enable_second_dependent and target2),
                "media": media,
                "non_media": non_media,
                "price": price_vars if use_price else [],
                "comp_media": comp_media,
                "comp_nonmedia": comp_nonmedia,
                "dummy_cols": [],
                "intercept_effectors": intercept_effectors,
                "intercept_effectors_2": intercept_effectors_2,
                "cross_media_map": cross_map,
                "cross_media_map_2": cross_map_2,
                "different_predictors_2": bool(different_predictors_2),
                "media_2": media_2,
                "non_media_2": non_media_2,
                "comp_media_2": comp_media_2,
                "comp_nonmedia_2": comp_nonmedia_2,
                "price_2": price_vars_2 if use_price_2 else [],
                "use_price_2": use_price_2,
                "positive_beta_cols_2": positive_beta_cols_2,
                "negative_beta_cols_2": negative_beta_cols_2,
                "adstock_type": adstock_type_str,
                "transform_type": transform_type_str,
                "adstock_n_lags": int(n_lags),
                "use_organic": use_organic,
                "use_price": use_price,
                "train_ratio": train_ratio,
                "n_train": n_train,
                "n_test": n_test,
                "positive_beta_cols": positive_beta_cols,
                "negative_beta_cols": negative_beta_cols,
                "per_channel_bounds": per_channel_bounds,
                "per_channel_bounds_2": per_channel_bounds_2,
                "initial_media_betas":         {c: 0.0     for c in media},
                "initial_comp_betas":          {c: -0.0001 for c in comp_media},
                "initial_own_nonmedia_betas":  {c: 0.0     for c in non_media},
                "initial_comp_nonmedia_betas": {c: -0.01   for c in comp_nonmedia},
                "initial_price_beta":          {c: -0.1    for c in price_vars},
            }
            combo_label = f"{'Weibull' if use_weibull else 'Instant'} × {'Hill' if use_hill else 'Power'}"
            st.success(
                f"✅ Saved — {len(media)} own-media · {len(comp_media)} competitor · "
                f"{len(non_media)} non-media "
                f"({len(prophet_in_model)} prophet col{'s' if len(prophet_in_model)!=1 else ''}) · "
                f"adstock×transform: **{combo_label}** · "
                f"{'lags: ' + str(n_lags) + ' · ' if use_weibull else ''}"
                f"train/test: **{n_train}/{n_test}** · "
                f"positive-beta: **{len(positive_beta_cols)}** · "
                f"negative-beta: **{len(negative_beta_cols)}** · "
                f"per-variable bound params: **{n_bounds_set}**"
            )
            if prophet_in_model:
                prophet_info(
                    f"📌 Prophet controls included in model: "
                    f"<b>{', '.join(prophet_in_model)}</b>"
                )
            if st.session_state.config["enable_second_dependent"]:
                st.success(
                    f"➕ Second dependent variable enabled: **{target2}** — will be "
                    f"fitted **jointly** with **{target}** in Tab 5 using a bivariate "
                    f"Kalman filter (shared predictors x_t, correlated errors)."
                )
                if different_predictors_2:
                    st.info(
                        f"🔀 Dependent 2 uses its **own predictor set**: "
                        f"{len(media_2)} media · {len(non_media_2)} non-media · "
                        f"{len(price_vars_2) if use_price_2 else 0} price · "
                        f"{len(comp_media_2)} comp-media · {len(comp_nonmedia_2)} comp-non-media."
                    )
