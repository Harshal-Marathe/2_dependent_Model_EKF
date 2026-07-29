"""
Tab 6 — Run RBE Model: optimizer selection (L-BFGS-B / SLSQP / Nevergrad)
and kicking off the full pipeline.
"""

import streamlit as st

from modules.ui_helpers import section, info, ng_info, prophet_info, need_data, need_config
from modules.pipeline import run_multi_dependent_pipeline


def render_tab5(nevergrad_available: bool):
    section("05", "Run RBE Model")
    if st.session_state.df is None:     need_data()
    if st.session_state.config is None: need_config()

    config = st.session_state.config
    if "media" not in config:
        st.error("Configuration is incomplete — please re-save it in **Tab 5**.")
        st.stop()

    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media channels", len(config["media"]))
    c2.metric("Price vars",     len(config.get("price", [])))
    adstock_map = config.get("adstock_map", {})
    if adstock_map:
        n_w = sum(1 for v in adstock_map.values() if v == "weibull")
        n_i = len(adstock_map) - n_w
        adstock_label = f"Mixed ({n_w}W/{n_i}I)" if (n_w and n_i) else ("Weibull" if n_w else "Instant")
    else:
        adstock_label = config.get("adstock_type", "instant").title()
    combo = f"{adstock_label} × {config.get('transform_type','hill').title()}"
    c3.metric("Adstock×Transform", combo)
    c4.metric("Train / Test",   f"{config['n_train']} / {config['n_test']}")

    pb_cols = config.get("positive_beta_cols", [])
    pcb     = config.get("per_channel_bounds", {})
    if pb_cols: st.info(f"🔒 Positive-beta: {', '.join(pb_cols)}")
    if pcb:
        n_pcb = sum(len(v) for v in pcb.values())
        st.info(f"🎛️ Per-variable bounds: {len(pcb)} channel(s), {n_pcb} params")

    prophet_in_model = [c for c in config.get("non_media", []) if c.startswith("prophet_")]
    if prophet_in_model:
        prophet_info(f"📌 Prophet control variables in model: <b>{', '.join(prophet_in_model)}</b>")

    # Validate all config columns exist in df
    dep2_active = config.get("enable_second_dependent") and config.get("target2")
    dep2_cols = (
        config.get("media_2", []) + config.get("non_media_2", []) +
        config.get("comp_media_2", []) + config.get("comp_nonmedia_2", []) +
        config.get("price_2", [])
    ) if dep2_active else []
    missing_cols = [
        col for col in (
            config.get("media", []) + config.get("non_media", []) +
            config.get("comp_media", []) + config.get("comp_nonmedia", []) +
            config.get("price", []) + dep2_cols +
            [config.get("target", "")] +
            ([config["target2"]] if dep2_active else [])
        )
        if col and col not in df.columns
    ]
    if missing_cols:
        st.error(
            f"❌ Columns in saved config are missing from dataset: "
            f"`{'`, `'.join(missing_cols)}`. Re-save configuration in Tab 5."
        )
        st.stop()

    if config.get("enable_second_dependent") and config.get("target2"):
        st.info(
            f"➕ **Joint (bivariate) mode**: Dependent 1 (`{config['target']}`) and "
            f"Dependent 2 (`{config['target2']}`) will be fitted **together** in a "
            f"single bivariate Kalman filter — one optimizer run over both equations' "
            f"parameters plus the correlation (ρ) between their errors, rather than "
            f"two separate independent fits."
        )
        if config.get("different_predictors_2"):
            st.caption(
                f"🔀 Dependent 2 uses its own predictor set: "
                f"{len(config.get('media_2', []))} media · {len(config.get('non_media_2', []))} non-media · "
                f"{len(config.get('price_2', []))} price · {len(config.get('comp_media_2', []))} comp-media · "
                f"{len(config.get('comp_nonmedia_2', []))} comp-non-media."
            )

    st.divider()
    st.markdown("### Optimizer Selection")
    OPTIMIZER_OPTIONS = ["L-BFGS-B", "SLSQP"]
    if nevergrad_available: OPTIMIZER_OPTIONS.append("Nevergrad")
    method = st.selectbox("Optimizer", OPTIMIZER_OPTIONS,
                           help="L-BFGS-B/SLSQP: gradient-based. "
                                "Nevergrad: derivative-free multi-objective.")

    ng_cfg = None
    if method == "Nevergrad":
        if not nevergrad_available:
            st.error("Nevergrad not installed. Run `pip install nevergrad`."); st.stop()
        ng_info(
            "🟣 <b>Nevergrad Optimizer</b><br>"
            "Loss = <code>−loglik</code> — same objective as L-BFGS-B / SLSQP, "
            "just optimised with a derivative-free search strategy instead of a gradient-based one."
        )
        ng_col1, ng_col2 = st.columns(2)
        with ng_col1:
            ng_strategy = st.selectbox("Strategy",
                ["TwoPointsDE","NGOpt","CMA","PSO","DE","OnePlusOne","RandomSearch","MetaModel"])
            ng_budget   = st.number_input("Budget (evaluations)", 100, 10000, 500, 50)
        with ng_col2:
            ng_workers  = st.number_input("Parallel workers", 1, 8, 1, 1)
            max_iter    = ng_budget

        ng_cfg = {
            "strategy": ng_strategy, "budget": int(ng_budget), "num_workers": int(ng_workers),
        }
    else:
        col1, _ = st.columns(2)
        with col1: max_iter = st.number_input("Max iterations", 100, 5000, 800, 100)

    if st.button("🚀 Run RBE MMM", type="primary", use_container_width=True):
        joint_mode = bool(config.get("enable_second_dependent") and config.get("target2"))
        spinner_text = ("Running joint bivariate RBE optimisation… (60–600 s)"
                         if joint_mode else "Running RBE optimisation… (30–300 s)")
        with st.spinner(spinner_text):
            try:
                results_1, results_2 = run_multi_dependent_pipeline(
                    df, config, max_iter, method, ng_cfg=ng_cfg)
                st.session_state.model_results   = results_1
                st.session_state.model_fitted    = True
                st.session_state.model_results_2 = results_2
                st.session_state.model_fitted_2  = results_2 is not None

                st.success("✅ Model fitted!")
                st.markdown(f"#### Dependent 1 · `{config['target']}`")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAPE",      f"{results_1['mape']:.2%}")
                c2.metric("R²",        f"{results_1['r2']:.4f}")
                c3.metric("Log-Lik",   f"{results_1['loglik']:.1f}")
                c4.metric("Converged", "Yes ✅" if results_1["success"] else "Partial ⚠️")

                if results_2 is not None:
                    st.markdown(f"#### Dependent 2 · `{config.get('target2')}` (joint bivariate fit)")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MAPE",      f"{results_2['mape']:.2%}")
                    d2.metric("R²",        f"{results_2['r2']:.4f}")
                    d3.metric("Log-Lik",   f"{results_2['loglik']:.1f}")
                    d4.metric("Converged", "Yes ✅" if results_2["success"] else "Partial ⚠️")
                    st.info(
                        f"🔗 Joint bivariate log-likelihood: **{results_2['joint_loglik']:.1f}** · "
                        f"estimated error correlation ρ(Dep1, Dep2) = **{results_2['rho_y']:.3f}** · "
                        f"cross-intercept coupling: φ₁ (Dep2→Dep1) = **{results_2['phi1']:.3f}**, "
                        f"φ₂ (Dep1→Dep2) = **{results_2['phi2']:.3f}**"
                    )
            except Exception as e:
                st.exception(e)

    if st.session_state.model_fitted:
        res = st.session_state.model_results; st.divider()
        st.markdown(f"**Dependent 1 · `{config['target']}`**")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{res['mape']:.2%}")
        c2.metric("R²",   f"{res['r2']:.4f}")
        c3.metric("Log-Lik", f"{res['loglik']:.2f}")

        if st.session_state.get("model_fitted_2") and st.session_state.get("model_results_2"):
            res2 = st.session_state.model_results_2
            st.markdown(f"**Dependent 2 · `{config.get('target2')}`** (joint bivariate fit)")
            e1, e2, e3 = st.columns(3)
            e1.metric("MAPE", f"{res2['mape']:.2%}")
            e2.metric("R²",   f"{res2['r2']:.4f}")
            e3.metric("Log-Lik", f"{res2['loglik']:.2f}")
            if res2.get("joint_fit"):
                st.caption(f"ρ(Dep1, Dep2) = {res2['rho_y']:.3f} · "
                           f"φ₁ (Dep2→Dep1) = {res2['phi1']:.3f} · φ₂ (Dep1→Dep2) = {res2['phi2']:.3f} · "
                           f"joint log-lik = {res2['joint_loglik']:.2f}")

        st.caption("Proceed to **Tab 7** for full results. Use the selector there to "
                   "switch between Dependent 1 and Dependent 2.")
