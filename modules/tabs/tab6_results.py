"""
Tab 6 — Results & ROI Analytics: performance metrics, fit charts,
residual diagnostics, channel contributions, ROI, parameters, and
interactive response curves.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section, need_model
from modules.transforms import hill_transform, power_transform, apply_transformation
from modules.exports import build_betas_df, build_master_workbook_bytes


def _render_tab7_promote_section():
    """If Tab 7 · Refine & Refit has produced a working model that differs
    from the currently-saved official model, surface a compare-and-save
    panel here so the whole 'refit → review → save' loop lives in one
    place: refits happen in Tab 7, saving happens here in Tab 6."""
    refit_result = st.session_state.get("refit_result")
    refit_config = st.session_state.get("refit_config")
    if refit_result is None or refit_config is None:
        return

    steps_taken = len(st.session_state.get("refit_history", [])) - 1
    if steps_taken <= 0:
        return  # nothing refined yet — nothing to promote

    already_saved = refit_result is st.session_state.model_results
    with st.container(border=True):
        st.markdown("### 🔧 Refined Model Available (from Tab 7 · Refine & Refit)")
        base_mape = st.session_state.model_results["mape"]
        base_r2   = st.session_state.model_results["r2"]
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Tab 5 baseline MAPE", f"{base_mape:.2%}")
        dc2.metric("Refined MAPE", f"{refit_result['mape']:.2%}",
                   delta=f"{(base_mape - refit_result['mape'])*100:+.2f} pp (lower is better)")
        dc3.metric("Tab 5 baseline R²", f"{base_r2:.4f}")
        dc4.metric("Refined R²", f"{refit_result['r2']:.4f}",
                   delta=f"{refit_result['r2']-base_r2:+.4f}")
        st.caption(f"{steps_taken} refinement step(s) taken in Tab 7.")

        if already_saved:
            st.success("✅ This refined model is already saved and is what's shown below.")
        else:
            if st.button("💾 Save this refined model as the official model",
                         type="primary", use_container_width=True, key="tab6_promote_btn"):
                st.session_state.config = refit_config
                st.session_state.model_results = refit_result
                st.session_state.model_fitted = True
                st.success("✅ Saved. Results below now reflect the refined model.")
                st.rerun()
    st.divider()


def render_tab6():
    section("06", "Results & ROI Analytics")
    if not st.session_state.model_fitted: need_model()

    config = st.session_state.config
    df     = st.session_state.df

    _render_tab7_promote_section()

    has_dep2 = bool(st.session_state.get("model_fitted_2") and st.session_state.get("model_results_2"))
    if has_dep2:
        dep_choice = st.radio(
            "📊 Viewing results for:",
            [f"Dependent 1 · {config['target']}", f"Dependent 2 · {config.get('target2')} (joint bivariate fit)"],
            horizontal=True, key="tab6_dep_choice",
        )
        if dep_choice.startswith("Dependent 2"):
            res = st.session_state.model_results_2
            target = config.get("target2")
        else:
            res = st.session_state.model_results
            target = config["target"]
        if res.get("joint_fit"):
            st.caption(
                f"🔗 Jointly fitted with a bivariate Kalman filter · "
                f"ρ(Dep1, Dep2) = **{res['rho_y']:.3f}** · "
                f"φ₁ (Dep2→Dep1 intercept) = **{res['phi1']:.3f}** · "
                f"φ₂ (Dep1→Dep2 intercept) = **{res['phi2']:.3f}** · "
                f"joint log-likelihood = **{res['joint_loglik']:.2f}**"
            )
        st.divider()
    else:
        res    = st.session_state.model_results
        target = config["target"]
    g = res["g"]

    st.markdown("### A · Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE",         f"{res['mape']:.2%}")
    c2.metric("R²",           f"{res['r2']:.4f}")
    c3.metric("Log-Lik",      f"{res['loglik']:.2f}")
    c4.metric("Observations", len(df))

    st.markdown("### B · Actual vs Predicted")
    x_axis  = np.arange(len(df)); n_train = config["n_train"]
    fig_avp = go.Figure()
    fig_avp.add_vrect(x0=n_train, x1=len(df)-1, fillcolor="#fef3c7", opacity=0.35,
                      layer="below", line_width=0,
                      annotation_text="Test period", annotation_position="top left")
    fig_avp.add_trace(go.Scatter(x=x_axis, y=df[target].values,
                                  name="Actual", line=dict(color="#1e40af", width=2)))
    fig_avp.add_trace(go.Scatter(x=x_axis, y=res["yhat_smooth"],
                                  name="EKF Smoothed", line=dict(color="#f59e0b", width=2, dash="dash")))
    fig_avp.update_layout(height=460, template="plotly_white",
                           title="Actual vs EKF Smoothed",
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("### C · Smoothed Intercept State")
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=x_axis, y=res["x_smooth"][:, 0],
                                  name="Intercept", line=dict(color="#10b981", width=2)))
    fig_int.update_layout(height=340, template="plotly_white",
                           title="Smoothed Intercept (Base Demand) Over Time")
    st.plotly_chart(fig_int, use_container_width=True)

    st.markdown("### D · Residual Diagnostics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_h = px.histogram(res["residuals"], nbins=40, title="Residual Distribution",
                              color_discrete_sequence=["#3b82f6"])
        fig_h.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_h, use_container_width=True)
    with col_b:
        sr = np.sort(res["residuals"])
        tq = np.sort(np.random.normal(0, np.std(res["residuals"]), len(sr)))
        fig_qq = px.scatter(x=tq, y=sr, title="Q-Q Plot",
                             labels={"x": "Theoretical Normal", "y": "Sample Residuals"})
        fig_qq.add_trace(go.Scatter(x=tq, y=tq, mode="lines",
                                     name="Ideal", line=dict(color="red", dash="dash")))
        fig_qq.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("### E · Channel Contributions")
    contrib_df = res["contrib_df"]
    short_cols = [c for c in contrib_df.columns if c.startswith("ShortTerm_")]
    long_cols  = [c for c in contrib_df.columns if c.startswith("LongTerm_")]
    totals_st  = contrib_df[short_cols].sum()
    totals_lt  = contrib_df[long_cols].sum()

    def _contribution_table(totals, prefix):
        """
        Share % logic:
        1. raw_pct_i = |value_i| / sum(|all values|) * 100
        2. Positives: scale so they sum to 100  =>  raw_pct_i / sum(raw_pct of positives) * 100
        3. Negatives: keep as negative raw_pct  =>  -raw_pct_i
        Result: positive shares sum to +100%, negatives are their true negative weight.
        """
        names  = [c.replace(prefix, "") for c in totals.index]
        values = totals.values.astype(float)

        abs_vals  = np.abs(values)
        total_abs = abs_vals.sum()

        if total_abs < 1e-12:
            shares = np.zeros_like(values)
        else:
            raw_pct     = abs_vals / total_abs * 100
            pos_raw_sum = raw_pct[values > 0].sum()
            shares = np.where(
                values >= 0,
                np.where(pos_raw_sum > 1e-12, raw_pct / pos_raw_sum * 100, 0.0),
                -raw_pct,
            )

        pos_sum = values[values > 0].sum()
        neg_sum = values[values < 0].sum()

        df_out = pd.DataFrame({
            "Channel":       names,
            "Total Contrib": np.round(values, 2),
            "Share (%)":     np.round(shares, 1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_out.insert(0, "Rank", range(1, len(df_out)+1))
        return df_out, pos_sum, neg_sum

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Short-Term Contribution Summary")
        df_st, pos_st, neg_st = _contribution_table(totals_st, "ShortTerm_")
        st.dataframe(df_st, use_container_width=True, hide_index=True)
        check_st = pos_st + neg_st
        c1s, c2s, c3s = st.columns(3)
        c1s.metric("Positive pool", f"{pos_st:,.2f}")
        c2s.metric("Negative pool", f"{neg_st:,.2f}")
        c3s.metric("Net", f"{check_st:,.2f}")
        st.caption("✅ Positive shares sum to **+100 %** · Negative shares sum to **−100 %**")
    with t2:
        st.markdown("#### Long-Term Contribution Summary")
        df_lt, pos_lt, neg_lt = _contribution_table(totals_lt, "LongTerm_")
        st.dataframe(df_lt, use_container_width=True, hide_index=True)
        check_lt = pos_lt + neg_lt
        c1l, c2l, c3l = st.columns(3)
        c1l.metric("Positive pool", f"{pos_lt:,.2f}")
        c2l.metric("Negative pool", f"{neg_lt:,.2f}")
        c3l.metric("Net", f"{check_lt:,.2f}")
        st.caption("✅ Positive shares sum to **+100 %** · Negative shares sum to **−100 %**")

    # FIX: cross-media synergy contributions (configured in Tab 4 ·
    # Section C) were computed by the filter and present in contrib_df
    # (see pipeline.py) but nothing in this tab ever read out the
    # Synergy_ columns — they were invisible in the UI even though the
    # numbers existed. Adding a dedicated summary here.
    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_cols:
        st.markdown("#### 🔗 Cross-Media Synergy Contribution Summary")
        totals_syn = contrib_df[synergy_cols].sum()
        syn_vals = totals_syn.values
        syn_tot  = np.abs(syn_vals).sum() + 1e-12
        syn_abs_vals  = np.abs(syn_vals)
        syn_total_abs = syn_abs_vals.sum()
        if syn_total_abs < 1e-12:
            syn_shares = np.zeros_like(syn_vals)
        else:
            syn_raw_pct     = syn_abs_vals / syn_total_abs * 100
            syn_pos_raw_sum = syn_raw_pct[syn_vals > 0].sum()
            syn_shares = np.where(
                syn_vals >= 0,
                np.where(syn_pos_raw_sum > 1e-12, syn_raw_pct / syn_pos_raw_sum * 100, 0.0),
                -syn_raw_pct,
            )
        df_syn = pd.DataFrame({
            "Synergy Pair": [c.replace("Synergy_", "").replace("_from_", " <- from ")
                              for c in synergy_cols],
            "Total Contrib": syn_vals.round(2),
            "Share (%)": np.round(syn_shares, 1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_syn.insert(0, "Rank", range(1, len(df_syn)+1))
        st.dataframe(df_syn, use_container_width=True, hide_index=True)

        fig_syn = px.bar(
            df_syn.sort_values("Total Contrib"),
            x="Total Contrib", y="Synergy Pair", orientation="h",
            title="Cross-Media Synergy Contribution by Pair",
            color="Total Contrib", color_continuous_scale="Purples",
        )
        fig_syn.update_layout(height=max(300, 45*len(synergy_cols)), template="plotly_white")
        st.plotly_chart(fig_syn, use_container_width=True)
    else:
        st.caption("No cross-media synergy pairs were configured in Tab 4 · Section C.")

    with st.expander("📊 Pie Charts — Positive Contributions Only"):
        st.caption(
            "Pie charts show only **positive** contributions (share of positive pool = 100%). "
            "Negative channels (competitor, price) are excluded from pies — they are shown "
            "in the summary tables above with negative share %."
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            st_names  = [c.replace("ShortTerm_", "") for c in short_cols]
            st_vals   = totals_st.values
            pos_mask  = st_vals > 0
            if pos_mask.any():
                fig_pie_st = px.pie(
                    values=st_vals[pos_mask],
                    names=[n for n, m in zip(st_names, pos_mask) if m],
                    title="Short-term positive share",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                st.plotly_chart(fig_pie_st, use_container_width=True)
            else:
                st.info("No positive short-term contributions.")
        with pc2:
            lt_names  = [c.replace("LongTerm_", "") for c in long_cols]
            lt_vals   = totals_lt.values
            pos_mask_lt = lt_vals > 0
            if pos_mask_lt.any():
                fig_pie_lt = px.pie(
                    values=lt_vals[pos_mask_lt],
                    names=[n for n, m in zip(lt_names, pos_mask_lt) if m],
                    title="Long-term positive share",
                    color_discrete_sequence=px.colors.sequential.Greens_r,
                )
                st.plotly_chart(fig_pie_lt, use_container_width=True)
            else:
                st.info("No positive long-term contributions.")

    _cplot = contrib_df.copy().reset_index(drop=True)
    _cplot["_period"] = np.arange(len(_cplot))
    fig_st_area = px.area(
        _cplot, x="_period", y=short_cols,
        title="Short-term Contributions Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_st_area.update_xaxes(title_text="Period")
    fig_st_area.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_st_area, use_container_width=True)

    with st.expander("Full contribution table"):
        st.dataframe(contrib_df.style.format("{:,.4f}"), use_container_width=True)
    st.download_button("📥 Download Contribution Table",
                       contrib_df.to_csv().encode(), "contributions.csv", "text/csv")

    st.markdown("### E2 · Synergy / Cross-Media Effects")
    synergy_df = res.get("synergy_df")
    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_df is None or synergy_df.empty or not synergy_cols:
        st.info(
            "No cross-media synergy relationships are configured for this model. "
            "Set them up in **Tab 4 · Section C (Cross-media Learning)** and re-run."
        )
    else:
        st.caption(
            "One row per configured source → target relationship "
            "(per-variable level, not a single aggregated number)."
        )
        st.dataframe(
            synergy_df.style.format({
                "Total Synergy Contribution": "{:,.4f}",
                "Avg Synergy / Period":       "{:,.4f}",
                "Cross Delta":                "{:.4f}",
                "Cross Hill n":                "{:.3f}",
                "Cross Hill S":                "{:.3f}",
                "Share of Target's Total Contrib (%)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.download_button("📥 Download Synergy Table",
                           synergy_df.to_csv(index=False).encode(),
                           "synergy_table.csv", "text/csv")

        pair_labels = [f"{row['Source Channel']} → {row['Target Channel']}"
                        for _, row in synergy_df.iterrows()]
        fig_syn_bar = px.bar(
            x=synergy_df["Total Synergy Contribution"].values, y=pair_labels,
            orientation="h", title="Total Synergy Contribution by Pair",
            labels={"x": "Total Contribution", "y": ""},
            color=synergy_df["Total Synergy Contribution"].values,
            color_continuous_scale="Purples",
        )
        fig_syn_bar.update_layout(height=max(300, 50*len(pair_labels)), template="plotly_white")
        st.plotly_chart(fig_syn_bar, use_container_width=True)

        _cplot_syn = contrib_df.copy().reset_index(drop=True)
        _cplot_syn["_period"] = np.arange(len(_cplot_syn))
        fig_syn_area = px.area(
            _cplot_syn, x="_period", y=synergy_cols,
            title="Synergy Contributions Over Time (per pair)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_syn_area.update_xaxes(title_text="Period")
        fig_syn_area.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_syn_area, use_container_width=True)

    st.markdown("### F · Estimated Parameters")
    param_df = res["param_df"]
    st.dataframe(param_df.style.format({"Value": "{:.6f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("📥 Download Parameters",
                       param_df.to_csv(index=False).encode(), "parameters.csv", "text/csv")

    st.markdown("### G · ROI Analytics")
    roi_df = res["roi_df"]
    if not roi_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_roi = px.bar(roi_df.sort_values("ROI", ascending=True),
                              x="ROI", y="Channel", orientation="h",
                              title="ROI by Media Channel",
                              color="ROI", color_continuous_scale="RdYlGn")
            fig_roi.add_vline(x=1.0, line_dash="dash", line_color="black",
                               annotation_text="Break-even")
            fig_roi.update_layout(height=360, template="plotly_white")
            st.plotly_chart(fig_roi, use_container_width=True)
        with c2:
            st.dataframe(roi_df.style.format({
                "TotalSpend":   "{:,.2f}",
                "TotalContrib": "{:,.2f}",
                "ROI":          "{:.4f}",
            }), use_container_width=True, hide_index=True)
        best  = roi_df.loc[roi_df["ROI"].idxmax()]
        worst = roi_df.loc[roi_df["ROI"].idxmin()]
        st.success(f"🏆 Best ROI: **{best['Channel']}** ({best['ROI']:.4f})")
        st.warning(f"⚠️ Lowest ROI: **{worst['Channel']}** ({worst['ROI']:.4f})")
        st.download_button("📥 Download ROI Report",
                           roi_df.to_csv(index=False).encode(), "roi_report.csv", "text/csv")

    st.markdown("### H · Response Curves")
    params       = res["params"]
    media_list   = g["MEDIA_COLS"]
    adstock_type = g["ADSTOCK_TYPE"]

    if not media_list:
        st.info("No media channels configured.")
    else:
        sel = st.selectbox("Select channel", media_list, key="rc_sel")
        idx = media_list.index(sel)

        rc1, rc2, rc3 = st.columns(3)
        with rc1: x_max_pct = st.slider("X-axis max (%)", 50, 300, 150, 10, key="rc_xmax")
        with rc2: n_points  = st.slider("Curve resolution", 50, 500, 200, 50, key="rc_npts")
        with rc3: show_ci   = st.checkbox("Show beta band", value=True, key="rc_ci")

        col_data = df[sel]; x_max = col_data.max() * x_max_pct / 100
        x_range  = np.linspace(0, x_max, n_points)
        beta_med = float(np.median(res["x_smooth"][:, idx+1]))
        beta_p25 = float(np.percentile(res["x_smooth"][:, idx+1], 25))
        beta_p75 = float(np.percentile(res["x_smooth"][:, idx+1], 75))
        n_v = params["n_params"][idx]; S_v = params["S_params"][idx]

        transform_type = g.get("TRANSFORM_TYPE", "hill")
        if adstock_type == "weibull":
            from modules.transforms import weibull_lag_weights
            ak = params["adstock_shape"][idx]; al = params["adstock_scale"][idx]
            n_lags = int(params.get("adstock_n_lags", 8))
            w_lags = weibull_lag_weights(ak, al, n_lags)
            am = w_lags.max()
            adstock_label = f"Weibull (k={ak:.3f}, λ={al:.3f}, L={n_lags})"
        else:
            ls_v = params["Ls"][idx]
            adstock_label = f"Instant / Nerlove-Arrow (no separate λ — carryover via Ls={ls_v:.3f})"

        def make_resp(b):
            return np.array([b * apply_transformation(
                np.array([x]), transform_type, n_v, S_v)[0] for x in x_range])

        resp_med = make_resp(beta_med)
        resp_p25 = make_resp(beta_p25)
        resp_p75 = make_resp(beta_p75)
        marginal = np.gradient(resp_med, x_range)

        pct_10 = np.percentile(col_data[col_data > 0], 10) if (col_data > 0).any() else 0
        pct_50 = np.percentile(col_data[col_data > 0], 50) if (col_data > 0).any() else 0
        pct_90 = np.percentile(col_data[col_data > 0], 90) if (col_data > 0).any() else 0
        mean_v = col_data.mean()
        def resp_at(v): return float(np.interp(v, x_range, resp_med))

        # ── Simple clean response curve (dark theme) ─────────────────
        fig_rc = go.Figure()

        if show_ci:
            fig_rc.add_trace(go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([resp_p75, resp_p25[::-1]]),
                fill="toself", fillcolor="rgba(99,179,237,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))

        fig_rc.add_trace(go.Scatter(
            x=x_range, y=resp_med, mode="lines",
            line=dict(color="#3b82f6", width=3),
            showlegend=False,
        ))

        marker_specs = [
            (pct_10, "Min",    "#10b981"),
            (mean_v, "Avg",    "#f59e0b"),
            (pct_50, "Median", "#a855f7"),
            (pct_90, "Max",    "#ef4444"),
        ]
        for pv, pn, clr in marker_specs:
            if 0 < pv <= x_max:
                rv = resp_at(pv)
                fig_rc.add_trace(go.Scatter(
                    x=[pv], y=[rv], mode="markers+text",
                    marker=dict(color=clr, size=13,
                                line=dict(color="white", width=1.5)),
                    text=[f"<b>{pn}<br>{pv:.1f}</b>"],
                    textposition="top center",
                    textfont=dict(size=10, color="white"),
                    showlegend=False,
                ))

        fig_rc.update_layout(
            title=dict(
                text=f"Response Curve: {sel}  ·  beta = {beta_med:.6f}",
                font=dict(color="white", size=14), x=0.5,
            ),
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font=dict(color="#cbd5e1"),
            xaxis=dict(
                title=f"Input - {sel}",
                gridcolor="#334155", zerolinecolor="#334155",
            ),
            yaxis=dict(
                title="Response (KPI per unit Input)",
                gridcolor="#334155", zerolinecolor="#334155",
            ),
            height=460,
            margin=dict(t=60, b=60, l=70, r=30),
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        # Parameter summary
        pc = st.columns(6)
        pc[0].metric("beta median", f"{beta_med:.4f}")
        pc[1].metric("beta P25",    f"{beta_p25:.4f}")
        pc[2].metric("beta P75",    f"{beta_p75:.4f}")
        transform_type_label = g.get("TRANSFORM_TYPE", "hill")
        if transform_type_label == "hill":
            pc[3].metric("Hill n", f"{n_v:.3f}")
            pc[4].metric("Hill S", f"{S_v:.3f}")
        else:
            pc[3].metric("Power n", f"{n_v:.3f}")
            pc[4].metric("", "")
        if adstock_type == "weibull":
            pc[5].metric("Weibull k/lam",
                         f"{params['adstock_shape'][idx]:.3f}/{params['adstock_scale'][idx]:.3f}")
        else:
            pc[5].metric("Ls (persistence)", f"{params['Ls'][idx]:.3f}")
        st.caption(f"Adstock: **{adstock_label}**")

        pcb_key = "per_channel_bounds_2" if (has_dep2 and dep_choice.startswith("Dependent 2")) else "per_channel_bounds"
        pcb = config.get(pcb_key, {}).get(sel, {})
        if pcb:
            st.markdown("#### Per-Variable Bounds Applied")
            st.dataframe(
                pd.DataFrame([{"Parameter": k,
                               "Min": f"{v[0]:.4g}",
                               "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
                              for k, v in pcb.items()]),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.markdown("### I · Export Center")
    info_text = (
        "Three ways to take results outside the app: just the betas, just the "
        "hyperparameters, or one structured Excel workbook with everything laid "
        "out as <b>Raw variables → Transformed variables → Betas → "
        "Contributions (Beta × Transformed)</b>, plus hyperparameters / ROI / "
        "synergy on their own sheets."
    )
    st.markdown(f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        betas_df = build_betas_df(res, df)
        st.download_button(
            "📥 Download Betas (Time Series)",
            betas_df.to_csv(index=False).encode(),
            "betas_timeseries.csv", "text/csv",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "📥 Download Hyperparameters",
            param_df.to_csv(index=False).encode(),
            "hyperparameters.csv", "text/csv",
            use_container_width=True,
        )
    with ec3:
        export_config = dict(config); export_config["target"] = target
        workbook_bytes = build_master_workbook_bytes(res, export_config, df)
        st.download_button(
            "📊 Download Full Excel Workbook",
            workbook_bytes,
            "rainbrain_model_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
