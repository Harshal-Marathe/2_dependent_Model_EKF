"""
Tab 4 — Correlation & Synergy Analysis.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sp_stats

from modules.ui_helpers import section, info, prophet_info, need_data, safe_multiselect


def render_tab3():
    section("03", "Correlation & Synergy Analysis")
    if st.session_state.df is None: need_data()

    # Always read from session state — captures prophet cols after rerun
    df_corr = st.session_state.df.copy()
    num_cols_all = df_corr.select_dtypes(include=np.number).columns.tolist()

    if st.session_state.prophet_cols_added:
        missing_now = [c for c in st.session_state.prophet_cols_added if c not in num_cols_all]
        if missing_now:
            st.error(
                f"⚠️ Prophet column(s) were added but are no longer numeric / present in the "
                f"dataset: `{'`, `'.join(missing_now)}`. Re-run the prophet merge in Tab 2."
            )
        prophet_info(
            f"📌 Prophet columns in dataset: "
            f"<b>{', '.join(st.session_state.prophet_cols_added)}</b> — "
            f"they are included in the variable list below."
        )

    info("Explore pairwise correlations and identify media channels that run concurrently.")

    st.markdown("### A · Variable Correlation Matrix")
    # FIX: routed through safe_multiselect(). options=num_cols_all is the
    # full, current numeric-column list (always includes new prophet cols).
    # safe_multiselect sanitizes any stored selection against these options
    # on every render, so Streamlit can never throw here, and prophet cols
    # that are present in `options` are never wiped out by an unrelated
    # stale value elsewhere in the stored list.
    sel_vars = safe_multiselect(
        "Select variables to include",
        options=num_cols_all,
        default=num_cols_all,   # all cols, including new prophet ones
        key="corr_vars",
    )
    corr_method = st.radio("Correlation method", ["Pearson","Spearman","Kendall"],
                            horizontal=True, key="corr_method")

    if sel_vars and len(sel_vars) >= 2:
        corr_mat = df_corr[sel_vars].corr(method=corr_method.lower())
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_mat.values, x=corr_mat.columns.tolist(), y=corr_mat.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_mat.values, 2), texttemplate="%{text}",
            textfont={"size": 10}, hoverongaps=False, colorbar=dict(title="r"),
        ))
        fig_heat.update_layout(height=max(500, 30*len(sel_vars)),
                                title=f"{corr_method} Correlation Matrix",
                                template="plotly_white", xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_heat, use_container_width=True)

        pairs = []
        for i, v1 in enumerate(corr_mat.columns):
            for j, v2 in enumerate(corr_mat.columns):
                if i < j:
                    r = corr_mat.loc[v1, v2]
                    pairs.append({
                        "Variable A": v1, "Variable B": v2,
                        "Correlation (r)": round(r, 4), "|r|": round(abs(r), 4),
                        "Strength": ("Very Strong" if abs(r) >= 0.8 else
                                     "Strong" if abs(r) >= 0.6 else
                                     "Moderate" if abs(r) >= 0.4 else
                                     "Weak" if abs(r) >= 0.2 else "Negligible"),
                        "Direction": "Positive" if r > 0 else "Negative",
                    })
        pairs_df = pd.DataFrame(pairs).sort_values("|r|", ascending=False)
        st.dataframe(pairs_df.drop(columns="|r|"), use_container_width=True, hide_index=True)
        st.download_button("📥 Download Correlation Table",
                           pairs_df.to_csv(index=False).encode(), "correlations.csv", "text/csv")
    elif sel_vars:
        st.info("Select at least 2 variables.")

    st.divider()
    st.markdown("### B · Per-Variable Correlation with KPI")
    if num_cols_all:
        kpi_col = st.selectbox("Select KPI / target variable", num_cols_all, key="corr_kpi")
        others  = [c for c in num_cols_all if c != kpi_col]
        if others:
            r_vals = df_corr[others].corrwith(df_corr[kpi_col], method=corr_method.lower())
            r_df   = pd.DataFrame({
                "Variable": r_vals.index,
                f"Correlation with {kpi_col}": r_vals.values.round(4),
                "|r|": r_vals.abs().values.round(4),
            }).sort_values("|r|", ascending=False)
            fig_bar_r = px.bar(
                r_df.sort_values(f"Correlation with {kpi_col}"),
                x=f"Correlation with {kpi_col}", y="Variable", orientation="h",
                color=f"Correlation with {kpi_col}", color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title=f"Correlation of each variable with {kpi_col}",
            )
            fig_bar_r.update_layout(height=max(350, 25*len(others)), template="plotly_white")
            st.plotly_chart(fig_bar_r, use_container_width=True)
            st.dataframe(r_df.drop(columns="|r|"), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### C · Media Concurrency (Synergy Detection)")
    info("Channels that are <b>active at the same time</b> are candidates for synergy effects.")

    media_default = [c for c in num_cols_all
                      if any(kw in c.lower() for kw in [
                          "tv", "digital", "social", "search", "display", "radio",
                          "ooh", "print", "email", "media", "spend", "cost", "paid",
                          "brand", "perf"])][:min(10, len(num_cols_all))]
    media_candidates = safe_multiselect(
        "Select media / spend columns to analyse", options=num_cols_all,
        default=media_default,
        key="media_cand",
    )
    activity_thresh = st.slider("Activity threshold (% of column max)", 1, 50, 5, 1,
                                 key="act_thresh")

    if media_candidates and len(media_candidates) >= 2:
        act_df = pd.DataFrame({
            col: (df_corr[col] > df_corr[col].max() * activity_thresh / 100).astype(int)
            for col in media_candidates
        }, index=df_corr.index)
        T_periods = len(act_df); n_chan = len(media_candidates)
        concurrency = np.array([[
            (act_df[media_candidates[i]] & act_df[media_candidates[j]]).sum() / T_periods * 100
            for j in range(n_chan)] for i in range(n_chan)])
        fig_conc = go.Figure(data=go.Heatmap(
            z=concurrency, x=media_candidates, y=media_candidates,
            colorscale="YlOrRd", zmin=0, zmax=100,
            text=np.round(concurrency, 1), texttemplate="%{text}%",
            textfont={"size": 11}, colorbar=dict(title="% periods<br>both active"),
        ))
        fig_conc.update_layout(height=max(450, 40*n_chan), title="Media Concurrency Matrix",
                                template="plotly_white", xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_conc, use_container_width=True)
        conc_pairs = [
            {"Channel A": media_candidates[i], "Channel B": media_candidates[j],
             "Overlap % (both active)": round(concurrency[i, j], 1),
             "Synergy candidate?": ("✅ Yes" if concurrency[i, j] >= 30 else
                                    "⚠️ Maybe" if concurrency[i, j] >= 10 else "❌ Low")}
            for i in range(n_chan) for j in range(i+1, n_chan)
        ]
        st.dataframe(
            pd.DataFrame(conc_pairs).sort_values("Overlap % (both active)", ascending=False),
            use_container_width=True, hide_index=True,
        )
    elif media_candidates:
        st.info("Select at least 2 media columns.")

    st.divider()
    st.markdown("### D · Apriori Check (Univariate OLS: y = b0 + b1·x)")
    info(
        "For each selected variable, fits a <b>simple OLS regression</b> against the KPI "
        "on its own — <code>y = b0 + b1&middot;x</code> — so you can sanity-check the "
        "individual (univariate) direction and strength of effect <b>before</b> going into "
        "the full multivariate model. This is a bivariate check only; it ignores "
        "interactions/confounding with other variables."
    )

    if num_cols_all:
        apriori_kpi = st.selectbox(
            "Select KPI / target variable", num_cols_all, key="apriori_kpi"
        )
        apriori_x_options = [c for c in num_cols_all if c != apriori_kpi]
        apriori_vars = safe_multiselect(
            "Select variables to run univariate OLS against the KPI",
            options=apriori_x_options,
            default=apriori_x_options,
            key="apriori_vars",
        )

        if apriori_vars:
            y = df_corr[apriori_kpi].astype(float)
            rows = []
            for x_col in apriori_vars:
                x = df_corr[x_col].astype(float)
                mask = x.notna() & y.notna()
                x_clean, y_clean = x[mask], y[mask]
                if len(x_clean) < 3 or x_clean.std() == 0:
                    rows.append({
                        "Variable": x_col, "b0 (Intercept)": np.nan, "b1 (Slope)": np.nan,
                        "R²": np.nan, "p-value": np.nan, "Std Err (b1)": np.nan,
                        "Expected Sign": "—", "Significant (p<0.05)": "—",
                        "n": int(mask.sum()),
                    })
                    continue
                res = sp_stats.linregress(x_clean, y_clean)
                rows.append({
                    "Variable": x_col,
                    "b0 (Intercept)": round(res.intercept, 4),
                    "b1 (Slope)": round(res.slope, 4),
                    "R²": round(res.rvalue ** 2, 4),
                    "p-value": round(res.pvalue, 4),
                    "Std Err (b1)": round(res.stderr, 4),
                    "Expected Sign": "Positive" if res.slope > 0 else "Negative",
                    "Significant (p<0.05)": "✅ Yes" if res.pvalue < 0.05 else "❌ No",
                    "n": int(mask.sum()),
                })

            apriori_df = pd.DataFrame(rows).sort_values("R²", ascending=False)
            st.dataframe(apriori_df, use_container_width=True, hide_index=True)

            plot_df = apriori_df.dropna(subset=["b1 (Slope)"])
            if not plot_df.empty:
                fig_b1 = px.bar(
                    plot_df.sort_values("b1 (Slope)"),
                    x="b1 (Slope)", y="Variable", orientation="h",
                    color="b1 (Slope)", color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    title=f"Univariate OLS slope (b1) of each variable on {apriori_kpi}",
                    hover_data=["R²", "p-value"],
                )
                fig_b1.update_layout(height=max(350, 25 * len(plot_df)), template="plotly_white")
                st.plotly_chart(fig_b1, use_container_width=True)

            st.download_button(
                "📥 Download Apriori OLS Table",
                apriori_df.to_csv(index=False).encode(), "apriori_ols.csv", "text/csv",
                key="apriori_dl",
            )

            with st.expander("🔎 Inspect a single variable's fit"):
                inspect_var = st.selectbox(
                    "Variable", apriori_vars, key="apriori_inspect_var"
                )
                x = df_corr[inspect_var].astype(float)
                mask = x.notna() & y.notna()
                x_clean, y_clean = x[mask], y[mask]
                if len(x_clean) >= 3 and x_clean.std() > 0:
                    res = sp_stats.linregress(x_clean, y_clean)
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=x_clean, y=y_clean, mode="markers", name="Observed",
                        marker=dict(color="#4C72B0", opacity=0.7),
                    ))
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = res.intercept + res.slope * x_line
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode="lines", name="OLS fit",
                        line=dict(color="#C44E52", width=2),
                    ))
                    fig_scatter.update_layout(
                        title=f"{apriori_kpi} = {res.intercept:.4f} + {res.slope:.4f} × {inspect_var}  "
                              f"(R²={res.rvalue**2:.3f}, p={res.pvalue:.4f})",
                        xaxis_title=inspect_var, yaxis_title=apriori_kpi,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough variation / data to fit a line for this variable.")
        else:
            st.info("Select at least 1 variable.")
