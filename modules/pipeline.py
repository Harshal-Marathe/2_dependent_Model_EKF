"""
Full EKF MMM pipeline: optimize on the train split, run the filter +
smoother on the full dataset, and assemble contributions / ROI / parameter
tables for the Results tab.

Two entry points:
  - run_full_ekf_pipeline: single dependent variable (univariate EKF).
  - run_multi_dependent_pipeline: one or two dependent variables. When a
    second dependent variable is configured, the two equations are fitted
    JOINTLY with a bivariate Kalman filter (shared time index, correlated
    observation errors) rather than as two separate, independent fits —
    see modules/kalman.py::run_bivariate_kalman_filter for the state-space
    derivation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.dependencies import NEVERGRAD_AVAILABLE
from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds
from modules.optimizer import run_nevergrad_optimizer, run_nevergrad_optimizer_joint
from modules.kalman import (
    run_kalman_filter, run_bivariate_kalman_filter, rts_smoother,
    _precompute_adstocked, _build_observation_matrix, build_static_cache,
)
from modules.transforms import apply_transformation, hill_transform_vec
from modules.metrics import safe_mape


# ── Shared post-processing (contributions / ROI / parameter tables) ─────────

def _postprocess_equation(df_full, g, params, x_smooth, adstocked_media,
                           cross_beta_contrib, opt_success, opt_nit, loglik):
    """
    Given a fitted/smoothed state trajectory for ONE equation (one
    dependent variable), builds: smoothed yhat, MAPE/R2, the contribution
    table, ROI table, synergy table and parameter table. Used identically
    whether that equation came from the single-dependent univariate filter
    or from one half of the joint bivariate filter.
    """
    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    ADSTOCK_TYPE   = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    x_smooth = x_smooth.copy()

    # Re-apply positivity / negativity floors after RTS smoothing
    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    for col in positive_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
    for col in negative_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["COMP_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["COMP_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in PRICE_COLS:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["N_COMP_NONMEDIA"] + PRICE_COLS.index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)

    # Baseline floor — a market-mix baseline shouldn't be negative or
    # near-zero. The forward filter already floors it (see
    # modules/kalman.py::_apply_beta_floors), but the RTS backward pass
    # can still pull it below the floor again since smoothing is an
    # unconstrained blend of filtered + next-period-smoothed values.
    min_base_fraction = float(g.get("MIN_BASE_FRACTION", 0.0))
    if min_base_fraction > 0:
        intercept_floor = min_base_fraction * float(df_full[TARGET_COL].mean())
        x_smooth[:, 0] = np.maximum(x_smooth[:, 0], intercept_floor)

    L_mat_full  = _build_observation_matrix(df_full, g, adstocked_media)
    yhat_smooth = (L_mat_full * x_smooth).sum(axis=1)

    target_vals  = df_full[TARGET_COL].values
    resid_smooth = target_vals - yhat_smooth
    # Zero-safe MAPE — see modules/metrics.py for why the naive
    # mean(|resid| / (|actual| + 1e-12)) formula is wrong here: any
    # period where the dependent variable is 0 (common for Dependent 2
    # KPIs like trial counts / leads, unlike a Sales Dependent 1) makes
    # that formula explode to billions of percent and swamp the metric.
    mape  = safe_mape(resid_smooth, target_vals)
    ss_res = np.sum(resid_smooth**2); ss_tot = np.sum((target_vals - target_vals.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    contrib_df = df_full[[TARGET_COL]].copy()

    G0 = float(params["G0"])
    prev_intercept = np.empty(len(df_full))
    prev_intercept[1:] = x_smooth[:-1, 0]
    prev_intercept[0]  = x_smooth[0, 0]
    intercept_carryover = G0 * prev_intercept

    # Short-term view: the intercept as it actually enters the observation
    # equation, Y_t = intercept_t + Σ beta_i,t * media_i,t + ...  (i.e. the
    # full smoothed intercept level, not a residual).
    contrib_df["ShortTerm_Intercept"] = x_smooth[:, 0]

    # Long-term view: decompose that SAME intercept per its own state
    # equation, I_t = G0 * I_(t-1) + Σ_k gamma_k * media_k,t^n_k_intercept,
    # into a carryover piece and (below) a per-effector boost piece. Named
    # "Intercept Carryover" (not "Intercept") so it doesn't collide with the
    # short-term "Intercept" row when Short-Term + Long-Term are combined.
    contrib_df["LongTerm_Intercept Carryover"] = intercept_carryover

    for i, col in enumerate(MEDIA_COLS):
        # Matches the observation equation: β_i,t is multiplied by RAW spend/
        # impressions, not adstocked media (carryover already lives in β_i,t
        # via its own λ_i decay / Weibull lag-weighting in the state equation).
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, i+1] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for j, col in enumerate(COMP_MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, 1+g["N_MEDIA"]+j] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for p, col in enumerate(PRICE_COLS):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+g["N_COMP_NONMEDIA"]+p
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0

    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        ni_int = params["n_intercept"][k]
        if col in MEDIA_COLS:
            raw = df_full[col].values.astype(float)
            transformed = apply_transformation(
                raw, TRANSFORM_TYPE, ni_int, params["S_params"][MEDIA_COLS.index(col)])
            contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * transformed
        elif col in df_full.columns:
            contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * df_full[col].values.astype(float)

    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        contrib_df[f"Synergy_{tgt}_from_{src}"] = cross_beta_contrib[:, k]

    # ROI denominator: for a channel whose raw input is GRP/impressions
    # (not currency), summing that column itself is meaningless as
    # "spend". MEDIA_SPEND_MAP (built in modules/params.py from
    # per_channel_bounds[col]["__spend_col__"], set in Tab 5 · D2 / Tab 8)
    # maps such a channel to its real spend column, whose TOTAL is used
    # as the ROI denominator instead. Channels left as "Spend" (the
    # default) fall back to summing themselves, unchanged from before.
    media_spend_map = g.get("MEDIA_SPEND_MAP", {})
    roi_rows = []
    for col in MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        spend_col = media_spend_map.get(col, col)
        if spend_col in df_full.columns:
            ts = df_full[spend_col].sum()
        else:
            spend_col = col
            ts = df_full[col].sum()
        roi_rows.append({"Channel": col,
                         "InputType": "GRP/Impressions" if spend_col != col else "Spend",
                         "SpendColumn": spend_col, "TotalSpend": ts, "TotalContrib": tc,
                         "ROI": tc/ts if ts > 0 else 0})
    roi_df = pd.DataFrame(roi_rows)

    synergy_rows = []
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        col_name = f"Synergy_{tgt}_from_{src}"
        total_synergy = float(contrib_df[col_name].sum())
        tgt_total = (contrib_df[f"ShortTerm_{tgt}"].sum()
                     + contrib_df[f"LongTerm_{tgt}"].sum()) if tgt in MEDIA_COLS else np.nan
        synergy_rows.append({
            "Source Channel": src,
            "Target Channel": tgt,
            "Total Synergy Contribution": total_synergy,
            "Avg Synergy / Period": float(contrib_df[col_name].mean()),
            "Cross Delta": float(params["cross_delta"][k]),
            "Cross Hill n": float(params["cross_n"][k]),
            "Cross Hill S": float(params["cross_S"][k]),
            "Share of Target's Total Contrib (%)": (
                round(100 * total_synergy / tgt_total, 2)
                if tgt_total and tgt_total != 0 and not np.isnan(tgt_total) else np.nan
            ),
        })
    synergy_df = pd.DataFrame(synergy_rows)

    # ── Parameter table ───────────────────────────────────────────────
    param_rows = []
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        effector_kind = "Media" if col in MEDIA_COLS else "Non-media"
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "Gamma (boost coeff.)", "Value": params["gamma"][k],
        })
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "n_intercept (exponent)", "Value": params["n_intercept"][k],
        })

    for i, col in enumerate(MEDIA_COLS):
        if TRANSFORM_TYPE == "hill":
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Hill slope)","Value":params["n_params"][i]},
                {"Category":"Media","Variable":col,"Parameter":"S (Half-sat)",  "Value":params["S_params"][i]},
            ]
        else:
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Power exponent)","Value":params["n_params"][i]},
            ]
        param_rows += [
            {"Category":"Media","Variable":col,"Parameter":"Ls",    "Value":params["Ls"][i]},
            {"Category":"Media","Variable":col,"Parameter":"Delta", "Value":params["delta"][i]},
        ] + transform_rows

        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"Media","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][i]},
                {"Category":"Media","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][i]},
            ]
        # Instant mode: no separate adstock row — carryover is the "Ls" row above.

    for j, col in enumerate(COMP_MEDIA_COLS):
        param_rows += [
            {"Category":"CompMedia","Variable":col,"Parameter":"Ls_comp",   "Value":params["Ls_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"Delta_comp", "Value":params["delta_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"n_comp",     "Value":params["n_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"S_comp",     "Value":params["S_comp"][j]},
        ]
        ci = g["N_MEDIA"] + j
        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ci]},
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ci]},
            ]
        # Instant mode: no separate adstock row — carryover is the "Ls_comp" row above.

    for i, col in enumerate(PRICE_COLS):
        param_rows += [
            {"Category":"Price","Variable":col,"Parameter":"Ls_price",  "Value":params["Ls_price"][i]},
            {"Category":"Price","Variable":col,"Parameter":"Delta_price","Value":params["delta_price"][i]},
        ]
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        pair_label = f"{src}→{tgt}"
        param_rows += [
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Delta", "Value":params["cross_delta"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill n","Value":params["cross_n"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill S","Value":params["cross_S"][k]},
        ]
    param_rows.append({"Category":"Global","Variable":"Intercept","Parameter":"G0",     "Value":params["G0"]})
    param_rows.append({"Category":"Global","Variable":"Noise",    "Parameter":"sigma_y","Value":params["sigma_y"]})
    if g["USE_ORGANIC_DRIFT"]:
        param_rows.append({"Category":"Global","Variable":"Organic drift","Parameter":"mu","Value":params["mu"]})

    if ADSTOCK_TYPE == "weibull":
        param_rows.append({"Category":"Global","Variable":"Weibull adstock",
                            "Parameter":"n_lags", "Value": g["ADSTOCK_N_LAGS"]})

    param_df = pd.DataFrame(param_rows)

    return {
        "params":params,"yhat_smooth":yhat_smooth,"residuals":resid_smooth,
        "x_smooth":x_smooth,"adstocked_media":adstocked_media,
        "contrib_df":contrib_df,"roi_df":roi_df,"param_df":param_df,"synergy_df":synergy_df,
        "loglik":loglik,"mape":mape,"r2":r2,
        "success":opt_success,"nit":opt_nit,"g":g,
    }


# ── Single-dependent-variable pipeline (univariate EKF) ──────────────────────

def run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    g = _make_globals(config)
    n_train  = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
    theta0, bounds = _build_theta0_and_bounds(df_train, g)

    # The observation matrix depends only on (df, g), never on the theta
    # being searched over — build it once per optimization run instead of
    # on every single candidate evaluation.
    static_cache_train = build_static_cache(df_train, g)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta, _ = run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg,
                                                  static_cache=static_cache_train)
        opt_success = True; opt_nit = ng_cfg.get("budget", 500)
    else:
        def objective(theta):
            p = unpack_theta(theta, g)
            _, _, _, _, _, _, _, _, loglik = run_kalman_filter(
                df_train, p, g, static_cache=static_cache_train)
            return -loglik
        opt = minimize(objective, theta0, method=method,
                       bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-9})
        best_theta = opt.x; opt_success = opt.success; opt_nit = opt.nit

    params = unpack_theta(best_theta, g)
    static_cache_full = build_static_cache(df_full, g)
    yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik = \
        run_kalman_filter(df_full, params, g, static_cache=static_cache_full)
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat)

    adstocked_media = _precompute_adstocked(df_full, g, params)
    result = _postprocess_equation(
        df_full, g, params, x_smooth, adstocked_media, cross_beta_contrib,
        opt_success, opt_nit, loglik,
    )
    result["P_smooth"] = P_smooth
    return result


# ── Multi-dependent pipeline — now a genuine JOINT bivariate fit ────────────

def run_multi_dependent_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    """
    Fits Dependent 1 (config["target"]) and, if a second dependent variable
    is configured (config["target2"], e.g. Top-of-Mind / Consideration),
    fits it TOGETHER with Dependent 1 using a single joint bivariate Kalman
    filter:

        [ y1_t ]   [ Intercept_1_t ]   [ beta_1_1_t ... beta_1_M_t ]
        [ y2_t ] = [ Intercept_2_t ] + [ beta_2_1_t ... beta_2_M_t ] · x_t
                                                            + correlated errors

    The two equations share the same time index and the same raw
    regressors x_t, but each keeps its own state dynamics (its own Ls, G0,
    adstock, transform, and per-channel bounds via config["per_channel_bounds_2"]
    if provided). What makes the fit "joint" rather than two independent
    fits stitched together is:
      1. A single optimizer call estimates BOTH equations' parameters
         (theta_1, theta_2), the cross-equation error correlation rho,
         AND the cross-intercept coupling coefficients phi_1/phi_2
         simultaneously, by maximising the bivariate log-likelihood:
             Intercept_1,t = G0_1·Intercept_1,t-1 + phi_1·Intercept_2,t-1 + effectors_1,t
             Intercept_2,t = G0_2·Intercept_2,t-1 + phi_2·Intercept_1,t-1 + effectors_2,t
      2. At every time step, the Kalman gain is computed from the full
         2x2 observation-noise covariance, so a surprising observation in
         one equation also updates the other equation's state estimate
         (through the off-diagonal covariance terms) at that same t.

    If no second dependent variable is configured, this transparently
    falls back to the single-equation pipeline (results_2 is None).

    Returns
    -------
    (results_1, results_2)
        results_2 is None if no second dependent variable is configured.
        When both are fitted, each result dict also carries "rho_y"
        (estimated error correlation), "phi1"/"phi2" (estimated
        cross-intercept coupling coefficients), and "joint_loglik"
        (shared bivariate log-likelihood) for display/diagnostics.
    """
    target2 = config.get("target2")
    if not (config.get("enable_second_dependent") and target2):
        results_1 = run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=ng_cfg)
        return results_1, None

    # ── Build per-equation configs / globals ─────────────────────────
    # Dependent 2 can reuse Dependent 1's exact predictor set (default,
    # backward-compatible with older saved configs that have no _2 keys),
    # or use its own independently-selected — and potentially overlapping —
    # set of media / non-media / price / competitor variables, configured
    # in Tab 5 · Section A3.
    config_1 = dict(config)
    config_2 = dict(config)
    config_2["target"]          = target2
    config_2["media"]           = config.get("media_2")           or config["media"]
    config_2["non_media"]       = config.get("non_media_2", config["non_media"])
    config_2["comp_media"]      = config.get("comp_media_2", config["comp_media"])
    config_2["comp_nonmedia"]   = config.get("comp_nonmedia_2", config["comp_nonmedia"])
    config_2["price"]           = config.get("price_2", config["price"])
    config_2["use_price"]       = config.get("use_price_2", config["use_price"])
    config_2["cross_media_map"] = config.get("cross_media_map_2", config["cross_media_map"])
    config_2["positive_beta_cols"] = config.get("positive_beta_cols_2", config["positive_beta_cols"])
    config_2["negative_beta_cols"] = config.get("negative_beta_cols_2", config["negative_beta_cols"])
    # Re-derive Dep 2's own initial-beta defaults against ITS OWN (possibly
    # different) channel lists rather than reusing Dep 1's, which may not
    # even contain the same columns.
    config_2["initial_media_betas"]         = {c: 0.0     for c in config_2["media"]}
    config_2["initial_comp_betas"]          = {c: -0.0001 for c in config_2["comp_media"]}
    config_2["initial_own_nonmedia_betas"]  = {c: 0.0     for c in config_2["non_media"]}
    config_2["initial_comp_nonmedia_betas"] = {c: -0.01   for c in config_2["comp_nonmedia"]}
    config_2["initial_price_beta"]          = {c: -0.1    for c in config_2["price"]}

    ie2 = config.get("intercept_effectors_2")
    if ie2 is not None:
        config_2["intercept_effectors"] = ie2
    pcb_2 = config.get("per_channel_bounds_2")
    if pcb_2:
        config_2["per_channel_bounds"] = pcb_2

    g1 = _make_globals(config_1)
    g2 = _make_globals(config_2)

    n_train  = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)

    theta0_1, bounds1 = _build_theta0_and_bounds(df_train, g1)
    theta0_2, bounds2 = _build_theta0_and_bounds(df_train, g2)
    n1 = len(theta0_1)
    n2 = len(theta0_2)

    rho0 = 0.0
    rho_bounds = (-0.95, 0.95)
    # Cross-intercept coupling — see modules/kalman.py module docstring:
    #   Intercept_1,t = G0_1·Intercept_1,t-1 + phi_1·Intercept_2,t-1 + effectors_1,t
    #   Intercept_2,t = G0_2·Intercept_2,t-1 + phi_2·Intercept_1,t-1 + effectors_2,t
    phi1_0, phi2_0 = 0.0, 0.0
    phi_bounds = (0.1, None)
    theta0_joint = np.concatenate([theta0_1, theta0_2, [rho0, phi1_0, phi2_0]])
    bounds_joint = list(bounds1) + list(bounds2) + [rho_bounds, phi_bounds, phi_bounds]

    static_cache1_train = build_static_cache(df_train, g1)
    static_cache2_train = build_static_cache(df_train, g2)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta_joint, _ = run_nevergrad_optimizer_joint(
            df_train, g1, g2, theta0_joint, bounds_joint, n1, n2, ng_cfg,
            static_cache1=static_cache1_train, static_cache2=static_cache2_train)
        opt_success = True; opt_nit = ng_cfg.get("budget", 500)
    else:
        def objective(theta_joint):
            theta1 = theta_joint[:n1]
            theta2 = theta_joint[n1:n1+n2]
            rho    = theta_joint[n1+n2]
            phi1   = theta_joint[n1+n2+1]
            phi2   = theta_joint[n1+n2+2]
            p1 = unpack_theta(theta1, g1)
            p2 = unpack_theta(theta2, g2)
            (_, _, _, _, _, _, _, _, _, loglik, _, _) = \
                run_bivariate_kalman_filter(df_train, p1, g1, p2, g2, rho, phi1, phi2,
                                             static_cache1=static_cache1_train,
                                             static_cache2=static_cache2_train)
            return -loglik
        opt = minimize(objective, theta0_joint, method=method,
                        bounds=bounds_joint, options={"maxiter": max_iter, "ftol": 1e-9})
        best_theta_joint = opt.x; opt_success = opt.success; opt_nit = opt.nit

    best_theta1 = best_theta_joint[:n1]
    best_theta2 = best_theta_joint[n1:n1+n2]
    best_rho    = float(np.clip(best_theta_joint[n1+n2], -0.995, 0.995))
    best_phi1   = float(best_theta_joint[n1+n2+1])
    best_phi2   = float(best_theta_joint[n1+n2+2])
    params1 = unpack_theta(best_theta1, g1)
    params2 = unpack_theta(best_theta2, g2)

    # ── Run the joint filter on the FULL dataset with the fitted params ──
    static_cache1_full = build_static_cache(df_full, g1)
    static_cache2_full = build_static_cache(df_full, g2)
    (yhat_joint, residuals_joint, x_filt, P_filt, x_pred, P_pred, Tmat_joint,
     cross1, cross2, joint_loglik, dim1, dim2) = run_bivariate_kalman_filter(
        df_full, params1, g1, params2, g2, best_rho, best_phi1, best_phi2,
        static_cache1=static_cache1_full, static_cache2=static_cache2_full)

    # RTS smoother is dimension-agnostic — run once on the joint state
    x_smooth_joint, P_smooth_joint = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat_joint)
    x_smooth_1 = x_smooth_joint[:, :dim1]
    x_smooth_2 = x_smooth_joint[:, dim1:]

    adstocked_media_1 = _precompute_adstocked(df_full, g1, params1)
    adstocked_media_2 = _precompute_adstocked(df_full, g2, params2)

    results_1 = _postprocess_equation(
        df_full, g1, params1, x_smooth_1, adstocked_media_1, cross1,
        opt_success, opt_nit, joint_loglik,
    )
    results_2 = _postprocess_equation(
        df_full, g2, params2, x_smooth_2, adstocked_media_2, cross2,
        opt_success, opt_nit, joint_loglik,
    )

    for res in (results_1, results_2):
        res["rho_y"] = best_rho
        res["phi1"] = best_phi1  # coefficient of Intercept_Dep2,t-1 in Dep1's intercept eq
        res["phi2"] = best_phi2  # coefficient of Intercept_Dep1,t-1 in Dep2's intercept eq
        res["joint_loglik"] = joint_loglik
        res["joint_fit"] = True
        res["P_smooth"] = P_smooth_joint  # joint covariance (block layout: dim1 then dim2)

    return results_1, results_2
