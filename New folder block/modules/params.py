"""
Parameter handling: building the globals dict (`g`) from a saved config,
and unpacking the flat theta vector used by the optimizer into named
parameter arrays.
"""

import numpy as np


def safe_median(series, default=1.0):
    val = np.nanmedian(series.values)
    return val if np.isfinite(val) and val > 0 else default


def _make_globals(cfg: dict):
    g = {}
    g["TARGET_COL"]          = cfg["target"]
    g["MEDIA_COLS"]          = cfg["media"]
    g["COMP_MEDIA_COLS"]     = cfg.get("comp_media", [])
    g["OWN_NONMEDIA_COLS"]   = cfg.get("non_media", [])
    g["COMP_NONMEDIA_COLS"]  = cfg.get("comp_nonmedia", [])
    g["PRICE_COLS"]          = cfg.get("price", []) if cfg.get("use_price", False) else []
    g["DUMMY_COLS"]          = cfg.get("dummy_cols", [])
    g["INTERCEPT_EFFECTORS"] = cfg.get("intercept_effectors", cfg["media"])
    g["CROSS_MEDIA_MAP"]     = cfg.get("cross_media_map", {})
    g["USE_ORGANIC_DRIFT"]   = cfg.get("use_organic", False)
    g["USE_PRICE"]           = cfg.get("use_price", False)

    # ── Adstock & transformation configuration ──────────────────────
    g["ADSTOCK_TYPE"]        = cfg.get("adstock_type", "instant")   # "instant" | "weibull"
    g["TRANSFORM_TYPE"]      = cfg.get("transform_type", "hill")    # "power"   | "hill"
    g["ADSTOCK_N_LAGS"]      = int(cfg.get("adstock_n_lags", 8))    # only for weibull

    g["POSITIVE_BETA_COLS"]  = cfg.get("positive_beta_cols", [])
    g["NEGATIVE_BETA_COLS"]  = cfg.get("negative_beta_cols", [])
    g["PER_CHANNEL_BOUNDS"]  = cfg.get("per_channel_bounds", {})

    # ── State-space Q / P0 block structure ───────────────────────────
    # Q and P0 are built as BLOCK-DIAGONAL matrices with two named blocks:
    # an intercept block (1x1) and a betas block (every media/comp-media/
    # non-media/comp-non-media/price coefficient state, together) — see
    # modules/kalman.py::_build_process_noise / _initial_state. The two
    # blocks never share off-diagonal (cross-block) covariance; within the
    # betas block, BETA_PROCESS_CORR/BETA_PRIOR_CORR optionally add a
    # shared (equicorrelated) off-diagonal structure. Defaults reproduce
    # the previous pure-diagonal behaviour exactly (rho = 0).
    g["Q_INTERCEPT_VAR"]     = float(cfg.get("q_intercept_var", 1e-4))
    g["Q_BETA_VAR"]          = float(cfg.get("q_beta_var", 1e-6))
    g["BETA_PROCESS_CORR"]   = float(cfg.get("beta_process_corr", 0.0))
    g["BETA_PRIOR_CORR"]     = float(cfg.get("beta_prior_corr", 0.0))

    g["N_MEDIA"]         = len(g["MEDIA_COLS"])
    g["N_COMP"]          = len(g["COMP_MEDIA_COLS"])
    g["N_OWN_NONMEDIA"]  = len(g["OWN_NONMEDIA_COLS"])
    g["N_COMP_NONMEDIA"] = len(g["COMP_NONMEDIA_COLS"])
    g["N_PRICE"]         = len(g["PRICE_COLS"])
    g["N_DUMMIES"]       = len(g["DUMMY_COLS"])
    g["N_EFFECTORS"]     = len(g["INTERCEPT_EFFECTORS"])
    g["N_ADSTOCK"]       = g["N_MEDIA"] + g["N_COMP"]
    g["SEASONAL_DIM"]    = 0

    g["CROSS_MEDIA_PAIRS"] = [
        (tgt, src)
        for tgt, srcs in g["CROSS_MEDIA_MAP"].items()
        for src in srcs
    ]
    g["N_CROSS"] = len(g["CROSS_MEDIA_PAIRS"])

    g["INITIAL_MEDIA_BETAS"]         = cfg.get("initial_media_betas", {})
    g["INITIAL_COMP_BETAS"]          = cfg.get("initial_comp_betas", {})
    g["INITIAL_OWN_NONMEDIA_BETAS"]  = cfg.get("initial_own_nonmedia_betas", {})
    g["INITIAL_COMP_NONMEDIA_BETAS"] = cfg.get("initial_comp_nonmedia_betas", {})
    g["INITIAL_PRICE_BETA"]          = cfg.get("initial_price_beta", {})
    return g


def unpack_theta(theta, g: dict):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_CROSS = g["N_CROSS"]
    N_EFFECTORS = g["N_EFFECTORS"]; N_ADSTOCK = g["N_ADSTOCK"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]
    ADSTOCK_TYPE  = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    idx = 0

    # ── Beta-persistence (Ls) for own media ─────────────────────────
    Ls       = theta[idx:idx+N_MEDIA];     idx += N_MEDIA
    G0       = theta[idx];                 idx += 1
    delta    = theta[idx:idx+N_MEDIA];     idx += N_MEDIA
    gamma    = theta[idx:idx+N_EFFECTORS]; idx += N_EFFECTORS

    # ── Transformation parameters ────────────────────────────────────
    # n_params always present (power exponent OR Hill slope n)
    n_params = theta[idx:idx+N_MEDIA];     idx += N_MEDIA
    # S_params only used for Hill; present in theta regardless (bounds
    # keep it irrelevant for power — optimizer still needs a slot)
    S_params = theta[idx:idx+N_MEDIA];     idx += N_MEDIA

    # ── Intercept effector transformation exponent ni ────────────────
    # Used in: I_t = G0 * I_{t-1} + Σ gamma_i * media_i^ni
    n_intercept = theta[idx:idx+N_EFFECTORS]; idx += N_EFFECTORS

    # ── Adstock parameters ────────────────────────────────────────────
    if ADSTOCK_TYPE == "weibull":
        adstock_shape  = theta[idx:idx+N_ADSTOCK]; idx += N_ADSTOCK
        adstock_scale  = theta[idx:idx+N_ADSTOCK]; idx += N_ADSTOCK
        adstock_lambda = np.zeros(N_ADSTOCK)
    else:
        # Instant mode: no adstock lambda is estimated — carryover lives
        # entirely in Ls (own/comp) persistence. Kept as a zero array only
        # so downstream code that reads params["adstock_lambda"] (e.g.
        # legacy display code) doesn't break; it is not consumed from theta.
        adstock_lambda = np.zeros(N_ADSTOCK)
        adstock_shape  = np.full(N_ADSTOCK, 1.5)
        adstock_scale  = np.full(N_ADSTOCK, 1.0)

    # ── Non-media / organic ───────────────────────────────────────────
    Ls_own_nonmedia     = theta[idx:idx+N_OWN_NONMEDIA];  idx += N_OWN_NONMEDIA
    Ls_comp_nonmedia    = theta[idx:idx+N_COMP_NONMEDIA]; idx += N_COMP_NONMEDIA
    delta_own_nonmedia  = theta[idx:idx+N_OWN_NONMEDIA];  idx += N_OWN_NONMEDIA
    delta_comp_nonmedia = theta[idx:idx+N_COMP_NONMEDIA]; idx += N_COMP_NONMEDIA

    # ── Competitor media ──────────────────────────────────────────────
    Ls_comp    = theta[idx:idx+N_COMP]; idx += N_COMP
    delta_comp = theta[idx:idx+N_COMP]; idx += N_COMP
    n_comp     = theta[idx:idx+N_COMP]; idx += N_COMP
    S_comp     = theta[idx:idx+N_COMP]; idx += N_COMP

    # ── Cross-media synergy ───────────────────────────────────────────
    cross_delta = theta[idx:idx+N_CROSS]; idx += N_CROSS
    cross_n     = theta[idx:idx+N_CROSS]; idx += N_CROSS
    cross_S     = theta[idx:idx+N_CROSS]; idx += N_CROSS

    # ── Price ─────────────────────────────────────────────────────────
    Ls_price    = theta[idx:idx+N_PRICE]; idx += N_PRICE
    delta_price = theta[idx:idx+N_PRICE]; idx += N_PRICE

    # ── Organic drift ─────────────────────────────────────────────────
    mu = theta[idx] if USE_ORGANIC_DRIFT else 0.0
    if USE_ORGANIC_DRIFT: idx += 1

    sigma_y = abs(theta[idx])

    return dict(
        Ls=Ls, G0=G0, delta=delta, gamma=gamma,
        n_params=n_params, S_params=S_params,
        n_intercept=n_intercept,
        adstock_lambda=adstock_lambda,
        adstock_shape=adstock_shape,
        adstock_scale=adstock_scale,
        adstock_n_lags=g["ADSTOCK_N_LAGS"],
        Ls_comp=Ls_comp, delta_comp=delta_comp, n_comp=n_comp, S_comp=S_comp,
        cross_delta=cross_delta, cross_n=cross_n, cross_S=cross_S,
        delta_own_nonmedia=delta_own_nonmedia, delta_comp_nonmedia=delta_comp_nonmedia,
        Ls_own_nonmedia=Ls_own_nonmedia, Ls_comp_nonmedia=Ls_comp_nonmedia,
        Ls_price=Ls_price, delta_price=delta_price,
        mu=mu, sigma_y=sigma_y,
    )
