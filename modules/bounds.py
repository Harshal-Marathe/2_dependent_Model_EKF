"""
Build the initial theta vector (theta0) and its optimizer bounds.
Supports both adstock types (instant/weibull) and both transformation
types (power/hill) with appropriate parameter bounds per spec.
"""

import numpy as np

from modules.params import safe_median


def _build_theta0_and_bounds(df, g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_CROSS = g["N_CROSS"]; N_EFFECTORS = g["N_EFFECTORS"]
    MEDIA_COLS = g["MEDIA_COLS"]; COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]
    ADSTOCK_TYPE  = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]
    POSITIVE_BETA_COLS = g.get("POSITIVE_BETA_COLS", [])
    NEGATIVE_BETA_COLS = g.get("NEGATIVE_BETA_COLS", [])
    PCB = g.get("PER_CHANNEL_BOUNDS", {})

    def _b(col, param, lo_def, hi_def):
        if col in PCB and param in PCB[col]:
            return PCB[col][param]
        return (lo_def, hi_def)

    # ── Beta persistence ──────────────────────────────────────────────
    ls_bounds = [_b(c, "ls", 0.2, 0.8) for c in MEDIA_COLS]
    ls_init   = [np.clip(0.5, lo, hi) for lo, hi in ls_bounds]

    # G0 bound
    G0_bound = (0.7, 0.99)
    G0_init  = 1.0  # or similar, just keep it inside the new bound

    # delta bounds: positive or negative constraint per media col
    def _delta_bound(col):
        if col in POSITIVE_BETA_COLS: return (0.0, None)
        if col in NEGATIVE_BETA_COLS: return (None, 0.0)
        return (None, None)
    delta_bounds = [_delta_bound(c) for c in MEDIA_COLS]
    delta_init   = np.full(N_MEDIA, 0.05)

    # gamma bounds (intercept effectors — always positive)
    gamma_bounds = [(0.0, None)] * N_EFFECTORS
    gamma_init   = np.full(N_EFFECTORS, 0.02)

    # ── Transformation parameters ─────────────────────────────────────
    if TRANSFORM_TYPE == "power":
        # n: (0, 1) per spec
        n_bounds = [_b(c, "transform_n", 0.01, 1.0) for c in MEDIA_COLS]
        n_init   = [np.clip(0.5, lo, hi) for lo, hi in n_bounds]
        # S is unused for power but keep a slot so theta layout is consistent
        S_bounds = [(1e-6, 1e8)] * N_MEDIA
        S_init   = [safe_median(df[c]) for c in MEDIA_COLS]
    else:
        # Hill: n: (1, 15), S > 0 per spec
        n_bounds = [_b(c, "hill_n", 1.0, 15.0) for c in MEDIA_COLS]
        n_init   = [np.clip(2.0, lo, hi) for lo, hi in n_bounds]
        S_bounds = [_b(c, "hill_s", 1e-6, 1e8) for c in MEDIA_COLS]
        S_init   = [safe_median(df[c]) for c in MEDIA_COLS]

    # ── Intercept effector transformation (ni, Si) ────────────────────
    # Independent of TRANSFORM_TYPE (which only governs the media betas)
    # — the intercept equation has its own Power/Hill switch, set via
    # config "intercept_transform_type".
    INTERCEPT_TRANSFORM_TYPE = g.get("INTERCEPT_TRANSFORM_TYPE", "power")
    INTERCEPT_EFFECTORS = g["INTERCEPT_EFFECTORS"]
    if INTERCEPT_TRANSFORM_TYPE == "power":
        n_int_bounds = [(0.01, 1.0)] * N_EFFECTORS
        n_int_init   = [0.5] * N_EFFECTORS
        # S is unused for power but keep a slot so theta layout is consistent
        S_int_bounds = [(1e-6, 1e8)] * N_EFFECTORS
        S_int_init   = [safe_median(df[c]) if c in df.columns else 1.0
                        for c in INTERCEPT_EFFECTORS]
    else:
        # Hill: n: (1, 15), S > 0 — mirrors the media Hill bounds above
        n_int_bounds = [(1.0, 15.0)] * N_EFFECTORS
        n_int_init   = [2.0] * N_EFFECTORS
        S_int_bounds = [(1e-6, 1e8)] * N_EFFECTORS
        S_int_init   = [safe_median(df[c]) if c in df.columns else 1.0
                        for c in INTERCEPT_EFFECTORS]

    # ── Adstock parameters ────────────────────────────────────────────
    all_adstock_cols = list(MEDIA_COLS) + list(COMP_MEDIA_COLS)
    N_ADSTOCK = len(all_adstock_cols)

    if ADSTOCK_TYPE == "weibull":
        # shape k: (0.1, 5.0), scale lambda: (0.1, 5.0) — user can tighten via PCB
        shape_bounds = [_b(c, "adstock_shape", 0.1, 5.0) for c in all_adstock_cols]
        scale_bounds = [_b(c, "adstock_scale", 0.1, 5.0) for c in all_adstock_cols]
        adstock_init   = ([np.clip(1.5, lo, hi) for lo, hi in shape_bounds] +
                          [np.clip(1.0, lo, hi) for lo, hi in scale_bounds])
        adstock_bounds = shape_bounds + scale_bounds
    else:
        # Instant (Nerlove-Arrow): no adstock lambda is estimated at all.
        # Carryover for every state (own media, comp media, and — via the
        # target beta it lands in — synergy) is carried entirely by that
        # state's own Ls persistence. Adding a second, separately-fitted
        # decay here on top of Ls would double-count the same carryover.
        adstock_init   = []
        adstock_bounds = []

    # ── Non-media / organic ───────────────────────────────────────────
    own_nm_ls_bounds    = [_b(c, "ls", 0.2, 0.8) for c in g["OWN_NONMEDIA_COLS"]]
    comp_nm_ls_bounds   = [_b(c, "ls", 0.2, 0.8) for c in g["COMP_NONMEDIA_COLS"]]
    def _nm_delta_bound(col):
        if col in POSITIVE_BETA_COLS: return (0.0, None)
        if col in NEGATIVE_BETA_COLS: return (None, 0.0)
        return (None, None)
    own_nm_delta_bounds = [_nm_delta_bound(c) for c in g["OWN_NONMEDIA_COLS"]]
    comp_nm_delta_bounds = [(None, 0)] * N_COMP_NONMEDIA

    # ── Competitor media ──────────────────────────────────────────────
    ls_comp_bounds  = [_b(c, "ls",     0.2, 0.8)  for c in COMP_MEDIA_COLS]
    n_comp_bounds   = [_b(c, "hill_n", 0.3, 5.0)  for c in COMP_MEDIA_COLS]
    s_comp_bounds   = [_b(c, "hill_s", 1e-6, 1e8) for c in COMP_MEDIA_COLS]
    price_ls_bounds = [_b(c, "ls", 0.2, 0.8) for c in g["PRICE_COLS"]]

    # ── theta0 assembly ───────────────────────────────────────────────
    theta0 = np.concatenate([
        ls_init,
        [G0_init],
        delta_init,
        gamma_init,
        n_init,
        S_init,
        n_int_init,
        S_int_init,
        adstock_init,
        [np.clip(0.5, lo, hi) for lo, hi in own_nm_ls_bounds] if N_OWN_NONMEDIA else [],
        [np.clip(0.5, lo, hi) for lo, hi in comp_nm_ls_bounds] if N_COMP_NONMEDIA else [],
        np.full(N_OWN_NONMEDIA, 0.01),
        np.full(N_COMP_NONMEDIA, -0.01),
        [np.clip(0.5, lo, hi) for lo, hi in ls_comp_bounds] if N_COMP else [],
        np.full(N_COMP, 0.02),
        np.full(N_COMP, 1.5),
        [safe_median(df[c]) for c in COMP_MEDIA_COLS] if COMP_MEDIA_COLS else [],
        np.full(N_CROSS, 0.02),
        np.full(N_CROSS, 1.5),
        np.full(N_CROSS, 1.0),
        [np.clip(0.5, lo, hi) for lo, hi in price_ls_bounds] if N_PRICE else [],
        np.full(N_PRICE, -0.01),
        [0.0] if USE_ORGANIC_DRIFT else [],
        [max(df[g["TARGET_COL"]].std() * 0.3, 1e-3)],
    ])

    bounds = (
        ls_bounds +
        [G0_bound] +
        delta_bounds +
        gamma_bounds +
        n_bounds +
        S_bounds +
        n_int_bounds +
        S_int_bounds +
        adstock_bounds +
        own_nm_ls_bounds +
        comp_nm_ls_bounds +
        own_nm_delta_bounds +
        comp_nm_delta_bounds +
        ls_comp_bounds +
        [(None, 0)] * N_COMP +           # delta_comp
        n_comp_bounds +
        s_comp_bounds +
        [(0, None)] * N_CROSS +          # cross_delta
        [(0.3, 5.0)] * N_CROSS +         # cross_n
        [(1e-6, None)] * N_CROSS +       # cross_S
        price_ls_bounds +                # Ls_price
        [(None, 0)] * N_PRICE +          # delta_price
        ([(-1.0, 1.0)] if USE_ORGANIC_DRIFT else []) +
        [(1e-3, None)]                   # sigma_y
    )
    return theta0, bounds
