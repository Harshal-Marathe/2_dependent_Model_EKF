"""
Extended Kalman Filter core: observation/transition/process-noise matrix
builders, the forward filter, and the RTS (Rauch-Tung-Striebel) smoother.

State equations (all 4 adstock × transform combinations), per dependent
variable:

Instant + Power:
  β_i,t = Ls_i · β_i,t-1  +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j
  (no adstock λ: carryover lives entirely in Ls_i, the beta-persistence
  term. Multiplying the shock by an additionally-decayed/adstocked series
  here would double-count the same carryover twice — once via Ls_i's own
  AR(1) memory, once via λ's geometric decay — so the shock always uses
  RAW spend/impressions, never adstocked media.)

Instant + Hill:
  β_i,t = Ls_i · β_i,t-1  +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)
  (same reasoning — x_i,t and every synergy x_j,t are RAW, not adstocked.
  Competitor-media betas follow the identical pattern: Ls_comp · β_comp,t-1
  + δ_comp · Hill(raw comp spend).)

Weibull + Power:
  β_i,t = Σ_{l=1}^{j} w_l · x_i,t-l   (weighted PAST-lag sum, j = user-selected lags, 0–8)
           +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j

Weibull + Hill:
  β_i,t = Σ_{l=1}^{j} w_l · x_i,t-l
           +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)

  Weibull per-lag weight (l = 1, 2, …, j), normalised to sum to 1
  (w_1 + w_2 + … + w_j = 1). The current period (l=0 / x_t) is NOT part
  of this lag sum — it only enters via the δ_i·f(x_i,t) shock term above:
    w_l = (k/λ) · (l/λ)^(k−1) · exp( −(l/λ)^k )

Intercept (all modes):
  I_t = G0 · I_{t-1}  +  Σ_k γ_k · media_k,t^{n_k_intercept}

──────────────────────────────────────────────────────────────────────────
Process noise Q / initial covariance P0 — block-diagonal structure
──────────────────────────────────────────────────────────────────────────
Both Q (process noise) and P0 (initial state covariance) are built as
BLOCK-DIAGONAL matrices with exactly two named blocks, with zero
cross-block covariance between them:

    Q  = [ Q_intercept (1×1)          0                         ]
         [ 0                          Q_betas (n_beta×n_beta)    ]

    P0 = [ P0_intercept (1×1)         0                         ]
         [ 0                          P0_betas (n_beta×n_beta)   ]

- The intercept block is a single scalar (its own noise/uncertainty
  level, independent of every beta).
- The betas block covers every media / comp-media / non-media /
  comp-non-media / price coefficient state TOGETHER as one block. It is
  `variance · [(1-ρ)·I + ρ·J]` — a diagonal matrix by default (ρ=0,
  reproducing the historical behaviour exactly), or an equicorrelated
  (compound-symmetry) matrix if ρ>0, letting betas share a common
  process-noise / prior-uncertainty component instead of evolving with
  fully independent noise.
- Dummy/seasonal indicator states sit outside both named blocks and keep
  their own independent (unblocked) diagonal noise.

See `_build_process_noise` and `_initial_state` below. Controlled via
config keys q_intercept_var, q_beta_var, beta_process_corr (Q) and
beta_prior_corr (P0) — see Tab 4 · Configuration.

──────────────────────────────────────────────────────────────────────────
Joint (bivariate) Kalman-filter fit
──────────────────────────────────────────────────────────────────────────
When a second dependent variable is configured (e.g. Consideration /
Top-of-Mind alongside Sales), the two equations above are NOT fitted as
two separate, independent EKF runs. Instead they are stacked into a single
joint (bivariate) state-space model and filtered together:

    [ sales_t         ]   [ Intercept_sales_t         ]   [ error_sales_t         ]
    [ consideration_t ] = [ Intercept_consideration_t ] + [ error_consideration_t ]

                            [ beta_sales_1_t         ... beta_sales_M_t         ]
                          + [ beta_consid_1_t        ... beta_consid_M_t       ] · x_t

where x_t is the vector of raw media/control regressors (e.g. raw TV
GRPs) — media/comp-media columns are RAW spend/impressions, not
adstocked; carryover is carried entirely by β_i,t's own Ls_i persistence
(instant mode) or, in Weibull mode, the weighted-lag sum, in the state
equation — never by adstocking the observation-side regressor, and
never by adstocking the shock term either (that would double-count the
same carryover twice). Concretely:

  • The joint transition matrix and process-noise matrix are block
    diagonal EXCEPT for two off-diagonal cross-intercept coupling terms
    (see "Cross-intercept coupling" below) — each equation's betas evolve
    according to their own dynamics (own Ls/adstock), but both blocks are
    propagated and updated within the same filter recursion, at the same
    time index t.
  • The joint observation matrix L_t is 2 × dim_joint, with the sales row
    loading only on the sales-equation block and the consideration row
    loading only on the consideration-equation block:
        L_t = [ L_sales,t        0            ]
              [ 0                L_consid,t   ]
  • The observation noise is a full 2×2 covariance matrix
        R = [ σ_sales²                  ρ·σ_sales·σ_consid ]
            [ ρ·σ_sales·σ_consid        σ_consid²           ]
    where ρ is a freely-estimated correlation between the two equations'
    contemporaneous shocks. This is what makes the fit genuinely "joint"
    rather than two separate univariate fits glued together: information
    about one equation's residual at time t informs the state update of
    the other equation at the same t (via the off-diagonal Kalman gain
    terms), and ρ itself is estimated jointly with all other parameters
    by maximising the bivariate Gaussian log-likelihood.
  • The RTS smoother is dimension-agnostic and is re-used unmodified on
    the joint (dim_1+dim_2)-length state.

──────────────────────────────────────────────────────────────────────────
Cross-intercept coupling (phi_1, phi_2)
──────────────────────────────────────────────────────────────────────────
Each equation's intercept also depends on the OTHER equation's previous
intercept, in addition to its own persistence and its own intercept
effectors:

  Intercept_1,t = G0_1 · Intercept_1,t-1  +  phi_1 · Intercept_2,t-1
                  + Σ_k γ_1,k · effector_1,k,t^{n_1,k}

  Intercept_2,t = G0_2 · Intercept_2,t-1  +  phi_2 · Intercept_1,t-1
                  + Σ_k γ_2,k · effector_2,k,t^{n_2,k}

phi_1 and phi_2 are estimated jointly with everything else (they only
exist/are fitted when a second dependent variable is configured). This
is implemented as two off-diagonal entries in the joint transition
matrix — Tmat_joint[0, dim1] = phi_1 and Tmat_joint[dim1, 0] = phi_2 —
and the matching additive terms in the mean prediction step.
"""

import numpy as np
import pandas as pd

from modules.transforms import (
    apply_transformation,
    hill_transform_vec,
    adstock_weibull_lagged,
    adstock_nerlove_arrow,
    weibull_lag_weights,
)


# ── Adstock pre-computation ──────────────────────────────────────────────────

def _precompute_adstocked(df, g, params):
    """
    For Weibull adstock: compute the weighted-lag sum for each channel.
    For Instant (Nerlove-Arrow): NOT computed. In instant mode, carryover
    is carried entirely by each state's own Ls persistence (β_i,t = Ls_i ·
    β_i,t-1 + shock), so a separately-decayed adstocked series is no longer
    needed anywhere in the pipeline — building it would only invite the
    double-carryover bug (Ls persistence stacked on top of λ decay) this
    function used to enable. Returns dict col -> np.ndarray of length T
    (empty dict in instant mode).
    """
    adstock_type = g["ADSTOCK_TYPE"]
    if adstock_type != "weibull":
        return {}

    n_lags = int(params.get("adstock_n_lags", 8))
    adstocked = {}

    for i, col in enumerate(g["MEDIA_COLS"]):
        adstocked[col] = adstock_weibull_lagged(
            df[col], params["adstock_shape"][i], params["adstock_scale"][i], n_lags)

    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        ci = g["N_MEDIA"] + j
        adstocked[col] = adstock_weibull_lagged(
            df[col], params["adstock_shape"][ci], params["adstock_scale"][ci], n_lags)

    return adstocked


# ── Observation matrix ───────────────────────────────────────────────────────

def _build_observation_matrix(df, g, adstocked_media):
    """
    y_t = I_t · 1
          + Σ β_i,t · raw_spend_i,t             (own media — raw spend/impressions,
                                                  NOT adstocked; carryover is already
                                                  carried by β_i,t's own λ_i decay in
                                                  the state equation, so multiplying by
                                                  adstocked media here would double-count
                                                  the carryover effect)
          + Σ β_j,t · raw_spend_j,t             (comp media — raw, same reasoning)
          + Σ β_k  · nonmedia_k,t               (own non-media — raw)
          + Σ β_k  · comp_nonmedia_k,t          (comp non-media — raw)
          + Σ β_p  · price_p,t
          + Σ β_d  · dummy_d,t

    Note: `adstocked_media` is still passed in and still used elsewhere (Weibull
    per-lag state transitions, Hill-on-adstocked comp/synergy shock terms in
    _prepare_equation) — it is simply no longer what the observation equation
    multiplies β by.
    """
    T = len(df)
    cols = [np.ones(T)]
    for c in g["MEDIA_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["COMP_MEDIA_COLS"]:    cols.append(df[c].values.astype(float))
    for c in g["OWN_NONMEDIA_COLS"]:  cols.append(df[c].values.astype(float))
    for c in g["COMP_NONMEDIA_COLS"]: cols.append(df[c].values.astype(float))
    for c in g["PRICE_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["DUMMY_COLS"]:         cols.append(df[c].values.astype(float))
    return np.column_stack(cols)


# ── Transition matrix ────────────────────────────────────────────────────────

def _build_transition_matrix(g, params):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    Tmat = np.eye(dim)
    Tmat[0, 0] = params["G0"]
    ADSTOCK_TYPE = g["ADSTOCK_TYPE"]
    for i in range(N_MEDIA):
        # Weibull mode: β_i,t = Σ_l w_l·x_i,t-l + δ_i·f(x_i,t) + synergy  (no λ·β_t-1 term —
        # the weighted-lag sum itself supplies the state's "memory", per the docstring).
        # Instant mode: β_i,t = λ_i·β_i,t-1 + δ_i·f(x_i,t) + synergy.
        Tmat[i+1, i+1] = 0.0 if ADSTOCK_TYPE == "weibull" else params["Ls"][i]
    for j in range(N_COMP):  Tmat[1+N_MEDIA+j, 1+N_MEDIA+j] = params["Ls_comp"][j]
    for k in range(N_OWN_NONMEDIA):
        idx = 1+N_MEDIA+N_COMP+k; Tmat[idx, idx] = params["Ls_own_nonmedia"][k]
    for k in range(N_COMP_NONMEDIA):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k; Tmat[idx, idx] = params["Ls_comp_nonmedia"][k]
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Tmat[idx, idx] = 0.98
    for p in range(N_PRICE):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p; Tmat[idx, idx] = params["Ls_price"][p]
    return Tmat


# ── Process noise ────────────────────────────────────────────────────────────

def _build_process_noise(g):
    """
    Q is built as a BLOCK-DIAGONAL matrix with two named blocks — an
    intercept block and a betas block — with zero cross-block covariance:

        Q = [ Q_intercept (1x1)         0                        ]
            [ 0                          Q_betas (n_beta x n_beta) ]

    Q_intercept = g["Q_INTERCEPT_VAR"] (own scalar noise level).
    Q_betas covers every media / comp-media / non-media / comp-non-media /
    price coefficient state together, as a single block:
        Q_betas = Q_BETA_VAR · [ (1-ρ)·I + ρ·J ]
    where ρ = g["BETA_PROCESS_CORR"] (0 ⇒ pure diagonal, i.e. the previous
    behaviour; >0 ⇒ betas share a common/equicorrelated process-noise
    component, on top of each having its own Q_BETA_VAR variance).

    Dummy/seasonal indicator states sit outside both named blocks and keep
    their own independent (unblocked) diagonal noise, as before.
    """
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    n_beta = N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE

    q_intercept = g.get("Q_INTERCEPT_VAR", 1e-4)
    q_beta      = g.get("Q_BETA_VAR", 1e-6)
    rho_beta    = float(np.clip(g.get("BETA_PROCESS_CORR", 0.0), 0.0, 0.95))

    Q = np.zeros((dim, dim))

    # ── Block 1: intercept ────────────────────────────────────────────
    Q[0, 0] = q_intercept

    # ── Block 2: betas (media, comp-media, non-media, comp-non-media, price) ──
    if n_beta > 0:
        Q[1:1+n_beta, 1:1+n_beta] = q_beta * (
            (1 - rho_beta) * np.eye(n_beta) + rho_beta * np.ones((n_beta, n_beta))
        )

    # ── Dummy / seasonal indicator states: unblocked, own diagonal noise ──
    for d in range(N_DUMMIES):
        idx = 1 + n_beta + d
        Q[idx, idx] = 5e-3
    for s in range(SEASONAL_DIM):
        idx = 1 + n_beta + N_DUMMIES + s
        Q[idx, idx] = q_beta

    return Q


# ── Transformation helper ────────────────────────────────────────────────────

def _transform_media(x: np.ndarray, transform_type: str,
                     n: float, S: float) -> np.ndarray:
    """Apply configured transformation to a media array."""
    return apply_transformation(x, transform_type, n, S)


# ── Per-equation precompute (shared by single & joint filters) ──────────────

def _prepare_equation(df, g, params):
    """
    Builds every array needed to run one equation's (one dependent
    variable's) state-space recursion: observation/transition/process-noise
    matrices, adstocked media, transformed shock terms, intercept boosts,
    Weibull lag sums, and the positivity/negativity index sets.

    This is called once per equation. For a single-dependent model it is
    called once; for the joint bivariate model it is called twice (once
    per dependent variable) and the two results are combined into the
    joint state-space system by run_bivariate_kalman_filter.
    """
    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; OWN_NONMEDIA_COLS = g["OWN_NONMEDIA_COLS"]
    COMP_NONMEDIA_COLS = g["COMP_NONMEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    CROSS_MEDIA_PAIRS = g["CROSS_MEDIA_PAIRS"]
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]
    N_CROSS = g["N_CROSS"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    ADSTOCK_TYPE  = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    T_len = len(df)
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM

    adstocked_media = _precompute_adstocked(df, g, params)
    L_mat = _build_observation_matrix(df, g, adstocked_media)
    Tmat  = _build_transition_matrix(g, params)
    Q     = _build_process_noise(g)

    # Applied to RAW spend (not adstocked) — this is the δ_i · f(x_i,t) term
    # in the state transition equation.
    transformed_own = np.stack([
        _transform_media(
            df[c].values.astype(float),
            TRANSFORM_TYPE,
            params["n_params"][i],
            params["S_params"][i],
        )
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((T_len, 0))

    # Competitor media: Hill on RAW spend in instant mode (Ls_comp already
    # supplies the carryover — Hill-ing an adstocked series on top would
    # double count it, same as own media). Weibull mode keeps Hill-on-
    # adstocked, since there the weighted-lag series IS the delay model.
    transformed_comp = np.stack([
        hill_transform_vec(
            (adstocked_media[c] if ADSTOCK_TYPE == "weibull" else df[c].values.astype(float)),
            params["n_comp"][j], params["S_comp"][j])
        for j, c in enumerate(COMP_MEDIA_COLS)
    ], axis=1) if COMP_MEDIA_COLS else np.zeros((T_len, 0))

    # Cross-media synergy: Hill on RAW source in instant mode, for the same
    # reason — the synergy contribution folds into the target channel's own
    # beta, which already persists forward via its own Ls; adstocking the
    # source on top of that would carry the source's effect over twice.
    transformed_cross = np.zeros((T_len, N_CROSS))
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        src_series = adstocked_media[src] if ADSTOCK_TYPE == "weibull" else df[src].values.astype(float)
        transformed_cross[:, k] = hill_transform_vec(
            src_series, params["cross_n"][k], params["cross_S"][k])

    # Intercept boosts: I_t += Σ γ_k · media_k,t^{n_k_intercept}
    intercept_boost = np.zeros(T_len)
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        ni_int = params["n_intercept"][k]
        if col in MEDIA_COLS:
            raw = df[col].values.astype(float)
            intercept_boost += params["gamma"][k] * _transform_media(
                raw, TRANSFORM_TYPE, ni_int, params["S_params"][MEDIA_COLS.index(col)])
        elif col in df.columns:
            intercept_boost += params["gamma"][k] * df[col].values.astype(float)

    # Weibull weighted-lag arrays for state transition
    weibull_lagsum_own = (
        np.stack([adstocked_media[c] for c in MEDIA_COLS], axis=1)
        if (ADSTOCK_TYPE == "weibull" and MEDIA_COLS) else np.zeros((T_len, N_MEDIA))
    )

    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    positive_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in positive_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in positive_cols]
    )
    negative_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + k
         for k, col in enumerate(COMP_NONMEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + p
         for p, col in enumerate(PRICE_COLS) if col in negative_cols]
    )

    return dict(
        g=g, params=params, dim=dim, T_len=T_len,
        TARGET_COL=TARGET_COL, MEDIA_COLS=MEDIA_COLS, COMP_MEDIA_COLS=COMP_MEDIA_COLS,
        OWN_NONMEDIA_COLS=OWN_NONMEDIA_COLS, COMP_NONMEDIA_COLS=COMP_NONMEDIA_COLS,
        PRICE_COLS=PRICE_COLS, CROSS_MEDIA_PAIRS=CROSS_MEDIA_PAIRS,
        N_MEDIA=N_MEDIA, N_COMP=N_COMP, N_OWN_NONMEDIA=N_OWN_NONMEDIA,
        N_COMP_NONMEDIA=N_COMP_NONMEDIA, N_PRICE=N_PRICE, N_CROSS=N_CROSS,
        ADSTOCK_TYPE=ADSTOCK_TYPE, USE_ORGANIC_DRIFT=g["USE_ORGANIC_DRIFT"],
        adstocked_media=adstocked_media, L_mat=L_mat, Tmat=Tmat, Q=Q,
        transformed_own=transformed_own, transformed_comp=transformed_comp,
        transformed_cross=transformed_cross, intercept_boost=intercept_boost,
        weibull_lagsum_own=weibull_lagsum_own,
        positive_cols=positive_cols, negative_cols=negative_cols,
        positive_state_idx=positive_state_idx, negative_state_idx=negative_state_idx,
        target_vals=df[TARGET_COL].values,
    )


def _initial_state(df, g, params, pc):
    """Builds x0 / P0 for one equation, identical to the previous
    single-equation initialisation logic."""
    dim = pc["dim"]; TARGET_COL = g["TARGET_COL"]
    MEDIA_COLS = g["MEDIA_COLS"]; COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]
    OWN_NONMEDIA_COLS = g["OWN_NONMEDIA_COLS"]; COMP_NONMEDIA_COLS = g["COMP_NONMEDIA_COLS"]
    PRICE_COLS = g["PRICE_COLS"]
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]

    x0 = np.zeros(dim)
    x0[0] = df[TARGET_COL].mean() * 0.8
    for i, col in enumerate(MEDIA_COLS):
        x0[i+1] = g["INITIAL_MEDIA_BETAS"].get(col, 0.0)
    for j, col in enumerate(COMP_MEDIA_COLS):
        x0[1+N_MEDIA+j] = g["INITIAL_COMP_BETAS"].get(col, -0.0001)
    for k, col in enumerate(OWN_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+k] = g["INITIAL_OWN_NONMEDIA_BETAS"].get(col, 0.0)
    for k, col in enumerate(COMP_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k] = g["INITIAL_COMP_NONMEDIA_BETAS"].get(col, -0.01)
    for p, col in enumerate(PRICE_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p] = g["INITIAL_PRICE_BETA"].get(col, -0.1)

    target_var = float(np.var(df[TARGET_COL].values))
    base_var   = max(target_var, 1e-6)
    n_beta = N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + g["N_PRICE"]
    rho0   = float(np.clip(g.get("BETA_PRIOR_CORR", 0.0), 0.0, 0.95))

    # P0 mirrors Q's block-diagonal structure: an intercept block (1x1,
    # higher initial uncertainty) and a betas block (n_beta x n_beta,
    # optionally sharing an equicorrelated ρ0 component across betas).
    P0 = np.zeros((dim, dim))
    P0[0, 0] = base_var * 4.0
    if n_beta > 0:
        P0[1:1+n_beta, 1:1+n_beta] = base_var * (
            (1 - rho0) * np.eye(n_beta) + rho0 * np.ones((n_beta, n_beta))
        )
    for i in range(1 + n_beta, dim):   # dummy/seasonal states: unblocked
        P0[i, i] = base_var
    return x0, P0


def _predict_step(t, x_prev, df, pc):
    """
    One equation's nonlinear state-prediction step (the x_p = f(x_{t-1}, x_t)
    part of the EKF), identical to the per-timestep logic previously inlined
    in run_kalman_filter's forward loop. Used by both the single-equation
    filter and (twice per t, once per dependent variable) by the joint
    bivariate filter.

    Returns (x_p, cross_contrib_row) where cross_contrib_row has length
    N_CROSS (synergy contribution booked against the target channel at t).
    """
    params = pc["params"]
    MEDIA_COLS = pc["MEDIA_COLS"]; OWN_NONMEDIA_COLS = pc["OWN_NONMEDIA_COLS"]
    COMP_NONMEDIA_COLS = pc["COMP_NONMEDIA_COLS"]; PRICE_COLS = pc["PRICE_COLS"]
    CROSS_MEDIA_PAIRS = pc["CROSS_MEDIA_PAIRS"]
    N_MEDIA = pc["N_MEDIA"]; N_COMP = pc["N_COMP"]
    ADSTOCK_TYPE = pc["ADSTOCK_TYPE"]
    positive_cols = pc["positive_cols"]; negative_cols = pc["negative_cols"]

    x_p = pc["Tmat"] @ x_prev
    cross_contrib_row = np.zeros(pc["N_CROSS"])

    # ── Own-media state equations ────────────────────────────
    for i in range(N_MEDIA):
        if ADSTOCK_TYPE == "weibull":
            x_p[i+1] = pc["weibull_lagsum_own"][t, i] + params["delta"][i] * pc["transformed_own"][t, i]
        else:
            shock = params["delta"][i] * pc["transformed_own"][t, i]
            x_p[i+1] += shock
        if MEDIA_COLS[i] in positive_cols:
            x_p[i+1] = max(x_p[i+1], 1e-8)
        elif MEDIA_COLS[i] in negative_cols:
            x_p[i+1] = min(x_p[i+1], -1e-8)

    # ── Competitor media ─────────────────────────────────────
    for j in range(N_COMP):
        x_p[1+N_MEDIA+j] += params["delta_comp"][j] * pc["transformed_comp"][t, j]
        x_p[1+N_MEDIA+j]  = min(x_p[1+N_MEDIA+j], -1e-8)

    # ── Cross-media synergy ───────────────────────────────────
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        tgt_idx = MEDIA_COLS.index(tgt)
        contrib = params["cross_delta"][k] * pc["transformed_cross"][t, k]
        x_p[tgt_idx+1] += contrib
        cross_contrib_row[k] = contrib

    # ── Own non-media ─────────────────────────────────────────
    for k, col in enumerate(OWN_NONMEDIA_COLS):
        si = 1+N_MEDIA+N_COMP+k
        x_p[si] += params["delta_own_nonmedia"][k] * df[col].iloc[t]
        if col in positive_cols:
            x_p[si] = max(x_p[si], 1e-8)

    # ── Competitor non-media ──────────────────────────────────
    for k, col in enumerate(COMP_NONMEDIA_COLS):
        si = 1+N_MEDIA+N_COMP+pc["N_OWN_NONMEDIA"]+k
        x_p[si] += params["delta_comp_nonmedia"][k] * df[col].iloc[t]
        x_p[si]  = min(x_p[si], -1e-8)

    # ── Price ─────────────────────────────────────────────────
    for p, col in enumerate(PRICE_COLS):
        si = 1+N_MEDIA+N_COMP+pc["N_OWN_NONMEDIA"]+pc["N_COMP_NONMEDIA"]+p
        x_p[si] += params["delta_price"][p] * df[col].iloc[t]
        x_p[si]  = min(x_p[si], -1e-8)

    # ── Intercept boost ───────────────────────────────────────
    x_p[0] += pc["intercept_boost"][t]
    if pc["USE_ORGANIC_DRIFT"]:
        x_p[0] += params["mu"]

    return x_p, cross_contrib_row


def _apply_beta_floors(x_vec, pc):
    for idx in pc["positive_state_idx"]:
        if x_vec[idx] < 0:
            x_vec[idx] = 1e-8
    for idx in pc["negative_state_idx"]:
        if x_vec[idx] > 0:
            x_vec[idx] = -1e-8
    return x_vec


# ── Main (single-equation) Kalman filter ─────────────────────────────────────

def run_kalman_filter(df, params, g):
    """Single-dependent-variable EKF (used when no second dependent
    variable is configured, and internally re-uses the same building
    blocks as the joint bivariate filter below)."""
    pc = _prepare_equation(df, g, params)
    dim = pc["dim"]; T_len = pc["T_len"]
    L_mat = pc["L_mat"]; Tmat = pc["Tmat"]; Q = pc["Q"]
    R = params["sigma_y"] ** 2

    x_filt = np.zeros((T_len, dim)); P_filt = np.zeros((T_len, dim, dim))
    x_pred = np.zeros((T_len, dim)); P_pred = np.zeros((T_len, dim, dim))
    cross_beta_contrib = np.zeros((T_len, pc["N_CROSS"]))
    yhat = np.zeros(T_len); residuals = np.zeros(T_len); loglik = 0.0

    x0, P0 = _initial_state(df, g, params, pc)
    x_filt[0] = x0; P_filt[0] = P0
    target_vals = pc["target_vals"]

    for t in range(T_len):
        if t > 0:
            x_p, cross_row = _predict_step(t, x_filt[t-1], df, pc)
            cross_beta_contrib[t] = cross_row
            x_pred[t] = x_p
            P_pred[t] = Tmat @ P_filt[t-1] @ Tmat.T + Q
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        L_t = L_mat[t]
        yhat[t]      = L_t @ x_pred[t]
        residuals[t] = target_vals[t] - yhat[t]
        LP  = L_t @ P_pred[t]
        S_t = max(LP @ L_t + R, 1e-8)
        K   = (P_pred[t] @ L_t) / S_t
        x_filt[t] = x_pred[t] + K * residuals[t]
        x_filt[t] = _apply_beta_floors(x_filt[t], pc)
        P_filt[t] = (np.eye(dim) - np.outer(K, L_t)) @ P_pred[t]
        loglik += -0.5 * (np.log(2 * np.pi * S_t) + residuals[t]**2 / S_t)

    return yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik


# ── Joint (bivariate) Kalman filter ──────────────────────────────────────────

def run_bivariate_kalman_filter(df, params1, g1, params2, g2, rho, phi1=0.0, phi2=0.0):
    """
    Joint (bivariate) Kalman-filter fit of:

        [ sales_t         ]   [ Intercept_sales_t         ]   [ error_sales_t         ]
        [ consideration_t ] = [ Intercept_consideration_t ] + [ error_consideration_t ]

                                [ beta_sales_1_t         ... beta_sales_M_t         ]
                              + [ beta_consid_1_t        ... beta_consid_M_t       ] · x_t

    Both dependent variables are fitted SIMULTANEOUSLY in one state-space
    system: a single (dim_1+dim_2)-length state vector is propagated and
    updated at every t, with a full 2×2 observation-noise covariance matrix
    R (parameterised by σ_1, σ_2 and a jointly-estimated correlation ρ)
    coupling the two equations' contemporaneous errors, AND each equation's
    intercept feeds into the other's next-period intercept via phi_1 / phi_2
    (see the module docstring's "Cross-intercept coupling" section):

        Intercept_1,t = G0_1 · Intercept_1,t-1 + phi_1 · Intercept_2,t-1 + effectors_1,t
        Intercept_2,t = G0_2 · Intercept_2,t-1 + phi_2 · Intercept_1,t-1 + effectors_2,t

    Parameters
    ----------
    phi1 : coefficient of Intercept_2,t-1 in equation 1's intercept update.
    phi2 : coefficient of Intercept_1,t-1 in equation 2's intercept update.

    Returns
    -------
    yhat, residuals : (T, 2) arrays — column 0 = dep 1, column 1 = dep 2
    x_filt, P_filt, x_pred, P_pred : joint filtered/predicted state & covariance
    Tmat_joint : joint transition matrix (block-diagonal plus the two
                 phi_1/phi_2 cross-intercept off-diagonal entries)
    cross1, cross2 : (T, N_CROSS) per-equation synergy contribution arrays
    loglik  : joint bivariate log-likelihood (single scalar, NOT split per equation)
    dim1, dim2 : state dimensions of equation 1 / equation 2 (for splitting x_smooth later)
    """
    pc1 = _prepare_equation(df, g1, params1)
    pc2 = _prepare_equation(df, g2, params2)
    dim1, dim2 = pc1["dim"], pc2["dim"]
    dim = dim1 + dim2
    T_len = pc1["T_len"]

    Tmat_joint = np.zeros((dim, dim))
    Tmat_joint[:dim1, :dim1] = pc1["Tmat"]
    Tmat_joint[dim1:, dim1:] = pc2["Tmat"]
    # Cross-intercept coupling: intercept is always state index 0 within
    # each equation's own block, i.e. joint indices 0 (eq 1) and dim1 (eq 2).
    Tmat_joint[0, dim1] = phi1
    Tmat_joint[dim1, 0] = phi2

    Q_joint = np.zeros((dim, dim))
    Q_joint[:dim1, :dim1] = pc1["Q"]
    Q_joint[dim1:, dim1:] = pc2["Q"]

    x0_1, P0_1 = _initial_state(df, g1, params1, pc1)
    x0_2, P0_2 = _initial_state(df, g2, params2, pc2)
    x0 = np.concatenate([x0_1, x0_2])
    P0 = np.zeros((dim, dim))
    P0[:dim1, :dim1] = P0_1
    P0[dim1:, dim1:] = P0_2

    x_filt = np.zeros((T_len, dim)); P_filt = np.zeros((T_len, dim, dim))
    x_pred = np.zeros((T_len, dim)); P_pred = np.zeros((T_len, dim, dim))
    cross1 = np.zeros((T_len, pc1["N_CROSS"])); cross2 = np.zeros((T_len, pc2["N_CROSS"]))
    yhat = np.zeros((T_len, 2)); residuals = np.zeros((T_len, 2)); loglik = 0.0

    x_filt[0] = x0; P_filt[0] = P0
    targets = np.column_stack([pc1["target_vals"], pc2["target_vals"]])

    sigma1 = float(params1["sigma_y"]); sigma2 = float(params2["sigma_y"])
    rho = float(np.clip(rho, -0.995, 0.995))
    R = np.array([
        [sigma1 ** 2,            rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2,  sigma2 ** 2],
    ])

    for t in range(T_len):
        if t > 0:
            x_p1, cross_row1 = _predict_step(t, x_filt[t-1, :dim1], df, pc1)
            x_p2, cross_row2 = _predict_step(t, x_filt[t-1, dim1:], df, pc2)
            # Cross-intercept coupling: each equation's intercept also picks
            # up phi_i · (other equation's PREVIOUS intercept).
            x_p1[0] += phi1 * x_filt[t-1, dim1]
            x_p2[0] += phi2 * x_filt[t-1, 0]
            cross1[t] = cross_row1; cross2[t] = cross_row2
            x_pred[t] = np.concatenate([x_p1, x_p2])
            P_pred[t] = Tmat_joint @ P_filt[t-1] @ Tmat_joint.T + Q_joint
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        # ── Joint observation matrix (2 × dim) ────────────────────────
        L_t = np.zeros((2, dim))
        L_t[0, :dim1] = pc1["L_mat"][t]
        L_t[1, dim1:] = pc2["L_mat"][t]

        yhat[t] = L_t @ x_pred[t]
        residuals[t] = targets[t] - yhat[t]

        PLt = P_pred[t] @ L_t.T                    # (dim, 2)
        S_t = L_t @ PLt + R                         # (2, 2) innovation covariance
        S_t = S_t + np.eye(2) * 1e-10               # numerical safety
        S_inv = np.linalg.inv(S_t)
        K = PLt @ S_inv                             # (dim, 2) joint Kalman gain

        x_filt[t] = x_pred[t] + K @ residuals[t]
        x_filt[t, :dim1] = _apply_beta_floors(x_filt[t, :dim1].copy(), pc1)
        x_filt[t, dim1:] = _apply_beta_floors(x_filt[t, dim1:].copy(), pc2)

        P_filt[t] = (np.eye(dim) - K @ L_t) @ P_pred[t]

        sign, logdet = np.linalg.slogdet(S_t)
        if sign <= 0:
            logdet = np.log(1e-12)
        quad = residuals[t] @ S_inv @ residuals[t]
        loglik += -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)

    return (yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat_joint,
            cross1, cross2, loglik, dim1, dim2)


def rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat):
    """Dimension-agnostic RTS smoother — used as-is for both the
    single-equation state and the joint (dim_1+dim_2) bivariate state."""
    T_len, dim = x_filt.shape
    x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T_len - 2, -1, -1):
        C = P_filt[t] @ Tmat.T @ np.linalg.inv(P_pred[t+1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t+1] - P_pred[t+1]) @ C.T
    return x_smooth, P_smooth
