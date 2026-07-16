"""
Nevergrad multi-objective optimizer support.

Loss = -w_loglik * loglik
       + w_mape * max(0, MAPE - mape_target)^2
       + w_r2   * max(0, r2_target - R2)^2
       + w_pos  * positivity_violations
"""

import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from modules.params import unpack_theta
from modules.kalman import run_kalman_filter, run_bivariate_kalman_filter, build_static_cache
from modules.metrics import safe_mape


def _composite_loss(theta, df_train, g, ng_weights, static_cache=None):
    w_loglik = ng_weights.get("w_loglik", 1.0); w_mape = ng_weights.get("w_mape", 10.0)
    w_r2     = ng_weights.get("w_r2",     5.0); w_pos  = ng_weights.get("w_pos",  100.0)
    mape_tgt = ng_weights.get("mape_target", 0.10); r2_tgt = ng_weights.get("r2_target", 0.80)
    try:
        p = unpack_theta(theta, g)
        yhat, resid, _, _, _, _, _, _, loglik = run_kalman_filter(
            df_train, p, g, static_cache=static_cache)
        tv   = df_train[g["TARGET_COL"]].values
        mape = safe_mape(resid, tv)
        r2   = 1.0 - np.sum(resid**2) / (np.sum((tv - tv.mean())**2) + 1e-12)
        loss = -w_loglik * loglik
        loss += w_mape * max(0, mape - mape_tgt) ** 2
        loss += w_r2  * max(0, r2_tgt - r2) ** 2
        for i, col in enumerate(g["MEDIA_COLS"]):
            if col in g.get("POSITIVE_BETA_COLS", []):
                loss += w_pos * max(0, -p["delta"][i]) ** 2
        for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
            if col in g.get("POSITIVE_BETA_COLS", []):
                loss += w_pos * max(0, -p["delta_own_nonmedia"][k]) ** 2
        for j in range(g["N_COMP"]):
            loss += w_pos * max(0, p["delta_comp"][j]) ** 2
        # Negative beta constraints
        for i, col in enumerate(g["MEDIA_COLS"]):
            if col in g.get("NEGATIVE_BETA_COLS", []):
                loss += w_pos * max(0, p["delta"][i]) ** 2
        for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
            if col in g.get("NEGATIVE_BETA_COLS", []):
                loss += w_pos * max(0, p["delta_own_nonmedia"][k]) ** 2
        return float(loss)
    except Exception:
        return 1e12


def _ask_eval_tell_loop(optimizer, budget, num_workers, strategy_name, loss_fn, progress_label):
    """
    Shared ask/evaluate/tell driver for both the single-equation and joint
    Nevergrad optimizers.

    Previously this always asked for and evaluated ONE candidate at a time,
    regardless of the `num_workers` setting in the UI — so raising "workers"
    had zero effect on wall-clock time. This now genuinely batches
    `num_workers` candidates per round and evaluates them concurrently via a
    thread pool (nevergrad's ask-many/tell-many pattern is explicitly
    designed for this). With num_workers=1 (the default) this is exactly
    the old serial behaviour — nothing changes unless you raise it.

    Note: the Kalman filter's per-timestep loop is Python-level, not a
    single vectorized numpy call, so GIL contention limits how much thread
    parallelism can help — the numpy matrix multiplies inside each step do
    release the GIL, so there's a real, if likely partial (not N×), speedup
    from raising num_workers. It is, at minimum, no longer a no-op.
    """
    best_loss = np.inf; best_theta = None
    progress = st.progress(0, text=f"{progress_label} — 0/{budget} evals")
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    evaluated = 0
    report_every = max(1, budget // 50)
    try:
        while evaluated < budget:
            batch_n = min(num_workers, budget - evaluated)
            cands = [optimizer.ask() for _ in range(batch_n)]
            if executor is not None:
                losses = list(executor.map(lambda c: loss_fn(c.value), cands))
            else:
                losses = [loss_fn(c.value) for c in cands]
            for cand, loss in zip(cands, losses):
                optimizer.tell(cand, loss)
                if loss < best_loss:
                    best_loss = loss; best_theta = cand.value.copy()
            evaluated += batch_n
            if (evaluated // report_every) != ((evaluated - batch_n) // report_every):
                progress.progress(min(100, int(evaluated / budget * 100)),
                                  text=f"{progress_label} — {evaluated}/{budget} | best: {best_loss:.4f}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
    progress.progress(100, text=f"✅ {progress_label} done — best loss: {best_loss:.4f}")
    return best_theta, best_loss


def run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg, static_cache=None):
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NgIohh"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1))); ng_weights = ng_cfg.get("ng_weights", {})
    if static_cache is None:
        static_cache = build_static_cache(df_train, g)
    lows  = np.array([b[0] if b[0] is not None else -1e6 for b in bounds])
    highs = np.array([b[1] if b[1] is not None else  1e6 for b in bounds])
    param = ng.p.Array(init=theta0).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NgIohh
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta: _composite_loss(theta, df_train, g, ng_weights, static_cache)
    best_theta, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}]", loss_fn)
    if best_theta is None:
        best_theta = theta0.copy()
    return best_theta, best_loss


# ── Joint (bivariate) composite loss & optimizer ─────────────────────────────

def _composite_loss_joint(theta_joint, df_train, g1, g2, n1, n2, ng_weights,
                           static_cache1=None, static_cache2=None):
    """
    Same composite-loss idea as `_composite_loss`, but evaluated on the
    JOINT bivariate Kalman filter so both dependent variables (the error
    correlation rho, and the cross-intercept coupling phi_1/phi_2) are
    optimised together in a single Nevergrad run, rather than as separate
    sequential optimizer calls.

    theta_joint = [theta_1 (len n1) | theta_2 (len n2) | rho | phi_1 | phi_2]
    """
    w_loglik = ng_weights.get("w_loglik", 1.0); w_mape = ng_weights.get("w_mape", 10.0)
    w_r2     = ng_weights.get("w_r2",     5.0); w_pos  = ng_weights.get("w_pos",  100.0)
    mape_tgt = ng_weights.get("mape_target", 0.10); r2_tgt = ng_weights.get("r2_target", 0.80)
    try:
        theta1 = theta_joint[:n1]
        theta2 = theta_joint[n1:n1+n2]
        rho    = theta_joint[n1+n2]
        phi1   = theta_joint[n1+n2+1]
        phi2   = theta_joint[n1+n2+2]
        p1 = unpack_theta(theta1, g1)
        p2 = unpack_theta(theta2, g2)
        (yhat, resid, _, _, _, _, _, _, _, loglik, dim1, dim2) = \
            run_bivariate_kalman_filter(df_train, p1, g1, p2, g2, rho, phi1, phi2,
                                         static_cache1=static_cache1, static_cache2=static_cache2)

        loss = -w_loglik * loglik
        for j, (g, p, col_idx) in enumerate([(g1, p1, 0), (g2, p2, 1)]):
            tv = df_train[g["TARGET_COL"]].values
            r  = resid[:, col_idx]
            mape = safe_mape(r, tv)
            r2   = 1.0 - np.sum(r**2) / (np.sum((tv - tv.mean())**2) + 1e-12)
            loss += w_mape * max(0, mape - mape_tgt) ** 2
            loss += w_r2  * max(0, r2_tgt - r2) ** 2
            for i, col in enumerate(g["MEDIA_COLS"]):
                if col in g.get("POSITIVE_BETA_COLS", []):
                    loss += w_pos * max(0, -p["delta"][i]) ** 2
                if col in g.get("NEGATIVE_BETA_COLS", []):
                    loss += w_pos * max(0, p["delta"][i]) ** 2
            for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
                if col in g.get("POSITIVE_BETA_COLS", []):
                    loss += w_pos * max(0, -p["delta_own_nonmedia"][k]) ** 2
                if col in g.get("NEGATIVE_BETA_COLS", []):
                    loss += w_pos * max(0, p["delta_own_nonmedia"][k]) ** 2
            for jj in range(g["N_COMP"]):
                loss += w_pos * max(0, p["delta_comp"][jj]) ** 2
        return float(loss)
    except Exception:
        return 1e12


def run_nevergrad_optimizer_joint(df_train, g1, g2, theta0_joint, bounds_joint, n1, n2, ng_cfg,
                                   static_cache1=None, static_cache2=None):
    """Joint-mode counterpart of run_nevergrad_optimizer: optimises
    theta_1, theta_2, rho, and the cross-intercept coupling phi_1/phi_2
    together against the bivariate loglik."""
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NgIohh"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1))); ng_weights = ng_cfg.get("ng_weights", {})
    if static_cache1 is None:
        static_cache1 = build_static_cache(df_train, g1)
    if static_cache2 is None:
        static_cache2 = build_static_cache(df_train, g2)
    lows  = np.array([b[0] if b[0] is not None else -1e6 for b in bounds_joint])
    highs = np.array([b[1] if b[1] is not None else  1e6 for b in bounds_joint])
    param = ng.p.Array(init=theta0_joint).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NgIohh
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta: _composite_loss_joint(
        theta, df_train, g1, g2, n1, n2, ng_weights, static_cache1, static_cache2)
    best_theta, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}] (joint bivariate)", loss_fn)
    if best_theta is None:
        best_theta = theta0_joint.copy()
    return best_theta, best_loss
