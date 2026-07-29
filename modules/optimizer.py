"""
Nevergrad optimizer support.

Loss = -loglik   (identical objective to the L-BFGS-B / SLSQP path)
"""

import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from modules.params import unpack_theta
from modules.kalman import run_kalman_filter, run_bivariate_kalman_filter, build_static_cache


def _composite_loss(theta, df_train, g, static_cache=None):
    try:
        p = unpack_theta(theta, g)
        _, _, _, _, _, _, _, _, loglik = run_kalman_filter(
            df_train, p, g, static_cache=static_cache)
        return -loglik
    except Exception:
        return 1e12


def _ask_eval_tell_loop(optimizer, budget, num_workers, progress_label, loss_fn):
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
    strategy_name = ng_cfg.get("strategy", "NGOpt"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1)))
    if static_cache is None:
        static_cache = build_static_cache(df_train, g)
    lows  = np.array([b[0] if b[0] is not None else -1e6 for b in bounds])
    highs = np.array([b[1] if b[1] is not None else  1e6 for b in bounds])
    param = ng.p.Array(init=theta0).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta: _composite_loss(theta, df_train, g, static_cache)
    best_theta, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}]", loss_fn)
    if best_theta is None:
        best_theta = theta0.copy()
    return best_theta, best_loss


# ── Joint (bivariate) composite loss & optimizer ─────────────────────────────

def _composite_loss_joint(theta_joint, df_train, g1, g2, n1, n2,
                           static_cache1=None, static_cache2=None):
    """
    Same composite-loss idea as `_composite_loss`, but evaluated on the
    JOINT bivariate Kalman filter so both dependent variables (the error
    correlation rho, and the cross-intercept coupling phi_1/phi_2) are
    optimised together in a single Nevergrad run, rather than as separate
    sequential optimizer calls.

    theta_joint = [theta_1 (len n1) | theta_2 (len n2) | rho | phi_1 | phi_2]

    The trailing phi_1/phi_2 pair is OMITTED entirely (theta_joint ends
    right after rho) when the model is configured with "simple" (no
    carryover) intercept dynamics — cross-intercept coupling is itself a
    carryover mechanism, so it doesn't apply there. Detected here from
    theta_joint's actual length rather than a separate flag, so this stays
    correct regardless of which caller (scipy or Nevergrad) built it.
    """
    try:
        theta1 = theta_joint[:n1]
        theta2 = theta_joint[n1:n1+n2]
        rho    = theta_joint[n1+n2]
        if len(theta_joint) - (n1 + n2) >= 3:
            phi1 = theta_joint[n1+n2+1]
            phi2 = theta_joint[n1+n2+2]
        else:
            phi1 = phi2 = 0.0
        p1 = unpack_theta(theta1, g1)
        p2 = unpack_theta(theta2, g2)
        (_, _, _, _, _, _, _, _, _, loglik, _, _) = \
            run_bivariate_kalman_filter(df_train, p1, g1, p2, g2, rho, phi1, phi2,
                                         static_cache1=static_cache1, static_cache2=static_cache2)
        return -loglik
    except Exception:
        return 1e12


def run_nevergrad_optimizer_joint(df_train, g1, g2, theta0_joint, bounds_joint, n1, n2, ng_cfg,
                                   static_cache1=None, static_cache2=None):
    """Joint-mode counterpart of run_nevergrad_optimizer: optimises
    theta_1, theta_2, rho, and the cross-intercept coupling phi_1/phi_2
    together against the bivariate loglik."""
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NGOpt"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1)))
    if static_cache1 is None:
        static_cache1 = build_static_cache(df_train, g1)
    if static_cache2 is None:
        static_cache2 = build_static_cache(df_train, g2)
    lows  = np.array([b[0] if b[0] is not None else -1e6 for b in bounds_joint])
    highs = np.array([b[1] if b[1] is not None else  1e6 for b in bounds_joint])
    param = ng.p.Array(init=theta0_joint).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta: _composite_loss_joint(
        theta, df_train, g1, g2, n1, n2, static_cache1, static_cache2)
    best_theta, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}] (joint bivariate)", loss_fn)
    if best_theta is None:
        best_theta = theta0_joint.copy()
    return best_theta, best_loss
