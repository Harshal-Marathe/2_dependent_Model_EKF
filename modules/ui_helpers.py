"""
Small UI helpers shared across tabs:
  - styled info/box renderers
  - "need_*" guard functions that stop the script with a warning
  - safe_multiselect(), the core fix that prevents Streamlit from raising
    StreamlitAPIException when a stored multiselect value goes stale
    (e.g. after prophet columns are merged into the dataset, or after
    an upstream widget like `media` changes the available `options`).
"""

import streamlit as st


# ────────────────────────────────────────────────────────────────────
# Styled boxes
# ────────────────────────────────────────────────────────────────────
def section(badge, title):
    st.markdown(
        f'<div class="section-header"><span class="section-badge">{badge}</span>'
        f'<h2>{title}</h2></div>', unsafe_allow_html=True)


def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def positive_info(text):
    st.markdown(f'<div class="positive-beta-box">{text}</div>', unsafe_allow_html=True)


def ng_info(text):
    st.markdown(f'<div class="ng-box">{text}</div>', unsafe_allow_html=True)


def per_channel_info(text):
    st.markdown(f'<div class="per-channel-box">{text}</div>', unsafe_allow_html=True)


def prophet_info(text):
    st.markdown(f'<div class="prophet-box">{text}</div>', unsafe_allow_html=True)


def weibull_placeholder():
    st.markdown(
        '<div class="weibull-zone">▼ WEIBULL ADSTOCK ZONE — '
        'replace <code>adstock_weibull()</code> body with your implementation.</div>',
        unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# Guard functions
# ────────────────────────────────────────────────────────────────────
def need_data():
    st.warning("⬆️ Upload data in **Step 1** first."); st.stop()


def need_config():
    st.warning("⚙️ Configure the model in **Step 5** first."); st.stop()


def need_model():
    st.warning("🚀 Run the RBE model in **Step 6** first."); st.stop()


# ════════════════════════════════════════════════════════════════════
# THE CORE FIX: safe_multiselect
# ════════════════════════════════════════════════════════════════════
# Streamlit raises StreamlitAPIException if st.session_state[key] holds
# any value not currently present in `options`. That happens routinely
# here because `options` for corr_vars / cfg_nonmedia / cfg_media / etc.
# are *derived* (filtered from the dataframe's columns, or filtered by
# what other widgets currently hold) and change shape across reruns —
# e.g. right after a prophet merge, or after the user edits an upstream
# widget like `media`.
#
# This wrapper sanitizes the stored selection to the intersection with
# `options` BEFORE the widget is instantiated, every single render, so
# the exception can never fire and a previously-valid value is never
# silently lost just because Streamlit choked on one stale entry among
# many valid ones.
def safe_multiselect(label, options, key, default=None, require=None, **kwargs):
    """
    Drop-in replacement for st.multiselect that never lets Streamlit raise
    on a stale stored value, and that can FORCE certain values (e.g. newly
    merged prophet columns) to stay selected across every rerun.

    Two distinct concepts, often confused in earlier attempts at this fix:

    - `default`: ONLY used the very first time this key is ever created
      (i.e. before the user has interacted with the widget at all). This
      matches normal Streamlit semantics — once the key exists, Streamlit
      owns the value and `default` is irrelevant on every later run.

    - `require`: a set of values that must ALWAYS be present in the
      selection, on every single render, regardless of whether the key
      already exists or what the user has since clicked. This is what
      prophet columns need: they were added to the dataset out-of-band
      (via a button in another tab), not via user interaction with THIS
      widget, so they must be re-injected every run rather than relying
      on a one-time default that Streamlit will ignore forever after
      the key is first created (this was the bug in the prior version —
      `key in st.session_state` becomes True almost immediately, e.g. as
      soon as the widget renders with an empty selection, which silently
      kills `default` on every subsequent run).

    Sanitization (dropping values no longer in `options`) always happens
    first, so Streamlit can never throw on a stale value.
    """
    options = list(options)
    if key not in st.session_state:
        st.session_state[key] = [v for v in (default or []) if v in options]

    current = st.session_state[key]
    cleaned = [v for v in current if v in options]
    if require:
        for v in require:
            if v in options and v not in cleaned:
                cleaned.append(v)
    if cleaned != current:
        st.session_state[key] = cleaned

    return st.multiselect(label, options=options, key=key, **kwargs)
