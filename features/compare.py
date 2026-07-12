"""
Side-by-side job comparison for Rolyx.

Users can add up to MAX_COMPARE jobs from any results list (resume matches,
skills search, lookup) into a comparison tray held in st.session_state, then
view them side by side in a dedicated section.

Self-contained by design — does NOT import from app.py (see skill_gap.py
docstring for why). `stable_job_key` is duplicated here on purpose so the
identifiers this module generates always agree with the ones app.py uses
for bookmarks, even though the two modules never import each other.
"""
import hashlib
import html
import pandas as pd
import streamlit as st

COMPARE_KEY = "compare_jobs"  # st.session_state[COMPARE_KEY] -> {job_key: row_dict}
MAX_COMPARE = 4


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _safe(value, fmt=None, default="N/A"):
    """Renders a value for display, handling None/NaN uniformly so we never
    print 'nan' or 'None' in the comparison table."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if fmt is not None:
        try:
            return fmt(value)
        except (TypeError, ValueError):
            return default
    return str(value)


def _fmt_salary(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return f"${float(val):,.0f}"
    except (TypeError, ValueError):
        return None


def stable_job_key(row) -> str:
    """Mirrors app.py's stable_job_key exactly, so a job added to compare
    from any section resolves to the same key as its bookmark, if any."""
    for id_col in ("job_id", "id", "job_posting_id"):
        if id_col in row and pd.notna(row.get(id_col)):
            return str(row.get(id_col))
    fingerprint = "|".join(str(row.get(c, "")) for c in (
        "title", "company_name", "description", "min_salary_yr", "max_salary_yr"
    ))
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


def _ensure_state():
    if COMPARE_KEY not in st.session_state:
        st.session_state[COMPARE_KEY] = {}


def is_in_compare(row) -> bool:
    _ensure_state()
    return stable_job_key(row) in st.session_state[COMPARE_KEY]


def compare_count() -> int:
    _ensure_state()
    return len(st.session_state[COMPARE_KEY])


def render_compare_toggle(row, idx, section: str):
    """Companion button to app.py's 'Save role' button — call this inside
    render_job_card (e.g. in a third column) so users can queue a job for
    comparison right from its card. Handles its own st.button + st.rerun(),
    matching the bookmark button's existing pattern."""
    _ensure_state()
    tray = st.session_state[COMPARE_KEY]
    job_key = stable_job_key(row)
    in_tray = job_key in tray

    # Section is included in the widget key for the same reason app.py's
    # bookmark_key includes it: the same job can appear in multiple result
    # lists (resume matches vs. manual search) without colliding.
    btn_key = f"compare_{section}_{idx}_{job_key}_btn"

    if in_tray:
        if st.button("Remove from compare", key=btn_key, use_container_width=True):
            tray.pop(job_key, None)
            st.rerun()
    elif len(tray) >= MAX_COMPARE:
        st.button(
            f"Compare (max {MAX_COMPARE})", key=btn_key,
            use_container_width=True, disabled=True,
            help=f"Remove a job from the comparison tray first — up to {MAX_COMPARE} at a time.",
        )
    else:
        if st.button("Add to compare", key=btn_key, use_container_width=True):
            tray[job_key] = row.to_dict()
            st.rerun()


def render_compare_bar():
    """Small persistent status pill — call this once near the top of the
    page (e.g. right after the hero/stats) so users always know what's
    queued, even while scrolling through other sections."""
    _ensure_state()
    n = len(st.session_state[COMPARE_KEY])
    if n == 0:
        return
    st.markdown(
        f'<div style="background:var(--accent-soft);border:1px solid var(--accent);'
        f'border-radius:999px;padding:8px 18px;font-size:0.82rem;color:var(--ink);'
        f'display:inline-block;margin-bottom:18px;">'
        f'<strong>{n}</strong> job{"s" if n != 1 else ""} queued for comparison — '
        f'<a href="#compare-section" style="color:var(--accent);font-weight:600;">jump to comparison</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


_FIELDS = [
    ("Company", "company_name", None),
    ("Work type", "formatted_work_type", None),
    ("Experience", "formatted_experience_level", None),
    ("Remote", "remote_allowed", lambda v: "Yes" if v == 1 else "No"),
    ("Industry", "industry", None),
    ("Match score", "match_score", lambda v: f"{v}%"),
]


def render_compare_section():
    """Renders the full side-by-side comparison view. Call this once,
    wherever the comparison section should live (give it its own anchor,
    consistent with the other modules in app.py)."""
    _ensure_state()
    tray = st.session_state[COMPARE_KEY]

    if not tray:
        st.markdown(
            '<div class="empty-state">No jobs added yet — use "Add to compare" '
            'on any job card above to queue it here (up to '
            f'{MAX_COMPARE} at a time).</div>',
            unsafe_allow_html=True,
        )
        return

    rows = list(tray.items())  # [(job_key, row_dict), ...]
    cols = st.columns(len(rows))

    for col, (job_key, row_dict) in zip(cols, rows):
        with col:
            title = _safe(row_dict.get("title"))
            min_s = _fmt_salary(row_dict.get("min_salary_yr"))
            max_s = _fmt_salary(row_dict.get("max_salary_yr"))
            salary_str = f"{min_s} – {max_s}" if min_s and max_s else "Not disclosed"

            body_rows = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                f'font-size:0.8rem;border-bottom:1px solid var(--border);">'
                f'<span style="color:var(--muted);">{_esc(label)}</span>'
                f'<span style="color:var(--ink);font-weight:600;text-align:right;">'
                f'{_esc(_safe(row_dict.get(field), fmt))}</span></div>'
                for label, field, fmt in _FIELDS
            )

            st.markdown(
                f'<div style="background:var(--surface);border:1px solid var(--border);'
                f'border-radius:14px;padding:16px;">'
                f'<div style="font-weight:700;font-size:0.95rem;color:var(--ink);'
                f'margin-bottom:10px;min-height:2.6em;">{_esc(title)}</div>'
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                f'font-size:0.8rem;border-bottom:1px solid var(--border);">'
                f'<span style="color:var(--muted);">Salary</span>'
                f'<span style="color:var(--accent);font-weight:700;text-align:right;">{_esc(salary_str)}</span>'
                f'</div>{body_rows}</div>',
                unsafe_allow_html=True,
            )

            if st.button("Remove", key=f"compare_remove_{job_key}", use_container_width=True):
                tray.pop(job_key, None)
                st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("Clear all comparisons", key="compare_clear_all"):
        st.session_state[COMPARE_KEY] = {}
        st.rerun()