"""
Skill-gap analysis for RoleSense.

Given a user's skill set (list or comma-separated string) and a job row
(pandas Series with a 'required_skills' column, comma-separated), this
module works out which of the job's required skills the user already has
and which they're missing, and renders that as a compact tag panel with a
coverage bar.

Self-contained by design — it does NOT import from app.py, because app.py
runs top-level Streamlit calls (st.set_page_config, model loading, etc.) as
soon as it's imported. Keeping this module import-safe means app.py can
`import` it near the top without side effects.
"""
import html
import pandas as pd
import streamlit as st


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def parse_skill_list(raw) -> set:
    """Normalize a comma-separated string (or list/set/tuple) of skills into
    a lower-cased set for comparison. Safe against None/NaN/non-string input."""
    if raw is None:
        return set()
    if isinstance(raw, float) and pd.isna(raw):
        return set()
    items = raw if isinstance(raw, (list, set, tuple)) else str(raw).split(",")
    out = set()
    for item in items:
        piece = str(item).strip().lower()
        if piece and piece != "not specified":
            out.add(piece)
    return out


def compute_skill_gap(user_skills, required_skills_raw) -> dict:
    """
    user_skills: set/list/comma-string of skills the candidate has
    required_skills_raw: the job's 'required_skills' cell (comma string or NaN)

    Returns:
      matched   - sorted list of required skills the user already has (title-cased)
      missing   - sorted list of required skills the user doesn't have
      total     - number of distinct required skills listed for the job
      match_pct - 0-100 share of required skills covered, or None if the job
                  lists no required skills at all (so callers can tell
                  "nothing to compare" apart from "0% overlap")
    """
    have = parse_skill_list(user_skills)
    need = parse_skill_list(required_skills_raw)

    if not need:
        return {"matched": [], "missing": [], "total": 0, "match_pct": None}

    matched = sorted(s.title() for s in need if s in have)
    missing = sorted(s.title() for s in need if s not in have)
    match_pct = round(100 * len(matched) / len(need), 1)

    return {"matched": matched, "missing": missing, "total": len(need), "match_pct": match_pct}


def _tag_row(items, kind: str) -> str:
    """kind is 'matched' or 'missing' — controls the tag color. Uses the
    same CSS custom properties app.py already defines on :root, so no extra
    CSS needs to be injected."""
    if not items:
        empty_msg = "Nothing missing — full coverage" if kind == "missing" else "No overlapping skills found"
        return f'<span style="color:var(--muted);font-size:0.8rem;">{_esc(empty_msg)}</span>'

    if kind == "matched":
        bg, border, fg = "var(--success-soft)", "var(--success)", "var(--success)"
    else:
        bg, border, fg = "var(--warning-soft)", "var(--warning)", "var(--warning)"

    return "".join(
        f'<span style="display:inline-block;padding:5px 11px;border-radius:999px;'
        f'font-size:0.76rem;margin:2px 6px 2px 0;font-weight:600;'
        f'background:{bg};color:{fg};border:1px solid {border};">{_esc(item)}</span>'
        for item in items
    )


def render_skill_gap(user_skills, row: pd.Series) -> dict:
    """
    Renders a coverage bar plus 'you have' / 'missing' tag columns for a
    single job row. Returns the gap dict so the caller can reuse it (e.g.
    to sort results by coverage, or include it in a CSV export).

    `user_skills` can be a raw comma-separated string (e.g. straight from
    the "Find matches" text input) or a pre-parsed set/list.
    """
    gap = compute_skill_gap(user_skills, row.get("required_skills"))

    if gap["total"] == 0:
        st.caption("This role doesn't list specific required skills to compare against.")
        return gap

    pct = gap["match_pct"]
    bar_color = "var(--success)" if pct >= 60 else "var(--warning)"
    st.markdown(
        f"""
        <div style="margin:6px 0 14px;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;
                      color:var(--muted);margin-bottom:4px;">
            <span>Skill coverage</span>
            <span style="font-weight:700;color:var(--ink);">{pct}%</span>
          </div>
          <div style="width:100%;height:6px;background:var(--surface-alt);border-radius:3px;overflow:hidden;">
            <div style="height:100%;border-radius:3px;width:{pct}%;background:{bar_color};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:var(--muted);margin-bottom:6px;">'
            f'YOU HAVE ({len(gap["matched"])})</div>{_tag_row(gap["matched"], "matched")}',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:var(--muted);margin-bottom:6px;">'
            f'MISSING ({len(gap["missing"])})</div>{_tag_row(gap["missing"], "missing")}',
            unsafe_allow_html=True,
        )

    return gap