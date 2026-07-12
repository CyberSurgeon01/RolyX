# Rolyx — Integration Guide

New files, drop these next to `app.py`:

```
your_project/
├── app.py
├── features/
│   ├── __init__.py
│   ├── skill_gap.py
│   ├── compare.py
│   └── export.py
└── requirements.txt   (add: reportlab)
```

Install the one new dependency:
```
pip install reportlab
```
(CSV export needs nothing extra — pandas already does that.)

All three modules are self-contained: they don't import `app.py`, so
importing them at the top of `app.py` is safe even though `app.py` calls
`st.set_page_config()` and loads models at import time.

Every step below is a **find this block → add this block** edit to your
existing `app.py`. Nothing existing is deleted except where noted.

---

## 1. Add imports

Find:
```python
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
```
Add directly after it:
```python
from features import skill_gap, compare, export
```

---

## 2. Add a "Compare" link to the navbar

Find (inside `render_navbar`):
```python
        <a href="#resume-section">Resume AI</a>
        <a href="#find-section">Find matches</a>
        <a href="#predict-section">Salary AI</a>
        <a href="#lookup-section">Role lookup</a>
```
Replace with:
```python
        <a href="#resume-section">Resume AI</a>
        <a href="#find-section">Find matches</a>
        <a href="#predict-section">Salary AI</a>
        <a href="#lookup-section">Role lookup</a>
        <a href="#compare-section">Compare</a>
```

---

## 3. Show the compare tray pill under the hero

Find:
```python
render_navbar()
render_hero()
render_stats(compute_dataset_stats(jobs))
```
Replace with:
```python
render_navbar()
render_hero()
render_stats(compute_dataset_stats(jobs))
compare.render_compare_bar()
```

---

## 4. Update `render_job_card` — add compare button + skill-gap panel

This replaces the whole function. The only behavioral changes: the button
row is now 3 columns instead of 2 (Save / Compare / — description expander
moves to its own row), and an optional skill-gap panel renders when you
pass `user_skills`.

Find the entire existing `render_job_card` function and replace it with:

```python
def render_job_card(row, idx, section="job", user_skills=None):
    score = row.get('match_score', 0)
    title = row.get('title', 'N/A')
    company = row.get('company_name', 'Unknown')
    wtype = row.get('formatted_work_type', 'N/A')
    exp = row.get('formatted_experience_level', 'N/A')
    is_remote = row.get('remote_allowed', 0) == 1
    industry = row.get('industry', 'N/A')
    skills = row.get('required_skills', 'Not specified')
    benefits = row.get('job_benefits', 'Not specified')
    min_s = fmt_salary(row.get('min_salary_yr'))
    max_s = fmt_salary(row.get('max_salary_yr'))
    salary_str = f"{min_s} – {max_s} / yr" if min_s and max_s else "Not disclosed"
    remote_text = "Remote friendly" if is_remote else "On-site / hybrid"

    with st.container(border=False):
        st.markdown(f"""
        <div class="job-card">
          <div class="job-card-top">
            <div class="job-card-left">
              <div class="job-logo">{initials(company)}</div>
              <div>
                <div class="job-title">{esc(title)}</div>
                <div class="job-meta">{esc(company)} · {esc(wtype)}</div>
              </div>
            </div>
            <div class="job-card-right">
              <div class="match-score-num">{esc(score)}%</div>
              <div class="match-score-label">{esc(match_tier(score))}</div>
            </div>
          </div>
          <div class="kv-row">
            <span class="kv-item"><span class="kv-label">Salary</span>{esc(salary_str)}</span>
            <span class="kv-item"><span class="kv-label">Experience</span>{esc(exp)}</span>
            <span class="kv-item"><span class="kv-label">Location</span>{esc(remote_text)}</span>
            <span class="kv-item"><span class="kv-label">Industry</span>{esc(industry)}</span>
          </div>
          <span class="tag-block-label">Skills</span>
          <div style="margin-bottom:10px">{tags_html(skills)}</div>
          <span class="tag-block-label">Benefits</span>
          <div>{tags_html(benefits)}</div>
        </div>
        """, unsafe_allow_html=True)

        # Skill-gap panel — only shown when the caller supplied the user's
        # skills (resume text or the "Find matches" skills field).
        if user_skills:
            with st.expander("Skill gap for this role", expanded=False):
                skill_gap.render_skill_gap(user_skills, row)

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            with st.expander("View full description"):
                desc_text = str(row.get('description', 'Not available'))
                st.write(desc_text[:1500] + ("..." if len(desc_text) > 1500 else ""))
        with bcol2:
            bookmark_key = f"bookmark_{section}_{idx}_{stable_job_key(row)}"
            saved = st.session_state.get(bookmark_key, False)
            label = "Saved" if saved else "Save role"
            if st.button(label, key=bookmark_key + "_btn", use_container_width=True):
                st.session_state[bookmark_key] = not saved
                st.rerun()
        with bcol3:
            compare.render_compare_toggle(row, idx, section)
```

**What changed vs. your original:** added the `user_skills=None` parameter,
inserted the skill-gap expander, split the button row into 3 columns, and
added the compare toggle in the third column. `stable_job_key` stays
defined in `app.py` exactly as before — it's still used for the bookmark
key.

---

## 5. Pass `user_skills` into the calls that already know it, and add export buttons

### Resume section
Find:
```python
            if results.empty:
                render_empty_state("No matching roles found from resume.")
            else:
                st.markdown(f"**Top {len(results)} matching roles based on your resume**")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="resume")
```
Replace with:
```python
            if results.empty:
                render_empty_state("No matching roles found from resume.")
            else:
                st.markdown(f"**Top {len(results)} matching roles based on your resume**")
                dl1, dl2 = st.columns(2)
                with dl1:
                    export.render_csv_download_button(results, "rolesense_resume_matches.csv", key="resume_csv")
                with dl2:
                    export.render_pdf_download_button(
                        results, "rolesense_resume_matches.pdf", key="resume_pdf",
                        report_title="Rolyx — Resume Matches"
                    )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                resume_skill_set = ", ".join(hard_skills)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="resume", user_skills=resume_skill_set)
```
(`hard_skills` already exists in that scope from the resume-extraction step
right above it.)

### Find-matches section
Find:
```python
            if results.empty:
                render_empty_state("No matches found. Try loosening a filter or adding a few more skills.")
            else:
                st.markdown(f"**{len(results)} suggested matches**")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="find")
```
Replace with:
```python
            if results.empty:
                render_empty_state("No matches found. Try loosening a filter or adding a few more skills.")
            else:
                st.markdown(f"**{len(results)} suggested matches**")
                dl1, dl2 = st.columns(2)
                with dl1:
                    export.render_csv_download_button(results, "rolesense_matches.csv", key="find_csv")
                with dl2:
                    export.render_pdf_download_button(
                        results, "rolesense_matches.pdf", key="find_pdf",
                        report_title="Rolyx — Job Matches"
                    )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="find", user_skills=user_skills)
```
(`user_skills` here is the raw text-input string already collected above —
`skill_gap.compute_skill_gap` parses raw comma strings directly, no
pre-processing needed.)

### Salary prediction — add a PDF report button
Find:
```python
            panel_html = (
                '<div class="ai-panel">'
                ...
                '</div>'
            )
            st.markdown(panel_html, unsafe_allow_html=True)
```
Add directly after the `st.markdown(panel_html, unsafe_allow_html=True)` line:
```python
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            export.render_salary_pdf_button(
                description, exp_level, wt_sel, med, low, high, industries,
                key="salary_pdf"
            )
```

---

## 6. Add the Compare section

Find:
```python
render_footer()
```
Replace with:
```python
st.markdown('<div id="compare-section"></div>', unsafe_allow_html=True)
render_section_header(
    "SIDE BY SIDE", "Compare jobs",
    f"Jobs you've queued for comparison (up to {compare.MAX_COMPARE} at a time) — "
    "use \"Add to compare\" on any job card above."
)
compare.render_compare_section()

render_footer()
```

---

## Notes / things I deliberately did *not* change

- `render_lookup_card` (the "Role lookup" module) is untouched — it doesn't
  take a `user_skills` argument, so it won't show the skill-gap panel or
  compare button. If you want compare/export there too, apply the same
  pattern from steps 4–5; I left it out because "Role lookup" isn't scoped
  to a specific candidate's skills, so a skill-gap comparison wouldn't mean
  much there.
- I didn't touch `stable_job_key` in `app.py` — `features/compare.py`
  keeps its own copy so the module has zero import dependency on `app.py`,
  but the hashing logic is identical, so keys always agree.
- The compare tray lives in `st.session_state["compare_jobs"]` for the
  session only, same lifetime as your existing bookmarks — no new storage
  layer needed.
- Every new module was import-checked, byte-compiled, linted (pyflakes,
  zero warnings), and exercised against mock data (skill-gap matching/NaN
  handling, stable key generation, CSV/PDF generation including empty and
  partial-column dataframes) before being handed over.