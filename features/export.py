"""
Export helpers for RoleSense — CSV export of job result sets, and PDF
export of either a job result set or a single salary-prediction report.

CSV export only needs pandas (already an app.py dependency).
PDF export needs reportlab:  pip install reportlab
If reportlab isn't installed, PDF buttons render a friendly install hint
instead of crashing the app.

Self-contained by design — does NOT import from app.py (see skill_gap.py
docstring for why).
"""
import io
import html
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


# (dataframe column, display label) — only columns present in the results
# dataframe are included, so this works across resume/find/lookup results
# even though they don't all carry identical columns.
EXPORT_COLUMNS = [
    ("title", "Title"),
    ("company_name", "Company"),
    ("formatted_work_type", "Work Type"),
    ("formatted_experience_level", "Experience"),
    ("industry", "Industry"),
    ("min_salary_yr", "Min Salary"),
    ("max_salary_yr", "Max Salary"),
    ("required_skills", "Required Skills"),
    ("match_score", "Match Score (%)"),
]


def _prep_export_df(results: pd.DataFrame) -> pd.DataFrame:
    available = [(c, label) for c, label in EXPORT_COLUMNS if c in results.columns]
    df = results[[c for c, _ in available]].copy()
    df.columns = [label for _, label in available]
    return df


def jobs_to_csv_bytes(results: pd.DataFrame) -> bytes:
    """Returns UTF-8 CSV bytes for a results dataframe, ready for
    st.download_button."""
    df = _prep_export_df(results)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def render_csv_download_button(results: pd.DataFrame, filename: str, key: str):
    """Convenience wrapper: renders a CSV download button for a results
    dataframe. No-ops if results is empty/None."""
    if results is None or results.empty:
        return
    st.download_button(
        label="Download results as CSV",
        data=jobs_to_csv_bytes(results),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def _pdf_unavailable_notice():
    st.info(
        "PDF export needs the `reportlab` package. Install it with "
        "`pip install reportlab` and restart the app to enable this."
    )


def jobs_to_pdf_bytes(results: pd.DataFrame, report_title: str = "RoleSense Job Matches") -> bytes:
    """Builds a tabular PDF report of job results using reportlab. Long
    text fields are truncated so rows stay readable in a table cell."""
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed — run `pip install reportlab`.")

    df = _prep_export_df(results)

    def _truncate(val, n=60):
        s = "" if pd.isna(val) else str(val)
        return (s[:n] + "…") if len(s) > n else s

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=9)
    header_style = ParagraphStyle(
        "Header", parent=styles["Normal"], fontSize=7, leading=9,
        textColor=colors.white, fontName="Helvetica-Bold",
    )

    header_row = [Paragraph(html.escape(str(c)), header_style) for c in df.columns]
    body_rows = [
        [Paragraph(html.escape(_truncate(v)), cell_style) for v in r]
        for _, r in df.iterrows()
    ]
    table_data = [header_row] + body_rows

    col_count = max(len(df.columns), 1)
    page_width = letter[0] - 1.2 * inch
    col_widths = [page_width / col_count] * col_count

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B82F6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    story = [
        Paragraph(html.escape(report_title), title_style),
        Paragraph(
            f"{len(df)} result{'s' if len(df) != 1 else ''} &middot; "
            f"Estimates are informational, not guarantees.",
            styles["Normal"],
        ),
        Spacer(1, 14),
        table,
    ]
    doc.build(story)
    return buf.getvalue()


def render_pdf_download_button(results: pd.DataFrame, filename: str, key: str,
                                report_title: str = "RoleSense Job Matches"):
    """Convenience wrapper for a PDF download button. No-ops on empty
    results; shows an install hint if reportlab isn't available."""
    if results is None or results.empty:
        return
    if not _REPORTLAB_AVAILABLE:
        _pdf_unavailable_notice()
        return
    st.download_button(
        label="Download results as PDF",
        data=jobs_to_pdf_bytes(results, report_title=report_title),
        file_name=filename,
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )


def salary_report_to_pdf_bytes(description_snippet: str, exp_level: str, work_type: str,
                                med: float, low: float, high: float, industries: list) -> bytes:
    """Builds a one-page PDF summarizing a single salary/industry
    prediction (the output of app.py's predict())."""
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed — run `pip install reportlab`.")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748B"))
    value_style = ParagraphStyle("Value", parent=styles["Heading2"], fontSize=14, spaceAfter=10)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#94A3B8"))

    snippet = (description_snippet or "").strip().replace("\n", " ")
    if len(snippet) > 400:
        snippet = snippet[:400] + "…"

    industry_rows = [["Industry", "Confidence"]] + [[name, f"{pct}%"] for name, pct in industries]
    industry_table = Table(industry_rows, colWidths=[3.5 * inch, 1.5 * inch])
    industry_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B82F6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story = [
        Paragraph("RoleSense Salary Estimate", title_style),
        Paragraph("Experience level / Work type", label_style),
        Paragraph(html.escape(f"{exp_level} \u00b7 {work_type}"), value_style),
        Paragraph("Estimated range", label_style),
        Paragraph(html.escape(f"${low:,.0f} \u2013 ${high:,.0f} / year"), value_style),
        Paragraph("Estimated median", label_style),
        Paragraph(html.escape(f"${med:,.0f} / year"), value_style),
        Spacer(1, 6),
        Paragraph("Top predicted industries", label_style),
        Spacer(1, 6),
        industry_table,
        Spacer(1, 16),
        Paragraph("Description excerpt", label_style),
        Paragraph(html.escape(snippet), styles["Normal"]),
        Spacer(1, 16),
        Paragraph(
            "Estimate based on patterns in the training dataset \u2014 treat it as a "
            "directional starting point, not an offer or guarantee.",
            caption_style,
        ),
    ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    doc.build(story)
    return buf.getvalue()


def render_salary_pdf_button(description: str, exp_level: str, work_type: str,
                              med: float, low: float, high: float, industries: list, key: str):
    """Convenience wrapper for the salary-report PDF download button."""
    if not _REPORTLAB_AVAILABLE:
        _pdf_unavailable_notice()
        return
    st.download_button(
        label="Download salary report as PDF",
        data=salary_report_to_pdf_bytes(description, exp_level, work_type, med, low, high, industries),
        file_name="rolesense_salary_estimate.pdf",
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )