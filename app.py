import pdfplumber
import docx
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import re
import hashlib
import html
import logging
import sys
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# ═════════════════════════════════════════════════════════════
# LOGGING — server-side only, never shown to the end user.
# Streamlit's default logger is easy to lose in noisy console output,
# so this app gets its own named logger with a clear format.
# ═════════════════════════════════════════════════════════════
logger = logging.getLogger("rolesense")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RoleSense — AI Career Matching",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODELS_DIR = "models/"
APP_VERSION = "2.1"
MAX_RESUME_MB = 5
MAX_RESUME_BYTES = MAX_RESUME_MB * 1024 * 1024

# Input length caps — generous enough for real use, small enough to keep
# TF-IDF transforms and regex scans fast and bounded per request.
MAX_SKILLS_CHARS = 500
MAX_TITLE_CHARS = 150
MAX_DESCRIPTION_CHARS = 8000
MAX_JOB_NAME_CHARS = 150


# ═════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS  (minimal, Notion/Linear/Stripe-inspired)
# ═════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/[email protected]/font/bootstrap-icons.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg:            #0B1220;
            --surface:       #131B2E;
            --surface-alt:   #1A2338;
            --border:        #263049;
            --border-strong: #34405E;
            --ink:           #F1F5F9;
            --ink-soft:      #C3CBDC;
            --muted:         #8B96AF;

            --accent:        #3B82F6;
            --accent-hover:  #2563EB;
            --accent-soft:   rgba(59,130,246,0.12);

            --success:       #22C55E;
            --success-soft:  rgba(34,197,94,0.12);
            --warning:       #F59E0B;
            --warning-soft:  rgba(245,158,11,0.12);

            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 20px;

            --shadow-sm: 0 1px 2px rgba(0,0,0,0.2);
            --shadow-md: 0 8px 24px rgba(0,0,0,0.35);
        }

        html { scroll-behavior: smooth; }

        html, body { background-color: #0B1220 !important; }
        .stApp { background-color: #0B1220 !important; }
        [data-testid="stAppViewContainer"] { background-color: #0B1220 !important; }
        [data-testid="stMain"] { background-color: #0B1220 !important; }
        [data-testid="stHeader"] { background-color: #0B1220 !important; }
        [data-testid="stBottomBlockContainer"] { background-color: #0B1220 !important; }
        [data-testid="stSidebar"] { background-color: #131B2E !important; }
        section.main { background-color: #0B1220 !important; }
        .main .block-container { background-color: #0B1220 !important; }
        [data-testid="stVerticalBlock"] { background-color: transparent !important; }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
        }
        .main { padding-top: 0; }
        .block-container { padding: 0 2rem 3rem; max-width: 1080px; }
        #MainMenu, footer, header { visibility: hidden; }
        section[id] { scroll-margin-top: 80px; }

        h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: #F1F5F9 !important; letter-spacing: -0.02em; font-weight: 800; }
        p  { line-height: 1.65; color: #F1F5F9; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stText"],
        label, .stCaption, small { color: #8B96AF !important; }
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b { color: #F1F5F9 !important; }

        /* ── Navbar ────────────────────────────────────────── */
        .navbar {
            position: sticky; top: 0; z-index: 999;
            display: flex; align-items: center; justify-content: space-between;
            padding: 16px 0;
            margin: 0 0 56px;
            background: rgba(11,18,32,0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
        }
        .navbar .brand { display: flex; align-items: center; gap: 12px; }
        .navbar .brand .logo-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 38px; height: 38px; border-radius: 11px;
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 55%, #2563EB 100%);
            color: #fff; font-weight: 800; font-size: 1rem;
            box-shadow: 0 4px 14px rgba(59,130,246,0.35);
        }
        .navbar .brand .brand-name { font-weight: 800; font-size: 1.28rem; color: var(--ink); letter-spacing: -0.02em; }
        .navbar .links { display: flex; align-items: center; gap: 30px; }
        .navbar .links a {
            display: inline-flex; align-items: center; gap: 7px;
            text-decoration: none; font-size: 0.9rem; font-weight: 500;
            color: var(--ink-soft); transition: color 0.15s ease, filter 0.15s ease, transform 0.15s ease;
        }
        .navbar .links a i { font-size: 0.95rem; color: var(--muted); transition: color 0.15s ease; }
        .navbar .links a:hover { color: var(--ink); filter: brightness(1.3); transform: translateY(-1px); }
        .navbar .links a:hover i { color: var(--accent); }
        .navbar .nav-cta {
            background: var(--accent); color: #fff !important; font-size: 0.88rem; font-weight: 700;
            padding: 11px 24px; border-radius: 999px; text-decoration: none;
            box-shadow: 0 4px 14px rgba(59,130,246,0.3);
            transition: filter 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        }
        .navbar .nav-cta:hover {
            filter: brightness(1.18);
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(59,130,246,0.45);
        }
        .navbar .brand .logo-mark { transition: filter 0.15s ease, transform 0.15s ease; }
        .navbar .brand .logo-mark:hover { filter: brightness(1.25); transform: scale(1.05); }

        @media (max-width: 768px) { .navbar .links { display: none; } }

        /* ── Hero ─────────────────────────────────────────── */
        .hero { padding: 6px 0 8px; max-width: 700px; }
        .hero-eyebrow {
            font-size: 0.8rem; font-weight: 700; color: var(--accent);
            margin-bottom: 14px; letter-spacing: 0.03em; text-transform: uppercase;
        }
        .hero h1 {
            font-size: 2.6rem; font-weight: 800; line-height: 1.18; letter-spacing: -0.025em;
            margin-bottom: 14px; color: #F1F5F9 !important;
        }
        .hero h1 .accent-word { color: var(--accent) !important; }
        .hero p.lead { font-size: 1.02rem; color: var(--ink-soft); margin-bottom: 0; line-height: 1.65; max-width: 600px; }

        /* ── Stats grid ────────────────────────────────────── */
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 32px 0 48px; }
        .stat-card {
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 18px 20px;
            transition: filter 0.18s ease, transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .stat-card:hover {
            filter: brightness(1.15);
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: 0 8px 20px rgba(59,130,246,0.18);
        }
        .stat-num { font-weight: 800; font-size: 1.4rem; color: #F1F5F9; letter-spacing: -0.01em; }
        .stat-label { font-size: 0.78rem; color: #8B96AF; font-weight: 500; margin-top: 3px; }
        @media (max-width: 700px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }

        /* ── Section header ───────────────────────────────── */
        .sec-header { margin: 0 0 20px; padding-top: 40px; border-top: 1px solid var(--border); }
        .sec-eyebrow { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; }
        .sec-title { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.015em; color: var(--ink); margin-bottom: 4px; }
        .sec-sub { font-size: 0.9rem; color: var(--muted); max-width: 560px; line-height: 1.55; }

        /* ── Form helper text ─────────────────────────────── */
        .field-label { font-size: 0.83rem; font-weight: 700; color: var(--ink); margin-bottom: 4px; }
        .chip-hint { font-size: 0.8rem; color: var(--ink-soft); margin: 0 0 10px; font-weight: 500; }

        /* ── Job cards ────────────────────────────────────── */
        .job-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 22px 24px;
            margin-bottom: 14px;
            transition: filter 0.18s ease, transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .job-card:hover {
            filter: brightness(1.1);
            transform: translateY(-2px);
            border-color: var(--border-strong);
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        }
        .job-card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
        .job-card-left { display: flex; gap: 14px; align-items: flex-start; }
        .job-logo {
            flex-shrink: 0; width: 42px; height: 42px; border-radius: var(--radius-sm);
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 1rem; color: var(--accent);
            background: var(--accent-soft); border: 1px solid var(--border);
        }
        .job-title { font-size: 1.02rem; font-weight: 700; color: var(--ink); margin-bottom: 2px; letter-spacing: -0.005em; }
        .job-meta { font-size: 0.82rem; color: var(--muted); }
        .job-card-right { flex-shrink: 0; text-align: right; }
        .match-score-num { font-size: 1.1rem; font-weight: 800; color: var(--accent); line-height: 1.1; }
        .match-score-label { font-size: 0.74rem; color: var(--muted); margin-top: 1px; }

        .kv-row { display: flex; flex-wrap: wrap; gap: 5px 24px; margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .kv-item { font-size: 0.84rem; color: var(--ink-soft); }
        .kv-item .kv-label { color: var(--muted); margin-right: 5px; }

        .tag-block-label { font-size: 0.72rem; font-weight: 700; color: var(--muted); margin-bottom: 6px; display: block; }
        .tag {
            display: inline-block; padding: 5px 11px; border-radius: 999px;
            font-size: 0.76rem; margin: 2px 6px 2px 0; font-weight: 600;
            background: var(--surface-alt); color: var(--ink-soft); border: 1px solid var(--border);
            transition: filter 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
        }
        .tag:hover { filter: brightness(1.3); border-color: var(--accent); transform: translateY(-1px); }
        .tag-empty { color: var(--muted); font-size: 0.8rem; }

        /* ── AI analytics panel ──────────────────────────── */
        .ai-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 28px 30px;
            margin-top: 4px;
        }
        .ai-panel-label { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }
        .ai-salary { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 3px; color: var(--ink); }
        .ai-salary-sub { font-size: 0.86rem; color: var(--muted); margin-bottom: 24px; }
        .ai-industry-row { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
        .ai-industry-name { font-size: 0.85rem; min-width: 190px; font-weight: 500; color: var(--ink-soft); }
        .ai-industry-track { flex: 1; height: 6px; background: var(--surface-alt); border-radius: 3px; overflow: hidden; }
        .ai-industry-fill { height: 100%; border-radius: 3px; background: var(--accent); }
        .ai-industry-pct { font-size: 0.82rem; font-weight: 700; color: var(--ink); min-width: 38px; text-align: right; }
        .ai-caption { font-size: 0.78rem; color: var(--muted); margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--border); }

        /* ── Lookup cards ─────────────────────────────────── */
        .lookup-hero {
            display: flex; justify-content: space-between; align-items: center;
            padding: 18px 22px; border-radius: var(--radius-md) var(--radius-md) 0 0;
            background: var(--surface-alt); border: 1px solid var(--border); border-bottom: none;
        }
        .lookup-hero .lh-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); }
        .lookup-hero .lh-score { font-size: 0.8rem; font-weight: 700; color: var(--accent); }
        .info-card {
            background: var(--surface-alt); border: 1px solid var(--border); border-radius: var(--radius-md);
            padding: 16px 18px; height: 100%;
            transition: filter 0.18s ease, border-color 0.18s ease;
        }
        .info-card:hover { filter: brightness(1.12); border-color: var(--accent); }
        .info-card .ic-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
        .info-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.85rem; border-bottom: 1px solid var(--border); }
        .info-row:last-child { border-bottom: none; }
        .info-key { color: var(--muted); font-weight: 400; }
        .info-val { color: var(--ink); font-weight: 600; text-align: right; }
        .salary-strip {
            background: var(--surface-alt); border: 1px solid var(--border); border-radius: var(--radius-md);
            padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; margin: 0 0 14px;
        }
        .salary-strip .ss-label { font-size: 0.78rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
        .salary-strip .ss-value { font-size: 1.2rem; font-weight: 800; color: var(--accent); }

        /* ── Empty state ──────────────────────────────────── */
        .empty-state {
            text-align: center; padding: 36px 20px; color: var(--muted);
            background: var(--surface); border: 1px dashed var(--border-strong); border-radius: var(--radius-md);
            font-size: 0.88rem;
        }

        /* ── Streamlit widget theming ─────────────────────── */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] > div {
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border) !important;
            background: var(--surface-alt) !important;
            color: var(--ink) !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-soft) !important;
        }
        .stButton > button,
        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: var(--accent) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 700 !important;
            font-size: 0.86rem !important;
            box-shadow: none !important;
            transition: filter 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover,
        button[data-testid="baseButton-primary"]:hover {
            filter: brightness(1.2) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(59,130,246,0.4) !important;
        }
        .stButton > button:active,
        button[data-testid="baseButton-primary"]:active {
            filter: brightness(0.95) !important;
            transform: translateY(0) !important;
        }
        .stButton > button[kind="secondary"] {
            background: var(--surface-alt) !important;
            color: var(--ink) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="secondary"]:hover {
            filter: brightness(1.25) !important;
            border-color: var(--accent) !important;
        }

        div[data-testid="stExpander"] {
            border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important;
            background: var(--surface) !important; box-shadow: none !important;
            transition: filter 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stExpander"]:hover { filter: brightness(1.08); border-color: var(--border-strong) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radius-md) !important; border: 1px solid var(--border) !important;
            background: var(--surface) !important; box-shadow: none !important;
            transition: border-color 0.2s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--border-strong) !important; }
        div[data-testid="stFileUploader"] section {
            transition: filter 0.15s ease, border-color 0.15s ease !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            filter: brightness(1.12) !important;
            border-color: var(--accent) !important;
        }
        hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 36px 0 !important; }

        [data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
        [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 500; font-size: 0.8rem; }

        /* ── Footer ───────────────────────────────────────── */
        .app-footer { margin-top: 56px; padding: 24px 0 4px; border-top: 1px solid var(--border); }
        .footer-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
        .footer-brand { font-size: 0.85rem; font-weight: 700; color: var(--ink); }
        .footer-meta { font-size: 0.78rem; color: var(--muted); }

        /* ── Responsive ───────────────────────────────────── */
        @media (max-width: 900px) {
            .block-container { padding: 0 1.1rem 2.2rem; }
            .navbar { padding: 14px 0; }
            .hero h1 { font-size: 2rem; }
        }
        @media (max-width: 640px) {
            .job-card-top { flex-direction: column; }
            .job-card-right { text-align: left; }
            .footer-row { flex-direction: column; align-items: flex-start; }
        }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ═════════════════════════════════════════════════════════════
# MODEL / DATA LOADING
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    with open(MODELS_DIR + "tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    tfidf_matrix = sp.load_npz(MODELS_DIR + "tfidf_matrix.npz")
    with open(MODELS_DIR + "salary_model.pkl", "rb") as f:
        salary_model = pickle.load(f)
    with open(MODELS_DIR + "title_tfidf.pkl", "rb") as f:
        title_tfidf = pickle.load(f)
    with open(MODELS_DIR + "le_exp.pkl", "rb") as f:
        le_exp = pickle.load(f)
    with open(MODELS_DIR + "le_wtype.pkl", "rb") as f:
        le_wtype = pickle.load(f)
    with open(MODELS_DIR + "cat_model.pkl", "rb") as f:
        cat_model = pickle.load(f)
    with open(MODELS_DIR + "cat_tfidf.pkl", "rb") as f:
        cat_tfidf = pickle.load(f)
    with open(MODELS_DIR + "le_industry.pkl", "rb") as f:
        le_industry = pickle.load(f)
    jobs = pd.read_csv(MODELS_DIR + "enriched_jobs.csv", low_memory=False)
    jobs.reset_index(drop=True, inplace=True)
    return (tfidf, tfidf_matrix, salary_model, title_tfidf,
            le_exp, le_wtype, cat_model, cat_tfidf, le_industry, jobs)


try:
    (tfidf, tfidf_matrix, salary_model, title_tfidf,
     le_exp, le_wtype, cat_model, cat_tfidf, le_industry, jobs) = load_all()
except FileNotFoundError as e:
    logger.error("Model/data file missing: %s", e.filename, exc_info=True)
    st.error(
        f"Couldn't load one of the model/data files from `{MODELS_DIR}` "
        f"({e.filename}). Make sure the `models/` folder is present next to this app."
    )
    st.stop()
except Exception:
    # Don't leak internal tracebacks (paths, library internals, etc.) to end users —
    # but do capture the real error server-side so it's debuggable.
    logger.error("Failed to load models/data on startup", exc_info=True)
    st.error(
        "RoleSense couldn't start because the model/data files failed to load. "
        "Please contact support or check the deployment logs for details."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════
SOFT_SKILLS = [
    'communication', 'teamwork', 'leadership', 'problem solving',
    'time management', 'adaptability', 'creativity', 'critical thinking',
    'collaboration', 'attention to detail', 'organizational', 'multitasking',
    'interpersonal', 'presentation', 'negotiation', 'decision making',
    'analytical', 'project management', 'customer service', 'mentoring'
]

EXAMPLE_SKILLSETS = {
    "Data & AI":      "Python, Machine Learning, SQL, TensorFlow, Data Analysis",
    "Software Dev":   "JavaScript, React, Node.js, APIs, Git",
    "Marketing":      "SEO, Content Strategy, Google Analytics, Social Media",
    "Product Design": "Figma, UX Research, Prototyping, Design Systems",
}

SORT_OPTIONS = {
    "Best match":     None,
    "Highest salary": "max_salary_yr",
}


# ═════════════════════════════════════════════════════════════
# ML / DATA HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s,./()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def extract_soft_skills(description):
    if not isinstance(description, str):
        return []
    desc_lower = description.lower()
    return [s.title() for s in SOFT_SKILLS if s in desc_lower]


def fmt_salary(val):
    if pd.isna(val):
        return None
    return f"${val:,.0f}"


def match_tier(score):
    if score >= 75:
        return "Excellent fit"
    if score >= 55:
        return "Strong fit"
    if score >= 35:
        return "Good fit"
    return "Possible fit"


@st.cache_data(show_spinner=False)
def compute_dataset_stats(_jobs):
    total_jobs = len(_jobs)
    companies = _jobs['company_name'].dropna().nunique() if 'company_name' in _jobs.columns else 0
    industries = _jobs['industry'].dropna().nunique() if 'industry' in _jobs.columns else 0
    skill_set = set()
    if 'required_skills' in _jobs.columns:
        for cell in _jobs['required_skills'].dropna():
            for piece in str(cell).split(','):
                piece = piece.strip().lower()
                if piece and piece != 'not specified':
                    skill_set.add(piece)
    skills = len(skill_set)
    return {"jobs": total_jobs, "skills": skills, "companies": companies, "industries": industries}


def recommend_jobs(user_skills, job_title=None,
                    work_type=None, remote_only=False, top_n=10,
                    sort_by=None):
    parts = [user_skills]
    if job_title:
        parts = [job_title] * 3 + parts
    query_vec = tfidf.transform([clean_text(" ".join(parts))])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    mask = np.ones(len(jobs), dtype=bool)
    if work_type and work_type != "Any" and 'formatted_work_type' in jobs.columns:
        mask &= jobs['formatted_work_type'].str.contains(work_type, case=False, na=False)
    if remote_only and 'remote_allowed' in jobs.columns:
        mask &= (jobs['remote_allowed'] == 1)
    filtered = scores.copy()
    filtered[~mask] = -1
    pool_size = min(len(jobs), max(top_n * 5, top_n))
    pool_idx = np.argsort(filtered)[::-1][:pool_size]
    pool_idx = pool_idx[filtered[pool_idx] > -1]
    result = jobs.iloc[pool_idx].copy()
    result['match_score'] = (filtered[pool_idx] * 100).round(1)
    if sort_by and sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False, na_position='last')
    result = result.head(top_n)
    return result, pool_idx[:top_n]


def predict(description, experience_level, work_type):
    desc = clean_text(description)
    if experience_level in le_exp.classes_:
        exp_enc = le_exp.transform([experience_level])[0]
    else:
        logger.warning("Unrecognized experience_level %r, defaulting to class 0", experience_level)
        exp_enc = 0
    if work_type in le_wtype.classes_:
        wtype_enc = le_wtype.transform([work_type])[0]
    else:
        logger.warning("Unrecognized work_type %r, defaulting to class 0", work_type)
        wtype_enc = 0
    title_feat = title_tfidf.transform([desc])
    struct_feat = csr_matrix([[exp_enc, wtype_enc, 0]])
    X = hstack([struct_feat, title_feat])
    pred = float(np.clip(salary_model.predict(X)[0], 0, None))
    sal_low = pred * 0.85
    sal_high = pred * 1.15
    X_cat = cat_tfidf.transform([desc])
    proba = cat_model.predict_proba(X_cat)[0]
    top3 = np.argsort(proba)[::-1][:3]
    industries = [(le_industry.classes_[i], round(proba[i] * 100, 1)) for i in top3]
    return pred, sal_low, sal_high, industries


def lookup_job(job_name, top_n=3):
    q = clean_text((" ".join([job_name] * 3)))
    vec = tfidf.transform([q])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    result = jobs.iloc[top_idx].copy()
    result['match_score'] = (scores[top_idx] * 100).round(1)
    result['soft_skills'] = result['description'].apply(
        lambda x: ", ".join(extract_soft_skills(x)) or "Not specified"
    )
    return result


# ─────────────────────────────────────────────────────────────
# RESUME PARSING HELPERS
# ─────────────────────────────────────────────────────────────
def extract_text_from_resume(uploaded_file):
    """Extract raw text from an uploaded PDF or DOCX file."""
    try:
        if uploaded_file.name.lower().endswith('.pdf'):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text
        elif uploaded_file.name.lower().endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return ""
    except Exception:
        logger.error("Failed to extract text from uploaded resume: %s", getattr(uploaded_file, "name", "?"), exc_info=True)
        st.error("There was a problem reading that resume file. Please try a different PDF or DOCX file.")
        return ""


@st.cache_data(show_spinner=False)
def build_skill_vocabulary(_jobs):
    """Builds a master skill vocabulary from the dataset's required_skills column."""
    skill_set = set()
    if 'required_skills' in _jobs.columns:
        for cell in _jobs['required_skills'].dropna():
            for piece in str(cell).split(','):
                piece = piece.strip().lower()
                if piece and piece != 'not specified' and len(piece) > 1:
                    skill_set.add(piece)
    return skill_set


def extract_hard_skills(resume_text, skill_vocab):
    """Finds which vocabulary skills appear in the resume text."""
    text_lower = resume_text.lower()
    found = []
    for skill in skill_vocab:
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return sorted(set(found))


def extract_qualifications(resume_text):
    """Simple keyword-based degree/certification detection."""
    quals = [
        "Bachelor", "Master", "PhD", "MBA", "B.Sc", "M.Sc", "B.Tech", "M.Tech",
        "Diploma", "Certified", "Certification", "Associate Degree"
    ]
    text_lower = resume_text.lower()
    found = [q for q in quals if q.lower() in text_lower]
    return found


# ═════════════════════════════════════════════════════════════
# PRESENTATIONAL HELPERS
# ═════════════════════════════════════════════════════════════
def esc(value) -> str:
    """HTML-escape any value that gets interpolated into an unsafe_allow_html block.
    Everything rendered here can originate from the jobs CSV or a user upload, so
    nothing is trusted by default."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def tags_html(items):
    if items is None or items == "Not specified" or (isinstance(items, float) and pd.isna(items)):
        return "<span class='tag-empty'>Not specified</span>"
    if isinstance(items, str):
        items = [i.strip() for i in items.split(",") if i.strip()]
    if not items:
        return "<span class='tag-empty'>Not specified</span>"
    return " ".join(f'<span class="tag">{esc(i)}</span>' for i in items[:8])


def initials(name: str):
    name = (name or "?").strip()
    return esc(name[0].upper()) if name else "?"


def render_navbar():
    logo_svg = """
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4 8.5C4 7.67157 4.67157 7 5.5 7H18.5C19.3284 7 20 7.67157 20 8.5V17.5C20 18.3284 19.3284 19 18.5 19H5.5C4.67157 19 4 18.3284 4 17.5V8.5Z"
            stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9 7V6C9 5.44772 9.44772 5 10 5H14C14.5523 5 15 5.44772 15 6V7"
            stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M4 12.5C6.2 13.8 9 14.5 12 14.5C15 14.5 17.8 13.8 20 12.5"
            stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="12" cy="12.5" r="0.9" fill="white"/>
    </svg>
    """
    st.markdown(f"""
    <div class="navbar">
      <div class="brand">
        <a class="logo-mark" href="." style="cursor:pointer; text-decoration:none;" title="Back to home">{logo_svg}</a>
        <a class="brand-name" href="." style="cursor:pointer; text-decoration:none;">RoleSense</a>
      </div>
      <div class="links">
        <a href="#resume-section">Resume AI</a>
        <a href="#find-section">Find matches</a>
        <a href="#predict-section">Salary AI</a>
        <a href="#lookup-section">Role lookup</a>
      </div>
      <a class="nav-cta" href="#resume-section">Get started</a>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">AI CAREER MATCHING</div>
      <h1>Find roles you're actually a <span class="accent-word">fit</span> for.</h1>
      <p class="lead">Match your skills against real job market data, estimate salary from any
      description, and look up typical pay and requirements for any role — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)


def render_stats(stats: dict):
    items = [
        (stats["jobs"], "Jobs indexed", None),
        (stats["skills"], "Skills recognized", "18,500+"),
        (stats["companies"], "Companies", None),
        (stats["industries"], "Industries", None),
    ]
    cards = []
    for value, label, override in items:
        if override:
            display = override
        else:
            display = f"{value/1_000_000:.1f}M" if value >= 1_000_000 else f"{value:,}"
        cards.append(
            '<div class="stat-card"><div class="stat-num">' + esc(display) +
            '</div><div class="stat-label">' + esc(label) + '</div></div>'
        )
    grid_html = '<div class="stats-grid">' + "".join(cards) + '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def render_section_header(eyebrow, title, sub):
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-eyebrow">{esc(eyebrow)}</div>
      <div class="sec-title">{esc(title)}</div>
      <div class="sec-sub">{esc(sub)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(message):
    st.markdown(f"""<div class="empty-state">{esc(message)}</div>""", unsafe_allow_html=True)


def stable_job_key(row) -> str:
    """Prefer a real dataset identifier so bookmarks don't collide when two
    different postings share the same title + company (very common for
    large employers). Falls back to a hash of more fields if no id column
    exists, and finally to the row's own dataframe index."""
    for id_col in ("job_id", "id", "job_posting_id"):
        if id_col in row and pd.notna(row.get(id_col)):
            return str(row.get(id_col))
    fingerprint = "|".join(str(row.get(c, "")) for c in (
        "title", "company_name", "description", "min_salary_yr", "max_salary_yr"
    ))
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


def render_job_card(row, idx, section="job"):
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

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            with st.expander("View full description"):
                desc_text = str(row.get('description', 'Not available'))
                st.write(desc_text[:1500] + ("..." if len(desc_text) > 1500 else ""))
        with bcol2:
            # Include the section name so the same job appearing in multiple
            # result lists (resume matches vs. manual search, etc.) never
            # collides on a duplicate widget id, and use a stable per-job
            # identifier so two different postings with the same title +
            # company don't share bookmark state.
            bookmark_key = f"bookmark_{section}_{idx}_{stable_job_key(row)}"
            saved = st.session_state.get(bookmark_key, False)
            label = "Saved" if saved else "Save role"
            if st.button(label, key=bookmark_key + "_btn", use_container_width=True):
                st.session_state[bookmark_key] = not saved
                st.rerun()


def render_lookup_card(row):
    score = row.get('match_score', 0)
    title = row.get('title', 'N/A')
    company = row.get('company_name', 'Unknown')
    wtype = row.get('formatted_work_type', 'N/A')
    exp = row.get('formatted_experience_level', 'N/A')
    remote = "Yes" if row.get('remote_allowed', 0) == 1 else "No"
    industry = row.get('industry', 'Not specified')
    skills = row.get('required_skills', 'Not specified')
    soft = row.get('soft_skills', 'Not specified')
    benefits = row.get('job_benefits', 'Not specified')
    min_s = fmt_salary(row.get('min_salary_yr'))
    max_s = fmt_salary(row.get('max_salary_yr'))
    salary_str = f"{min_s} – {max_s} / yr" if min_s and max_s else "Not disclosed"

    st.markdown(f"""
    <div class="lookup-hero">
      <div class="lh-title">{esc(title)}</div>
      <div class="lh-score">{esc(score)}% reference match</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"""
        <div class="salary-strip">
          <span class="ss-label">Typical salary</span>
          <span class="ss-value">{esc(salary_str)}</span>
        </div>
        """, unsafe_allow_html=True)

        tab_overview, tab_desc = st.tabs(["Overview", "Full description"])

        with tab_overview:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Role overview</div>
                  <div class="info-row"><span class="info-key">Example employer</span><span class="info-val">{esc(company)}</span></div>
                  <div class="info-row"><span class="info-key">Work type</span><span class="info-val">{esc(wtype)}</span></div>
                  <div class="info-row"><span class="info-key">Experience</span><span class="info-val">{esc(exp)}</span></div>
                  <div class="info-row"><span class="info-key">Often remote</span><span class="info-val">{esc(remote)}</span></div>
                  <div class="info-row"><span class="info-key">Industry</span><span class="info-val">{esc(industry)}</span></div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Required skills</div>
                  <div>{tags_html(skills)}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Soft skills</div>
                  <div>{tags_html(soft)}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Benefits</div>
                  <div>{tags_html(benefits)}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab_desc:
            desc_text = str(row.get('description', 'Not available'))
            st.write(desc_text[:2500] + ("..." if len(desc_text) > 2500 else ""))

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(f"""
    <div class="app-footer">
      <div class="footer-row">
        <span class="footer-brand">RoleSense</span>
        <span class="footer-meta">v{esc(APP_VERSION)} · Built with Streamlit + scikit-learn · Estimates are informational, not guarantees</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_splash_injector():
    """Injects a full-screen branded splash / landing page that shows on first
    load (auto-dismisses after 3 s) and whenever the navbar logo is clicked.
    The splash shows the RoleSense branding image filling the ENTIRE viewport
    (object-fit: cover), with the animated loading bar overlaid near the
    bottom of the screen."""
    import base64, pathlib
    # Embed the landing image as base64 so the splash is fully self-contained.
    # Place the branding image as either:
    #   • assets/rolesense_logo.png   (preferred)
    #   • rolesense_logo.png          (next to app.py)
    _base = pathlib.Path(__file__).parent
    _candidates = [
        _base / "assets" / "rolesense_logo.png",
        _base / "rolesense_logo.png",
    ]
    _img_src = ""
    for _p in _candidates:
        if _p.exists():
            try:
                _img_b64 = base64.b64encode(_p.read_bytes()).decode()
                _img_src = f"data:image/png;base64,{_img_b64}"
            except Exception:
                pass
            break

    components.html(f"""
    <script>
    (function() {{
        var doc   = window.parent.document;
        var win   = window.parent;
        var IMG_SRC = {repr(_img_src)};

        /* ── Inject styles into PARENT document head ─────── */
        if (!doc.getElementById('rs-splash-styles')) {{
            var style = doc.createElement('style');
            style.id  = 'rs-splash-styles';
            style.textContent = [
                /* force every ancestor to not clip our overlay */
                '#rs-splash,#rs-splash *{{box-sizing:border-box;}}',

                /* The overlay itself — must live in parent doc, cover true viewport */
                '#rs-splash{{',
                '  position:fixed!important;',
                '  top:0!important;left:0!important;',
                '  width:100vw!important;height:100vh!important;',
                '  z-index:2147483647!important;',   /* max z-index */
                '  display:flex;flex-direction:column;',
                '  align-items:stretch;justify-content:flex-start;',
                '  background:#030c18;',
                '  font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;',
                '  opacity:0;',
                '  transition:opacity .55s cubic-bezier(.4,0,.2,1);',
                '  overflow:hidden;',
                '  margin:0!important;padding:0!important;',
                '  transform:none!important;',
                '}}',

                /* dot grid texture, sits above the full-bleed image */
                '.rs-dots{{',
                '  position:absolute;inset:0;pointer-events:none;z-index:1;',
                '  background-image:radial-gradient(rgba(147,197,253,.10) 1px,transparent 1px);',
                '  background-size:30px 30px;',
                '}}',

                /* top-left arc */
                '.rs-arc-tl{{',
                '  position:absolute;width:560px;height:560px;border-radius:50%;',
                '  border:1px solid rgba(255,255,255,.14);',
                '  top:-200px;left:-200px;pointer-events:none;z-index:1;',
                '}}',
                /* bottom-right arc */
                '.rs-arc-br{{',
                '  position:absolute;width:420px;height:420px;border-radius:50%;',
                '  border:1px solid rgba(255,255,255,.12);',
                '  bottom:-140px;right:-100px;pointer-events:none;z-index:1;',
                '}}',

                /* full-bleed image layer — the image IS the full-screen background */
                '.rs-image-layer{{',
                '  position:absolute;top:0;left:0;right:0;bottom:0;',
                '  z-index:2;',
                '  display:flex;align-items:center;justify-content:center;',
                '  overflow:hidden;',
                '  opacity:0;transform:scale(1.04);',
                '  transition:opacity .7s .12s cubic-bezier(.22,1,.36,1),',
                '             transform .7s .12s cubic-bezier(.22,1,.36,1);',
                '}}',
                '.rs-image-layer.rs-visible{{opacity:1;transform:scale(1);}}',

                /* the brand image is a full-bleed 1536x1024 splash graphic (already contains
                   its own dark background, arcs, and dot-grid) — cover the ENTIRE viewport */
                '#rs-img{{',
                '  width:100%;height:100%;',
                '  object-fit:cover;object-position:center;',
                '  display:block;',
                '  user-select:none;-webkit-user-drag:none;',
                '}}',

                /* fallback wordmark, sized big if no image is found */
                '.rs-fallback{{',
                '  font-size:min(9vw,6rem);font-weight:800;color:#F1F5F9;',
                '  letter-spacing:-.03em;',
                '}}',

                /* subtle bottom scrim so the bar reads clearly over any image */
                '.rs-scrim{{',
                '  position:absolute;left:0;right:0;bottom:0;height:34vh;z-index:3;',
                '  background:linear-gradient(180deg,rgba(3,12,24,0) 0%,rgba(3,12,24,.55) 55%,rgba(3,12,24,.92) 100%);',
                '  pointer-events:none;',
                '}}',

                /* bar section — pinned near the bottom, overlaying the image */
                '.rs-bar-section{{',
                '  position:absolute;left:50%;bottom:64px;z-index:4;',
                '  transform:translateX(-50%) translateY(14px);',
                '  width:min(360px,72vw);',
                '  display:flex;flex-direction:column;align-items:stretch;gap:9px;',
                '  opacity:0;',
                '  transition:opacity .6s .3s cubic-bezier(.22,1,.36,1),',
                '             transform .6s .3s cubic-bezier(.22,1,.36,1);',
                '}}',
                '.rs-bar-section.rs-visible{{opacity:1;transform:translateX(-50%) translateY(0);}}',
                '.rs-bar-label{{',
                '  font-size:.7rem;font-weight:700;',
                '  letter-spacing:.14em;text-transform:uppercase;',
                '  color:rgba(255,255,255,.75);',
                '  text-align:center;',
                '  text-shadow:0 1px 6px rgba(0,0,0,.5);',
                '}}',
                '.rs-track{{',
                '  width:100%;height:2px;',  /* slim, refined */
                '  background:rgba(255,255,255,.16);',
                '  border-radius:99px;overflow:hidden;position:relative;',
                '}}',
                '.rs-fill{{',
                '  position:absolute;left:0;top:0;bottom:0;',
                '  width:0%;border-radius:99px;',
                '  background:linear-gradient(90deg,#1e40af 0%,#3b82f6 45%,#60a5fa 75%,#bfdbfe 100%);',
                '  transition:width .28s linear;',
                '  box-shadow:0 0 10px rgba(96,165,250,.55),0 0 22px rgba(59,130,246,.25);',
                '}}',
                /* shimmer sweep */
                '.rs-fill::after{{',
                '  content:"";',
                '  position:absolute;top:0;bottom:0;left:-70%;width:70%;',
                '  background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);',
                '  animation:rs-shim 1.2s linear infinite;',
                '}}',
                '@keyframes rs-shim{{from{{left:-70%}}to{{left:120%}}}}',

                /* pct counter */
                '.rs-pct{{',
                '  font-size:.66rem;font-weight:700;',
                '  letter-spacing:.05em;',
                '  color:rgba(191,219,254,.85);',
                '  text-align:right;',
                '  font-variant-numeric:tabular-nums;',
                '}}',
            ].join('');
            doc.head.appendChild(style);
        }}

        /* ── Inject overlay into PARENT document body ────── */
        if (!doc.getElementById('rs-splash')) {{
            var el = doc.createElement('div');
            el.id  = 'rs-splash';
            el.innerHTML =
                '<div class="rs-image-layer" id="rs-image-layer">' +
                  (IMG_SRC
                    ? '<img id="rs-img" src="' + IMG_SRC + '" alt="RoleSense" draggable="false">'
                    : '<div class="rs-fallback">RoleSense</div>'
                  ) +
                '</div>' +
                '<div class="rs-scrim"></div>' +
                '<div class="rs-bar-section" id="rs-inner">' +
                  '<div class="rs-bar-label" id="rs-label">Initializing</div>' +
                  '<div class="rs-track"><div class="rs-fill" id="rs-fill"></div></div>' +
                  '<div class="rs-pct" id="rs-pct">0%</div>' +
                '</div>';
            /* prepend so it's on top of everything */
            doc.body.insertBefore(el, doc.body.firstChild);
        }}

        /* ── Core runner ─────────────────────────────────── */
        function runSplash(duration) {{
            var overlay   = doc.getElementById('rs-splash');
            var imageLayer= doc.getElementById('rs-image-layer');
            var inner     = doc.getElementById('rs-inner');
            var fill      = doc.getElementById('rs-fill');
            var pctEl     = doc.getElementById('rs-pct');
            var labelEl   = doc.getElementById('rs-label');
            if (!overlay) return;

            var labels = ['Initializing','Loading models','Calibrating engine','Almost ready','Launching'];
            var labelIdx = 0;

            /* ── show overlay ── */
            overlay.style.cssText += ';display:flex!important;';
            /* force body not to scroll while splash is up */
            doc.body.style.overflow = 'hidden';

            requestAnimationFrame(function() {{
                overlay.style.opacity = '1';
                if (imageLayer) imageLayer.classList.add('rs-visible');
                if (inner) inner.classList.add('rs-visible');
            }});

            /* scroll to top */
            try {{ win.scrollTo({{top:0,behavior:'instant'}}); }} catch(e) {{}}

            /* label cycling */
            var labelTimer = setInterval(function() {{
                labelIdx = Math.min(labelIdx + 1, labels.length - 1);
                if (labelEl) labelEl.textContent = labels[labelIdx];
            }}, duration / labels.length);

            /* rAF-driven progress with ease-out-cubic */
            var t0 = performance.now();
            function tick() {{
                var t = Math.min((performance.now() - t0) / duration, 1);
                var pct = Math.round((1 - Math.pow(1 - t, 2.8)) * 100);
                if (fill)   fill.style.width  = pct + '%';
                if (pctEl)  pctEl.textContent = pct + '%';
                if (t < 1) {{ requestAnimationFrame(tick); }}
                else        {{ dismiss(); }}
            }}
            requestAnimationFrame(tick);

            function dismiss() {{
                clearInterval(labelTimer);
                overlay.style.opacity = '0';
                if (imageLayer) imageLayer.classList.remove('rs-visible');
                if (inner) inner.classList.remove('rs-visible');
                setTimeout(function() {{
                    overlay.style.display = 'none';
                    doc.body.style.overflow = '';
                    if (fill)    fill.style.width    = '0%';
                    if (pctEl)   pctEl.textContent   = '0%';
                    if (labelEl) labelEl.textContent  = labels[0];
                }}, 560);
            }}
        }}

        /* ── Auto-run on first page load ─────────────────── */
        if (!win.__rsSplashShown) {{
            win.__rsSplashShown = true;
            /* slight delay so DOM is ready */
            setTimeout(function() {{ runSplash(3000); }}, 80);
        }}

        /* ── Expose for logo / brand-name clicks ─────────── */
        win.showSplash = function() {{ runSplash(3000); }};

    }})();
    </script>
    """, height=0)


# ═════════════════════════════════════════════════════════════
# PAGE ASSEMBLY
# ═════════════════════════════════════════════════════════════
try:
    render_splash_injector()
except Exception:
    # This splash reaches into the parent document's DOM, which isn't a
    # supported Streamlit API — if a future Streamlit version changes that
    # structure, fail quietly instead of taking the whole page down with it.
    logger.warning("Splash injector failed to render; continuing without it", exc_info=True)
render_navbar()
render_hero()
render_stats(compute_dataset_stats(jobs))

# ─────────────────────────────────────────────────────────────
# MODULE 00 — Resume upload & auto-match
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="resume-section"></div>', unsafe_allow_html=True)
render_section_header(
    "GET STARTED", "Upload your resume",
    "Upload a PDF or DOCX resume — RoleSense extracts your skills and instantly matches you against real roles."
)

skill_vocab = build_skill_vocabulary(jobs)

with st.container(border=True):
    uploaded_resume = st.file_uploader(
        f"Upload your resume (PDF or DOCX, max {MAX_RESUME_MB}MB)",
        type=["pdf", "docx"],
        label_visibility="collapsed"
    )
    st.caption("Your resume is processed in memory for this session only and is not stored.")

    analyze_clicked = st.button("Analyze resume", key="resume_btn", use_container_width=True)

    if analyze_clicked:
        if not uploaded_resume:
            st.warning("Please upload a resume file first.")
        elif uploaded_resume.size > MAX_RESUME_BYTES:
            st.warning(
                f"That file is {uploaded_resume.size / (1024*1024):.1f}MB — "
                f"please upload a resume under {MAX_RESUME_MB}MB."
            )
        else:
            with st.status("Analyzing your resume...", expanded=False) as status:
                st.write("Extracting text from file...")
                resume_text = extract_text_from_resume(uploaded_resume)

                if not resume_text.strip():
                    status.update(label="Couldn't extract text.", state="error")
                    st.error("Couldn't extract text from the uploaded file. Please try another file.")
                    st.stop()

                status.update(label="Extracting hard skills...")
                hard_skills = extract_hard_skills(resume_text, skill_vocab)

                status.update(label="Extracting soft skills...")
                soft_skills = extract_soft_skills(resume_text)

                status.update(label="Detecting qualifications...")
                qualifications = extract_qualifications(resume_text)

                status.update(label="Matching against job roles...")
                cleaned = clean_text(resume_text)
                results, _ = recommend_jobs(user_skills=cleaned, top_n=10)

                status.update(label="Analysis complete.", state="complete")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Hard skills found ({len(hard_skills)})</div>
                  <div>{tags_html(hard_skills)}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Soft skills found ({len(soft_skills)})</div>
                  <div>{tags_html(soft_skills)}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="info-card">
                  <div class="ic-label">Qualifications found ({len(qualifications)})</div>
                  <div>{tags_html(qualifications)}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

            if results.empty:
                render_empty_state("No matching roles found from resume.")
            else:
                st.markdown(f"**Top {len(results)} matching roles based on your resume**")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="resume")

# ─────────────────────────────────────────────────────────────
# MODULE 01 — Find matching jobs
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="find-section"></div>', unsafe_allow_html=True)
render_section_header(
    "SKILLS MATCH", "Find your best-fit roles",
    "Enter your skills and RoleSense surfaces roles from the dataset you're likely to match well with."
)

with st.container(border=True):
    st.markdown("<div class='chip-hint'>Quick start — click a skillset to autofill</div>", unsafe_allow_html=True)
    chip_cols = st.columns(len(EXAMPLE_SKILLSETS))
    for col, (label, skills_str) in zip(chip_cols, EXAMPLE_SKILLSETS.items()):
        if col.button(label, key=f"chip_{label}", use_container_width=True):
            st.session_state["user_skills_key"] = skills_str

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("<div class='field-label'>Your skills</div>", unsafe_allow_html=True)
        user_skills = st.text_input(
            "Your skills (comma separated)",
            key="user_skills_key",
            placeholder="e.g. Python, Machine Learning, SQL, TensorFlow",
            label_visibility="collapsed",
            max_chars=MAX_SKILLS_CHARS
        )
        st.caption("Comma-separated. The more specific, the better the match.")
    with col_b:
        st.markdown("<div class='field-label'>Desired title</div>", unsafe_allow_html=True)
        desired_title = st.text_input(
            "Desired job title (optional)",
            placeholder="e.g. Data Scientist",
            label_visibility="collapsed",
            max_chars=MAX_TITLE_CHARS
        )
        st.caption("Optional — sharpens title relevance.")

    col_d, col_e, col_f, col_g = st.columns([1, 1, 1, 1])
    with col_d:
        work_type = st.selectbox("Work type", ["Any", "Full-time", "Part-time", "Contract", "Internship"])
    with col_e:
        sort_choice = st.selectbox("Sort by", list(SORT_OPTIONS.keys()))
    with col_f:
        top_n = st.selectbox("Results", [5, 10, 20], index=1)
    with col_g:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        remote_only = st.checkbox("Remote only", value=False)

    find_clicked = st.button("Find matching jobs", key="find_btn", use_container_width=True)

    if find_clicked:
        if not user_skills.strip():
            st.warning("Please enter at least one skill.")
        else:
            with st.status("Running match engine...", expanded=False) as status:
                st.write("Parsing your skill profile...")
                status.update(label="Scoring roles against the dataset...")
                results, _ = recommend_jobs(
                    user_skills=user_skills,
                    job_title=desired_title or None,
                    work_type=work_type,
                    remote_only=remote_only,
                    top_n=top_n,
                    sort_by=SORT_OPTIONS[sort_choice]
                )
                status.update(label=f"Done — {len(results)} matches ranked.", state="complete")

            if results.empty:
                render_empty_state("No matches found. Try loosening a filter or adding a few more skills.")
            else:
                st.markdown(f"**{len(results)} suggested matches**")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    render_job_card(row, i, section="find")

# ─────────────────────────────────────────────────────────────
# MODULE 02 — Predict salary & industry
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="predict-section"></div>', unsafe_allow_html=True)
render_section_header(
    "SALARY INSIGHTS", "Predict salary & industry",
    "Paste any job description and RoleSense estimates its pay range and the most likely industries."
)

with st.container(border=True):
    st.markdown("<div class='field-label'>Job description</div>", unsafe_allow_html=True)
    description = st.text_area(
        "Job description",
        height=150,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
        max_chars=MAX_DESCRIPTION_CHARS
    )
    st.caption(
        f"More detail (responsibilities, requirements, seniority cues) improves accuracy. "
        f"Max {MAX_DESCRIPTION_CHARS:,} characters."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        exp_options = list(le_exp.classes_)
        exp_level = st.selectbox("Experience level", exp_options,
                                  index=exp_options.index('Mid-Senior level')
                                  if 'Mid-Senior level' in exp_options else 0)
    with col_p2:
        wt_options = list(le_wtype.classes_)
        wt_sel = st.selectbox("Work type", wt_options,
                               index=wt_options.index('Full-time')
                               if 'Full-time' in wt_options else 0,
                               key="predict_wtype")

    predict_clicked = st.button("Predict salary & industry", key="pred_btn", use_container_width=True)

    if predict_clicked:
        if not description.strip():
            st.warning("Please paste a job description.")
        elif len(description.strip()) < 30:
            st.warning("That description looks a bit short — paste more detail for a reliable estimate.")
        else:
            try:
                with st.status("Running AI analysis...", expanded=False) as status:
                    st.write("Reading the description...")
                    status.update(label="Estimating salary range...")
                    med, low, high, industries = predict(description, exp_level, wt_sel)
                    status.update(label="Classifying industry...")
                    status.update(label="Analysis complete.", state="complete")
            except Exception:
                logger.error("Salary/industry prediction failed", exc_info=True)
                st.error("Something went wrong while analyzing that description. Please try again.")
                st.stop()

            top_industry_name, top_industry_pct = industries[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated median salary", f"${med:,.0f}")
            m2.metric("Estimated range", f"${low:,.0f} – ${high:,.0f}")
            m3.metric("Top industry confidence", f"{top_industry_pct}%", help=f"Model confidence that this role belongs to {top_industry_name}")

            ind_bars = "".join(
                '<div class="ai-industry-row"><span class="ai-industry-name">' + esc(name) +
                '</span><div class="ai-industry-track"><div class="ai-industry-fill" style="width:' +
                str(int(prob)) + '%"></div></div><span class="ai-industry-pct">' + esc(prob) +
                '%</span></div>'
                for name, prob in industries
            )

            panel_html = (
                '<div class="ai-panel">'
                '<div class="ai-panel-label">AI salary estimate</div>'
                '<div class="ai-salary">$' + f"{low:,.0f}" + ' – $' + f"{high:,.0f}" + ' / year</div>'
                '<div class="ai-salary-sub">Estimated median: $' + f"{med:,.0f}" + ' / year</div>'
                '<div class="ai-panel-label">Top predicted industries</div>'
                + ind_bars +
                '<div class="ai-caption">Estimate based on patterns in the training dataset — '
                'treat it as a directional starting point, not an offer or guarantee.</div>'
                '</div>'
            )
            st.markdown(panel_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODULE 03 — Job lookup
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="lookup-section"></div>', unsafe_allow_html=True)
render_section_header(
    "ROLE DATA", "Look up a role",
    "Search any job title to see typical skills, pay, and benefits for that role in the dataset."
)

with st.container(border=True):
    col_l1, col_l2 = st.columns([3, 1])
    with col_l1:
        st.markdown("<div class='field-label'>Job title</div>", unsafe_allow_html=True)
        job_name = st.text_input(
            "Job title", placeholder="e.g. Data Scientist, Product Manager, DevOps Engineer",
            label_visibility="collapsed",
            max_chars=MAX_JOB_NAME_CHARS
        )
    with col_l2:
        top_n_lookup = st.selectbox("Results", [1, 2, 3, 5], index=1)

    lookup_clicked = st.button("Search job details", key="lookup_btn", use_container_width=True)

    if lookup_clicked:
        if not job_name.strip():
            st.warning("Please enter a job title.")
        else:
            with st.spinner("Looking up typical role details..."):
                results = lookup_job(job_name, top_n=top_n_lookup)

            if results.empty:
                render_empty_state("No close matches for that title. Try a broader or more common job name.")
            else:
                for _, row in results.iterrows():
                    render_lookup_card(row)

render_footer()