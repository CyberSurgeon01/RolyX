# RoleSense — AI Career Matching

RoleSense is a Streamlit web app that helps job seekers match their skills against real job market data, estimate salaries from job descriptions, look up typical requirements for any role, and now — analyze a resume directly to find the best-fit jobs.

---

## Features

### 1. Resume Upload & Analysis (Module 00)
- Upload a resume in **PDF or DOCX** format.
- Automatically extracts:
  - **Hard skills** (matched against a skill vocabulary built from the job dataset)
  - **Soft skills** (communication, leadership, teamwork, etc.)
  - **Qualifications** (degrees, certifications)
- Instantly matches the resume against the job dataset using TF-IDF + cosine similarity and shows the **top matching roles** (e.g. Engineer, Developer, Data Scientist).

### 2. Find Matching Jobs (Module 01)
- Enter your skills (comma-separated) and optionally a desired job title.
- Filter by work type (Full-time, Part-time, Contract, Internship) and remote-only.
- Sort by best match or highest salary.
- Returns ranked job cards with salary, experience level, industry, required skills, and benefits.

### 3. Predict Salary & Industry (Module 02)
- Paste any job description.
- Get an estimated salary range (median, low, high).
- See the top 3 predicted industries with confidence scores.

### 4. Role Lookup (Module 03)
- Search any job title (e.g. "Data Scientist", "DevOps Engineer").
- View typical salary, required skills, soft skills, benefits, and full description for similar roles in the dataset.

---

## Tech Stack

- **Frontend/UI:** Streamlit (custom CSS for a dark, Notion/Linear/Stripe-inspired design)
- **ML/NLP:** scikit-learn (TF-IDF vectorization, cosine similarity, salary regression, industry classification)
- **Resume Parsing:** `pdfplumber` (PDF), `python-docx` (DOCX)
- **Data:** pandas, NumPy, SciPy sparse matrices

---

## Project Structure

```
JOB_INTELLIGENCE_APP/
├── app.py                     # Main Streamlit application
├── config.toml                # Streamlit theme configuration
└── models/
    ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer for job matching
    ├── tfidf_matrix.npz       # Precomputed TF-IDF matrix of job dataset
    ├── salary_model.pkl       # Trained salary prediction model
    ├── title_tfidf.pkl        # TF-IDF vectorizer for salary model input
    ├── le_exp.pkl             # Label encoder — experience level
    ├── le_wtype.pkl           # Label encoder — work type
    ├── cat_model.pkl          # Trained industry classification model
    ├── cat_tfidf.pkl          # TF-IDF vectorizer for industry model input
    ├── le_industry.pkl        # Label encoder — industry
    └── enriched_jobs.csv      # Job dataset (titles, skills, salary, benefits, etc.)
```

---

## Setup & Installation

### 1. Clone/download the project
Make sure `app.py`, `config.toml`, and the `models/` folder are all in the same directory.

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn scipy pdfplumber python-docx
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser, typically at `http://localhost:8501`.

---

## Requirements

- Python 3.9+
- The `models/` directory must be present next to `app.py` with all required `.pkl`, `.npz`, and `.csv` files listed above — the app will show an error and stop if any file is missing.

---

## Notes

- All salary and industry predictions are **estimates** based on patterns in the training dataset — they are directional guidance, not guarantees or job offers.
- Resume parsing works best with text-based PDFs/DOCX files. Scanned/image-based resumes without OCR may not extract text correctly.
- Uploaded resumes are processed in-memory for the current session only; nothing is persisted to disk.

---

## Version

Current version: **2.1**
