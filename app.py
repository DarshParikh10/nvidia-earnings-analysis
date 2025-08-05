import os
import re
from typing import List, Tuple

import pandas as pd
import streamlit as st

# Keep HF quiet on Windows-like envs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------- Streamlit page setup ----------
st.set_page_config(page_title="NVIDIA Earnings Call Analysis", layout="centered")
st.title("NVIDIA Earnings Call Analysis")
st.write(
    "Upload four transcripts (Q1–Q4) or use sample placeholders, then run the analysis. "
    "Results will display below and a CSV will be downloadable."
)

# ---------- Paths & constants ----------
TRANSCRIPTS_DIR = "transcripts"
CSV_PATH = "analysis.csv"

# ---------- Small, robust helpers ----------
def ensure_clean_transcripts_dir():
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    # Remove old files to avoid mixing runs
    for f in os.listdir(TRANSCRIPTS_DIR):
        try:
            os.remove(os.path.join(TRANSCRIPTS_DIR, f))
        except Exception:
            pass

def write_placeholder_quarter(q_name: str, text: str):
    with open(os.path.join(TRANSCRIPTS_DIR, f"{q_name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

def safe_chunk(s: str, limit: int) -> str:
    """Trim text to a max character limit (rough proxy for tokens)."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit]

def split_prepared_qa(text: str) -> Tuple[str, str]:
    """
    Try to split into prepared remarks vs Q&A using common markers.
    If not found, treat all as prepared remarks.
    """
    if not text:
        return "", ""
    markers = [
        r"Question[-\s]and[-\s]Answer Session",
        r"Q&A",
        r"Question and Answer",
        r"Questions and Answers",
    ]
    for m in markers:
        parts = re.split(m, text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return parts[0], parts[1]
    # fallback: try a "Prepared Remarks:" header, then rest
    prep_match = re.search(r"Prepared Remarks:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if prep_match:
        prepared = prep_match.group(0)
        rest = text.replace(prepared, "")
        return prepared, rest
    return text, ""

def compute_qoq_change(sent_scores: List[float]) -> List[str]:
    """Return QoQ arrows based on deltas (↑/↓/→)."""
    changes = ["—"]
    for i in range(1, len(sent_scores)):
        delta = sent_scores[i] - sent_scores[i - 1]
        if delta > 0.03:
            changes.append("↑")
        elif delta < -0.03:
            changes.append("↓")
        else:
            changes.append("→")
    return changes

# ---------- Cached model loading (fast & memory-light) ----------
# Using smaller models that work well on CPU on Streamlit Community Cloud.
@st.cache_resource(show_spinner=False)
def load_pipelines():
    # Imported here so Streamlit only loads them once per session.
    from transformers import pipeline
    import torch

    SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # light summarizer
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # 2-class sentiment

    DEVICE = 0 if torch.cuda.is_available() else -1  # -1 = CPU

    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        device=DEVICE,
    )
    sentiment_cls = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=DEVICE,
    )
    return summarizer, sentiment_cls

summarizer, sentiment_cls = load_pipelines()

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Return (label, score) using cached classifier; limit text for speed."""
    text = safe_chunk(text, 3000)  # keep small to avoid OOM
    if not text.strip():
        return "NEUTRAL", 0.0
    res = sentiment_cls(text)[0]
    label = res.get("label", "NEUTRAL").upper()
    score = float(res.get("score", 0.0))
    # Normalize labels to 3 classes
    if "NEG" in label:
        label = "NEGATIVE"
    elif "POS" in label:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"
    return label, score

def extract_focus(text: str) -> str:
    """
    Summarize prepared remarks into 3–5 short focus items.
    Keep inputs & outputs small to avoid memory spikes.
    """
    chunk = safe_chunk(text, 3200)  # conservative cap
    if not chunk:
        return "-"
    try:
        out = summarizer(chunk, max_length=110, min_length=40, do_sample=False)
        summary = out[0]["summary_text"].strip()
    except Exception:
        # Very long or odd input—fall back to a trimmed first paragraph
        first_para = chunk.split("\n\n")[0]
        return safe_chunk(first_para, 240)

    # Light cleanup to look like focus bullets
    parts = re.split(r";\s+|\. ", summary)
    parts = [p.strip(" -•") for p in parts if p and len(p) > 3]
    return " | ".join(parts[:5]) if parts else safe_chunk(summary, 240)

def run_analysis(q_texts: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    q_texts: list of (Qname, text) like [("Q4", "..."), ("Q3", "..."), ...]
    Returns a DataFrame with the analysis.
    """
    rows = []
    mgmt_scores = []
    for quarter, text in q_texts:
        prepared, qa = split_prepared_qa(text)
        mgmt_label, mgmt_score = analyze_sentiment(prepared)
        qa_label, qa_score = analyze_sentiment(qa if qa else prepared[-1000:])
        focus = extract_focus(prepared)
        rows.append([quarter, mgmt_label, round(mgmt_score, 3), qa_label, round(qa_score, 3), focus])
        mgmt_scores.append(mgmt_score)

    # Compute QoQ arrows
    arrows = compute_qoq_change(mgmt_scores)
    for i in range(len(rows)):
        rows[i].insert(3, arrows[i])

    df = pd.DataFrame(
        rows,
        columns=[
            "Quarter",
            "Mgmt Sentiment",
            "Mgmt Score",
            "QoQ Tone",
            "Q&A Sentiment",
            "Q&A Score",
            "Strategic Focuses",
        ],
    )
    return df

# ---------- Sample placeholders ----------
PLACEHOLDER = {
    "Q4": """Prepared Remarks:
We delivered strong growth in data center and AI platforms. Demand remained robust across hyperscalers and enterprise. Margins expanded and we executed well on our roadmap.

Question-and-Answer Session:
Q: Can you talk about demand visibility?
A: We continue to see broad-based strength in training and inference workloads.
""",
    "Q3": """Prepared Remarks:
We saw sustained acceleration in AI infrastructure and networking. Supply improved and we are ramping next-gen architectures. We returned capital to shareholders.

Question-and-Answer Session:
Q: How do you see pricing and mix?
A: Mix is favorable toward high-performance SKUs; pricing remains disciplined.
""",
    "Q2": """Prepared Remarks:
Enterprise adoption of AI is expanding. Partnerships and ecosystem momentum continue. We are investing in software stacks and platforms.

Question-and-Answer Session:
Q: What about competition?
A: The market is large and growing; we differentiate on performance and platform depth.
""",
    "Q1": """Prepared Remarks:
We executed well despite supply constraints earlier in the year. Gaming normalized while data center led overall growth. We are optimistic about the pipeline.

Question-and-Answer Session:
Q: How should we think about gross margin?
A: Efficiency and scale benefits should support healthy margins going forward.
"""
}

# ---------- Input UI ----------
st.subheader("Input transcripts")
use_samples = st.checkbox("Use sample placeholder transcripts (works instantly)", value=True)

st.write("**OR** upload your own four .txt files (named Q1/Q2/Q3/Q4 if possible):")
uploads = st.file_uploader(
    "Upload 4 transcript .txt files",
    type="txt",
    accept_multiple_files=True
)
st.caption("Tip: If filenames include Q1/Q2/Q3/Q4, I’ll map them automatically. Otherwise, I’ll map by upload order Q1→Q4.")

run = st.button("Run Analysis")

# ---------- Run ----------
if run:
    ensure_clean_transcripts_dir()

    # Collect Q texts
    q_texts: List[Tuple[str, str]] = []

    if use_samples:
        for q in ["Q4", "Q3", "Q2", "Q1"]:  # show most recent first
            text = PLACEHOLDER[q]
            write_placeholder_quarter(q, text)
            q_texts.append((q, text))
    else:
        if not uploads or len(uploads) != 4:
            st.error("Please upload exactly 4 .txt files or check 'Use sample placeholder transcripts'.")
            st.stop()

        # Map uploads to Q1..Q4 using filename hint, else by order
        mapping = {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        # filename-based mapping
        for file in uploads:
            name_upper = file.name.upper()
            for q in mapping.keys():
                if q in name_upper and mapping[q] is None:
                    mapping[q] = file
                    break
        # fill remaining by order
        remaining = [f for f in uploads if f not in mapping.values()]
        for q in mapping.keys():
            if mapping[q] is None and remaining:
                mapping[q] = remaining.pop(0)

        # Persist to /transcripts and build q_texts (Q4..Q1 order for display)
        for q in ["Q4", "Q3", "Q2", "Q1"]:
            f = mapping[q]
            content = f.read().decode("utf-8", errors="ignore")
            write_placeholder_quarter(q, content)
            q_texts.append((q, content))

    with st.status(
        "Running analysis (first run may download small models; this can take up to ~1–2 minutes)...",
        expanded=True
    ):
        try:
            df = run_analysis(q_texts)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # Save CSV (optional for local debugging)
    try:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    except Exception:
        # Ignore write errors in read-only environments
        pass

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysis.csv",
        data=csv_bytes,
        file_name="analysis.csv",
        mime="text/csv"
    )
    st.success("Done! If the app feels slow the very first time, it’s just downloading models; subsequent runs will be faster.")

    with st.expander("Why this version uses less memory"):
        st.markdown(
            "- Uses **distil** models (smaller, CPU-friendly)\n"
            "- **Caches** the models for the whole session\n"
            "- Trims very long inputs before summarization/sentiment\n"
            "- Avoids loading large frameworks repeatedly"
        )
