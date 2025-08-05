import os
import glob
import re
import csv
from typing import List, Tuple

# Keep HF tokenizers from spawning extra threads (less RAM on Streamlit)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import pipeline
import torch
from tabulate import tabulate

TRANSCRIPTS_DIR = "transcripts"
OUTPUT_CSV = "analysis.csv"

# ---------- Utilities ----------
def read_transcripts(folder: str) -> List[Tuple[str, str]]:
    """
    Returns list of (quarter_name, text) from all .txt files in folder.
    Sorted by filename descending so Q4, Q3, Q2, Q1 prints in that order if named that way.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.txt")), reverse=True)
    data = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]  # e.g., Q4
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data.append((name, f.read()))
    return data

def split_prepared_qa(text: str) -> Tuple[str, str]:
    """
    Split transcript into prepared remarks vs Q&A using common markers.
    Falls back to heuristic if marker not found.
    """
    markers = [
        r"Question[-\s]and[-\s]Answer Session",
        r"Question\s*[-–]\s*Answer",
        r"Questions and Answers",
        r"Question and Answer",
        r"Q&A",
    ]
    for m in markers:
        parts = re.split(m, text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return parts[0], parts[1]
    # Fallback: try header 'Prepared Remarks:' then everything after as Q&A
    prep_match = re.search(r"Prepared Remarks:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if prep_match:
        prepared = prep_match.group(0)
        rest = text.replace(prepared, "")
        return prepared, rest
    # If no split found, treat all as prepared
    return text, ""

def safe_chunk(s: str, limit: int) -> str:
    """Trim text to a safe char length (rough proxy for tokens)."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit]

# ---------- Lightweight Models ----------
# Smaller, CPU-friendly models
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # ~300–400MB
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # ~250MB

DEVICE = 0 if torch.cuda.is_available() else -1

summarizer_pipe = pipeline(
    "summarization",
    model=SUMMARIZER_MODEL,
    device=DEVICE,
)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=DEVICE,
)

def summarize_text(txt: str, max_chars: int = 2000) -> str:
    """
    Short, conservative summary to avoid memory spikes.
    """
    chunk = safe_chunk(txt or "", max_chars)
    if not chunk:
        return ""
    try:
        out = summarizer_pipe(
            chunk,
            max_length=120,  # keep outputs tight
            min_length=32,
            do_sample=False,
        )
        return out[0]["summary_text"].strip()
    except Exception:
        # If the model chokes on something, return a trimmed snippet instead of failing
        return safe_chunk(chunk, 300)

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Return ("POSITIVE"/"NEGATIVE"/"NEUTRAL", confidence).
    DistilBERT SST-2 is binary; treat low confidence as NEUTRAL.
    """
    txt = safe_chunk(text or "", 1000)
    if not txt:
        return "NEUTRAL", 0.0
    try:
        res = sentiment_pipe(txt)[0]
        label = res.get("label", "").upper()
        score = float(res.get("score", 0.0))
        if score < 0.60:  # weak confidence -> call it neutral
            return "NEUTRAL", score
        # Map to consistent labels
        if "POS" in label:
            return "POSITIVE", score
        if "NEG" in label:
            return "NEGATIVE", score
        return "NEUTRAL", score
    except Exception:
        return "NEUTRAL", 0.0

def extract_focus(text: str) -> str:
    """
    Produce 3–5 short focus items from prepared remarks by summarizing,
    then splitting into concise fragments.
    """
    summary = summarize_text(text, max_chars=2400)
    if not summary:
        return "-"
    # Split into bullet-ish fragments, keep concise ones
    parts = re.split(r";\s+|\. |\n|\r", summary)
    parts = [p.strip("-•  ") for p in parts if p and len(p) > 3]
    # Prefer unique, shortish phrases
    seen = set()
    focuses = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        focuses.append(p)
        if len(focuses) >= 5:
            break
    return " | ".join(focuses) if focuses else "-"

def compute_qoq_change(sent_scores: List[float]) -> List[str]:
    """
    Given list of management sentiment scores [Q4, Q3, Q2, Q1],
    return arrows vs previous quarter.
    """
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

# ---------- Main ----------
def main():
    if not os.path.isdir(TRANSCRIPTS_DIR):
        print(f"Missing folder: {TRANSCRIPTS_DIR}. Create it and put .txt transcripts inside.")
        return

    items = read_transcripts(TRANSCRIPTS_DIR)
    if not items:
        print(f"No .txt files found in {TRANSCRIPTS_DIR}.")
        return

    rows = []
    mgmt_scores = []

    print("Analyzing local transcripts...\n")
    for quarter, text in items:
        prepared, qa = split_prepared_qa(text)
        mgmt_label, mgmt_score = analyze_sentiment(prepared)
        # Fallback to tail of prepared remarks if no Q&A section detected
        qa_label, qa_score = analyze_sentiment(qa if qa else prepared[-1000:])
        focus = extract_focus(prepared)

        rows.append([
            quarter,
            mgmt_label,
            round(mgmt_score, 3),
            # QoQ arrow inserted later
            qa_label,
            round(qa_score, 3),
            focus
        ])
        mgmt_scores.append(mgmt_score)

    # Compute QoQ arrows relative to prior row
    arrows = compute_qoq_change(mgmt_scores)
    for i in range(len(rows)):
        rows[i].insert(3, arrows[i])  # insert after Mgmt Score

    headers = [
        "Quarter",
        "Mgmt Sentiment",
        "Mgmt Score",
        "QoQ Tone",
        "Q&A Sentiment",
        "Q&A Score",
        "Strategic Focuses"
    ]
    print(tabulate(rows, headers=headers))

    # Save CSV safely
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nSaved CSV -> {OUTPUT_CSV}")
    except PermissionError:
        print(f"\n[Warn] Could not write {OUTPUT_CSV} (file open or permission). Close it and re-run.")

if __name__ == "__main__":
    main()

