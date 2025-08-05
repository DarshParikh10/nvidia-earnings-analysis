import os
import glob
import re
from typing import List, Tuple
import nltk
from tabulate import tabulate

from transformers import pipeline

# Make sure tokenizer is available
nltk.download('punkt', quiet=True)

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
        r"Q&A",
        r"Question and Answer",
        r"Questions and Answers",
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
    """Trim text to token-friendly chunk length."""
    return s.strip()[:limit]

# ---------- Models ----------
# Smaller, faster models to keep CPU-friendly.
# If you've already downloaded BART large, you can switch model names back if you prefer.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
summarizer_pipe = pipeline("summarization", model=SUMMARIZER_MODEL)

def analyze_sentiment(text: str) -> Tuple[str, float]:
    text = safe_chunk(text, 1000)
    if not text:
        return "NEUTRAL", 0.0
    out = sentiment_pipe(text)[0]
    # Normalize label names a bit
    label = out.get("label", "").upper()
    score = float(out.get("score", 0.0))
    if "NEG" in label:
        label = "NEGATIVE"
    elif "POS" in label:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"
    return label, score

def extract_focus(text: str) -> str:
    """Produce a short bullet-style summary for strategic focuses."""
    chunk = safe_chunk(text, 3500)  # distilbart can take ~1024 tokens; this is a safe char cap.
    if not chunk:
        return "-"
    summary = summarizer_pipe(chunk, max_length=110, min_length=45, do_sample=False)[0]["summary_text"]
    # Light cleanup to look like focuses
    bullets = re.split(r";\s+|\. ", summary.strip())
    bullets = [b.strip("- •") for b in bullets if b and len(b) > 3]
    top = bullets[:5]
    return " | ".join(top)

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
        qa_label, qa_score = analyze_sentiment(qa if qa else prepared[-1000:])  # fallback
        focus = extract_focus(prepared)

        rows.append([quarter, mgmt_label, round(mgmt_score, 3), qa_label, round(qa_score, 3), focus])
        mgmt_scores.append(mgmt_score)

    # Compute QoQ arrows relative to prior row
    arrows = compute_qoq_change(mgmt_scores)
    for i in range(len(rows)):
        rows[i].insert(3, arrows[i])  # insert after Mgmt Sentiment

    headers = ["Quarter", "Mgmt Sentiment", "Mgmt Score", "QoQ Tone", "Q&A Sentiment", "Q&A Score", "Strategic Focuses"]
    print(tabulate(rows, headers=headers))

    # Save CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nSaved CSV -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
