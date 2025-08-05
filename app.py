import os
import io
import pandas as pd
import streamlit as st

# Optional: quiet down HF symlink warning on Windows-like envs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="NVIDIA Earnings Call Analysis", layout="centered")
st.title("NVIDIA Earnings Call Analysis")
st.write("Upload four transcripts (Q1–Q4) or use the sample placeholders, then run the analysis. Results will display below and a CSV will be downloadable.")

# --- Helpers ---
TRANSCRIPTS_DIR = "transcripts"
CSV_PATH = "analysis.csv"

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

# --- Input UI ---
st.subheader("Input transcripts")
use_samples = st.checkbox("Use sample placeholder transcripts (works instantly)", value=True)

st.write("**OR** upload your own four .txt files (named Q1/Q2/Q3/Q4 if possible):")
uploads = st.file_uploader(
    "Upload 4 transcript .txt files", type="txt", accept_multiple_files=True
)

st.caption("Tip: If filenames include Q1/Q2/Q3/Q4, I’ll map them automatically. Otherwise, I’ll map by upload order Q1→Q4.")

run = st.button("Run Analysis")

# --- Run ---
if run:
    # Prepare transcripts dir
    ensure_clean_transcripts_dir()

    if use_samples:
        # Write placeholders
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            write_placeholder_quarter(q, PLACEHOLDER[q])
    else:
        if not uploads or len(uploads) != 4:
            st.error("Please upload exactly 4 .txt files or check 'Use sample placeholder transcripts'.")
            st.stop()

        # Map uploads to Q1..Q4 using filename hint, else by order
        mapping = {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        # First pass: filename-based mapping
        for file in uploads:
            name_upper = file.name.upper()
            for q in mapping.keys():
                if q in name_upper and mapping[q] is None:
                    mapping[q] = file
                    break
        # Second pass: fill remaining by order
        remaining = [f for f in uploads if f not in mapping.values()]
        for q in mapping.keys():
            if mapping[q] is None and remaining:
                mapping[q] = remaining.pop(0)

        # Save to transcripts/Q*.txt
        for q, file in mapping.items():
            content = file.read().decode("utf-8", errors="ignore")
            with open(os.path.join(TRANSCRIPTS_DIR, f"{q}.txt"), "w", encoding="utf-8") as f:
                f.write(content)

    # Import and run your existing analyzer
    try:
        import nvidia_earnings  # your module
    except Exception as e:
        st.error(f"Could not import nvidia_earnings.py. Make sure it exists in the repo. Error: {e}")
        st.stop()

    with st.status("Running analysis (first run may download models; this can take a few minutes)...", expanded=True):
        try:
            # Call your script's main() which reads transcripts/ and writes analysis.csv
            nvidia_earnings.main()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # Load and show results
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analysis.csv",
                data=csv_bytes,
                file_name="analysis.csv",
                mime="text/csv"
            )
            st.success("Done!")
        except Exception as e:
            st.error(f"Could not read {CSV_PATH}: {e}")
    else:
        st.warning("analysis.csv was not found. Did the analyzer complete successfully?")
