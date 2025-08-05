**GitHub Repository:** [https://github.com/DarshParikh10/nvidia-earnings-analysis](https://github.com/DarshParikh10/nvidia-earnings-analysis)  
**GitHub Profile:** [https://github.com/DarshParikh10](https://github.com/DarshParikh10)
**Live App:** [https://nvidia-earnings-analysis-jpr857mlc7h5dgzzlns64y.streamlit.app/](https://nvidia-earnings-analysis-jpr857mlc7h5dgzzlns64y.streamlit.app/)


# NVIDIA Earnings Call Analysis

## Purpose
This project reviews NVIDIA’s last four earnings call transcripts and provides:
- Management sentiment (tone from prepared remarks)
- Q&A sentiment (tone from analyst questions and answers)
- Quarter-to-quarter tone changes
- Main business focuses each quarter

It runs entirely in the terminal and saves the results in a CSV file for further review.

---

## How to Run (Locally)
1. **Set up Python environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or
   source venv/bin/activate   # Mac/Linux



2\. Install requirements



pip install -r requirements.txt



3\. Add transcripts



Q1.txt

Q2.txt

Q3.txt

Q4.txt



4\. Run the script



python nvidia\_earnings.py



5\. View results



The analysis table will appear in the terminal



A file named analysis.csv will be saved with the same data







nvidia\_earnings.py → Main script



transcripts/ → Transcript text files



analysis.csv → Output data



requirements.txt → Dependencies



README.md → Instructions



Example Output



Quarter    Mgmt. Sentiment  Mgmt. Score  QoQ Tone  Q\&A Sentiment  Q\&A Score  Strategic Focuses

Q4         POSITIVE        0.704       —         NEUTRAL        0.843      Data Center revenue reached $35.6B...

Q3         POSITIVE        0.749       ↑         NEUTRAL        0.701      Blackwell architecture enables 30x...

Q2         POSITIVE        0.610       ↓         NEUTRAL        0.522      Automotive revenue $350M...

Q1         POSITIVE        0.570       ↓         NEUTRAL        0.723      Data Center revenue $39.1B...



=======
# nvidia-earnings-analysis

