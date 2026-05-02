# Pakistan_National_Assembly_Discourse_NLP_Analysis
End-to-end NLP pipeline analyzing Pakistan National Assembly debates (2008–2023) to extract legislative priorities and map political discourse using transformer-based topic modeling.

# Legislative Priorities and Political Discourse in Pakistan 🇵🇰
**A Computational NLP Analysis of National Assembly Debates (2008–2023)**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)
![GovTech](https://img.shields.io/badge/Domain-GovTech%20%7C%20Public%20Policy-006400)

## 📌 Executive Summary
This repository contains an end-to-end Machine Learning pipeline that parses, processes, and models 15 years of Pakistani parliamentary discourse. By leveraging transformer-based embeddings (`all-MiniLM-L6-v2`), dimensionality reduction (`UMAP`), and density-based clustering (`HDBSCAN`/`BERTopic`), the system extracts latent legislative priorities and public policy trends from raw National Assembly PDF transcripts. 

Built as a decision-support tool to bring quantitative rigor to GovTech and public policy analysis.

---

## 📊 Dataset Statistics & Corpus Overview

The raw data consisted of unstructured, occasionally corrupted PDF transcripts from the National Assembly. After robust preprocessing, the final corpus metrics are:

* **Timeframe:** 3 Parliamentary Terms (2008–2013, 2013–2018, 2018–2023)
* **Total Pages Parsed:** 116,145 pages
* **Total Valid Documents:** 1,367 debate transcripts
* **Date Metadata Coverage:** 92.54% (Recovered via Regex pipelines)
* **Outlier / Noise Rate:** ~30% (Intentionally isolated via HDBSCAN to prevent forced clustering of procedural noise)

### Top Extracted Macro-Themes (by Document Frequency)
1. **Corruption & NAB:** 637 sessions
2. **Terrorism & Security:** 176 sessions
3. **CPEC & China Relations:** 140 sessions
4. **Afghanistan & Foreign Policy:** 33 sessions

---

## 🏗️ Methodology & Pipeline Architecture

The pipeline is completely modularized to handle the transition from unstructured government PDFs to structured semantic maps.

### 1. Data Ingestion & Preprocessing
* **Extraction:** Automated scraping and parsing of PDFs. Implemented error handling to bypass 475 corrupted/unreadable scanned image files.
* **Metadata Recovery:** Utilized regular expressions to successfully reconstruct missing date and session metadata for over 180 transcripts.
* **Multilingual Handling:** Assessed Urdu/English text ratios to isolate English segments, navigating heavy parliamentary code-switching.

### 2. Semantic Modeling & NLP
* **Vectorization:** Encoded cleaned text into a `(1364, 384)` dimensional vector space using the `sentence-transformers/all-MiniLM-L6-v2` model.
* **Topic Modeling (BERTopic):** * **UMAP:** Reduced 384D embeddings to optimize for density-based clustering while preserving local/global structures.
  * **HDBSCAN:** Grouped cohesive policy themes without hard-coding $k$-clusters. Evaluated models based on Silhouette scores and intra-topic coherence.
* **Named Entity Recognition (NER):** Extracted key political figures (e.g., Ayaz Sadiq, Faisal Karim Kundi, Asad Qaiser) to map speaker frequency to specific eras.
* **Sentiment Analysis:** Deployed HuggingFace transformers to map the emotional valence (Positive/Neutral/Negative) of debates regarding specific topics (e.g., Law & Justice, Infrastructure) across different parliamentary terms.

---

## 📂 Repository Structure

```text
pak-national-assembly-nlp/
├── data/
│   ├── raw/               # Raw PDF transcripts (Ignored in Git)
│   └── processed/         # Cleaned JSON/CSV corpora (1367 docs)
├── notebooks/
│   └── 01-eda-and-modeling.ipynb  # Primary execution & EDA layer
├── src/                   # Core Python package
│   ├── ingestion.py       # PDF parsing and regex metadata extraction
│   ├── preprocessing.py   # Code-switching handlers and text cleaners
│   └── modeling.py        # BERTopic, UMAP, and Sentiment wrappers
├── models/                # Saved transformer weights and UMAP state (Ignored)
├── outputs/
│   └── figures/           # Rendered DPI-optimized visual assets (UMAP plots, etc.)
├── requirements.txt       # Pinned dependencies
├── .gitignore             # Strict exclusion of heavy assets
└── README.md
