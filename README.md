# Pakistan_National_Assembly_Discourse_NLP_Analysis
End-to-end NLP pipeline analyzing Pakistan National Assembly debates (2008–2023) to extract legislative priorities and map political discourse using transformer-based topic modeling.

# Legislative Priorities and Political Discourse in Pakistan 🇵🇰
**A Computational NLP Analysis of National Assembly Debates (2008–2023)**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)
![GovTech](https://img.shields.io/badge/Domain-GovTech%20%7C%20Public%20Policy-006400)

## 📌 TL;DR
This repository contains an end-to-end Machine Learning pipeline that parses, processes, and models 15 years of Pakistani parliamentary discourse. By leveraging transformer-based embeddings (`sentence-transformers`), dimensionality reduction (`UMAP`), and density-based clustering (`HDBSCAN`/`BERTopic`), the system extracts latent legislative priorities and public policy trends from raw National Assembly PDF transcripts. 

Built as a decision-support tool to bring quantitative rigor to GovTech and public policy analysis.

---

## 🏗️ System Architecture & Pipeline

The pipeline is entirely modularized to separate data ingestion from intensive computational modeling.

1. **Document Ingestion (`pdfplumber`):** Automated extraction of text from unstructured parliamentary PDFs.
2. **Text Normalization (`spacy`, `langdetect`):** Multilingual detection, noise reduction, and domain-specific stop-word removal.
3. **Semantic Embedding (`sentence-transformers`):** Projection of discourse into high-dimensional vector space using pre-trained semantic models.
4. **Dimensionality Reduction (`UMAP`):** Non-linear manifold learning to optimize vector space for density-based clustering while preserving local/global structures.
5. **Topic Modeling (`BERTopic` & `HDBSCAN`):** Unsupervised extraction of cohesive policy themes without hard-coding $k$-clusters.
6. **Projection & Visualization:** 2D semantic maps (t-SNE/UMAP) for stakeholder reporting.

---

## 📂 Repository Structure

```text
pak-national-assembly-nlp/
├── data/
│   ├── raw/               # Raw PDF transcripts (Ignored in Git)
│   └── processed/         # Cleaned JSON/CSV corpora
├── notebooks/
│   └── 01-eda-and-modeling.ipynb  # Primary execution & presentation layer
├── src/                   # Core Python package
│   ├── ingestion.py       # PDF parsing logic
│   ├── preprocessing.py   # Spacy pipelines and regex cleaners
│   └── modeling.py        # BERTopic and UMAP wrapper classes
├── models/                # Saved weights and UMAP state (Ignored in Git)
├── outputs/
│   └── figures/           # Rendered DPI-optimized visual assets
├── requirements.txt       # Pinned dependencies
├── .gitignore             # Strict exclusion of heavy assets
└── README.md
