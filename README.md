# Uncovering Academic Profiles via Dimensionality Reduction

## Problem Statement
Raw academic grade averages often mask critical inter-disciplinary trade-offs—such as strong Humanities performance paired with STEM struggles, or high elective scores concealing core academic risk. Because rigid score cutoffs fail to capture holistic student learning patterns, academic advisors and curriculum designers need a data-driven way to visualize multi-subject performance.

This project applies unsupervised **Linear Dimensionality Reduction (PCA)** and **Non-Linear Manifold Learning (t-SNE / UMAP)** to 8-dimensional student grade vectors ($\mathbb{R}^8 \rightarrow \mathbb{R}^2$ and $\mathbb{R}^3$) across `math`, `science`, `cs`, `band`, `english`, `history`, `spanish`, and `physed`. By combining variance analysis, eigenvector interpretation, hyperparameter tuning, and K-Means clustering, this pipeline translates raw gradebooks into actionable advising and intervention strategies.

---

## Tech Stack
* **Language:** Python 3.10+
* **Data Manipulation & Matrix Algebra:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `PCA`, `TSNE`, `KMeans`), `umap-learn`
* **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Presentation Generation:** `python-pptx`
* **Development Environment:** Jupyter Notebook / Google Colab

---

## Repository Structure
```text
Dimentionality_Reduction_AcademicProfiles_TMP/
├── README.md                              # Project overview & key findings
├── student_grades.csv                     # Original dataset (100 students x 8 subjects)
├── 01_PCA_Analysis.ipynb                  # EDA, Standardization, 2D/3D PCA & Component Loadings
├── 02_tSNE_Analysis.ipynb                 # t-SNE Perplexity/LR Tuning, UMAP & PCA Benchmarking
├── generate_presentation.py               # Script to build the .pptx deck & export charts
├── Academic_Profiles_Presentation.pptx    # Final executive pitch deck
└── images/                                # High-resolution visualizations exported for reporting
```

---

## Dataset Overview
* **Records:** `100` students (`student_id` 1–100, isolated from modeling)
* **Features (8 numeric subjects):** `math`, `science`, `cs`, `band`, `english`, `history`, `spanish`, `physed`
* **Data Quality:** `0` missing values, `0` duplicate rows
* **Preprocessing:** Standardized via `StandardScaler` ($\mu = 0, \sigma = 1$) to equalize variance across wide-spread Humanities subjects ($\sigma \approx 17.3\text{–}18.4$) and tightly clustered Electives ($\sigma \approx 6.3\text{–}8.1$).

---

## Summary of Findings

### 1. Three Underlying Correlation Blocks
* **STEM Block (`math`, `science`, `cs`):** Pairwise correlations of **$0.97\text{–}0.98$**.
* **Humanities Block (`english`, `history`, `spanish`):** Pairwise correlations of **$0.96\text{–}0.97$**.
* **Electives Block (`band`, `physed`):** Strongly correlated with each other (**$0.80$**), yet nearly independent of core academics (**$-0.04$ to $+0.13$**).

### 2. PCA Captures 96.0% of Total Variance in 3 Dimensions
* **PC1 ($65.2\%$ variance) — Overall Core Academic Aptitude:** Equal positive loadings ($\approx +0.41$) across all 6 core subjects.
* **PC2 ($22.6\%$ variance) — Electives Performance:** Driven by `band` ($+0.706$) and `physed` ($+0.702$).
* **PC3 ($8.2\%$ variance) — STEM vs. Humanities Orientation:** Separates STEM ($-0.38$ to $-0.44$) from Humanities ($+0.40$ to $+0.41$).
* **Cumulative Coverage:** $87.8\%$ in 2D; **$96.0\%$ in 3D**.

### 3. Non-Linear t-SNE Outperforms Linear PCA in 2D Cluster Separation
* Systematic tuning across `perplexity` ($[5, 15, 20, 30, 45]$) and `learning_rate` ($[10, 50, 200, \text{"auto"}]$) identified **`perplexity=20, learning_rate="auto"`** as optimal ($\text{KL Divergence} = 0.264$).
* **Quantitative Benchmark (2D Silhouette Score):** **t-SNE ($0.499$)** outperforms **2D PCA PC1 vs. PC2 ($0.433$)** and **2D PCA PC1 vs. PC3 ($0.278$)**, cleanly isolating elite all-rounders and at-risk cohorts in a single 2D view.

### 4. Four Actionable Student Profiles (K-Means, $k=4$)

| Academic Profile | Share | STEM Avg | Humanities Avg | Electives Avg | Recommended Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **At-Risk Students** | **44%** ($n=44$) | $42.6$ | $46.7$ | $68.4$ | **Immediate Intervention & Targeted Tutoring:** Foundational Math, Science, and English support + bi-weekly advisor check-ins. |
| **Humanities-Focused** | **25%** ($n=25$) | $55.0$ | $74.1$ | $63.1$ | **Data-Driven Advising:** Humanities-track planning paired with targeted STEM bridge workshops. |
| **Humanities & Electives Leaders** | **22%** ($n=22$) | $58.0$ | $73.4$ | $76.3$ | **Engagement-Leveraged Support:** Channel high extracurricular engagement into peer-led STEM study groups. |
| **High-Achieving All-Rounders** | **9%** ($n=9$) | $86.1$ | $90.5$ | $68.4$ | **Enrichment & Mentorship:** Honors/AP placement, interdisciplinary projects, and peer-tutoring leadership roles. |

---

## How to Reproduce
1. Clone this repository and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly umap-learn python-pptx
   ```
2. Run **`01_PCA_Analysis.ipynb`** for EDA, standardization, 2D/3D PCA, and cluster profiling.
3. Run **`02_tSNE_Analysis.ipynb`** for t-SNE tuning, UMAP comparison, and PCA vs. t-SNE benchmarking.


