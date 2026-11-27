"""
MANIFOLD — AI-Powered Dimensionality Reduction & Clustering Platform
Complete 7-stage guided analytics workflow:
Overview -> Data Upload -> Preprocessing -> Clustering -> Visualization -> AI Insights -> Export
"""

from datetime import datetime
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ==========================================
# 1. PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="MANIFOLD — AI Analytics Platform",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = [
    "01 · Overview",
    "02 · Data Upload",
    "03 · Preprocess",
    "04 · Clustering",
    "05 · Visualizations",
    "06 · AI Insights",
    "07 · Export & Save",
]

DEFAULT_SUBJECTS = ["math", "science", "cs", "band", "english", "history", "spanish", "physed"]

CLUSTER_COLORS_DARK = [
    "#F43F5E",
    "#818CF8",
    "#34D399",
    "#FBBF24",
    "#38BDF8",
    "#C084FC",
    "#FB923C",
    "#F8FAFC",
]

CLUSTER_COLORS_LIGHT = [
    "#E11D48",
    "#4F46E5",
    "#059669",
    "#D97706",
    "#0284C7",
    "#9333EA",
    "#EA580C",
    "#0F172A",
]

ALGO_INFO = {
    "K-Means": {
        "summary": "Partitions records into k compact, spherical clusters by minimizing distance to each cluster centroid.",
        "best_for": "Well-separated numerical cohorts with roughly balanced sizes (recommended default).",
    },
    "DBSCAN": {
        "summary": "Groups densely packed points together and automatically flags isolated outliers as Noise (-1).",
        "best_for": "Irregularly shaped clusters and datasets containing anomalous or outlier records.",
    },
    "Hierarchical": {
        "summary": "Builds a bottom-up nested hierarchy of clusters by iteratively merging the closest pairs.",
        "best_for": "Small-to-medium datasets where nested subgroup structure matters.",
    },
}

THEMES = {
    "Dark": {
        "bg_root": "#07090f",
        "bg_surface": "#0f131f",
        "bg_elevated": "#161c2e",
        "bg_sidebar": "#0b0e17",
        "border": "rgba(255, 255, 255, 0.09)",
        "border_hover": "rgba(129, 140, 248, 0.45)",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "grid": "rgba(255, 255, 255, 0.06)",
        "zero": "rgba(255, 255, 255, 0.16)",
        "shadow": "0 8px 24px -6px rgba(0, 0, 0, 0.45)",
        "colors": CLUSTER_COLORS_DARK,
        "marker_edge": "#07090f",
        "vector_color": "rgba(248, 250, 252, 0.55)",
        "arrow_color": "#ffffff",
    },
    "Light": {
        "bg_root": "#f4f6fb",
        "bg_surface": "#ffffff",
        "bg_elevated": "#f8fafc",
        "bg_sidebar": "#ffffff",
        "border": "rgba(15, 23, 42, 0.10)",
        "border_hover": "rgba(79, 70, 229, 0.45)",
        "text_primary": "#0f172a",
        "text_secondary": "#334155",
        "text_muted": "#64748b",
        "grid": "rgba(15, 23, 42, 0.07)",
        "zero": "rgba(15, 23, 42, 0.18)",
        "shadow": "0 6px 20px -4px rgba(15, 23, 42, 0.08)",
        "colors": CLUSTER_COLORS_LIGHT,
        "marker_edge": "#ffffff",
        "vector_color": "rgba(15, 23, 42, 0.55)",
        "arrow_color": "#0f172a",
    },
}


# ==========================================
# 2. CACHED COMPUTATIONS & HELPERS
# ==========================================
@st.cache_data(show_spinner=False)
def load_sample_dataset():
    """Load the built-in student_grades.csv dataset."""
    return pd.read_csv("student_grades.csv")


@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_bytes, filename):
    """Parse uploaded CSV or Excel file safely."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(BytesIO(file_bytes))
    if ext in ("xlsx", "xls"):
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError(f"Unsupported file format '.{ext}'. Please upload a CSV or Excel (.xlsx) file.")


@st.cache_data(show_spinner=False)
def compute_embedding(x_matrix, method, n_comp, perp=20, lr="auto", n_neighbors=15, min_dist=0.1):
    """Compute cached 2D/3D PCA, t-SNE, or UMAP embedding and trustworthiness."""
    n_samples, n_features = x_matrix.shape
    eff_comp = min(n_comp, n_samples, n_features)
    if method == "PCA (Linear)":
        model = PCA(n_components=eff_comp, random_state=42)
        coords = model.fit_transform(x_matrix)
        evr = model.explained_variance_ratio_
        axes = [
            f"PC1 — Primary Variance Axis ({evr[0]*100:.1f}% var)",
            f"PC2 — Secondary Orthogonal Axis ({evr[1]*100:.1f}% var)" if eff_comp >= 2 else "",
            f"PC3 — Tertiary Contrast Axis ({evr[2]*100:.1f}% var)" if eff_comp >= 3 else "",
        ]
        loadings = model.components_[: min(2, eff_comp)].T
    elif method == "t-SNE (Non-Linear)":
        lr_val = "auto" if lr == "auto" else float(lr)
        safe_perp = min(perp, max(2, n_samples - 1))
        model = TSNE(n_components=eff_comp, perplexity=safe_perp, learning_rate=lr_val, init="pca", random_state=42)
        coords = model.fit_transform(x_matrix)
        axes = [
            "t-SNE Dim 1 — Local Neighborhood Axis",
            "t-SNE Dim 2 — Cluster Separation Axis" if eff_comp >= 2 else "",
            "t-SNE Dim 3 — Manifold Depth Axis" if eff_comp >= 3 else "",
        ]
        loadings = None
    else:
        safe_k = min(n_neighbors, max(2, n_samples - 1))
        model = umap.UMAP(n_neighbors=safe_k, min_dist=min_dist, n_components=eff_comp, random_state=42)
        coords = model.fit_transform(x_matrix)
        axes = [
            "UMAP Dim 1 — Global & Local Manifold Axis",
            "UMAP Dim 2 — Continuum Separation Axis" if eff_comp >= 2 else "",
            "UMAP Dim 3 — Manifold Depth Axis" if eff_comp >= 3 else "",
        ]
        loadings = None

    k_trust = min(15, max(1, n_samples // 2 - 1))
    trust = float(trustworthiness(x_matrix, coords, n_neighbors=k_trust)) if n_samples > 4 else 1.0
    return coords, axes, trust, loadings


def run_clustering_model(x_matrix, algo, params):
    """Fit K-Means, DBSCAN, or Hierarchical clustering and return labels + evaluation metrics."""
    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], random_state=42, n_init=10)
        labels = model.fit_predict(x_matrix)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        labels = model.fit_predict(x_matrix)
    else:
        model = AgglomerativeClustering(n_clusters=params["k"], linkage=params["linkage"])
        labels = model.fit_predict(x_matrix)

    unique_non_noise = [lbl for lbl in np.unique(labels) if lbl != -1]
    if len(unique_non_noise) >= 2 and len(unique_non_noise) < len(x_matrix):
        mask = labels != -1
        sil = float(silhouette_score(x_matrix[mask], labels[mask]))
        dbi = float(davies_bouldin_score(x_matrix[mask], labels[mask]))
    else:
        sil, dbi = None, None
    return labels, sil, dbi


def build_cluster_names(df_proc, labels, features):
    """Assign human-readable cluster names based on mean feature z-scores."""
    global_means = df_proc[features].mean()
    global_stds = df_proc[features].std().replace(0, 1)
    names = {}
    for cid in sorted(np.unique(labels)):
        if cid == -1:
            names[cid] = "Outliers / Noise"
            continue
        sub = df_proc.loc[labels == cid, features]
        z_diff = (sub.mean() - global_means) / global_stds
        top_feat = z_diff.idxmax()
        overall_z = z_diff.mean()
        if overall_z >= 0.75:
            tag = "High Performers"
        elif overall_z <= -0.55:
            tag = "At-Risk / Low"
        else:
            tag = f"{top_feat.capitalize()}-Oriented"
        names[cid] = f"Cluster {cid}: {tag}"
    return names


def generate_ai_insights(df_proc, labels, cluster_names, features, algo, sil, dbi):
    """Compute data-grounded insights, feature drivers, and recommendations."""
    n_total = len(df_proc)
    unique_ids = sorted(np.unique(labels))
    non_noise = [c for c in unique_ids if c != -1]
    noise_cnt = int((labels == -1).sum())

    # Identify top discriminating features via between-cluster centroid variance
    if len(non_noise) >= 2:
        z_df = (df_proc[features] - df_proc[features].mean()) / df_proc[features].std().replace(0, 1)
        z_df["_c"] = labels
        centroid_spread = z_df[z_df["_c"] != -1].groupby("_c")[features].mean().std().sort_values(ascending=False)
        top_drivers = centroid_spread.head(3).index.tolist()
    else:
        centroid_spread = pd.Series(dtype=float)
        top_drivers = features[:3]

    # Per-cluster profile details
    global_means = df_proc[features].mean()
    global_stds = df_proc[features].std().replace(0, 1)
    cluster_cards = []
    for cid in unique_ids:
        mask = labels == cid
        cnt = int(mask.sum())
        pct = cnt / n_total * 100
        sub_means = df_proc.loc[mask, features].mean()
        z_diff = ((sub_means - global_means) / global_stds).sort_values(ascending=False)
        strengths = [f"{f.upper()} (+{z_diff[f]:.2f}σ)" for f in z_diff.head(2).index if z_diff[f] > 0.1]
        weaknesses = [f"{f.upper()} ({z_diff[f]:.2f}σ)" for f in z_diff.tail(2).index if z_diff[f] < -0.1]
        cluster_cards.append(
            {
                "cid": cid,
                "name": cluster_names[cid],
                "count": cnt,
                "pct": pct,
                "strengths": strengths or ["Balanced near cohort average"],
                "weaknesses": weaknesses or ["No major below-average deficits"],
                "mean_overall": float(sub_means.mean()),
            }
        )

    # Quality interpretation
    if sil is None:
        sil_note = "Insufficient non-noise clusters (<2) to compute Silhouette Score."
    elif sil >= 0.45:
        sil_note = f"Strong separation ({sil:.3f}): clusters are compact and well-isolated."
    elif sil >= 0.25:
        sil_note = f"Moderate separation ({sil:.3f}): meaningful structure with slight boundary overlap."
    else:
        sil_note = f"Weak separation ({sil:.3f}): consider tuning k/eps or removing noisy features."

    if dbi is None:
        dbi_note = "Davies-Bouldin Index requires at least 2 clusters."
    elif dbi <= 1.0:
        dbi_note = f"Low Davies-Bouldin ({dbi:.3f}): tight intra-cluster density vs. inter-cluster distance."
    else:
        dbi_note = f"Moderate Davies-Bouldin ({dbi:.3f}): some clusters overlap in feature space."

    return {
        "n_clusters": len(non_noise),
        "noise_count": noise_cnt,
        "top_drivers": top_drivers,
        "cluster_cards": cluster_cards,
        "sil_note": sil_note,
        "dbi_note": dbi_note,
        "algo": algo,
    }


def generate_pdf_report(summary_dict, cluster_table_df, insights_dict):
    """Generate a clean multi-section PDF report using Matplotlib PdfPages."""
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")

        y = 0.96
        ax.text(0, y, "MANIFOLD — Clustering & Analytics Report", fontsize=18, fontweight="bold", color="#0f172a")
        y -= 0.035
        ax.text(0, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Dataset: {summary_dict['dataset_name']}", fontsize=10, color="#475569")
        y -= 0.04
        ax.plot([0, 1], [y, y], color="#cbd5e1", linewidth=1)
        y -= 0.04

        ax.text(0, y, "1. Pipeline Configuration & Evaluation", fontsize=13, fontweight="bold", color="#1e293b")
        y -= 0.03
        meta_lines = [
            f"Algorithm: {summary_dict['algo']}   |   Scaling: {summary_dict['scaling']}",
            f"Records Analyzed: {summary_dict['rows']}   |   Selected Features: {summary_dict['n_features']}",
            f"Clusters Discovered: {insights_dict['n_clusters']}   |   Noise Points: {insights_dict['noise_count']}",
            f"Silhouette Score: {summary_dict['sil']}   |   Davies-Bouldin Index: {summary_dict['dbi']}",
        ]
        for line in meta_lines:
            ax.text(0.02, y, line, fontsize=10, color="#334155")
            y -= 0.026

        y -= 0.02
        ax.text(0, y, "2. AI-Generated Insights", fontsize=13, fontweight="bold", color="#1e293b")
        y -= 0.03
        ax.text(0.02, y, f"• Separation: {insights_dict['sil_note']}", fontsize=9.5, color="#334155")
        y -= 0.024
        ax.text(0.02, y, f"• Compactness: {insights_dict['dbi_note']}", fontsize=9.5, color="#334155")
        y -= 0.024
        ax.text(0.02, y, f"• Top Discriminating Features: {', '.join(f.upper() for f in insights_dict['top_drivers'])}", fontsize=9.5, color="#334155")
        y -= 0.035

        ax.text(0, y, "3. Cluster Profile Breakdown", fontsize=13, fontweight="bold", color="#1e293b")
        y -= 0.03
        for c in insights_dict["cluster_cards"]:
            ax.text(0.02, y, f"{c['name']} — {c['count']} records ({c['pct']:.1f}%)", fontsize=10, fontweight="bold", color="#0f172a")
            y -= 0.022
            ax.text(0.04, y, f"Strengths: {', '.join(c['strengths'])}   |   Deficits: {', '.join(c['weaknesses'])}", fontsize=9, color="#475569")
            y -= 0.028

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Mean Feature Table
        if not cluster_table_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 5.5))
            ax2.axis("off")
            ax2.set_title("Cluster Mean Feature Summary", fontsize=14, fontweight="bold", pad=12)
            tbl = ax2.table(
                cellText=cluster_table_df.round(2).values,
                colLabels=cluster_table_df.columns,
                rowLabels=cluster_table_df.index,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1, 1.4)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
def init_session():
    """Initialize session state with a ready-to-explore default analysis."""
    if "stage_idx" not in st.session_state:
        st.session_state.stage_idx = 0
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    if "raw_df" not in st.session_state:
        sample_df = load_sample_dataset()
        st.session_state.raw_df = sample_df
        st.session_state.dataset_name = "student_grades.csv (Sample)"
        st.session_state.file_size_kb = 2.8
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = [c for c in DEFAULT_SUBJECTS if c in st.session_state.raw_df.columns]
    if "scaling_method" not in st.session_state:
        st.session_state.scaling_method = "StandardScaler (Z-Score)"
    if "missing_strategy" not in st.session_state:
        st.session_state.missing_strategy = "Drop rows with missing values"
    if "drop_duplicates" not in st.session_state:
        st.session_state.drop_duplicates = True
    if "proc_df" not in st.session_state:
        df_s = st.session_state.raw_df.copy()
        feats = st.session_state.selected_features
        x_s = StandardScaler().fit_transform(df_s[feats].values)
        labels, sil, dbi = run_clustering_model(x_s, "K-Means", {"k": 4})
        c_names = build_cluster_names(df_s, labels, feats)
        st.session_state.proc_df = df_s
        st.session_state.x_matrix = x_s
        st.session_state.scaler_obj = StandardScaler().fit(df_s[feats].values)
        st.session_state.cluster_labels = labels
        st.session_state.cluster_names = c_names
        st.session_state.sil_score = sil
        st.session_state.dbi_score = dbi
        st.session_state.algo_used = "K-Means"
        st.session_state.algo_params = {"k": 4}
        st.session_state.comparison_runs = [
            {
                "Run": "Run #1 (Baseline)",
                "Dataset": "student_grades.csv (Sample)",
                "Algorithm": "K-Means (k=4)",
                "Scaling": "StandardScaler",
                "Clusters": 4,
                "Silhouette": round(sil, 4),
                "Davies-Bouldin": round(dbi, 4),
            }
        ]
        st.session_state.saved_analyses = [
            {
                "name": "Student Academic Profiles (Baseline)",
                "timestamp": "Built-in Demo",
                "dataset": "student_grades.csv (Sample)",
                "rows": len(df_s),
                "features": len(feats),
                "algo": "K-Means (k=4)",
                "sil": round(sil, 4),
                "dbi": round(dbi, 4),
            }
        ]


init_session()


# ==========================================
# 4. SIDEBAR NAVIGATION & PROGRESS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; padding:4px 0 12px 0;">
            <span style="width:10px; height:10px; border-radius:50%; background:#ff2a3a; box-shadow:0 0 10px rgba(255,42,58,0.7); display:inline-block;"></span>
            <span style="font-family:'DotGothic16', monospace; font-size:1.3rem; letter-spacing:0.12em;">MANIFOLD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True,
    )
    st.session_state.theme_mode = theme_choice
    T = THEMES[theme_choice]

    st.markdown("---")
    stage_pct = int((st.session_state.stage_idx + 1) / len(STAGES) * 100)
    st.caption(f"WORKFLOW PROGRESS · STEP {st.session_state.stage_idx + 1} OF {len(STAGES)}")
    st.progress(stage_pct / 100)

    selected_stage = st.radio(
        "Journey Stages",
        STAGES,
        index=st.session_state.stage_idx,
        label_visibility="collapsed",
    )
    if STAGES.index(selected_stage) != st.session_state.stage_idx:
        st.session_state.stage_idx = STAGES.index(selected_stage)
        st.rerun()

    st.markdown("---")
    st.caption("ACTIVE DATASET")
    st.markdown(
        f"**{st.session_state.dataset_name}**  \n"
        f"{len(st.session_state.proc_df)} rows · {len(st.session_state.selected_features)} features  \n"
        f"Algorithm: `{st.session_state.algo_used}`"
    )


# ==========================================
# 5. GLOBAL THEME CSS
# ==========================================
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --bg-root: {T['bg_root']};
        --bg-surface: {T['bg_surface']};
        --bg-elevated: {T['bg_elevated']};
        --bg-sidebar: {T['bg_sidebar']};
        --border-subtle: {T['border']};
        --border-hover: {T['border_hover']};
        --text-primary: {T['text_primary']};
        --text-secondary: {T['text_secondary']};
        --text-muted: {T['text_muted']};
        --card-shadow: {T['shadow']};
        --ease-fluid: cubic-bezier(0.22, 1, 0.36, 1);
    }}

    #MainMenu, footer, [data-testid="stDecoration"] {{ display: none !important; }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: var(--bg-root) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}

    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}

    .block-container {{
        max-width: 1220px !important;
        padding: 1.6rem 2rem 3.5rem 2rem !important;
    }}

    @keyframes fadeSlideUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{ animation: fadeSlideUp 0.4s var(--ease-fluid) forwards; }}

    .hero-wrap {{
        margin-bottom: 1.5rem;
        padding-bottom: 1.15rem;
        border-bottom: 1px solid var(--border-subtle);
    }}
    .hero-eyebrow {{
        font-family: 'DotGothic16', monospace;
        font-size: 0.92rem;
        letter-spacing: 0.12em;
        color: #6366f1;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}
    .hero-h1 {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.15;
        color: var(--text-primary);
        margin: 0 0 0.45rem 0;
    }}
    .hero-p {{
        font-size: 1.02rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 780px;
        margin: 0;
    }}

    .bento {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        box-shadow: var(--card-shadow);
        transition: transform 0.28s var(--ease-fluid), border-color 0.28s ease, background-color 0.28s ease;
        margin-bottom: 0.75rem;
    }}
    .bento:hover {{
        border-color: var(--border-hover);
        background: var(--bg-elevated);
        transform: translateY(-2px);
    }}
    .bento-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .bento-num {{
        font-family: 'DotGothic16', monospace;
        font-size: 2.2rem;
        line-height: 1.1;
        color: var(--text-primary);
    }}
    .bento-meta {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.35rem;
        line-height: 1.45;
    }}
    .dot-ind {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }}

    .legend-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 12px;
    }}
    .legend-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 12px 15px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        box-shadow: var(--card-shadow);
    }}
    .legend-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }}
    .legend-desc {{
        font-size: 0.86rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }}

    [data-testid="stPlotlyChart"] {{
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 12px 14px 6px 14px !important;
        box-shadow: var(--card-shadow) !important;
    }}

    .sec-title {{
        font-family: 'DotGothic16', monospace;
        font-size: 1.18rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin: 0.75rem 0;
    }}
    .hairline {{
        height: 1px;
        background: var(--border-subtle);
        margin: 1.25rem 0;
    }}

    @media (max-width: 1024px) {{
        .block-container {{ padding: 1.2rem 1.1rem 2.5rem 1.1rem !important; }}
        .hero-h1 {{ font-size: 1.75rem; }}
    }}
    @media (max-width: 640px) {{
        .legend-grid {{ grid-template-columns: 1fr; }}
        .hero-h1 {{ font-size: 1.45rem; }}
        .bento-num {{ font-size: 1.8rem; }}
    }}
</style>
""",
    unsafe_allow_html=True,
)


def render_step_footer(current_idx):
    """Render consistent Back / Save Progress / Next navigation controls."""
    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 1.5, 1])
    with b1:
        if current_idx > 0:
            if st.button("← Back", use_container_width=True, key=f"back_{current_idx}"):
                st.session_state.stage_idx = current_idx - 1
                st.rerun()
    with b2:
        if st.button("Save Progress to History", use_container_width=True, key=f"save_{current_idx}"):
            sil_v = st.session_state.sil_score
            dbi_v = st.session_state.dbi_score
            st.session_state.saved_analyses.insert(
                0,
                {
                    "name": f"{st.session_state.dataset_name} ({st.session_state.algo_used})",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "dataset": st.session_state.dataset_name,
                    "rows": len(st.session_state.proc_df),
                    "features": len(st.session_state.selected_features),
                    "algo": st.session_state.algo_used,
                    "sil": round(sil_v, 4) if sil_v is not None else "N/A",
                    "dbi": round(dbi_v, 4) if dbi_v is not None else "N/A",
                },
            )
            st.toast("Analysis snapshot saved to Recent Analyses!")
    with b3:
        if current_idx < len(STAGES) - 1:
            if st.button("Next Stage →", type="primary", use_container_width=True, key=f"next_{current_idx}"):
                st.session_state.stage_idx = current_idx + 1
                st.rerun()


# ==========================================
# STAGE 1: DASHBOARD OVERVIEW
# ==========================================
if st.session_state.stage_idx == 0:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 01 // WORKSPACE OVERVIEW</div>
        <h1 class="hero-h1">AI-Powered Clustering &amp; Manifold Analytics</h1>
        <p class="hero-p">
            Move seamlessly from raw CSV or Excel data to preprocessed features, unsupervised clustering,
            interactive 2D/3D embeddings, automated insights, and exportable reports.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    n_clusters_active = len([c for c in np.unique(st.session_state.cluster_labels) if c != -1])
    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (k1, "DATASETS LOADED", str(len({a["dataset"] for a in st.session_state.saved_analyses})), "Active + Saved"),
        (k2, "ACTIVE RECORDS", str(len(st.session_state.proc_df)), st.session_state.dataset_name),
        (k3, "FEATURES", str(len(st.session_state.selected_features)), "Selected numeric cols"),
        (k4, "CLUSTERS", str(n_clusters_active), st.session_state.algo_used),
        (k5, "ANALYSES RUN", str(len(st.session_state.comparison_runs)), "Session runs"),
    ]
    for col, lbl, val, meta in cards:
        with col:
            st.markdown(
                f"""
            <div class="bento fade-in">
                <div class="bento-label">{lbl}</div>
                <div class="bento-num">{val}</div>
                <div class="bento-meta">{meta}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    cta1, cta2, cta3 = st.columns([1.4, 1.4, 1.2])
    with cta1:
        if st.button("Start New Analysis →", type="primary", use_container_width=True):
            st.session_state.stage_idx = 1
            st.rerun()
    with cta2:
        if st.button("Explore Active Visualizations", use_container_width=True):
            st.session_state.stage_idx = 4
            st.rerun()
    with cta3:
        if st.button("View AI Insights", use_container_width=True):
            st.session_state.stage_idx = 5
            st.rerun()

    st.markdown('<div class="sec-title">RECENT ANALYSES &amp; SAVED SNAPSHOTS</div>', unsafe_allow_html=True)
    recent_df = pd.DataFrame(st.session_state.saved_analyses)
    st.dataframe(recent_df, use_container_width=True, hide_index=True)


# ==========================================
# STAGE 2: DATA UPLOAD
# ==========================================
elif st.session_state.stage_idx == 1:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 02 // DATA INGESTION</div>
        <h1 class="hero-h1">Upload your dataset or use the built-in sample.</h1>
        <p class="hero-p">
            Drag and drop a CSV or Excel (.xlsx) file to inspect dimensions, schema, and sample rows.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    up_col, sample_col = st.columns([2, 1], gap="large")
    with up_col:
        uploaded = st.file_uploader(
            "Drag and drop CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: .csv, .xlsx, .xls",
        )
        if uploaded is not None:
            try:
                with st.spinner("Validating and parsing dataset..."):
                    df_new = parse_uploaded_file(uploaded.getvalue(), uploaded.name)
                if df_new.empty or df_new.shape[1] < 2:
                    st.error("Uploaded file must contain at least 1 row and 2 columns.")
                else:
                    num_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
                    if len(num_cols) < 2:
                        st.error("At least 2 numeric columns are required for dimensionality reduction and clustering.")
                    else:
                        st.session_state.raw_df = df_new
                        st.session_state.dataset_name = uploaded.name
                        st.session_state.file_size_kb = round(len(uploaded.getvalue()) / 1024, 2)
                        default_feats = [c for c in num_cols if "id" not in c.lower()]
                        st.session_state.selected_features = default_feats if len(default_feats) >= 2 else num_cols
                        st.success(f"Loaded `{uploaded.name}` ({df_new.shape[0]} rows × {df_new.shape[1]} columns).")
            except ImportError:
                st.error("Excel support requires `openpyxl`. Install it via `pip install openpyxl` or upload a `.csv` file.")
            except Exception as exc:
                st.error(f"Invalid or unreadable file: {exc}")

    with sample_col:
        st.markdown(
            """
        <div class="bento">
            <div class="bento-label">BUILT-IN DATASET</div>
            <div class="bento-meta">Load the 100-student, 8-subject academic gradebook (`student_grades.csv`).</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Reset to Sample Dataset", use_container_width=True):
            sample_df = load_sample_dataset()
            st.session_state.raw_df = sample_df
            st.session_state.dataset_name = "student_grades.csv (Sample)"
            st.session_state.file_size_kb = 2.8
            st.session_state.selected_features = [c for c in DEFAULT_SUBJECTS if c in sample_df.columns]
            st.success("Loaded built-in `student_grades.csv`.")

    raw = st.session_state.raw_df
    i1, i2, i3, i4 = st.columns(4)
    for col, lbl, val in [
        (i1, "FILENAME", st.session_state.dataset_name),
        (i2, "ROW COUNT", f"{raw.shape[0]:,}"),
        (i3, "COLUMN COUNT", f"{raw.shape[1]:,}"),
        (i4, "FILE SIZE", f"{st.session_state.file_size_kb} KB"),
    ]:
        with col:
            st.markdown(
                f'<div class="bento"><div class="bento-label">{lbl}</div><div class="bento-num" style="font-size:1.5rem;">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec-title">DATASET PREVIEW (FIRST 20 ROWS)</div>', unsafe_allow_html=True)
    st.dataframe(raw.head(20), use_container_width=True)
    render_step_footer(1)


# ==========================================
# STAGE 3: DATA PREPROCESSING
# ==========================================
elif st.session_state.stage_idx == 2:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 03 // DATA PREPROCESSING</div>
        <h1 class="hero-h1">Audit quality, select features, and scale values.</h1>
        <p class="hero-p">
            Inspect missing values, duplicates, and column types before standardizing features for clustering.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    raw = st.session_state.raw_df
    missing_total = int(raw.isnull().sum().sum())
    dup_total = int(raw.duplicated().sum())
    num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in raw.columns if c not in num_cols]

    q1, q2, q3, q4 = st.columns(4)
    for col, lbl, val in [
        (q1, "MISSING VALUES", str(missing_total)),
        (q2, "DUPLICATE ROWS", str(dup_total)),
        (q3, "NUMERIC COLUMNS", str(len(num_cols))),
        (q4, "NON-NUMERIC COLS", str(len(cat_cols))),
    ]:
        with col:
            st.markdown(f'<div class="bento"><div class="bento-label">{lbl}</div><div class="bento-num">{val}</div></div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.4, 1], gap="large")
    with c_left:
        feats = st.multiselect(
            "Select Numeric Features for Analysis (minimum 2)",
            options=num_cols,
            default=[f for f in st.session_state.selected_features if f in num_cols] or num_cols[:2],
        )
        missing_opt = st.selectbox(
            "Missing Value Handling",
            ["Drop rows with missing values", "Mean imputation", "Median imputation"],
            index=["Drop rows with missing values", "Mean imputation", "Median imputation"].index(st.session_state.missing_strategy),
        )
        scale_opt = st.selectbox(
            "Feature Scaling",
            ["StandardScaler (Z-Score)", "MinMaxScaler (0–1)", "None (Raw Values)"],
            index=["StandardScaler (Z-Score)", "MinMaxScaler (0–1)", "None (Raw Values)"].index(st.session_state.scaling_method),
            help="StandardScaler centers features to mean=0 and std=1 so high-variance columns do not dominate distances.",
        )
        drop_dups = st.checkbox("Remove duplicate rows", value=st.session_state.drop_duplicates)

    with c_right:
        dtype_df = pd.DataFrame(
            {
                "Column": raw.columns,
                "Dtype": [str(t) for t in raw.dtypes],
                "Missing": raw.isnull().sum().values,
            }
        )
        st.markdown('<div class="sec-title">SCHEMA &amp; DATA TYPES</div>', unsafe_allow_html=True)
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=260)

    if len(feats) < 2:
        st.warning("Please select at least 2 numeric features to continue.")
    else:
        df_clean = raw.copy()
        if drop_dups:
            df_clean = df_clean.drop_duplicates()
        if missing_opt == "Drop rows with missing values":
            df_clean = df_clean.dropna(subset=feats)
        elif missing_opt == "Mean imputation":
            df_clean[feats] = df_clean[feats].fillna(df_clean[feats].mean())
        else:
            df_clean[feats] = df_clean[feats].fillna(df_clean[feats].median())
        df_clean = df_clean.reset_index(drop=True)

        if scale_opt == "StandardScaler (Z-Score)":
            scaler_obj = StandardScaler()
            x_mat = scaler_obj.fit_transform(df_clean[feats].values)
        elif scale_opt == "MinMaxScaler (0–1)":
            scaler_obj = MinMaxScaler()
            x_mat = scaler_obj.fit_transform(df_clean[feats].values)
        else:
            scaler_obj = None
            x_mat = df_clean[feats].values.astype(float)

        st.markdown(
            f"""
        <div class="bento">
            <div class="bento-label">PREPROCESSING SUMMARY</div>
            <div class="bento-meta">
                Records: <strong>{len(raw)} → {len(df_clean)}</strong> &nbsp;·&nbsp;
                Features: <strong>{len(feats)}</strong> ({', '.join(feats)}) &nbsp;·&nbsp;
                Scaling: <strong>{scale_opt}</strong>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Continue to Analysis →", type="primary", use_container_width=True):
            st.session_state.selected_features = feats
            st.session_state.missing_strategy = missing_opt
            st.session_state.scaling_method = scale_opt
            st.session_state.drop_duplicates = drop_dups
            st.session_state.proc_df = df_clean
            st.session_state.x_matrix = x_mat
            st.session_state.scaler_obj = scaler_obj
            st.session_state.stage_idx = 3
            st.rerun()

    render_step_footer(2)


# ==========================================
# STAGE 4: CLUSTERING CONFIGURATION
# ==========================================
elif st.session_state.stage_idx == 3:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 04 // CLUSTERING CONFIGURATION</div>
        <h1 class="hero-h1">Select a clustering algorithm and tune parameters.</h1>
        <p class="hero-p">
            Compare K-Means, DBSCAN, and Hierarchical Clustering using recommended defaults or custom parameters.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cfg_l, cfg_r = st.columns([1.2, 1], gap="large")
    with cfg_l:
        algo = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical"],
            index=["K-Means", "DBSCAN", "Hierarchical"].index(st.session_state.algo_used),
        )
        params = {}
        if algo == "K-Means":
            params["k"] = st.slider("Number of Clusters (k)", 2, 10, st.session_state.algo_params.get("k", 4), help="Recommended default for student_grades.csv: k = 4.")
            param_label = f"K-Means (k={params['k']})"
        elif algo == "DBSCAN":
            params["eps"] = st.slider("Neighborhood Radius (eps)", 0.2, 3.5, float(st.session_state.algo_params.get("eps", 1.2)), 0.1, help="Maximum distance between two samples to be considered neighbors.")
            params["min_samples"] = st.slider("Minimum Samples per Core Point", 2, 15, int(st.session_state.algo_params.get("min_samples", 4)), help="Minimum points required to form a dense region.")
            param_label = f"DBSCAN (eps={params['eps']}, min={params['min_samples']})"
        else:
            params["k"] = st.slider("Number of Clusters (k)", 2, 10, st.session_state.algo_params.get("k", 4))
            params["linkage"] = st.selectbox("Linkage Criterion", ["ward", "complete", "average"], help="'ward' minimizes intra-cluster variance.")
            param_label = f"Hierarchical (k={params['k']}, {params['linkage']})"

        if st.button("Run Clustering →", type="primary", use_container_width=True):
            with st.spinner(f"Running {param_label}..."):
                x_mat = st.session_state.x_matrix
                df_proc = st.session_state.proc_df
                feats = st.session_state.selected_features
                labels, sil, dbi = run_clustering_model(x_mat, algo, params)
                c_names = build_cluster_names(df_proc, labels, feats)

                st.session_state.cluster_labels = labels
                st.session_state.cluster_names = c_names
                st.session_state.sil_score = sil
                st.session_state.dbi_score = dbi
                st.session_state.algo_used = algo
                st.session_state.algo_params = params

                n_found = len([c for c in np.unique(labels) if c != -1])
                st.session_state.comparison_runs.append(
                    {
                        "Run": f"Run #{len(st.session_state.comparison_runs) + 1}",
                        "Dataset": st.session_state.dataset_name,
                        "Algorithm": param_label,
                        "Scaling": st.session_state.scaling_method.split()[0],
                        "Clusters": n_found,
                        "Silhouette": round(sil, 4) if sil is not None else "N/A",
                        "Davies-Bouldin": round(dbi, 4) if dbi is not None else "N/A",
                    }
                )
                st.session_state.stage_idx = 4
                st.rerun()

    with cfg_r:
        info = ALGO_INFO[algo]
        st.markdown(
            f"""
        <div class="bento">
            <div class="bento-label">HOW {algo.upper()} WORKS</div>
            <div class="bento-meta">{info['summary']}</div>
            <div class="hairline"></div>
            <div class="bento-label">RECOMMENDED USE CASE</div>
            <div class="bento-meta">{info['best_for']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    render_step_footer(3)


# ==========================================
# STAGE 5: VISUALIZATION & ANALYSIS
# ==========================================
elif st.session_state.stage_idx == 4:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 05 // VISUALIZATION &amp; ANALYSIS</div>
        <h1 class="hero-h1">Explore interactive PCA, t-SNE, and UMAP manifolds.</h1>
        <p class="hero-p">
            Inspect cluster separation, evaluation metrics, feature profiles, and compare clustering runs side-by-side.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_proc = st.session_state.proc_df.copy()
    x_mat = st.session_state.x_matrix
    feats = st.session_state.selected_features
    labels = st.session_state.cluster_labels
    c_names = st.session_state.cluster_names
    df_proc["_cluster_id"] = labels
    df_proc["Cluster_Label"] = [c_names[c] for c in labels]

    unique_labels_sorted = sorted(np.unique(labels))
    color_map = {
        c_names[cid]: ("#71717a" if cid == -1 else T["colors"][idx % len(T["colors"])])
        for idx, cid in enumerate(unique_labels_sorted)
    }

    tab_manifold, tab_compare, tab_sim = st.tabs(["Manifold Explorer", "Cluster Comparison & Profiles", "What-If Simulator"])

    with tab_manifold:
        m_col, s_col = st.columns([2.4, 1], gap="large")
        with m_col:
            alg_opts = ["PCA (Linear)", "t-SNE (Non-Linear)"] + (["UMAP (Manifold)"] if HAS_UMAP else [])
            v1, v2, v3 = st.columns([1.2, 1.2, 0.9])
            with v1:
                emb_method = st.selectbox("Projection", alg_opts)
            with v2:
                color_opt = st.selectbox("Color By", ["Cluster"] + feats)
            with v3:
                dims = st.selectbox("Dimensions", ["2D Plane", "3D Space"])

            n_comp = 3 if dims == "3D Space" else 2
            perp, lr_c, u_nb, u_md = 20, "auto", 15, 0.1
            if emb_method == "t-SNE (Non-Linear)":
                tc1, tc2 = st.columns(2)
                perp = tc1.slider("Perplexity", 5, 45, 20, 5)
                lr_c = tc2.selectbox("Learning Rate", ["auto", "50", "200"])
            elif emb_method == "UMAP (Manifold)":
                uc1, uc2 = st.columns(2)
                u_nb = uc1.slider("n_neighbors", 5, 45, 15, 5)
                u_md = uc2.slider("min_dist", 0.05, 0.50, 0.10, 0.05)

            active_clusters = st.multiselect(
                "Filter Visible Clusters",
                options=list(color_map.keys()),
                default=list(color_map.keys()),
            )

            with st.spinner("Rendering projection..."):
                coords, axes_lbls, trust_val, loadings = compute_embedding(
                    x_mat, emb_method, n_comp, perp, lr_c, u_nb, u_md
                )

            df_proc["x"] = coords[:, 0]
            df_proc["y"] = coords[:, 1]
            if n_comp == 3 and coords.shape[1] >= 3:
                df_proc["z"] = coords[:, 2]

            vis_df = df_proc[df_proc["Cluster_Label"].isin(active_clusters)]
            x_lbl, y_lbl, z_lbl = axes_lbls

            if n_comp == 2:
                if color_opt == "Cluster":
                    fig = px.scatter(
                        vis_df, x="x", y="y", color="Cluster_Label",
                        color_discrete_map=color_map,
                        hover_data=feats[:5],
                    )
                else:
                    fig = px.scatter(
                        vis_df, x="x", y="y", color=color_opt,
                        color_continuous_scale="Plasma" if st.session_state.theme_mode == "Dark" else "Viridis",
                        hover_data=["Cluster_Label"] + feats[:4],
                    )
                fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color=T["marker_edge"])))

                if emb_method == "PCA (Linear)" and loadings is not None and loadings.shape[1] >= 2:
                    if st.checkbox("Overlay PCA Feature Vectors", value=False):
                        vecs = loadings * 4.0
                        for i, fname in enumerate(feats):
                            vx, vy = float(vecs[i, 0]), float(vecs[i, 1])
                            fig.add_shape(type="line", x0=0, y0=0, x1=vx, y1=vy, line=dict(color=T["vector_color"], width=1.5))
                            fig.add_annotation(x=vx * 1.12, y=vy * 1.12, text=fname.upper(), showarrow=False, font=dict(family="JetBrains Mono", size=11, color=T["text_primary"]))
            else:
                fig = px.scatter_3d(
                    vis_df, x="x", y="y", z="z",
                    color="Cluster_Label" if color_opt == "Cluster" else color_opt,
                    color_discrete_map=color_map if color_opt == "Cluster" else None,
                    hover_data=feats[:4],
                )
                fig.update_traces(marker=dict(size=6, opacity=0.9, line=dict(width=0.6, color=T["marker_edge"])))

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=13),
                height=500,
                margin=dict(l=20, r=20, t=38, b=20),
                xaxis=dict(title=dict(text=x_lbl, font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12, color=T["text_muted"]), showgrid=True, gridcolor=T["grid"], zerolinecolor=T["zero"]),
                yaxis=dict(title=dict(text=y_lbl, font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12, color=T["text_muted"]), showgrid=True, gridcolor=T["grid"], zerolinecolor=T["zero"]),
                scene=dict(
                    bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title=dict(text=x_lbl, font=dict(size=13)), showbackground=False, gridcolor=T["grid"]),
                    yaxis=dict(title=dict(text=y_lbl, font=dict(size=13)), showbackground=False, gridcolor=T["grid"]),
                    zaxis=dict(title=dict(text=z_lbl, font=dict(size=13)), showbackground=False, gridcolor=T["grid"]),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title=None, font=dict(size=12, color=T["text_primary"])),
            )
            st.session_state.last_fig_html = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

            # Cluster Legend & Meaning
            cards_html = ""
            for cid in unique_labels_sorted:
                cname = c_names[cid]
                cnt = int((labels == cid).sum())
                pct = cnt / len(df_proc) * 100
                cards_html += (
                    f'<div class="legend-card">'
                    f'<span class="dot-ind" style="background:{color_map[cname]}; margin-top:6px;"></span>'
                    f'<div><div class="legend-title">{cname}</div>'
                    f'<div class="legend-desc">{cnt} records ({pct:.1f}%)</div></div></div>'
                )
            st.markdown(f'<div class="legend-grid">{cards_html}</div>', unsafe_allow_html=True)

        with s_col:
            sil_disp = f"{st.session_state.sil_score:.3f}" if st.session_state.sil_score is not None else "N/A"
            dbi_disp = f"{st.session_state.dbi_score:.3f}" if st.session_state.dbi_score is not None else "N/A"
            st.markdown(
                f"""
            <div class="bento">
                <div class="bento-label">SILHOUETTE SCORE</div>
                <div class="bento-num">{sil_disp}</div>
                <div class="bento-meta">Range [-1, 1] · Higher = sharper cluster separation</div>
            </div>
            <div class="bento">
                <div class="bento-label">DAVIES-BOULDIN INDEX</div>
                <div class="bento-num">{dbi_disp}</div>
                <div class="bento-meta">Lower = tighter, better separated clusters</div>
            </div>
            <div class="bento">
                <div class="bento-label">TRUSTWORTHINESS</div>
                <div class="bento-num">{trust_val:.3f}</div>
                <div class="bento-meta">Neighborhood preservation in {emb_method.split()[0]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab_compare:
        st.markdown('<div class="sec-title">CLUSTERING RUN COMPARISON</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.comparison_runs), use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-title">MEAN FEATURE PROFILE BY CLUSTER</div>', unsafe_allow_html=True)
        means_tbl = df_proc.groupby("Cluster_Label")[feats].mean().round(2)
        st.dataframe(means_tbl, use_container_width=True)

        melted = means_tbl.reset_index().melt(id_vars="Cluster_Label", var_name="Feature", value_name="Mean Value")
        fig_prof = px.line(
            melted, x="Feature", y="Mean Value", color="Cluster_Label",
            markers=True, color_discrete_map=color_map,
        )
        fig_prof.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=13),
            height=380,
            xaxis=dict(title=dict(text="Feature", font=dict(size=14, color=T["text_primary"])), showgrid=False),
            yaxis=dict(title=dict(text="Mean Value", font=dict(size=14, color=T["text_primary"])), showgrid=True, gridcolor=T["grid"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title=None),
        )
        st.plotly_chart(fig_prof, use_container_width=True, config={"displayModeBar": False})

    with tab_sim:
        st.markdown('<div class="sec-title">COUNTERFACTUAL RECORD SIMULATOR</div>', unsafe_allow_html=True)
        if st.session_state.algo_used != "K-Means":
            st.info("Real-time nearest-centroid prediction is supported for K-Means. Switch to K-Means in Step 4 to simulate cluster transitions.")
        else:
            sim_l, sim_r = st.columns([1, 1.8], gap="large")
            with sim_l:
                row_idx = st.selectbox("Select Record Index", list(range(len(df_proc))))
                rec = df_proc.iloc[row_idx]
                st.markdown(
                    f'<div class="bento"><div class="bento-label">CURRENT ASSIGNMENT</div><div class="bento-num" style="font-size:1.5rem;">{rec["Cluster_Label"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with sim_r:
                sliders_vals = []
                sc1, sc2 = st.columns(2)
                for i, f in enumerate(feats):
                    col_min = float(df_proc[f].min())
                    col_max = float(df_proc[f].max())
                    if col_min == col_max:
                        col_max = col_min + 1.0
                    target_col = sc1 if i % 2 == 0 else sc2
                    sliders_vals.append(target_col.slider(f.upper(), col_min, col_max, float(rec[f])))

                vec = np.array(sliders_vals).reshape(1, -1)
                vec_scaled = st.session_state.scaler_obj.transform(vec) if st.session_state.scaler_obj is not None else vec
                km_sim = KMeans(n_clusters=st.session_state.algo_params.get("k", 4), random_state=42, n_init=10).fit(x_mat)
                pred_cid = int(km_sim.predict(vec_scaled)[0])
                pred_lbl = c_names.get(pred_cid, f"Cluster {pred_cid}")
                changed = pred_lbl != rec["Cluster_Label"]
                st.markdown(
                    f"""
                <div class="bento" style="margin-top:0.75rem;">
                    <div class="bento-label">SIMULATED OUTCOME</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{T['text_primary']};">
                        {rec['Cluster_Label']} → <span style="color:{color_map.get(pred_lbl, '#34d399')};">{pred_lbl}</span>
                    </div>
                    <div class="bento-meta">{"Boundary crossed into a new cluster." if changed else "Record remains within its current cluster."}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    render_step_footer(4)


# ==========================================
# STAGE 6: AI-GENERATED INSIGHTS
# ==========================================
elif st.session_state.stage_idx == 5:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 06 // AI-GENERATED INSIGHTS</div>
        <h1 class="hero-h1">Automated cluster narrative &amp; pattern analysis.</h1>
        <p class="hero-p">
            Plain-English interpretation of cluster drivers, relative strengths and deficits, and recommended next steps.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    insights = generate_ai_insights(
        st.session_state.proc_df,
        st.session_state.cluster_labels,
        st.session_state.cluster_names,
        st.session_state.selected_features,
        st.session_state.algo_used,
        st.session_state.sil_score,
        st.session_state.dbi_score,
    )

    st.markdown(
        f"""
    <div class="bento fade-in">
        <div class="bento-label">EXECUTIVE SUMMARY</div>
        <div class="bento-meta" style="font-size:1rem;">
            Running <strong>{insights['algo']}</strong> on <strong>{len(st.session_state.proc_df)} records</strong> across
            <strong>{len(st.session_state.selected_features)} features</strong> discovered
            <strong>{insights['n_clusters']} distinct clusters</strong>
            {f"(plus {insights['noise_count']} noise outliers)" if insights['noise_count'] > 0 else ""}.
            The primary features separating these groups are
            <strong>{', '.join(f.upper() for f in insights['top_drivers'])}</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
        <div class="bento">
            <div class="bento-label">SILHOUETTE SCORE INTERPRETATION</div>
            <div class="bento-meta">{insights['sil_note']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
        <div class="bento">
            <div class="bento-label">DAVIES-BOULDIN INTERPRETATION</div>
            <div class="bento-meta">{insights['dbi_note']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-title">CLUSTER CHARACTERISTICS &amp; PATTERNS</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, card in enumerate(insights["cluster_cards"]):
        c_color = "#71717a" if card["cid"] == -1 else T["colors"][idx % len(T["colors"])]
        with cols[idx % 2]:
            st.markdown(
                f"""
            <div class="bento">
                <div class="bento-label"><span class="dot-ind" style="background:{c_color};"></span> {card['name']} ({card['pct']:.1f}% · n={card['count']})</div>
                <div class="bento-meta"><strong>Strongest features:</strong> {', '.join(card['strengths'])}</div>
                <div class="bento-meta"><strong>Lowest features:</strong> {', '.join(card['weaknesses'])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    <div class="bento">
        <div class="bento-label">RECOMMENDED NEXT STEPS</div>
        <div class="bento-meta">
            1. Validate cluster stability by comparing K-Means and Hierarchical Clustering in <strong>Step 4</strong>.<br>
            2. Inspect non-linear neighborhood boundaries using t-SNE or UMAP in <strong>Step 5</strong>.<br>
            3. Export the labeled CSV and PDF executive report in <strong>Step 7</strong> for stakeholder review.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_step_footer(5)


# ==========================================
# STAGE 7: RESULTS & EXPORT
# ==========================================
else:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">STEP 07 // RESULTS &amp; EXPORT</div>
        <h1 class="hero-h1">Download datasets, interactive charts, and PDF reports.</h1>
        <p class="hero-p">
            Review the final summary and export your enriched dataset, interactive visualization, or executive PDF report.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_out = st.session_state.proc_df.copy()
    x_mat = st.session_state.x_matrix
    feats = st.session_state.selected_features
    labels = st.session_state.cluster_labels
    c_names = st.session_state.cluster_names

    df_out["Cluster_ID"] = labels
    df_out["Cluster_Label"] = [c_names[c] for c in labels]

    pca_coords, _, _, _ = compute_embedding(x_mat, "PCA (Linear)", min(3, len(feats)))
    for i in range(pca_coords.shape[1]):
        df_out[f"PC{i+1}"] = pca_coords[:, i].round(4)

    insights = generate_ai_insights(
        st.session_state.proc_df, labels, c_names, feats,
        st.session_state.algo_used, st.session_state.sil_score, st.session_state.dbi_score,
    )
    means_tbl = df_out.groupby("Cluster_Label")[feats].mean()

    sil_str = f"{st.session_state.sil_score:.4f}" if st.session_state.sil_score is not None else "N/A"
    dbi_str = f"{st.session_state.dbi_score:.4f}" if st.session_state.dbi_score is not None else "N/A"

    st.markdown(
        f"""
    <div class="bento fade-in">
        <div class="bento-label">FINAL ANALYSIS SUMMARY</div>
        <div class="bento-meta" style="font-size:1rem;">
            Dataset: <strong>{st.session_state.dataset_name}</strong> ({len(df_out)} rows, {len(feats)} features) &nbsp;·&nbsp;
            Algorithm: <strong>{st.session_state.algo_used}</strong> &nbsp;·&nbsp;
            Silhouette: <strong>{sil_str}</strong> &nbsp;·&nbsp;
            Davies-Bouldin: <strong>{dbi_str}</strong>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            label="Download Processed CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="manifold_clustered_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        pdf_bytes = generate_pdf_report(
            {
                "dataset_name": st.session_state.dataset_name,
                "algo": st.session_state.algo_used,
                "scaling": st.session_state.scaling_method,
                "rows": len(df_out),
                "n_features": len(feats),
                "sil": sil_str,
                "dbi": dbi_str,
            },
            means_tbl,
            insights,
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="manifold_analytics_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with e3:
        if "last_fig_html" not in st.session_state:
            fig_exp = px.scatter(df_out, x="PC1", y="PC2" if "PC2" in df_out.columns else "PC1", color="Cluster_Label")
            st.session_state.last_fig_html = fig_exp.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="Download Interactive Chart (HTML)",
            data=st.session_state.last_fig_html,
            file_name="manifold_visualization.html",
            mime="text/html",
            use_container_width=True,
        )

    st.markdown('<div class="sec-title">PROCESSED DATASET PREVIEW</div>', unsafe_allow_html=True)
    st.dataframe(df_out.head(15), use_container_width=True)
    render_step_footer(6)
