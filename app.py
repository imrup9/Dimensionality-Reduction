"""
MANIFOLD — Academic Profile Discovery & Dimensionality Reduction Platform
Industry-standard interactive analytics dashboard supporting PCA, t-SNE, UMAP,
K-Means clustering, Dark/Light themes, responsive layouts, and counterfactual simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ==========================================
# 1. PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="MANIFOLD — Academic Profile Discovery",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUBJECTS = ["math", "science", "cs", "band", "english", "history", "spanish", "physed"]
STEM = ["math", "science", "cs"]
HUM = ["english", "history", "spanish"]
ELEC = ["band", "physed"]

PROFILE_MAP = {
    0: "Humanities-Focused",
    1: "Electives Leaders",
    2: "At-Risk",
    3: "All-Rounders",
}

CLUSTER_META = {
    "At-Risk": {
        "desc": "Low core STEM & Humanities scores masked by high elective grades.",
        "action": "Prioritize foundational Math/English tutoring and bi-weekly milestone reviews.",
    },
    "Humanities-Focused": {
        "desc": "Strong verbal literacy (English, History, Spanish); moderate STEM.",
        "action": "Pair liberal-arts advising with applied quantitative bridge workshops.",
    },
    "Electives Leaders": {
        "desc": "Top Band & PhysEd engagement paired with solid Humanities scores.",
        "action": "Channel arts/athletic leadership into peer mentorship and flexible deadlines.",
    },
    "All-Rounders": {
        "desc": "High-achieving mastery across all 8 core and elective subjects.",
        "action": "Offer AP/IB acceleration, dual-enrollment, and peer-tutoring roles.",
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
        "palette": {
            "At-Risk": "#F43F5E",
            "Humanities-Focused": "#818CF8",
            "Electives Leaders": "#34D399",
            "All-Rounders": "#F8FAFC",
        },
        "fill_rgba": {
            "At-Risk": ("rgba(244, 63, 94, 0.08)", "rgba(244, 63, 94, 0.42)", "rgba(244, 63, 94, 0.16)"),
            "Humanities-Focused": ("rgba(129, 140, 248, 0.08)", "rgba(129, 140, 248, 0.42)", "rgba(129, 140, 248, 0.16)"),
            "Electives Leaders": ("rgba(52, 211, 153, 0.08)", "rgba(52, 211, 153, 0.42)", "rgba(52, 211, 153, 0.16)"),
            "All-Rounders": ("rgba(248, 250, 252, 0.06)", "rgba(248, 250, 252, 0.35)", "rgba(248, 250, 252, 0.14)"),
        },
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
        "palette": {
            "At-Risk": "#E11D48",
            "Humanities-Focused": "#4F46E5",
            "Electives Leaders": "#059669",
            "All-Rounders": "#0F172A",
        },
        "fill_rgba": {
            "At-Risk": ("rgba(225, 29, 72, 0.08)", "rgba(225, 29, 72, 0.45)", "rgba(225, 29, 72, 0.15)"),
            "Humanities-Focused": ("rgba(79, 70, 229, 0.08)", "rgba(79, 70, 229, 0.45)", "rgba(79, 70, 229, 0.15)"),
            "Electives Leaders": ("rgba(5, 150, 105, 0.08)", "rgba(5, 150, 105, 0.45)", "rgba(5, 150, 105, 0.15)"),
            "All-Rounders": ("rgba(15, 23, 42, 0.06)", "rgba(15, 23, 42, 0.35)", "rgba(15, 23, 42, 0.12)"),
        },
        "marker_edge": "#ffffff",
        "vector_color": "rgba(15, 23, 42, 0.55)",
        "arrow_color": "#0f172a",
    },
}


# ==========================================
# 2. CACHED DATA & MODEL COMPUTATIONS
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_cluster(file_bytes=None):
    """Load dataset, compute domain averages, standardize features, and fit K-Means (k=4)."""
    if file_bytes is not None:
        from io import BytesIO

        df_in = pd.read_csv(BytesIO(file_bytes))
    else:
        df_in = pd.read_csv("student_grades.csv")

    df_in = df_in.copy()
    df_in["GPA"] = df_in[SUBJECTS].mean(axis=1).round(1)
    df_in["STEM_Avg"] = df_in[STEM].mean(axis=1).round(1)
    df_in["Hum_Avg"] = df_in[HUM].mean(axis=1).round(1)
    df_in["Elec_Avg"] = df_in[ELEC].mean(axis=1).round(1)

    scaler_obj = StandardScaler()
    x_scaled = scaler_obj.fit_transform(df_in[SUBJECTS].values)

    km_obj = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_in["Cluster"] = km_obj.fit_predict(x_scaled)
    df_in["Archetype"] = df_in["Cluster"].map(PROFILE_MAP)
    return df_in, x_scaled, scaler_obj, km_obj


@st.cache_data(show_spinner=False)
def compute_embedding(x_scaled, labels, method, n_comp, perp=20, lr="auto", n_neighbors=15, min_dist=0.1):
    """Compute cached 2D/3D dimensionality reduction coordinates and validation scores."""
    if method == "PCA (Linear)":
        model = PCA(n_components=n_comp, random_state=42)
        coords = model.fit_transform(x_scaled)
        evr = model.explained_variance_ratio_
        axes_lbls = [
            f"PC1 — Core Academic Aptitude ({evr[0]*100:.1f}% var)",
            f"PC2 — Electives: Band & PhysEd ({evr[1]*100:.1f}% var)",
            f"PC3 — STEM vs. Humanities ({evr[2]*100:.1f}% var)" if n_comp == 3 else "",
        ]
        loadings = model.components_[:2].T
    elif method == "t-SNE (Non-Linear)":
        lr_val = "auto" if lr == "auto" else float(lr)
        model = TSNE(n_components=n_comp, perplexity=perp, learning_rate=lr_val, init="pca", random_state=42)
        coords = model.fit_transform(x_scaled)
        axes_lbls = [
            "t-SNE Dim 1 — Local Neighborhood Axis",
            "t-SNE Dim 2 — Cluster Separation Axis",
            "t-SNE Dim 3 — Manifold Depth Axis" if n_comp == 3 else "",
        ]
        loadings = None
    else:
        model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_comp, random_state=42)
        coords = model.fit_transform(x_scaled)
        axes_lbls = [
            "UMAP Dim 1 — Global & Local Manifold Axis",
            "UMAP Dim 2 — Continuum Separation Axis",
            "UMAP Dim 3 — Manifold Depth Axis" if n_comp == 3 else "",
        ]
        loadings = None

    sil = float(silhouette_score(coords, labels))
    trust = float(trustworthiness(x_scaled, coords, n_neighbors=min(15, len(x_scaled) - 1)))
    return coords, axes_lbls, sil, trust, loadings


# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; padding:4px 0 14px 0;">
            <span style="width:10px; height:10px; border-radius:50%; background:#ff2a3a; box-shadow:0 0 10px rgba(255,42,58,0.7); display:inline-block;"></span>
            <span style="font-family:'DotGothic16', monospace; font-size:1.3rem; letter-spacing:0.12em;">MANIFOLD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_mode = st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        help="Switch between high-contrast Dark and Light analytics themes.",
    )
    T = THEMES[theme_mode]

    st.markdown("---")
    nav_page = st.radio(
        "Navigation",
        [
            "01 · Manifold Explorer",
            "02 · Cohort Profiles",
            "03 · Student Simulator",
            "04 · Benchmark & Export",
        ],
        help="Navigate between dashboard sections.",
    )

    st.markdown("---")
    with st.expander("Dataset & Global Filters", expanded=True):
        uploaded_csv = st.file_uploader(
            "Upload Custom CSV",
            type=["csv"],
            help="Optional: upload a CSV containing the 8 subject columns (math, science, cs, band, english, history, spanish, physed).",
        )

    file_bytes = uploaded_csv.getvalue() if uploaded_csv is not None else None
    try:
        df, X_scaled, scaler, kmeans = load_and_cluster(file_bytes)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    missing_cols = [c for c in SUBJECTS if c not in df.columns]
    if missing_cols:
        st.error(f"Dataset is missing required columns: {missing_cols}")
        st.stop()

    all_archetypes = ["At-Risk", "Humanities-Focused", "Electives Leaders", "All-Rounders"]
    selected_archetypes = st.multiselect(
        "Filter Archetypes",
        all_archetypes,
        default=all_archetypes,
        help="Filter which student archetypes are highlighted across the dashboard.",
    )
    gpa_min, gpa_max = float(df["GPA"].min()), float(df["GPA"].max())
    gpa_range = st.slider(
        "Overall GPA Range",
        min_value=0.0,
        max_value=100.0,
        value=(gpa_min, gpa_max),
        step=1.0,
        help="Filter students by overall mean grade.",
    )


# ==========================================
# 4. DYNAMIC THEME CSS (RESPONSIVE & ANIMATED)
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

    /* Hide default Streamlit footer/decoration */
    #MainMenu, footer, [data-testid="stDecoration"] {{
        display: none !important;
    }}

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
        padding: 1.75rem 2rem 3.5rem 2rem !important;
    }}

    @keyframes fadeSlideUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .fade-in {{
        animation: fadeSlideUp 0.45s var(--ease-fluid) forwards;
    }}

    /* Hero Header */
    .hero-wrap {{
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--border-subtle);
    }}
    .hero-eyebrow {{
        font-family: 'DotGothic16', monospace;
        font-size: 0.92rem;
        letter-spacing: 0.12em;
        color: #6366f1;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}
    .hero-h1 {{
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.15;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
    }}
    .hero-p {{
        font-size: 1.03rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 740px;
        margin: 0;
    }}

    /* Bento Cards */
    .bento {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        box-shadow: var(--card-shadow);
        transition: transform 0.28s var(--ease-fluid), border-color 0.28s ease, background-color 0.28s ease;
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
        font-size: 2.25rem;
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

    /* Cluster Meaning Legend */
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
        transition: border-color 0.25s ease, transform 0.25s var(--ease-fluid);
    }}
    .legend-card:hover {{
        border-color: var(--border-hover);
        transform: translateY(-1px);
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

    /* Plotly Card Wrapper */
    [data-testid="stPlotlyChart"] {{
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 12px 14px 6px 14px !important;
        box-shadow: var(--card-shadow) !important;
        transition: border-color 0.3s ease !important;
    }}
    [data-testid="stPlotlyChart"]:hover {{
        border-color: var(--border-hover) !important;
    }}

    /* Subject Rows & Dividers */
    .hairline {{
        height: 1px;
        background: var(--border-subtle);
        margin: 1rem 0;
    }}
    .sec-title {{
        font-family: 'DotGothic16', monospace;
        font-size: 1.18rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }}
    .subj-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 7px 0;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 0.92rem;
    }}
    .subj-name {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 0.84rem;
    }}
    .subj-val {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.94rem;
    }}

    /* Responsive Breakpoints (Tablet & Mobile) */
    @media (max-width: 1024px) {{
        .block-container {{
            padding: 1.25rem 1.2rem 2.5rem 1.2rem !important;
        }}
        .hero-h1 {{
            font-size: 1.8rem;
        }}
    }}
    @media (max-width: 640px) {{
        .legend-grid {{
            grid-template-columns: 1fr;
        }}
        .hero-h1 {{
            font-size: 1.5rem;
        }}
        .bento-num {{
            font-size: 1.85rem;
        }}
    }}
</style>
""",
    unsafe_allow_html=True,
)

# Filtered mask for interactive exploration
filter_mask = (df["Archetype"].isin(selected_archetypes)) & (df["GPA"].between(gpa_range[0], gpa_range[1]))
filtered_df = df[filter_mask].copy()

pca_3d_ref = PCA(n_components=3, random_state=42).fit(X_scaled)
var_3d = float(np.sum(pca_3d_ref.explained_variance_ratio_) * 100)


def render_cluster_legend():
    """Render a responsive 4-card legend explaining each cluster color and meaning."""
    cards_html = ""
    for arch in all_archetypes:
        count = int((df["Archetype"] == arch).sum())
        pct = count / len(df) * 100
        color = T["palette"][arch]
        cards_html += (
            f'<div class="legend-card">'
            f'<span class="dot-ind" style="background:{color}; margin-top:6px;"></span>'
            f'<div>'
            f'<div class="legend-title">{arch} ({pct:.0f}% · n={count})</div>'
            f'<div class="legend-desc">{CLUSTER_META[arch]["desc"]}</div>'
            f'</div>'
            f'</div>'
        )
    st.markdown(f'<div class="legend-grid">{cards_html}</div>', unsafe_allow_html=True)


# ==========================================
# 5. PAGE 1: MANIFOLD EXPLORER
# ==========================================
if nav_page == "01 · Manifold Explorer":
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">01 // LATENT MANIFOLD EXPLORER</div>
        <h1 class="hero-h1">Uncover student archetypes hidden behind scalar GPAs.</h1>
        <p class="hero-p">
            Project 8 correlated subject scores into 2D or 3D space using PCA, t-SNE, or UMAP
            to inspect cluster separation, subject loadings, and at-risk cohorts.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    main_col, side_col = st.columns([2.4, 1], gap="large")

    with main_col:
        alg_options = ["PCA (Linear)", "t-SNE (Non-Linear)"] + (["UMAP (Manifold)"] if HAS_UMAP else [])
        c1, c2, c3 = st.columns([1.2, 1.2, 0.9])
        with c1:
            method = st.selectbox("Projection Algorithm", alg_options, help="Select linear (PCA) or non-linear (t-SNE / UMAP) dimensionality reduction.")
        with c2:
            color_by = st.selectbox("Color Encoding", ["Archetype", "Overall GPA", "STEM Avg", "Humanities Avg"], help="Color points by K-Means archetype or continuous grade metrics.")
        with c3:
            dims = st.selectbox("Dimensions", ["2D Plane", "3D Space"], help="Switch between 2D projection and interactive 3D space.")

        n_comp = 3 if dims == "3D Space" else 2

        # Optional hyperparameter controls for t-SNE / UMAP
        perp, lr_choice, u_neighbors, u_dist = 20, "auto", 15, 0.1
        if method == "t-SNE (Non-Linear)":
            tc1, tc2 = st.columns(2)
            perp = tc1.slider("t-SNE Perplexity", 5, 45, 20, 5, help="Controls local vs. global neighborhood balance.")
            lr_choice = tc2.selectbox("Learning Rate", ["auto", "50", "200"], help="Gradient descent step size for t-SNE.")
        elif method == "UMAP (Manifold)":
            uc1, uc2 = st.columns(2)
            u_neighbors = uc1.slider("UMAP n_neighbors", 5, 45, 15, 5, help="Size of local neighborhood for manifold approximation.")
            u_dist = uc2.slider("UMAP min_dist", 0.05, 0.50, 0.10, 0.05, help="Minimum distance between embedded points.")

        with st.spinner("Computing manifold projection..."):
            coords, axes_lbls, sil, trust, loadings = compute_embedding(
                X_scaled, df["Cluster"].values, method, n_comp, perp, lr_choice, u_neighbors, u_dist
            )

        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        if n_comp == 3:
            plot_df["z"] = coords[:, 2]

        # Apply sidebar filters to visible points
        vis_df = plot_df[filter_mask]
        x_lbl, y_lbl, z_lbl = axes_lbls

        color_col = (
            "Archetype"
            if color_by == "Archetype"
            else ("GPA" if color_by == "Overall GPA" else ("STEM_Avg" if color_by == "STEM Avg" else "Hum_Avg"))
        )

        if n_comp == 2:
            fig = go.Figure()
            if color_col == "Archetype":
                theta = np.linspace(0, 2 * np.pi, 60)
                for arch in selected_archetypes:
                    sub = vis_df[vis_df["Archetype"] == arch]
                    if sub.empty:
                        continue
                    fill_c, line_c, halo_c = T["fill_rgba"][arch]
                    if len(sub) > 2:
                        pts = sub[["x", "y"]].values
                        cov = np.cov(pts, rowvar=False)
                        vals, vecs = np.linalg.eigh(cov)
                        order_e = vals.argsort()[::-1]
                        vals, vecs = vals[order_e], vecs[:, order_e]
                        radii = 1.85 * np.sqrt(np.maximum(vals, 1e-6))
                        ell = (vecs @ (radii[:, None] * np.vstack([np.cos(theta), np.sin(theta)]))).T + pts.mean(axis=0)
                        fig.add_trace(
                            go.Scatter(
                                x=ell[:, 0], y=ell[:, 1], mode="lines", fill="toself",
                                fillcolor=fill_c, line=dict(color=line_c, width=1.1, dash="dot"),
                                hoverinfo="skip", showlegend=False,
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"], y=sub["y"], mode="markers",
                            marker=dict(size=18, color=halo_c, line=dict(width=0)),
                            hoverinfo="skip", showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"], y=sub["y"], mode="markers", name=arch,
                            marker=dict(size=9, color=T["palette"][arch], line=dict(width=1.2, color=T["marker_edge"])),
                            hovertext=sub["student_id"],
                            customdata=sub[["Archetype", "GPA", "STEM_Avg", "Hum_Avg", "Elec_Avg"]].values,
                            hovertemplate=(
                                "<b>Student #%{hovertext}</b><br>"
                                "Archetype: %{customdata[0]}<br>"
                                "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]} · Elec: %{customdata[4]}"
                                "<extra></extra>"
                            ),
                        )
                    )
            else:
                fig = px.scatter(
                    vis_df, x="x", y="y", color=color_col,
                    color_continuous_scale="Plasma" if theme_mode == "Dark" else "Viridis",
                    hover_name="student_id",
                    custom_data=["Archetype", "GPA", "STEM_Avg", "Hum_Avg", "Elec_Avg"],
                )
                fig.update_traces(
                    marker=dict(size=10, opacity=0.92, line=dict(width=1, color=T["marker_edge"])),
                    hovertemplate=(
                        "<b>Student #%{hovertext}</b><br>"
                        "Archetype: %{customdata[0]}<br>"
                        "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]} · Elec: %{customdata[4]}"
                        "<extra></extra>"
                    ),
                )

            if method == "PCA (Linear)" and loadings is not None:
                if st.checkbox("Overlay PCA Subject Loading Vectors", value=False, help="Show how each subject pulls students in PC1-PC2 space."):
                    scaled_vecs = loadings * 4.2
                    for i, subj in enumerate(SUBJECTS):
                        vx, vy = float(scaled_vecs[i, 0]), float(scaled_vecs[i, 1])
                        fig.add_shape(type="line", x0=0, y0=0, x1=vx, y1=vy, line=dict(color=T["vector_color"], width=1.6))
                        fig.add_annotation(
                            x=vx * 1.13, y=vy * 1.13, text=subj.upper(), showarrow=False,
                            font=dict(family="JetBrains Mono, monospace", size=11, color=T["text_primary"]),
                        )
        else:
            fig = px.scatter_3d(
                vis_df, x="x", y="y", z="z", color=color_col,
                color_discrete_map=T["palette"] if color_col == "Archetype" else None,
                color_continuous_scale="Plasma" if color_col != "Archetype" else None,
                hover_name="student_id",
                custom_data=["Archetype", "GPA", "STEM_Avg", "Hum_Avg", "Elec_Avg"],
            )
            fig.update_traces(
                marker=dict(size=6, opacity=0.92, line=dict(width=0.6, color=T["marker_edge"])),
                hovertemplate=(
                    "<b>Student #%{hovertext}</b><br>"
                    "Archetype: %{customdata[0]}<br>"
                    "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )

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
                xaxis=dict(title=dict(text=x_lbl, font=dict(size=13, color=T["text_primary"])), tickfont=dict(size=11), showbackground=False, gridcolor=T["grid"], showspikes=False),
                yaxis=dict(title=dict(text=y_lbl, font=dict(size=13, color=T["text_primary"])), tickfont=dict(size=11), showbackground=False, gridcolor=T["grid"], showspikes=False),
                zaxis=dict(title=dict(text=z_lbl, font=dict(size=13, color=T["text_primary"])), tickfont=dict(size=11), showbackground=False, gridcolor=T["grid"], showspikes=False),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title=None, font=dict(size=13, color=T["text_primary"]), itemsizing="constant"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
        render_cluster_legend()

    with side_col:
        at_risk_n = int((filtered_df["Archetype"] == "At-Risk").sum())
        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:{T['palette']['Humanities-Focused']};"></span> VISIBLE STUDENTS</div>
            <div class="bento-num">{len(filtered_df)}<span style="font-size:1.15rem; color:{T['text_muted']};">/{len(df)}</span></div>
            <div class="bento-meta">Active filter selection</div>
        </div>
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:#34d399;"></span> SILHOUETTE / TRUSTWORTHINESS</div>
            <div class="bento-num">{sil:.3f} <span style="font-size:1.15rem; color:{T['text_muted']};">/ {trust:.3f}</span></div>
            <div class="bento-meta">Cluster separation &amp; neighborhood fidelity</div>
        </div>
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:#818cf8;"></span> 3D PCA VARIANCE</div>
            <div class="bento-num">{var_3d:.1f}%</div>
            <div class="bento-meta">8 subjects compressed to 3 axes</div>
        </div>
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:{T['palette']['At-Risk']};"></span> AT-RISK COHORT</div>
            <div class="bento-num">{at_risk_n}</div>
            <div class="bento-meta">Core deficit masked by electives</div>
        </div>
        <div class="bento fade-in">
            <div class="bento-label">PCA LATENT AXES</div>
            <div class="subj-row"><span class="subj-name">PC1 (65.2%)</span><span class="subj-val">Core Aptitude</span></div>
            <div class="subj-row"><span class="subj-name">PC2 (22.6%)</span><span class="subj-val">Electives</span></div>
            <div class="subj-row" style="border:none;"><span class="subj-name">PC3 (8.2%)</span><span class="subj-val">STEM vs Hum</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==========================================
# 6. PAGE 2: COHORT PROFILES
# ==========================================
elif nav_page == "02 · Cohort Profiles":
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">02 // COHORT SEGMENTATION</div>
        <h1 class="hero-h1">Four distinct academic archetypes.</h1>
        <p class="hero-p">
            K-Means clustering (k=4) on standardized grades isolates clear performance signatures
            across STEM, Humanities, and Electives.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    stats = df.groupby("Archetype").agg(
        Count=("GPA", "count"),
        STEM=("STEM_Avg", "mean"),
        Hum=("Hum_Avg", "mean"),
        Elec=("Elec_Avg", "mean"),
        GPA=("GPA", "mean"),
    ).round(1)

    cols = st.columns(4)
    for idx, name in enumerate(all_archetypes):
        row = stats.loc[name]
        color = T["palette"][name]
        with cols[idx]:
            st.markdown(
                f"""
            <div class="bento fade-in">
                <div class="bento-label"><span class="dot-ind" style="background:{color};"></span> {name}</div>
                <div class="bento-num">{int(row['Count'])}%</div>
                <div class="bento-meta">Mean GPA: <strong>{row['GPA']}</strong></div>
                <div class="bento-meta" style="font-size:0.84rem;">{CLUSTER_META[name]['desc']}</div>
                <div class="hairline"></div>
                <div class="subj-row"><span class="subj-name">STEM</span><span class="subj-val">{row['STEM']}</span></div>
                <div class="subj-row"><span class="subj-name">Humanities</span><span class="subj-val">{row['Hum']}</span></div>
                <div class="subj-row" style="border:none;"><span class="subj-name">Electives</span><span class="subj-val">{row['Elec']}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    hdr_l, hdr_r = st.columns([2, 1])
    with hdr_l:
        st.markdown('<div class="sec-title">SUBJECT PROFILE BY ARCHETYPE</div>', unsafe_allow_html=True)
    with hdr_r:
        chart_type = st.selectbox("Chart View", ["Radar Fingerprint", "Subject Curve"], label_visibility="collapsed")

    means_df = df.groupby("Archetype")[SUBJECTS].mean().round(1)
    active_list = [a for a in all_archetypes if a in selected_archetypes] or all_archetypes

    if chart_type == "Radar Fingerprint":
        fig_cohort = go.Figure()
        cats = [s.upper() for s in SUBJECTS] + [SUBJECTS[0].upper()]
        for arch in active_list:
            vals = means_df.loc[arch].tolist()
            vals.append(vals[0])
            fill_c = T["fill_rgba"][arch][0]
            fig_cohort.add_trace(
                go.Scatterpolar(
                    r=vals, theta=cats, fill="toself", fillcolor=fill_c,
                    name=arch, line=dict(color=T["palette"][arch], width=2.4),
                    marker=dict(size=6, color=T["palette"][arch]),
                )
            )
        fig_cohort.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[25, 100], tickvals=[40, 60, 80, 100], gridcolor=T["grid"], linecolor=T["zero"], tickfont=dict(size=11, color=T["text_muted"])),
                angularaxis=dict(gridcolor=T["grid"], linecolor=T["zero"], tickfont=dict(size=13, color=T["text_primary"])),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=13),
            height=470,
            margin=dict(l=50, r=50, t=45, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0, font=dict(size=13, color=T["text_primary"])),
        )
        st.plotly_chart(fig_cohort, use_container_width=True, config={"displayModeBar": False})
    else:
        fig_curve = go.Figure()
        subj_labels = [s.upper() for s in SUBJECTS]
        for arch in active_list:
            fig_curve.add_trace(
                go.Scatter(
                    x=subj_labels, y=means_df.loc[arch].tolist(), mode="lines+markers",
                    name=arch, line=dict(color=T["palette"][arch], width=2.8, shape="spline", smoothing=0.8),
                    marker=dict(size=9, color=T["palette"][arch], line=dict(width=1.2, color=T["marker_edge"])),
                )
            )
        fig_curve.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=13),
            height=440,
            margin=dict(l=20, r=20, t=38, b=20),
            hovermode="x unified",
            xaxis=dict(title=dict(text="Academic Subject (8-Dimensional Feature Space)", font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12, color=T["text_secondary"]), showgrid=False),
            yaxis=dict(title=dict(text="Mean Cohort Grade (0–100 Scale)", font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12, color=T["text_muted"]), showgrid=True, gridcolor=T["grid"], range=[30, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title=None, font=dict(size=13, color=T["text_primary"])),
        )
        st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False})


# ==========================================
# 7. PAGE 3: STUDENT SIMULATOR
# ==========================================
elif nav_page == "03 · Student Simulator":
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">03 // COUNTERFACTUAL DIAGNOSTICS</div>
        <h1 class="hero-h1">Simulate grade interventions in real time.</h1>
        <p class="hero-p">
            Select a student and adjust subject scores to see how targeted improvements shift
            their predicted archetype and coordinates in PC1–PC2 space.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    left_c, right_c = st.columns([1, 1.85], gap="large")

    with left_c:
        id_options = filtered_df["student_id"].tolist() if not filtered_df.empty else df["student_id"].tolist()
        stu_id = st.selectbox("Student ID", id_options, help="Pick any student from the active filtered cohort.")
        stu = df[df["student_id"] == stu_id].iloc[0]
        dot_color = T["palette"][stu["Archetype"]]

        subj_html = "".join(
            f'<div class="subj-row"><span class="subj-name">{s}</span><span class="subj-val">{int(stu[s])}</span></div>'
            for s in SUBJECTS
        )
        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-top: 0.75rem;">
            <div class="bento-label"><span class="dot-ind" style="background:{dot_color};"></span> STUDENT #{int(stu['student_id']):03d}</div>
            <div class="bento-num" style="font-size: 1.7rem;">{stu['Archetype']}</div>
            <div class="bento-meta">Current GPA: <strong>{stu['GPA']}</strong></div>
            <div class="bento-meta" style="font-size:0.84rem;">{CLUSTER_META[stu['Archetype']]['action']}</div>
            <div class="hairline"></div>
            {subj_html}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with right_c:
        st.markdown('<div class="sec-title">INTERVENTION SLIDERS</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        s_math = sc1.slider("Math", 0, 100, int(stu["math"]))
        s_sci = sc2.slider("Science", 0, 100, int(stu["science"]))
        s_cs = sc1.slider("CS", 0, 100, int(stu["cs"]))
        s_eng = sc2.slider("English", 0, 100, int(stu["english"]))
        s_hist = sc1.slider("History", 0, 100, int(stu["history"]))
        s_span = sc2.slider("Spanish", 0, 100, int(stu["spanish"]))

        sim_vec = np.array([s_math, s_sci, s_cs, stu["band"], s_eng, s_hist, s_span, stu["physed"]]).reshape(1, -1)
        sim_scaled = scaler.transform(sim_vec)
        new_k = int(kmeans.predict(sim_scaled)[0])
        new_arch = PROFILE_MAP[new_k]
        new_gpa = float(np.mean(sim_vec).round(1))
        delta_gpa = new_gpa - float(stu["GPA"])

        changed = new_arch != stu["Archetype"]
        accent = "#34d399" if changed else T["text_muted"]
        status_txt = "ARCHETYPE TRANSITION" if changed else "NO BOUNDARY CROSSING"

        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-top: 0.75rem; margin-bottom: 1rem; border-color: {accent if changed else T['border']};">
            <div class="bento-label"><span class="dot-ind" style="background:{accent};"></span> {status_txt}</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {T['text_primary']}; margin: 4px 0;">
                {stu['Archetype']}  →  <span style="color: {T['palette'][new_arch]};">{new_arch}</span>
            </div>
            <div class="bento-meta" style="font-family: 'JetBrains Mono', monospace;">
                PROJECTED GPA: {new_gpa} ({delta_gpa:+.1f})
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        pca_sim = PCA(n_components=2, random_state=42).fit(X_scaled)
        bg_pts = pca_sim.transform(X_scaled)
        orig_pt = pca_sim.transform(scaler.transform(stu[SUBJECTS].values.reshape(1, -1)))[0]
        new_pt = pca_sim.transform(sim_scaled)[0]

        traj = go.Figure()
        for arch in all_archetypes:
            mask = df["Archetype"] == arch
            traj.add_trace(
                go.Scatter(
                    x=bg_pts[mask, 0], y=bg_pts[mask, 1], mode="markers",
                    marker=dict(size=7, color=T["palette"][arch], opacity=0.28),
                    hoverinfo="skip", name=arch,
                )
            )
        traj.add_trace(
            go.Scatter(
                x=[orig_pt[0]], y=[orig_pt[1]], mode="markers+text",
                marker=dict(size=12, color="#6366f1", line=dict(width=1.5, color=T["marker_edge"])),
                text=["Baseline"], textposition="bottom center",
                textfont=dict(size=12, color=T["text_primary"]), showlegend=False,
            )
        )
        traj.add_trace(
            go.Scatter(
                x=[new_pt[0]], y=[new_pt[1]], mode="markers+text",
                marker=dict(size=14, color=T["palette"][new_arch], symbol="diamond", line=dict(width=1.5, color=T["marker_edge"])),
                text=["Simulated"], textposition="top center",
                textfont=dict(size=12, color=T["text_primary"]), showlegend=False,
            )
        )
        traj.add_annotation(
            x=float(new_pt[0]), y=float(new_pt[1]), ax=float(orig_pt[0]), ay=float(orig_pt[1]),
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2.2, arrowcolor=T["arrow_color"],
        )
        traj.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=12),
            height=320,
            margin=dict(l=20, r=20, t=38, b=20),
            xaxis=dict(title=dict(text="PC1 — Core Academic Aptitude (Low → High)", font=dict(size=13, color=T["text_primary"])), tickfont=dict(size=11, color=T["text_muted"]), showgrid=True, gridcolor=T["grid"], zeroline=False),
            yaxis=dict(title=dict(text="PC2 — Electives Engagement (Low → High)", font=dict(size=13, color=T["text_primary"])), tickfont=dict(size=11, color=T["text_muted"]), showgrid=True, gridcolor=T["grid"], zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12, color=T["text_primary"])),
        )
        st.plotly_chart(traj, use_container_width=True, config={"displayModeBar": False})


# ==========================================
# 8. PAGE 4: BENCHMARK & EXPORT
# ==========================================
else:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">04 // VALIDATION &amp; EXPORT</div>
        <h1 class="hero-h1">Quantitative benchmark &amp; enriched dataset.</h1>
        <p class="hero-p">
            Compare linear PCA vs. non-linear manifold embeddings across Silhouette separation
            and Trustworthiness (k=15), and download the enriched cohort table.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    methods_to_eval = ["PCA (Linear)", "t-SNE (Non-Linear)"] + (["UMAP (Manifold)"] if HAS_UMAP else [])
    bench_rows = []
    with st.spinner("Evaluating embeddings..."):
        for m in methods_to_eval:
            _, _, s_val, t_val, _ = compute_embedding(X_scaled, df["Cluster"].values, m, 2)
            bench_rows.append({"Method": m, "2D Silhouette": round(s_val, 4), "Trustworthiness (k=15)": round(t_val, 4)})
    bench_df = pd.DataFrame(bench_rows)

    b_col1, b_col2 = st.columns([1.3, 1], gap="large")
    with b_col1:
        fig_bench = go.Figure()
        fig_bench.add_trace(
            go.Bar(name="2D Silhouette", x=bench_df["Method"], y=bench_df["2D Silhouette"], marker_color=T["palette"]["Humanities-Focused"])
        )
        fig_bench.add_trace(
            go.Bar(name="Trustworthiness (k=15)", x=bench_df["Method"], y=bench_df["Trustworthiness (k=15)"], marker_color=T["palette"]["Electives Leaders"])
        )
        fig_bench.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color=T["text_secondary"], size=13),
            height=360,
            margin=dict(l=20, r=20, t=38, b=20),
            xaxis=dict(title=dict(text="Dimensionality Reduction Method", font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12)),
            yaxis=dict(title=dict(text="Score (Higher is Better)", font=dict(size=14, color=T["text_primary"])), tickfont=dict(size=12), range=[0, 1.08], showgrid=True, gridcolor=T["grid"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=13, color=T["text_primary"])),
        )
        st.plotly_chart(fig_bench, use_container_width=True, config={"displayModeBar": False})

    with b_col2:
        st.markdown('<div class="sec-title">BENCHMARK SUMMARY</div>', unsafe_allow_html=True)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        pca_c, _, _, _, _ = compute_embedding(X_scaled, df["Cluster"].values, "PCA (Linear)", 3)
        tsne_c, _, _, _, _ = compute_embedding(X_scaled, df["Cluster"].values, "t-SNE (Non-Linear)", 2)
        export_df = filtered_df.copy()
        idx_mask = export_df.index
        export_df["PC1"] = pca_c[idx_mask, 0].round(4)
        export_df["PC2"] = pca_c[idx_mask, 1].round(4)
        export_df["PC3"] = pca_c[idx_mask, 2].round(4)
        export_df["tSNE_1"] = tsne_c[idx_mask, 0].round(4)
        export_df["tSNE_2"] = tsne_c[idx_mask, 1].round(4)

        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
        st.download_button(
            label=f"Download Enriched CSV ({len(export_df)} Students)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="student_profiles_enriched.csv",
            mime="text/csv",
            use_container_width=True,
        )
