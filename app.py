"""
MANIFOLD — Student Academic Profile Discovery
Scale.com inspired dark bento UI with Nothing OS dot-matrix typography.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="MANIFOLD — Academic Profile Discovery",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. DESIGN SYSTEM (SCALE.COM × NOTHING OS)
# ==========================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-root: #050507;
        --bg-surface: #0b0b0f;
        --bg-elevated: #111117;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(255, 255, 255, 0.22);
        --text-primary: #f4f4f5;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent-indigo: #818cf8;
        --accent-red: #ff2a3a;
        --ease-fluid: cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* Hide Streamlit default chrome */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }

    /* Root Canvas with subtle Nothing-style dot grid + Scale ambient glow */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: var(--bg-root) !important;
        background-image:
            radial-gradient(ellipse 80% 45% at 50% -15%, rgba(129, 140, 248, 0.12), transparent 70%),
            radial-gradient(ellipse 40% 30% at 85% 10%, rgba(255, 42, 58, 0.05), transparent 60%),
            radial-gradient(rgba(255, 255, 255, 0.055) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 24px 24px !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .block-container {
        max-width: 1180px !important;
        padding: 2rem 2rem 4rem 2rem !important;
    }

    /* Smooth entry animation */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeSlideUp 0.45s var(--ease-fluid) forwards;
    }

    /* Top Navigation Bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.5rem;
        margin-bottom: 1.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--accent-red);
        box-shadow: 0 0 12px rgba(255, 42, 58, 0.75);
    }

    .brand-name {
        font-family: 'DotGothic16', monospace;
        font-size: 1.35rem;
        letter-spacing: 0.14em;
        color: #ffffff;
        text-transform: uppercase;
    }

    .brand-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
        padding: 3px 9px;
        border-radius: 999px;
        letter-spacing: 0.06em;
    }

    /* Hero Typography */
    .hero-wrap {
        margin-bottom: 1.75rem;
    }

    .hero-eyebrow {
        font-family: 'DotGothic16', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        color: var(--accent-indigo);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-h1 {
        font-size: 2.35rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        line-height: 1.12;
        color: #ffffff;
        margin: 0 0 0.6rem 0;
    }

    .hero-p {
        font-size: 0.98rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 680px;
        margin: 0;
    }

    /* Bento Cards */
    .bento {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.04);
        transition: transform 0.3s var(--ease-fluid), border-color 0.3s ease, background-color 0.3s ease;
    }

    .bento:hover {
        border-color: var(--border-hover);
        background: var(--bg-elevated);
        transform: translateY(-2px);
    }

    .bento-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .bento-num {
        font-family: 'DotGothic16', monospace;
        font-size: 2.15rem;
        line-height: 1.1;
        color: #ffffff;
        letter-spacing: 0.02em;
    }

    .bento-meta {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.4rem;
    }

    .dot-ind {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Segmented Control (Radio Override) */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 6px !important;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        padding: 4px;
        border-radius: 10px;
        width: fit-content;
    }

    div[data-testid="stRadio"] label {
        background: transparent !important;
        padding: 6px 16px !important;
        border-radius: 7px !important;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.25s var(--ease-fluid) !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.04) !important;
    }

    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
        background: rgba(255, 255, 255, 0.09) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
    }

    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Selectbox & Inputs */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 9px !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
        transition: border-color 0.2s ease !important;
    }

    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--border-hover) !important;
    }

    /* Modern Slider & Toggle Accent */
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid var(--accent-indigo) !important;
        box-shadow: 0 0 10px rgba(129, 140, 248, 0.5) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, var(--accent-indigo), #38bdf8) !important;
    }
    div[data-testid="stCheckbox"] label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.74rem !important;
        color: var(--text-secondary) !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* Subtle Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-root); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.24); }

    /* Frame Plotly Charts inside Bento Surface */
    [data-testid="stPlotlyChart"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 10px 12px 4px 12px !important;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.04) !important;
        transition: border-color 0.3s ease !important;
    }
    [data-testid="stPlotlyChart"]:hover {
        border-color: var(--border-hover) !important;
    }

    /* Clean divider */
    .hairline {
        height: 1px;
        background: var(--border-subtle);
        margin: 1.5rem 0;
    }

    /* Section title */
    .sec-title {
        font-family: 'DotGothic16', monospace;
        font-size: 1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }

    /* Subject bar row */
    .subj-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.84rem;
    }
    .subj-name {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-size: 0.75rem;
    }
    .subj-val {
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. DATA & MODEL PIPELINE
# ==========================================
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

PALETTE = {
    "At-Risk": "#F43F5E",
    "Humanities-Focused": "#818CF8",
    "Electives Leaders": "#34D399",
    "All-Rounders": "#FFFFFF",
}


@st.cache_data
def load_data():
    df = pd.read_csv("student_grades.csv")
    df["GPA"] = df[SUBJECTS].mean(axis=1).round(1)
    df["STEM_Avg"] = df[STEM].mean(axis=1).round(1)
    df["Hum_Avg"] = df[HUM].mean(axis=1).round(1)
    df["Elec_Avg"] = df[ELEC].mean(axis=1).round(1)
    return df


df = load_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[SUBJECTS].values)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
df["Archetype"] = df["Cluster"].map(PROFILE_MAP)

pca_3d = PCA(n_components=3, random_state=42).fit(X_scaled)
var_3d = float(np.sum(pca_3d.explained_variance_ratio_) * 100)

# ==========================================
# 4. TOPBAR & NAVIGATION
# ==========================================
st.markdown(
    """
<div class="topbar fade-in">
    <div class="brand">
        <span class="brand-dot"></span>
        <span class="brand-name">MANIFOLD</span>
        <span class="brand-tag">N=100 · R⁸ → R²</span>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #71717a;">
        PCA · t-SNE · K-MEANS
    </div>
</div>
""",
    unsafe_allow_html=True,
)

nav_col, _ = st.columns([3, 1])
with nav_col:
    active_view = st.radio(
        "Navigation",
        ["01  MANIFOLD MAP", "02  COHORT PROFILES", "03  SIMULATOR"],
        label_visibility="collapsed",
        horizontal=True,
    )

st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)

# ==========================================
# VIEW 1: MANIFOLD MAP
# ==========================================
if active_view == "01  MANIFOLD MAP":
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">01 // LATENT SPACE PROJECTION</div>
        <h1 class="hero-h1">Uncover student archetypes hidden behind scalar GPAs.</h1>
        <p class="hero-p">
            Projecting 8 correlated subject grades into 2D reveals four distinct academic profiles
            and exposes high-risk students masked by elective grades.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    main_col, side_col = st.columns([2.4, 1], gap="large")

    with main_col:
        c1, c2, c3 = st.columns([1.2, 1.2, 0.9])
        with c1:
            method = st.selectbox("Projection", ["PCA (Linear)", "t-SNE (Non-Linear)"])
        with c2:
            color_by = st.selectbox("Color Encoding", ["Archetype", "Overall GPA", "STEM Avg", "Humanities Avg"])
        with c3:
            dims = st.selectbox("Dimensions", ["2D Plane", "3D Space"])

        n_comp = 3 if dims == "3D Space" else 2
        if method == "PCA (Linear)":
            pca = PCA(n_components=n_comp, random_state=42)
            coords = pca.fit_transform(X_scaled)
            x_lbl = f"PC1 · Core Aptitude ({pca.explained_variance_ratio_[0]*100:.1f}%)"
            y_lbl = f"PC2 · Electives ({pca.explained_variance_ratio_[1]*100:.1f}%)"
            z_lbl = f"PC3 · STEM vs Hum ({pca.explained_variance_ratio_[2]*100:.1f}%)" if n_comp == 3 else ""
        else:
            tsne = TSNE(n_components=n_comp, perplexity=20, learning_rate="auto", init="pca", random_state=42)
            coords = tsne.fit_transform(X_scaled)
            x_lbl = "t-SNE Dim 1"
            y_lbl = "t-SNE Dim 2"
            z_lbl = "t-SNE Dim 3" if n_comp == 3 else ""

        sil = silhouette_score(coords, df["Cluster"])

        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        if n_comp == 3:
            plot_df["z"] = coords[:, 2]

        color_col = (
            "Archetype"
            if color_by == "Archetype"
            else ("GPA" if color_by == "Overall GPA" else ("STEM_Avg" if color_by == "STEM Avg" else "Hum_Avg"))
        )

        if n_comp == 2:
            fig = go.Figure()
            if color_col == "Archetype":
                rgba_map = {
                    "At-Risk": ("rgba(244, 63, 94, 0.06)", "rgba(244, 63, 94, 0.38)", "rgba(244, 63, 94, 0.16)"),
                    "Humanities-Focused": ("rgba(129, 140, 248, 0.06)", "rgba(129, 140, 248, 0.38)", "rgba(129, 140, 248, 0.16)"),
                    "Electives Leaders": ("rgba(52, 211, 153, 0.06)", "rgba(52, 211, 153, 0.38)", "rgba(52, 211, 153, 0.16)"),
                    "All-Rounders": ("rgba(255, 255, 255, 0.04)", "rgba(255, 255, 255, 0.30)", "rgba(255, 255, 255, 0.12)"),
                }
                theta = np.linspace(0, 2 * np.pi, 60)
                for arch in ["At-Risk", "Humanities-Focused", "Electives Leaders", "All-Rounders"]:
                    sub = plot_df[plot_df["Archetype"] == arch]
                    fill_c, line_c, halo_c = rgba_map[arch]
                    # 1.8-sigma covariance ellipse for cluster boundary
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
                                x=ell[:, 0],
                                y=ell[:, 1],
                                mode="lines",
                                fill="toself",
                                fillcolor=fill_c,
                                line=dict(color=line_c, width=1, dash="dot"),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                    # Glow halo layer
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"],
                            y=sub["y"],
                            mode="markers",
                            marker=dict(size=18, color=halo_c, line=dict(width=0)),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    # Crisp core markers
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"],
                            y=sub["y"],
                            mode="markers",
                            name=arch,
                            marker=dict(size=8.5, color=PALETTE[arch], line=dict(width=1.2, color="#050507")),
                            hovertext=sub["student_id"],
                            customdata=sub[["Archetype", "GPA", "STEM_Avg", "Hum_Avg"]].values,
                            hovertemplate=(
                                "<b>Student #%{hovertext}</b><br>"
                                "Profile: %{customdata[0]}<br>"
                                "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]}"
                                "<extra></extra>"
                            ),
                        )
                    )
            else:
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color=color_col,
                    color_continuous_scale="Tealgrn" if color_col != "GPA" else "Magenta",
                    hover_name="student_id",
                    custom_data=["Archetype", "GPA", "STEM_Avg", "Hum_Avg", "Elec_Avg"],
                    labels={"x": x_lbl, "y": y_lbl},
                )
                fig.update_traces(
                    marker=dict(size=10, opacity=0.92, line=dict(width=1, color="#050507")),
                    hovertemplate=(
                        "<b>Student #%{hovertext}</b><br>"
                        "Profile: %{customdata[0]}<br>"
                        "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]}"
                        "<extra></extra>"
                    ),
                )

            if method == "PCA (Linear)":
                show_vectors = st.checkbox("Overlay PCA Subject Vectors", value=False)
                if show_vectors:
                    loadings = pca.components_[:2].T * 4.2
                    for i, subj in enumerate(SUBJECTS):
                        vx, vy = float(loadings[i, 0]), float(loadings[i, 1])
                        fig.add_shape(
                            type="line",
                            x0=0, y0=0, x1=vx, y1=vy,
                            line=dict(color="rgba(255,255,255,0.5)", width=1.4),
                        )
                        fig.add_annotation(
                            x=vx * 1.13, y=vy * 1.13,
                            text=subj.upper(),
                            showarrow=False,
                            font=dict(family="JetBrains Mono, monospace", size=10, color="#f4f4f5"),
                        )
        else:
            fig = px.scatter_3d(
                plot_df,
                x="x",
                y="y",
                z="z",
                color=color_col,
                color_discrete_map=PALETTE if color_col == "Archetype" else None,
                color_continuous_scale="Magenta" if color_col != "Archetype" else None,
                hover_name="student_id",
                custom_data=["Archetype", "GPA", "STEM_Avg", "Hum_Avg", "Elec_Avg"],
                labels={"x": x_lbl, "y": y_lbl, "z": z_lbl},
            )
            fig.update_traces(
                marker=dict(size=6, opacity=0.92, line=dict(width=0.6, color="#050507")),
                hovertemplate=(
                    "<b>Student #%{hovertext}</b><br>"
                    "Profile: %{customdata[0]}<br>"
                    "GPA: %{customdata[1]} · STEM: %{customdata[2]} · Hum: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=11),
            height=490,
            margin=dict(l=16, r=16, t=28, b=16),
            xaxis=dict(title=x_lbl, showgrid=True, gridcolor="rgba(255,255,255,0.045)", zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title=y_lbl, showgrid=True, gridcolor="rgba(255,255,255,0.045)", zerolinecolor="rgba(255,255,255,0.1)"),
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showbackground=False, gridcolor="rgba(255,255,255,0.07)", showspikes=False),
                yaxis=dict(showbackground=False, gridcolor="rgba(255,255,255,0.07)", showspikes=False),
                zaxis=dict(showbackground=False, gridcolor="rgba(255,255,255,0.07)", showspikes=False),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                title=None,
                font=dict(size=11, color="#e4e4e7"),
                itemsizing="constant",
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with side_col:
        at_risk_n = int((df["Archetype"] == "At-Risk").sum())
        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:#818cf8;"></span> 3D PCA VARIANCE</div>
            <div class="bento-num">{var_3d:.1f}%</div>
            <div class="bento-meta">8 subjects compressed to 3 axes</div>
        </div>
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:#34d399;"></span> 2D SILHOUETTE</div>
            <div class="bento-num">{sil:.3f}</div>
            <div class="bento-meta">Cluster separation ({method.split()[0]})</div>
        </div>
        <div class="bento fade-in" style="margin-bottom: 12px;">
            <div class="bento-label"><span class="dot-ind" style="background:#f43f5e;"></span> AT-RISK COHORT</div>
            <div class="bento-num">{at_risk_n}<span style="font-size:1.1rem; color:#71717a;">/{len(df)}</span></div>
            <div class="bento-meta">Core deficit masked by electives</div>
        </div>
        <div class="bento fade-in">
            <div class="bento-label">LATENT AXES</div>
            <div class="subj-row"><span class="subj-name">PC1 (65.2%)</span><span class="subj-val">Core Aptitude</span></div>
            <div class="subj-row"><span class="subj-name">PC2 (22.6%)</span><span class="subj-val">Electives</span></div>
            <div class="subj-row" style="border:none;"><span class="subj-name">PC3 (8.2%)</span><span class="subj-val">STEM vs Hum</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ==========================================
# VIEW 2: COHORT PROFILES
# ==========================================
elif active_view == "02  COHORT PROFILES":
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">02 // UNSUPERVISED SEGMENTATION</div>
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

    order = ["At-Risk", "Humanities-Focused", "Electives Leaders", "All-Rounders"]
    cols = st.columns(4)
    for idx, name in enumerate(order):
        row = stats.loc[name]
        color = PALETTE[name]
        with cols[idx]:
            st.markdown(
                f"""
            <div class="bento fade-in">
                <div class="bento-label"><span class="dot-ind" style="background:{color};"></span> {name}</div>
                <div class="bento-num">{int(row['Count'])}%</div>
                <div class="bento-meta">Mean GPA: <strong style="color:#fff;">{row['GPA']}</strong></div>
                <div class="hairline" style="margin: 0.9rem 0;"></div>
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
        st.markdown('<div class="sec-title">SUBJECT BREAKDOWN BY ARCHETYPE</div>', unsafe_allow_html=True)
    with hdr_r:
        chart_type = st.selectbox("View", ["Radar Fingerprint", "Subject Curve"], label_visibility="collapsed")

    means_df = df.groupby("Archetype")[SUBJECTS].mean().round(1)

    if chart_type == "Radar Fingerprint":
        fig_cohort = go.Figure()
        fill_map = {
            "At-Risk": "rgba(244, 63, 94, 0.14)",
            "Humanities-Focused": "rgba(129, 140, 248, 0.14)",
            "Electives Leaders": "rgba(52, 211, 153, 0.14)",
            "All-Rounders": "rgba(255, 255, 255, 0.08)",
        }
        cats = [s.upper() for s in SUBJECTS] + [SUBJECTS[0].upper()]
        for arch in order:
            vals = means_df.loc[arch].tolist()
            vals.append(vals[0])
            fig_cohort.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats,
                    fill="toself",
                    fillcolor=fill_map[arch],
                    name=arch,
                    line=dict(color=PALETTE[arch], width=2.2),
                    marker=dict(size=5, color=PALETTE[arch]),
                )
            )
        fig_cohort.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[25, 100],
                    tickvals=[40, 60, 80, 100],
                    gridcolor="rgba(255,255,255,0.07)",
                    linecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=9, color="#71717a"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.07)",
                    linecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=10, color="#d4d4d8"),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=11),
            height=440,
            margin=dict(l=40, r=40, t=35, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cohort, use_container_width=True, config={"displayModeBar": False})
    else:
        fig_curve = go.Figure()
        subj_labels = [s.upper() for s in SUBJECTS]
        for arch in order:
            vals = means_df.loc[arch].tolist()
            fig_curve.add_trace(
                go.Scatter(
                    x=subj_labels,
                    y=vals,
                    mode="lines+markers",
                    name=arch,
                    line=dict(color=PALETTE[arch], width=2.5, shape="spline", smoothing=0.8),
                    marker=dict(size=8, color=PALETTE[arch], line=dict(width=1.2, color="#050507")),
                )
            )
        fig_curve.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=11),
            height=410,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[30, 100], title="Mean Grade"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        )
        st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False})

# ==========================================
# VIEW 3: STUDENT SIMULATOR
# ==========================================
else:
    st.markdown(
        """
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">03 // COUNTERFACTUAL DIAGNOSTICS</div>
        <h1 class="hero-h1">Simulate grade interventions in real time.</h1>
        <p class="hero-p">
            Select a student and adjust subject scores to see how targeted improvements shift
            their predicted archetype and position in the latent space.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    left_c, right_c = st.columns([1, 1.8], gap="large")

    with left_c:
        stu_id = st.selectbox("Student ID", df["student_id"].tolist())
        stu = df[df["student_id"] == stu_id].iloc[0]
        dot_color = PALETTE[stu["Archetype"]]

        subj_html = "".join(
            f'<div class="subj-row"><span class="subj-name">{s}</span><span class="subj-val">{int(stu[s])}</span></div>'
            for s in SUBJECTS
        )

        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-top: 0.75rem;">
            <div class="bento-label"><span class="dot-ind" style="background:{dot_color};"></span> STUDENT #{int(stu['student_id']):03d}</div>
            <div class="bento-num" style="font-size: 1.6rem;">{stu['Archetype']}</div>
            <div class="bento-meta">Current GPA: <strong style="color:#fff;">{stu['GPA']}</strong></div>
            <div class="hairline" style="margin: 1rem 0;"></div>
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

        sim_vec = np.array(
            [s_math, s_sci, s_cs, stu["band"], s_eng, s_hist, s_span, stu["physed"]]
        ).reshape(1, -1)
        sim_scaled = scaler.transform(sim_vec)
        new_k = int(kmeans.predict(sim_scaled)[0])
        new_arch = PROFILE_MAP[new_k]
        new_gpa = float(np.mean(sim_vec).round(1))
        delta_gpa = new_gpa - float(stu["GPA"])

        changed = new_arch != stu["Archetype"]
        accent = "#34d399" if changed else "#71717a"
        status_txt = "ARCHETYPE TRANSITION" if changed else "NO BOUNDARY CROSSING"

        st.markdown(
            f"""
        <div class="bento fade-in" style="margin-top: 1rem; border-color: {accent if changed else 'rgba(255,255,255,0.08)'};">
            <div class="bento-label"><span class="dot-ind" style="background:{accent};"></span> {status_txt}</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; margin: 4px 0;">
                {stu['Archetype']}  →  <span style="color: {PALETTE[new_arch]};">{new_arch}</span>
            </div>
            <div class="bento-meta" style="font-family: 'JetBrains Mono', monospace;">
                PROJECTED GPA: {new_gpa} ({delta_gpa:+.1f})
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Live 2D PCA trajectory mini-map
        pca_sim = PCA(n_components=2, random_state=42).fit(X_scaled)
        bg_pts = pca_sim.transform(X_scaled)
        orig_pt = pca_sim.transform(scaler.transform(stu[SUBJECTS].values.reshape(1, -1)))[0]
        new_pt = pca_sim.transform(sim_scaled)[0]

        traj = go.Figure()
        for arch in ["At-Risk", "Humanities-Focused", "Electives Leaders", "All-Rounders"]:
            mask = df["Archetype"] == arch
            traj.add_trace(
                go.Scatter(
                    x=bg_pts[mask, 0],
                    y=bg_pts[mask, 1],
                    mode="markers",
                    marker=dict(size=6, color=PALETTE[arch], opacity=0.28),
                    hoverinfo="skip",
                    name=arch,
                )
            )
        traj.add_trace(
            go.Scatter(
                x=[orig_pt[0]],
                y=[orig_pt[1]],
                mode="markers+text",
                marker=dict(size=12, color="#818cf8", line=dict(width=1.5, color="#ffffff")),
                text=["Baseline"],
                textposition="bottom center",
                showlegend=False,
            )
        )
        traj.add_trace(
            go.Scatter(
                x=[new_pt[0]],
                y=[new_pt[1]],
                mode="markers+text",
                marker=dict(size=14, color=PALETTE[new_arch], symbol="diamond", line=dict(width=1.5, color="#ffffff")),
                text=["Simulated"],
                textposition="top center",
                showlegend=False,
            )
        )
        traj.add_annotation(
            x=float(new_pt[0]),
            y=float(new_pt[1]),
            ax=float(orig_pt[0]),
            ay=float(orig_pt[1]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.3,
            arrowwidth=2,
            arrowcolor="#ffffff",
        )
        traj.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=10),
            height=275,
            margin=dict(l=15, r=15, t=20, b=15),
            xaxis=dict(title="PC1", showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
            yaxis=dict(title="PC2", showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
        )
        st.plotly_chart(traj, use_container_width=True, config={"displayModeBar": False})
