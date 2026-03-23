"""
Semantic Mirrors — Streamlit App
Explores how humans and LLMs understand "oppositeness" in the embedding space.
Original sentence: "The quick brown fox jumps over the lazy dog"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Mirrors",
    page_icon="🪞",
    layout="wide"
)

# ── API client ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("No OpenAI API key found. Set OPENAI_API_KEY as an environment variable or Streamlit secret.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_client()

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle("sentence_embeddings.pkl")
        return df
    except FileNotFoundError:
        st.error("sentence_embeddings.pkl not found. Please run the notebook first to generate embeddings.")
        st.stop()

df = load_data()
embedding_matrix = np.stack(df["embedding"].values)
sim_matrix = cosine_similarity(embedding_matrix)
original_idx = df[df["source"] == "Original"].index[0]

# ── Color scheme ───────────────────────────────────────────────────────────────
COLOR_MAP = {
    "Original":        "#1f77b4",
    "LLM":             "#ff7f0e",
    "LLM (Cap-Aware)": "#d62728",
    "Consensus":       "#9467bd",
    "Human":           "#2ca02c"
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🪞 Semantic Mirrors")
st.markdown("""
Exploring how humans and LLMs conceptualize *oppositeness* — and how those intuitions
map onto OpenAI's embedding space.

**Original sentence:** *The quick brown fox jumps over the lazy dog*
""")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Embedding Space", "📐 Similarity Analysis",
    "📋 Sentence Explorer", "🔍 Try Your Own"
])

# ── Tab 1: Embedding space visualizations ─────────────────────────────────────
with tab1:
    st.subheader("Sentence Embeddings Visualized")
    st.markdown(
        "Each point is one sentence — the original, the LLM-generated opposite, "
        "or one of the 26 human-generated opposites — projected into 2D or 3D space."
    )

    viz_type = st.radio("Visualization method", ["t-SNE (2D)", "t-SNE (3D)", "UMAP (2D)"],
                        horizontal=True)

    plot_df = df.copy()
    plot_df["short_sentence"] = plot_df["sentence"].apply(
        lambda s: s[:60] + "..." if len(s) > 60 else s
    )

    if viz_type == "t-SNE (2D)":
        if "tsne_x" not in df.columns:
            st.warning("t-SNE coordinates not found. Run cells #10 in the notebook first.")
        else:
            fig = px.scatter(
                plot_df, x="tsne_x", y="tsne_y",
                color="source", color_discrete_map=COLOR_MAP,
                hover_data={"short_sentence": True, "tsne_x": False, "tsne_y": False},
                size=[15 if s != "Human" else 8 for s in plot_df["source"]],
                size_max=20,
                title="2D t-SNE Projection of Sentence Embeddings",
                template="plotly_dark"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "t-SNE (3D)":
        if "tsne_x" not in df.columns:
            st.warning("t-SNE coordinates not found. Run cell #10 in the notebook first.")
        else:
            fig = px.scatter_3d(
                plot_df, x="tsne_x", y="tsne_y", z="tsne_z",
                color="source", color_discrete_map=COLOR_MAP,
                hover_data={"short_sentence": True,
                            "tsne_x": False, "tsne_y": False, "tsne_z": False},
                title="3D t-SNE Projection of Sentence Embeddings",
                template="plotly_dark",
                opacity=0.85
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

    else:  # UMAP
        if "umap_x" not in df.columns:
            st.warning("UMAP coordinates not found. Run cell #11 in the notebook first.")
        else:
            fig = px.scatter(
                plot_df, x="umap_x", y="umap_y",
                color="source", color_discrete_map=COLOR_MAP,
                hover_data={"short_sentence": True, "umap_x": False, "umap_y": False},
                size=[15 if s != "Human" else 8 for s in plot_df["source"]],
                size_max=20,
                title="UMAP Projection of Sentence Embeddings",
                template="plotly_dark"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Similarity analysis ─────────────────────────────────────────────────
with tab2:
    st.subheader("Cosine Similarity to the Original Sentence")
    st.markdown(
        "Cosine similarity measures how close two sentences are in embedding space. "
        "A *lower* value means more semantically different — i.e. more 'opposite'."
    )

    sims = sim_matrix[original_idx]
    df_sim = df[["source", "sentence"]].copy()
    df_sim["similarity"] = sims
    df_sim = df_sim.sort_values("similarity")

    # Key stats
    llm_sim = df_sim[df_sim["source"] == "LLM"]["similarity"].values[0]
    human_sims = df_sim[df_sim["source"] == "Human"]["similarity"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM similarity", f"{llm_sim:.4f}")
    col2.metric("Human mean", f"{human_sims.mean():.4f}")
    col3.metric("Human min (most opposite)", f"{human_sims.min():.4f}")
    col4.metric("LLM > X% of humans",
                f"{(human_sims < llm_sim).mean()*100:.1f}%",
                help="Percentage of human responses that are MORE opposite than the LLM")

    # Bar chart
    bar_colors = [COLOR_MAP.get(s, "#2ca02c") for s in df_sim["source"]]
    fig = go.Figure(go.Bar(
        x=df_sim["similarity"],
        y=df_sim["sentence"].apply(lambda s: s[:50] + "..." if len(s) > 50 else s),
        orientation="h",
        marker_color=bar_colors,
        hovertext=df_sim["sentence"],
        hoverinfo="text+x"
    ))
    fig.add_vline(x=llm_sim, line_dash="dash", line_color="#ff7f0e",
                  annotation_text="LLM", annotation_position="top right")
    fig.update_layout(
        title="Sentences Ranked by Similarity to Original (low = more opposite)",
        xaxis_title="Cosine Similarity",
        height=750,
        template="plotly_dark",
        margin=dict(l=300)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Most and least opposite
    farthest = df_sim.iloc[0]
    st.info(
        f"**Most opposite sentence** (similarity {farthest['similarity']:.4f})  \n"
        f"Source: **{farthest['source']}**  \n"
        f"_{farthest['sentence']}_"
    )

# ── Tab 3: Sentence Explorer ───────────────────────────────────────────────────
with tab3:
    st.subheader("All Sentences")
    st.markdown("Browse every sentence in the dataset with its similarity to the original.")

    source_filter = st.multiselect(
        "Filter by source",
        options=df["source"].unique().tolist(),
        default=df["source"].unique().tolist()
    )

    display_df = df[df["source"].isin(source_filter)][
        ["source", "sentence", "similarity_to_original"]
    ].sort_values("similarity_to_original")

    st.dataframe(
        display_df.style.format({"similarity_to_original": "{:.4f}"}),
        use_container_width=True,
        height=500
    )

# ── Tab 4: Try Your Own ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Where Does Your Sentence Land?")
    st.markdown(
        "Enter your own opposite sentence and see where it falls in embedding space "
        "relative to all the others."
    )

    user_input = st.text_input(
        "Your opposite sentence:",
        placeholder="e.g. A sluggish grey wolf crawls below an energetic cat"
    )

    if st.button("Embed & Compare") and user_input.strip():
        with st.spinner("Generating embedding..."):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=user_input.strip()
                )
                user_emb = np.array(resp.data[0].embedding).reshape(1, -1)

                # Similarity to original
                orig_emb = embedding_matrix[original_idx].reshape(1, -1)
                user_sim = cosine_similarity(user_emb, orig_emb)[0][0]

                # Rank among existing sentences
                all_sims = sim_matrix[original_idx]
                pct_rank = (all_sims > user_sim).mean() * 100

                st.success(f"**Your similarity to original:** {user_sim:.4f}")
                st.markdown(
                    f"Your sentence is more opposite than **{100-pct_rank:.1f}%** "
                    f"of the sentences in the dataset."
                )

                # Show in context of all similarities
                df_with_user = df[["source", "sentence", "similarity_to_original"]].copy()
                user_row = pd.DataFrame([{
                    "source": "You",
                    "sentence": user_input.strip(),
                    "similarity_to_original": user_sim
                }])
                df_with_user = pd.concat([df_with_user, user_row]).sort_values(
                    "similarity_to_original"
                )

                color_map_ext = {**COLOR_MAP, "You": "#FFD700"}
                bar_colors = [color_map_ext.get(s, "#2ca02c") for s in df_with_user["source"]]

                fig = go.Figure(go.Bar(
                    x=df_with_user["similarity_to_original"],
                    y=df_with_user["sentence"].apply(
                        lambda s: s[:50] + "..." if len(s) > 50 else s
                    ),
                    orientation="h",
                    marker_color=bar_colors,
                ))
                fig.update_layout(
                    title="Your sentence vs. all others",
                    xaxis_title="Cosine Similarity to Original",
                    height=750,
                    template="plotly_dark",
                    margin=dict(l=300)
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating embedding: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Semantic Mirrors — Duke University, 2025 | "
    "Embeddings: OpenAI text-embedding-3-small | "
    "Dimensionality reduction: t-SNE, UMAP"
)
