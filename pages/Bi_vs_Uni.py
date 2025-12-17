import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import preprocess_text, build_unigram_graph

# =====================
# UI
# =====================
st.set_page_config(layout="wide")
st.title("📈 Unigram vs Bigram — Centrality Analysis")

if "pdf_text" not in st.session_state:
    st.warning("⚠️ Silakan upload PDF terlebih dahulu di halaman utama.")
    st.stop()

# =====================
# LOAD & PREPROCESS (CACHED)
# =====================
tokens, tokens_by_sentence = preprocess_text(st.session_state["pdf_text"])

# =====================
# BUILD GRAPHS
# =====================
# --- UNIGRAM GRAPH (sequential words)
G_uni = build_unigram_graph(tokens)

# --- BIGRAM GRAPH (within sentence)
bigrams = []
for sent in tokens_by_sentence:
    for i in range(len(sent) - 1):
        bigrams.append((sent[i], sent[i + 1]))

G_bi = nx.Graph()
for w1, w2 in bigrams:
    if G_bi.has_edge(w1, w2):
        G_bi[w1][w2]["weight"] += 1
    else:
        G_bi.add_edge(w1, w2, weight=1)

# =====================
# CENTRALITY FUNCTIONS
# =====================
def pagerank_safe(G):
    return nx.pagerank(G, weight="weight", max_iter=200)

metrics = {
    "PageRank": pagerank_safe,
    "Degree": nx.degree_centrality,
    "Betweenness": lambda G: nx.betweenness_centrality(G, weight="weight")
}

# =====================
# 1️⃣ DATA UNDERSTANDING
# =====================
st.subheader("1️⃣ Data Understanding")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🔹 Unigram Graph")
    st.write(f"Nodes: {G_uni.number_of_nodes()}")
    st.write(f"Edges: {G_uni.number_of_edges()}")

with c2:
    st.markdown("### 🔹 Bigram Graph")
    st.write(f"Nodes: {G_bi.number_of_nodes()}")
    st.write(f"Edges: {G_bi.number_of_edges()}")

# =====================
# 2️⃣ TOP-10 CENTRALITY TABLES
# =====================
st.subheader("2️⃣ Top-10 Central Words (Unigram vs Bigram)")

for metric_name, metric_func in metrics.items():
    st.markdown(f"### 🔸 {metric_name}")

    cent_uni = metric_func(G_uni)
    cent_bi = metric_func(G_bi)

    df_uni = (
        pd.DataFrame(cent_uni.items(), columns=["Word", metric_name])
        .sort_values(metric_name, ascending=False)
        .head(10)
    )

    df_bi = (
        pd.DataFrame(cent_bi.items(), columns=["Word", metric_name])
        .sort_values(metric_name, ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Unigram**")
        st.dataframe(df_uni)

    with col2:
        st.markdown("**Bigram**")
        st.dataframe(df_bi)

# =====================
# 3️⃣ GRAPH VISUALIZATION
# =====================
st.subheader("3️⃣ Graph Visualization (Centrality-based)")

def draw_graph(G, centrality, title, ax):
    pos = nx.spring_layout(G, seed=42)

    values = np.array(list(centrality.values()))
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)

    node_sizes = 600 + norm * 3000
    node_colors = norm

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.85,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)

    nx.draw_networkx_labels(
        G, pos,
        font_size=7,
        font_color="black",
        ax=ax
    )

    ax.set_title(title)
    ax.axis("off")

for metric_name, metric_func in metrics.items():
    st.markdown(f"### 🔸 {metric_name} Graph")

    cent_uni = metric_func(G_uni)
    cent_bi = metric_func(G_bi)

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))

    draw_graph(
        G_uni, cent_uni,
        f"Unigram Graph ({metric_name})",
        ax[0]
    )

    draw_graph(
        G_bi, cent_bi,
        f"Bigram Graph ({metric_name})",
        ax[1]
    )

    st.pyplot(fig)
