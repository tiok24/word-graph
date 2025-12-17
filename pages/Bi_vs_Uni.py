import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import preprocess_text, build_unigram_graph

st.set_page_config(layout="wide")
st.title("📈 Unigram vs Bigram — Centrality Analysis")

if "pdf_text" not in st.session_state:
    st.warning("Upload PDF di halaman utama")
    st.stop()

tokens, tokens_by_sentence = preprocess_text(st.session_state["pdf_text"])

# ===== GRAPHS =====
G_uni = build_unigram_graph(tokens)

bigrams = []
for s in tokens_by_sentence:
    bigrams += list(zip(s, s[1:]))

G_bi = nx.Graph()
for w1, w2 in bigrams:
    if G_bi.has_edge(w1, w2):
        G_bi[w1][w2]["weight"] += 1
    else:
        G_bi.add_edge(w1, w2, weight=1)

metrics = {
    "PageRank": lambda G: nx.pagerank(G, weight="weight", max_iter=200),
    "Degree": nx.degree_centrality,
    "Betweenness": lambda G: nx.betweenness_centrality(G, weight="weight")
}

st.subheader("1️⃣ Data Understanding")
c1, c2 = st.columns(2)
c1.write("Unigram Nodes:", G_uni.number_of_nodes())
c1.write("Unigram Edges:", G_uni.number_of_edges())
c2.write("Bigram Nodes:", G_bi.number_of_nodes())
c2.write("Bigram Edges:", G_bi.number_of_edges())

st.subheader("2️⃣ Top-10 Centrality Comparison")
for name, func in metrics.items():
    cu, cb = func(G_uni), func(G_bi)
    st.markdown(f"### 🔸 {name}")
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame(cu.items(), columns=["Word", name]).sort_values(name, ascending=False).head(10))
    c2.dataframe(pd.DataFrame(cb.items(), columns=["Word", name]).sort_values(name, ascending=False).head(10))

st.subheader("3️⃣ Graph Visualization")
def draw(G, cent, ax, title):
    pos = nx.spring_layout(G, seed=42)
    vals = np.array(list(cent.values()))
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=600 + norm * 3000,
        node_color=norm,
        cmap=plt.cm.viridis,
        alpha=0.85,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(title)
    ax.axis("off")

for name, func in metrics.items():
    cu, cb = func(G_uni), func(G_bi)
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    draw(G_uni, cu, ax[0], f"Unigram ({name})")
    draw(G_bi, cb, ax[1], f"Bigram ({name})")
    st.pyplot(fig)
