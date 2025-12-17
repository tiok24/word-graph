import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx.algorithms.community as nx_comm
from collections import Counter

from utils import preprocess_text, build_bigram_graph

st.set_page_config(layout="wide")
st.title("📊 Word Graph NLP — PageRank & Community")

if "pdf_text" not in st.session_state:
    st.warning("Upload PDF di halaman utama")
    st.stop()

# ===== DATA =====
tokens, _ = preprocess_text(st.session_state["pdf_text"])
G, bigram_freq = build_bigram_graph(tokens)

# ===== DATA UNDERSTANDING =====
st.subheader("1️⃣ Data Understanding")
c1, c2, c3 = st.columns(3)
c1.metric("Total Tokens", len(tokens))
c2.metric("Unique Tokens", len(set(tokens)))
c3.metric("Bigram Edges", len(bigram_freq))

# ===== CO-OCCURRENCE MATRIX =====
st.subheader("2️⃣ Sub-Matriks Co-occurrence")
top_words = [w for w, _ in Counter(tokens).most_common(10)]
matrix = pd.DataFrame(0, index=top_words, columns=top_words)
for (w1, w2), f in bigram_freq.items():
    if w1 in top_words and w2 in top_words:
        matrix.loc[w1, w2] = f
st.dataframe(matrix)

# ===== PAGERANK =====
pagerank = nx.pagerank(G, weight="weight", max_iter=200)

st.subheader("3️⃣ Top PageRank Words")
st.dataframe(
    pd.DataFrame(pagerank.items(), columns=["Word", "PageRank"])
    .sort_values("PageRank", ascending=False)
    .head(10)
)

# ===== WORD GRAPH PAGERANK =====
st.subheader("4️⃣ Word Graph PageRank")
pos = nx.spring_layout(G, seed=42)
values = np.array(list(pagerank.values()))
norm = (values - values.min()) / (values.max() - values.min() + 1e-9)

fig1, ax1 = plt.subplots(figsize=(14, 14))
nx.draw_networkx_nodes(
    G, pos,
    node_size=600 + norm * 3000,
    node_color=norm,
    cmap=plt.cm.viridis,
    alpha=0.85,
    ax=ax1
)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax1)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
ax1.axis("off")
st.pyplot(fig1)

# ===== COMMUNITY GRAPH =====
st.subheader("5️⃣ Word Graph Community (Louvain)")
communities = nx_comm.louvain_communities(G, weight="weight")

community_map = {}
for i, c in enumerate(communities):
    for n in c:
        community_map[n] = i

colors = [community_map[n] for n in G.nodes()]

fig2, ax2 = plt.subplots(figsize=(14, 14))
nx.draw_networkx_nodes(
    G, pos,
    node_size=600 + norm * 3000,
    node_color=colors,
    cmap=plt.cm.tab20,
    alpha=0.85,
    ax=ax2
)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax2)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
ax2.axis("off")
st.pyplot(fig2)
