import streamlit as st
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, words as nltk_words
from collections import Counter
import networkx.algorithms.community as nx_comm

# =====================
# NLTK SETUP (WAJIB)
# =====================
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("words")

STOP_WORDS = set(stopwords.words("english"))
ENGLISH_WORDS = set(nltk_words.words("en"))

def preprocess_sentence(sentence: str):
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens

# =====================
# UI
# =====================
st.set_page_config(layout="wide")
st.title("📊 Word Graph NLP — PageRank & Community Detection")

if "pdf_text" not in st.session_state:
    st.warning("⚠️ Silakan upload PDF terlebih dahulu di halaman utama.")
    st.stop()

# =====================
# PREPROCESSING
# =====================
sentences = sent_tokenize(st.session_state["pdf_text"])

tokens = []
for s in sentences:
    tokens.extend(preprocess_sentence(s))

bigrams = list(ngrams(tokens, 2))
bigram_freq = Counter(bigrams)

# =====================
# BUILD WORD GRAPH
# =====================
G = nx.Graph()
for (w1, w2), freq in bigram_freq.items():
    G.add_edge(w1, w2, weight=freq)

# =====================
# DATA UNDERSTANDING
# =====================
st.subheader("1️⃣ Data Understanding")

c1, c2, c3 = st.columns(3)
c1.metric("Total Tokens (Unigram)", len(tokens))
c2.metric("Unique Tokens", len(set(tokens)))
c3.metric("Total Bigram Edges", len(bigram_freq))

# =====================
# PREVIEW CO-OCCURRENCE
# =====================
st.subheader("2️⃣ Preview Sub-Matriks Co-occurrence (Bigram)")

top_words = [w for w, _ in Counter(tokens).most_common(10)]

co_matrix = pd.DataFrame(0, index=top_words, columns=top_words)
for (w1, w2), freq in bigram_freq.items():
    if w1 in top_words and w2 in top_words:
        co_matrix.loc[w1, w2] = freq

st.dataframe(co_matrix)

# =====================
# TOP PAGERANK
# =====================
st.subheader("3️⃣ Top PageRank Words")

pagerank = nx.pagerank(G, weight="weight", max_iter=200)

pr_df = (
    pd.DataFrame(pagerank.items(), columns=["Word", "PageRank"])
    .sort_values("PageRank", ascending=False)
    .head(10)
)

st.dataframe(pr_df)

# =====================
# WORD GRAPH — PAGERANK
# =====================
st.subheader("4️⃣ Word Graph PageRank")

pos = nx.spring_layout(G, k=0.15, seed=42)

pr_values = np.array(list(pagerank.values()))
pr_norm = (pr_values - pr_values.min()) / (pr_values.max() - pr_values.min() + 1e-9)

node_sizes = 600 + pr_norm * 3000
node_colors = pr_norm  # warna berdasarkan PageRank

fig1, ax1 = plt.subplots(figsize=(14, 14))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.viridis,
    alpha=0.85,
    ax=ax1
)

nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax1)

nx.draw_networkx_labels(
    G, pos,
    font_size=8,
    font_color="black",
    ax=ax1
)

ax1.set_title("Word Graph based on PageRank (Color & Size)")
ax1.axis("off")
st.pyplot(fig1)

# =====================
# WORD GRAPH — COMMUNITY (LOUVAIN)
# =====================
st.subheader("5️⃣ Word Graph Community (Louvain)")

communities = nx_comm.louvain_communities(G, weight="weight")

community_map = {}
for i, community in enumerate(communities):
    for node in community:
        community_map[node] = i

community_colors = [community_map[n] for n in G.nodes()]

fig2, ax2 = plt.subplots(figsize=(14, 14))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=community_colors,
    cmap=plt.cm.tab20,
    alpha=0.85,
    ax=ax2
)

nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax2)

nx.draw_networkx_labels(
    G, pos,
    font_size=8,
    font_color="black",
    ax=ax2
)

ax2.set_title("Word Graph Community Detection (Louvain)")
ax2.axis("off")
st.pyplot(fig2)
