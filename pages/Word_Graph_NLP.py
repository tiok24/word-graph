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

def preprocess_sentence(sentence):
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
st.title("📊 Word Graph NLP")

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
# SIDEBAR CONTROLS
# =====================
st.sidebar.header("⚙️ Pengaturan Visualisasi")

max_weight = max(bigram_freq.values()) if bigram_freq else 1
threshold = st.sidebar.slider(
    "Threshold Edge Weight (0 = semua node)",
    min_value=0,
    max_value=max_weight,
    value=0
)

# =====================
# BUILD GRAPH
# =====================
G = nx.Graph()
for (w1, w2), freq in bigram_freq.items():
    if freq >= threshold:
        G.add_edge(w1, w2, weight=freq)

# =====================
# 1️⃣ DATA UNDERSTANDING
# =====================
st.subheader("1️⃣ Data Understanding")

col1, col2, col3 = st.columns(3)
col1.metric("Total Tokens", len(tokens))
col2.metric("Unique Tokens", len(set(tokens)))
col3.metric("Total Bigrams", len(bigram_freq))

# =====================
# 2️⃣ PREVIEW SUB-MATRIX CO-OCCURRENCE
# =====================
st.subheader("2️⃣ Preview Sub-Matriks Co-occurrence (Bigram)")

top_words = [w for w, _ in Counter(tokens).most_common(10)]

co_matrix = pd.DataFrame(0, index=top_words, columns=top_words)

for (w1, w2), freq in bigram_freq.items():
    if w1 in top_words and w2 in top_words:
        co_matrix.loc[w1, w2] = freq

st.dataframe(co_matrix)

# =====================
# 3️⃣ TOP PAGERANK WORDS
# =====================
st.subheader("3️⃣ Top PageRank Words")

pagerank = nx.pagerank_numpy(G, weight="weight")

pr_df = (
    pd.DataFrame(pagerank.items(), columns=["Word", "PageRank"])
    .sort_values("PageRank", ascending=False)
    .head(10)
)

st.dataframe(pr_df)

# =====================
# 4️⃣ WORD GRAPH (PAGERANK)
# =====================
st.subheader("4️⃣ Word Graph PageRank (All Nodes)")

pos = nx.spring_layout(G, k=0.15, seed=42)

pr_values = np.array(list(pagerank.values()))
pr_norm = (pr_values - pr_values.min()) / (pr_values.max() - pr_values.min() + 1e-9)
node_sizes = 400 + pr_norm * 3500

fig1, ax1 = plt.subplots(figsize=(14, 14))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    alpha=0.8,
    ax=ax1
)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax1)

ax1.set_title("Node Size = PageRank")
ax1.axis("off")
st.pyplot(fig1)

# =====================
# 5️⃣ COMMUNITY GRAPH (LOUVAIN)
# =====================
st.subheader("5️⃣ Community Graph (Louvain)")

communities = nx_comm.louvain_communities(G, weight="weight")
main_community = max(communities, key=len)

community_map = {}
for i, c in enumerate(communities):
    for node in c:
        community_map[node] = i

colors = [
    "red" if n in main_community else "lightgray"
    for n in G.nodes()
]

fig2, ax2 = plt.subplots(figsize=(14, 14))

nx.draw_networkx_nodes(
    G, pos,
    node_color=colors,
    node_size=node_sizes,
    alpha=0.85,
    ax=ax2
)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax2)

ax2.set_title("Red = Main Community (Louvain)")
ax2.axis("off")
st.pyplot(fig2)
