import streamlit as st
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, words as nltk_words
import networkx.algorithms.community as nx_comm

# ===== NLTK FIX =====
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("words")

STOP_WORDS = set(stopwords.words("english"))
ENGLISH_WORDS = set(nltk_words.words("en"))

def preprocess_sentence(s):
    tokens = word_tokenize(s)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens

st.title("📊 Word Graph NLP")

if "pdf_text" not in st.session_state:
    st.warning("Silakan upload PDF di halaman utama.")
    st.stop()

# ===== PREPROCESS =====
tokens = []
for s in sent_tokenize(st.session_state["pdf_text"]):
    tokens.extend(preprocess_sentence(s))

bigrams = list(ngrams(tokens, 2))
bigram_freq = Counter(bigrams)

G = nx.Graph()
for (w1, w2), freq in bigram_freq.items():
    G.add_edge(w1, w2, weight=freq)

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider(
    "Threshold Edge Weight (0 = semua node)",
    min_value=0,
    max_value=max(bigram_freq.values()),
    value=0
)

if threshold > 0:
    G = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if d["weight"] >= threshold
    )

# ===== COMMUNITY =====
communities = nx_comm.louvain_communities(G, weight="weight")
main_community = max(communities, key=len)

# ===== PAGERANK =====
pagerank = nx.pagerank(G, weight="weight")

# ===== VISUAL =====
st.subheader("🕸️ Word Graph + Main Community Highlight")

pos = nx.spring_layout(G, k=0.15, seed=42)
node_sizes = [pagerank[n] * 30000 for n in G.nodes()]
node_colors = [
    "red" if n in main_community else "lightgray"
    for n in G.nodes()
]

fig, ax = plt.subplots(figsize=(14, 14))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.8,
    ax=ax
)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
ax.set_title("Node Size = PageRank | Red = Main Community")
ax.axis("off")
st.pyplot(fig)
