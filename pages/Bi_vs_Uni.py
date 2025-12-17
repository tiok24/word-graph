import streamlit as st
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, words as nltk_words

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

STOP_WORDS = set(stopwords.words('english'))
ENGLISH_WORDS = set(nltk_words.words('en'))

def preprocess_sentence(s):
    tokens = word_tokenize(s)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens

st.title("📈 Unigram vs Bigram — Centrality Comparison")

if "pdf_text" not in st.session_state:
    st.warning("Upload PDF terlebih dahulu di halaman utama.")
    st.stop()

# =====================
# PREPROCESS
# =====================
tokens = []
for s in sent_tokenize(st.session_state["pdf_text"]):
    tokens.extend(preprocess_sentence(s))

# =====================
# GRAPHS
# =====================
G_uni = nx.Graph()
for i in range(len(tokens)-1):
    G_uni.add_edge(tokens[i], tokens[i+1])

G_bi = nx.Graph()
for w1, w2 in ngrams(tokens, 2):
    if G_bi.has_edge(w1, w2):
        G_bi[w1][w2]["weight"] += 1
    else:
        G_bi.add_edge(w1, w2, weight=1)

# =====================
# CENTRALITY
# =====================
centrality_option = st.sidebar.radio(
    "Centrality untuk ukuran node",
    ["PageRank", "Degree", "Betweenness"]
)

def get_centrality(G):
    if centrality_option == "PageRank":
        return nx.pagerank(G, weight="weight")
    if centrality_option == "Degree":
        return nx.degree_centrality(G)
    return nx.betweenness_centrality(G, weight="weight")

cent_uni = get_centrality(G_uni)
cent_bi = get_centrality(G_bi)

# =====================
# VISUALIZATION
# =====================
st.subheader("🔀 Perbandingan Unigram vs Bigram")

fig, ax = plt.subplots(1, 2, figsize=(22,10))

for graph, cent, axis, title in [
    (G_uni, cent_uni, ax[0], "Unigram"),
    (G_bi, cent_bi, ax[1], "Bigram")
]:
    pos = nx.spring_layout(graph, seed=42)
    values = np.array(list(cent.values()))
    sizes = 400 + (values - values.min()) / (values.max() - values.min() + 1e-9) * 3000

    nx.draw_networkx_nodes(graph, pos, node_size=sizes, alpha=0.8, ax=axis)
    nx.draw_networkx_edges(graph, pos, alpha=0.15, ax=axis)

    axis.set_title(f"{title} Graph ({centrality_option})")
    axis.axis("off")

st.pyplot(fig)
