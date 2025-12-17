import streamlit as st
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, words as nltk_words

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
st.title("📈 Unigram vs Bigram — Centrality Analysis")

if "pdf_text" not in st.session_state:
    st.warning("⚠️ Silakan upload PDF terlebih dahulu di halaman utama.")
    st.stop()

# =====================
# PREPROCESS
# =====================
sentences = sent_tokenize(st.session_state["pdf_text"])

tokens_sentence = []
for s in sentences:
    tokens_sentence.append(preprocess_sentence(s))

# --- UNIGRAM (within sentence)
tokens_uni = [w for sent in tokens_sentence for w in sent]

# --- BIGRAM (gabungan sentence)
bigrams = []
for sent in tokens_sentence:
    bigrams.extend(list(ngrams(sent, 2)))

# =====================
# BUILD GRAPHS
# =====================
# UNIGRAM GRAPH
G_uni = nx.Graph()
for i in range(len(tokens_uni) - 1):
    G_uni.add_edge(tokens_uni[i], tokens_uni[i + 1])

# BIGRAM GRAPH
G_bi = nx.Graph()
for w1, w2 in bigrams:
    if G_bi.has_edge(w1, w2):
        G_bi[w1][w2]["weight"] += 1
    else:
        G_bi.add_edge(w1, w2, weight=1)

# =====================
# CENTRALITY METRICS
# =====================
metrics = {
    "PageRank": lambda G: nx.pagerank(G, weight="weight", max_iter=200),
    "Degree": nx.degree_centrality,
    "Betweenness": lambda G: nx.betweenness_centrality(G, weight="weight")
}

# =====================
# DATA UNDERSTANDING
# =====================
st.subheader("1️⃣ Data Understanding")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔹 Unigram")
    st.write("Nodes:", G_uni.number_of_nodes())
    st.write("Edges:", G_uni.number_of_edges())

with col2:
    st.markdown("### 🔹 Bigram")
    st.write("Nodes:", G_bi.number_of_nodes())
    st.write("Edges:", G_bi.number_of_edges())

# =====================
# TOP-10 TABLES
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

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Unigram**")
        st.dataframe(df_uni)

    with c2:
        st.markdown("**Bigram**")
        st.dataframe(df_bi)

# =====================
# GRAPH VISUALIZATION
# =====================
st.subheader("3️⃣ Graph Visualization (Unigram vs Bigram)")

def draw_graph(G, centrality, title, ax):
    pos = nx.spring_layout(G, seed=42)
    values = np.array(list(centrality.values()))
    sizes = 400 + (values - values.min()) / (values.max() - values.min() + 1e-9) * 3000

    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
    ax.set_title(title)
    ax.axis("off")

for metric_name, metric_func in metrics.items():
    st.markdown(f"### 🔸 {metric_name} Graph")

    cent_uni = metric_func(G_uni)
    cent_bi = metric_func(G_bi)

    fig, ax = plt.subplots(1, 2, figsize=(22, 10))

    draw_graph(G_uni, cent_uni, f"Unigram ({metric_name})", ax[0])
    draw_graph(G_bi, cent_bi, f"Bigram ({metric_name})", ax[1])

    st.pyplot(fig)
