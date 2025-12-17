import nltk
import streamlit as st
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, words as nltk_words
from collections import Counter

# NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("words")

STOP_WORDS = set(stopwords.words("english"))
ENGLISH_WORDS = set(nltk_words.words("en"))

@st.cache_data(show_spinner=False)
def preprocess_text(text: str):
    sentences = sent_tokenize(text)
    tokens = []
    tokens_by_sentence = []

    for s in sentences:
        t = word_tokenize(s)
        t = [w.lower() for w in t if w.isalpha()]
        t = [w for w in t if len(w) > 2]
        t = [w for w in t if w not in STOP_WORDS]
        t = [w for w in t if w in ENGLISH_WORDS]

        tokens.extend(t)
        tokens_by_sentence.append(t)

    return tokens, tokens_by_sentence


@st.cache_data(show_spinner=False)
def build_bigram_graph(tokens):
    bigrams = Counter(ngrams(tokens, 2))
    G = nx.Graph()
    for (w1, w2), freq in bigrams.items():
        G.add_edge(w1, w2, weight=freq)
    return G, bigrams


@st.cache_data(show_spinner=False)
def build_unigram_graph(tokens):
    G = nx.Graph()
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i + 1])
    return G
