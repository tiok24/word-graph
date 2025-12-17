import streamlit as st
import fitz
import nltk

# ===== NLTK FIX (WAJIB) =====
nltk.download("punkt")
nltk.download("punkt_tab")

st.set_page_config(page_title="NLP Word Graph App", layout="wide")
st.title("📚 NLP Word Graph Analysis")

uploaded_file = st.file_uploader(
    "📄 Upload PDF (digunakan untuk semua halaman)",
    type=["pdf"]
)

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    st.session_state["pdf_text"] = "".join([p.get_text() for p in doc])
    st.success("PDF berhasil dimuat & tersinkron ke semua halaman")

st.markdown("""
Gunakan **sidebar** untuk berpindah halaman:
- 📊 Word Graph NLP
- 📈 Unigram vs Bigram
""")
