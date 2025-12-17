import streamlit as st
import fitz

st.set_page_config(layout="wide")
st.title("📚 NLP Word Graph Application")

uploaded_file = st.file_uploader("📄 Upload PDF (sekali untuk semua halaman)", type=["pdf"])

if uploaded_file and "pdf_text" not in st.session_state:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    st.session_state["pdf_text"] = "".join(p.get_text() for p in doc)
    st.success("PDF berhasil dimuat & disimpan")

if "pdf_text" in st.session_state:
    st.sidebar.success("PDF loaded")
else:
    st.sidebar.warning("Upload PDF terlebih dahulu")

st.markdown("""
Gunakan sidebar untuk berpindah halaman:
- 📊 Word Graph NLP
- 📈 Bi vs Uni
""")
