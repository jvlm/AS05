import streamlit as st
from qa_engine import QASystem
import os
import PyPDF2

st.set_page_config(page_title="Assistente LLM com PDFs", layout="wide")
st.title("📄🤖 Assistente LLM com Documentos PDF")

# Criação de diretórios necessários
os.makedirs("docs", exist_ok=True)
os.makedirs("db", exist_ok=True)

@st.cache_resource
def load_qa():
    return QASystem()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

qa = load_qa()

uploaded_file = st.file_uploader("📄 Envie um arquivo PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extraindo conteúdo..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        qa.rebuild_index_from_text(pdf_text)
    st.success("Documento indexado com sucesso!")

query = st.text_input("🔍 Faça sua pergunta sobre o documento:")

if query:
    with st.spinner("Pensando..."):
        answer = qa.answer(query)
    st.markdown(f"**Resposta:** {answer}")
