import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def extract_text_from_pdfs(folder_path):
    all_texts = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                all_texts.append(text)
                filenames.append(filename)

    return all_texts, filenames

def build_vector_store(texts, filenames, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    with open("db/filenames.pkl", "wb") as f:
        pickle.dump(texts, f)

    faiss.write_index(index, "db/index.faiss")
    print(f"{len(embeddings)} documentos indexados com sucesso.")

if __name__ == "__main__":
    texts, filenames = extract_text_from_pdfs("docs")
    os.makedirs("db", exist_ok=True)
    build_vector_store(texts, filenames)
