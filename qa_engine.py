import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QASystem:
    def __init__(self):
        self.model_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.text_chunks = []
        self.index = None

    def rebuild_index_from_text(self, text, chunk_size=512):
        self.text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = self.model_embed.encode(self.text_chunks)
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search_similar(self, query, k=3):
        if self.index is None:
            return ["Nenhum documento carregado."]
        embedding = self.model_embed.encode([query])
        D, I = self.index.search(np.array(embedding), k)
        return [self.text_chunks[i] for i in I[0]]

    def answer(self, question):
        retrieved = self.search_similar(question)
        context = "\n".join(retrieved)

        prompt = f"""Você é um assistente útil. Baseado no contexto abaixo, responda à pergunta.

Contexto:
{context}

Pergunta: {question}
Resposta:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response.split("Resposta:")[-1].strip()
