# src/agentes/agente_analisis.py
from typing import List
import numpy as np
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore

class AgenteAnalisis:
    """
    Genera embeddings y administra el VectorStore.
    """
    def __init__(self, modelo_name: str = "all-MiniLM-L6-v2"):
        self.embedder = EmbeddingModel(modelo_name)
        self.store = None

    def indexar_chunks(self, chunks_meta: List[dict]):
        textos = [c["texto"] for c in chunks_meta]
        embs = self.embedder.embedir(textos)  # np.ndarray
        dim = embs.shape[1]
        self.store = VectorStore(dim)
        self.store.agregar(embs, chunks_meta)
        return self.store

    def buscar_similares(self, consulta: str, top_k: int = 3):
        consulta_emb = self.embedder.embedir([consulta])[0]
        resultados = self.store.buscar(consulta_emb, top_k)
        return resultados
