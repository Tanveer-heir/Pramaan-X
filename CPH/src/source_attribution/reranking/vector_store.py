"""
Persistent Vector Database using ChromaDB (§2.4).
Stores multimodal embeddings (text, images, video keyframes) with forensic metadata.
"""

from typing import List, Dict, Any, Optional
import os
import math
from src.common.logger import logger

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ForensicVectorStore:
    """
    Zero-cost persistent vector database for candidate posts and video keyframes.
    Uses ChromaDB persistent storage at data/chromadb/ with in-memory fallback.
    """

    def __init__(self, collection_name: str = "cph_forensic_store", persist_dir: str = "data/chromadb"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._fallback_store: List[Dict[str, Any]] = []

        if HAS_CHROMADB:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("vector_store.chromadb.initialized", path=self.persist_dir)
            except Exception as e:
                logger.warn("vector_store.chromadb.fallback", error=str(e))
                self.client = None
                self.collection = None

    def add_records(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """Indexes candidate documents or keyframes into the vector store."""
        if not ids:
            return

        if self.collection is not None:
            try:
                # Sanitize metadatas for ChromaDB (no None, no nested dicts)
                clean_metas = []
                for m in metadatas:
                    clean = {}
                    for k, v in m.items():
                        if v is None:
                            clean[k] = ""
                        elif isinstance(v, (str, int, float, bool)):
                            clean[k] = v
                        else:
                            clean[k] = str(v)
                    clean_metas.append(clean)

                # Batch upsert in chunks of 50
                chunk_size = 50
                for i in range(0, len(ids), chunk_size):
                    self.collection.upsert(
                        ids=ids[i:i + chunk_size],
                        embeddings=embeddings[i:i + chunk_size],
                        documents=documents[i:i + chunk_size],
                        metadatas=clean_metas[i:i + chunk_size]
                    )
                return
            except Exception as e:
                logger.warn("vector_store.upsert_error", error=str(e))

        # Fallback in-memory list
        for i, doc_id in enumerate(ids):
            self._fallback_store.append({
                "id": doc_id,
                "embedding": embeddings[i] if i < len(embeddings) else None,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {}
            })

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries nearest neighbors by cosine similarity."""
        if self.collection is not None:
            try:
                res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, max(1, self.collection.count())),
                    where=where
                )
                results = []
                ids = res.get("ids", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                distances = res.get("distances", [[]])[0]
                docs = res.get("documents", [[]])[0]

                for idx, item_id in enumerate(ids):
                    # In Chroma with cosine space, distance is 1 - cosine_similarity
                    distance = distances[idx] if idx < len(distances) else 0.5
                    sim = max(0.0, min(1.0, 1.0 - distance))
                    results.append({
                        "id": item_id,
                        "similarity": round(sim, 3),
                        "metadata": metas[idx] if idx < len(metas) else {},
                        "document": docs[idx] if idx < len(docs) else ""
                    })
                return results
            except Exception as e:
                logger.warn("vector_store.query_error", error=str(e))

        # Fallback brute-force cosine similarity
        scored = []
        for item in self._fallback_store:
            emb = item.get("embedding")
            if emb and len(emb) == len(query_embedding):
                dot = sum(a * b for a, b in zip(query_embedding, emb))
                norm_q = math.sqrt(sum(a * a for a in query_embedding)) or 1.0
                norm_e = math.sqrt(sum(b * b for b in emb)) or 1.0
                sim = max(0.0, min(1.0, dot / (norm_q * norm_e)))
            else:
                sim = 0.50

            scored.append({
                "id": item["id"],
                "similarity": round(sim, 3),
                "metadata": item["metadata"],
                "document": item["document"]
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:n_results]
