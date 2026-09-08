"""Tests for Multimodal Re-ranker and Vector Store."""

import pytest
from src.source_attribution.reranking.embedder import MultimodalEmbedder
from src.source_attribution.reranking.vector_store import ForensicVectorStore

def test_vector_store_persistence(tmp_path):
    store = ForensicVectorStore(collection_name="test_col", persist_dir=str(tmp_path / "chromadb"))
    store.add_records(
        ids=["id_001", "id_002"],
        embeddings=[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        documents=["Protest video in city center", "Cooking pasta recipe"],
        metadatas=[{"platform": "reddit"}, {"platform": "youtube"}]
    )
    query_vec = [0.1, 0.2, 0.3]
    res = store.query(query_vec, n_results=2)
    assert len(res) == 2
    assert res[0]["id"] == "id_001"
    assert res[0]["similarity"] > res[1]["similarity"]

def test_multimodal_embedder_reranking():
    embedder = MultimodalEmbedder()
    description = "A street demonstration with flags and public speeches"
    candidates = [
        {
            "platform": "reddit",
            "account": "u/activist",
            "title": "Massive street demonstration rally with flags",
            "text": "Footage from the scene"
        },
        {
            "platform": "open_web",
            "account": "cookingblog.com",
            "title": "Top 10 bakery desserts",
            "text": "Delicious chocolate cakes"
        }
    ]
    ranked = embedder.rerank(description, candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].account == "u/activist"
    assert ranked[0].similarity > ranked[1].similarity
    assert 0.0 <= ranked[0].similarity <= 1.0
