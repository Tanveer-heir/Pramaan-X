"""
Multimodal Embedding & Semantic Re-ranking Engine (§2.4d step 5).
Uses CLIP (ViT-B/32) in shared ℝ⁵¹² space for cross-modal Image-to-Text,
Image-to-Image visual matching, and TF-IDF fallback for lightweight environments.
Supports batch embedding of 200–300 candidate hits.
"""

import os
import sys

# Prevent HuggingFace Transformers from probing TensorFlow (which causes Protobuf mismatch)
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import List, Dict, Any, Optional, Tuple
import math
from PIL import Image
from src.common.schemas import ShortlistCandidate
from src.common.logger import logger
from src.source_attribution.reranking.vector_store import ForensicVectorStore

# Attempt to load local PyTorch & HuggingFace CLIP (Zero-Cost / Open Source)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MultimodalEmbedder:
    """
    Computes cross-modal embeddings for images, video keyframes, and candidate post texts.
    100% free, running locally on CPU or CUDA device.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if HAS_CLIP and torch.cuda.is_available() else "cpu"
        self._clip_model = None
        self._clip_processor = None
        self.vector_store = ForensicVectorStore()

    def _load_clip(self):
        if HAS_CLIP and self._clip_model is None:
            try:
                logger.info("embedder.loading_clip", model=self.model_name, device=self.device)
                self._clip_processor = CLIPProcessor.from_pretrained(self.model_name)
                self._clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self._clip_model.eval()
            except Exception as e:
                logger.warn("embedder.clip_load_failed", error=str(e))
                self._clip_model = None

    def embed_image(self, image_path: str) -> Optional[List[float]]:
        """Embeds raw image pixels into normalized 512-dim vector."""
        self._load_clip()
        if self._clip_model is not None and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self._clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy()[0].tolist()
            except Exception as e:
                logger.warn("embedder.image_embed_error", error=str(e))
        return None

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embeds a single text string into normalized 512-dim vector space."""
        res = self.embed_texts([text])
        return res[0] if res else None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Embeds a batch of texts into normalized 512-dim vector space efficiently."""
        self._load_clip()
        if not texts:
            return []

        if self._clip_model is None:
            return [None] * len(texts)

        all_embeddings: List[Optional[List[float]]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            cleaned_chunk = [t[:77] if t and t.strip() else "media post" for t in chunk]
            try:
                inputs = self._clip_processor(
                    text=cleaned_chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                with torch.no_grad():
                    text_features = self._clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.extend(text_features.cpu().numpy().tolist())
            except Exception as e:
                logger.warn("embedder.batch_embed_error", error=str(e))
                all_embeddings.extend([None] * len(chunk))

        return all_embeddings

    def score_candidates(
        self,
        media_description: str,
        candidates: List[Dict[str, Any]],
        query_image_path: Optional[str] = None,
        event_context: Optional[str] = None,
        top_k: int = 10
    ) -> Tuple[List[ShortlistCandidate], Dict[str, float]]:
        """
        Reranks candidate hits using Calibrated Dual-Anchor scoring across all candidate items:
        - Image-to-Text cross-modal cosine similarity against query image.
        - Event Context-to-Text semantic cosine similarity (text-to-text in 512-dim CLIP space).
        - Calibrated dynamic range expanding event matches to 80%-95% and suppressing noise below 35%.
        - Stores all vectors into ChromaDB.
        - Returns (ranked[:top_k], similarity_map).
        """
        if not candidates:
            return [], {}

        logger.info(
            "embedder.score_candidates.start",
            num_candidates=len(candidates),
            has_image=bool(query_image_path),
            has_event_context=bool(event_context)
        )

        query_image_vec = self.embed_image(query_image_path) if query_image_path else None
        query_text_vec = self.embed_text(media_description)
        event_context_vec = self.embed_text(event_context) if event_context else None

        cand_texts = [f"{c.get('title', '')} {c.get('text', '')}".strip() for c in candidates]
        clip_embeddings = self.embed_texts(cand_texts, batch_size=64)

        # Check if we should prepare TF-IDF fallback for texts without CLIP
        reference_text = f"{event_context or ''} {media_description}".strip()
        corpus = [reference_text] + cand_texts
        tfidf_matrix = None
        if HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(corpus)
            except Exception:
                tfidf_matrix = None

        ranked = []
        similarity_map: Dict[str, float] = {}
        valid_ids = []
        valid_vecs = []
        valid_docs = []
        valid_metas = []

        for idx, c in enumerate(candidates):
            cand_text = cand_texts[idx]
            cand_vec = clip_embeddings[idx] if idx < len(clip_embeddings) else None
            score = 0.50

            # 1. If CLIP vectors available, compute calibrated dual-anchor score
            if cand_vec is not None:
                valid_ids.append(f"cand_{idx}_{abs(hash(c.get('post_url', '')) % 100000)}")
                valid_vecs.append(cand_vec)
                valid_docs.append(cand_text)
                valid_metas.append({
                    "platform": c.get("platform", "unknown"),
                    "account": c.get("account", "unknown"),
                    "post_url": c.get("post_url", ""),
                    "created_utc": c.get("created_utc", "")
                })

                # Helper lambda for bound calibration
                def _cal(val: float, lo: float, hi: float) -> float:
                    return max(0.0, min(1.0, (val - lo) / max(1e-5, (hi - lo))))

                # A. Event semantic similarity (text-to-text in CLIP space)
                cal_event = None
                if event_context_vec:
                    sim_event = sum(a * b for a, b in zip(event_context_vec, cand_vec))
                    cal_event = _cal(sim_event, 0.45, 0.75)

                # B. Image visual cross-modal similarity
                cal_img = None
                if query_image_vec:
                    raw_img_sim = sum(a * b for a, b in zip(query_image_vec, cand_vec))
                    cal_img = _cal(raw_img_sim, 0.16, 0.32)

                # C. Generic description similarity
                cal_desc = None
                if query_text_vec:
                    raw_desc_sim = sum(a * b for a, b in zip(query_text_vec, cand_vec))
                    cal_desc = _cal(raw_desc_sim, 0.55, 0.90)

                # Dual-anchor fusion
                if cal_event is not None and cal_img is not None:
                    # Both event context and query image present
                    score = 0.15 + 0.80 * (0.65 * cal_event + 0.35 * cal_img)
                elif cal_event is not None:
                    score = 0.15 + 0.80 * cal_event
                elif cal_img is not None:
                    score = 0.15 + 0.80 * cal_img
                elif cal_desc is not None:
                    score = 0.15 + 0.80 * cal_desc

            # 2. Otherwise use TF-IDF cosine similarity
            elif tfidf_matrix is not None:
                sim_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[idx+1:idx+2])[0][0]
                score = round(min(0.95, max(0.15, sim_tfidf * 0.85 + 0.10)), 3)

            # 3. Simple token overlap fallback
            else:
                desc_words = set(reference_text.lower().split())
                cand_words = set(cand_text.lower().split())
                overlap = len(desc_words.intersection(cand_words))
                score = round(min(0.95, max(0.15, (overlap / max(1, len(desc_words))) * 0.85 + 0.10)), 3)

            score_rounded = round(min(0.98, max(0.05, score)), 3)

            # Media-content boost: prioritize posts that contain the actual image/video
            # over text-only articles about the event (§2.4 hackathon requirement)
            if c.get("has_media"):
                score_rounded = round(min(0.98, score_rounded + 0.05), 3)

            # Annotate candidate dict directly
            c["similarity"] = score_rounded

            # Populate similarity map
            url = c.get("post_url") or c.get("url") or ""
            plat = c.get("platform", "unknown")
            acc = c.get("account", "unknown")
            title = c.get("title", "")
            if url:
                similarity_map[url] = score_rounded
            norm_key = f"{plat}:{acc}:{title[:30]}"
            similarity_map[norm_key] = score_rounded
            similarity_map[f"cand_{idx}"] = score_rounded

            ranked.append(
                ShortlistCandidate(
                    platform=plat,
                    account=acc,
                    similarity=score_rounded,
                    post_url=url if url else None,
                    timestamp=c.get("created_utc"),
                    post_text=c.get("text") or title,
                    has_media=bool(c.get("has_media"))
                )
            )

        # Index all valid embeddings into ChromaDB vector store
        if valid_ids and valid_vecs:
            self.vector_store.add_records(
                ids=valid_ids,
                embeddings=valid_vecs,
                documents=valid_docs,
                metadatas=valid_metas
            )

        # Sort candidates descending by similarity score
        ranked.sort(key=lambda x: x.similarity, reverse=True)

        # Multi-platform diversity selection (§2.4d step 5):
        # Guarantee top suspects across distinct platforms (Reddit, X, YouTube, Web, Visual)
        top_candidates = []
        platform_buckets: Dict[str, List[ShortlistCandidate]] = {}
        for c in ranked:
            platform_buckets.setdefault(c.platform, []).append(c)

        # First pass: pick the top candidate from each searched platform
        for plat in sorted(platform_buckets.keys()):
            items = platform_buckets[plat]
            if items:
                top_candidates.append(items.pop(0))

        # Second pass: fill remaining top_k slots with highest similarity candidates
        remaining = []
        for items in platform_buckets.values():
            remaining.extend(items)
        remaining.sort(key=lambda x: x.similarity, reverse=True)

        for item in remaining:
            if len(top_candidates) >= top_k:
                break
            top_candidates.append(item)

        top_candidates.sort(key=lambda x: x.similarity, reverse=True)
        return top_candidates[:top_k], similarity_map

    def rerank(
        self,
        media_description: str,
        candidates: List[Dict[str, Any]],
        query_image_path: Optional[str] = None,
        event_context: Optional[str] = None,
        top_k: int = 5
    ) -> List[ShortlistCandidate]:
        """Reranks candidate hits and returns top_k most similar items."""
        ranked, _ = self.score_candidates(
            media_description=media_description,
            candidates=candidates,
            query_image_path=query_image_path,
            event_context=event_context,
            top_k=top_k
        )
        return ranked


# Alias for backward compatibility
SemanticReranker = MultimodalEmbedder
