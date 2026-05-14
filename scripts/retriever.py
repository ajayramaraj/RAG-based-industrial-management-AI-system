"""
retriever.py — FAISS + AllMiniLM-L6-v2 semantic retrieval engine.
Builds and queries a vector index over CSV rows.
Strictly retrieves ONLY from the uploaded CSV — no external knowledge.
"""
import os
import gc
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.utils import INDEX_DIR, df_to_texts, detect_column_roles, row_to_text

# ─── Model (loaded once, cached) ──────────────────────────────────────────────
_MODEL = None

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


# ─── Retriever Class ──────────────────────────────────────────────────────────
class FAISSRetriever:
    """
    Builds a FAISS flat index over CSV row embeddings.
    Retrieves top-k most relevant rows for a natural-language query.
    """

    def __init__(self, fhash: str):
        self.fhash      = fhash
        self.index      = None
        self.texts      = []       # raw text per row
        self.row_ids    = []       # original df row indices
        self.dim        = 384      # MiniLM output dim
        self._index_path = INDEX_DIR / f"{fhash}.faiss"
        self._meta_path  = INDEX_DIR / f"{fhash}_meta.pkl"

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame, col_roles: dict,
              batch_size: int = 512, progress_cb=None):
        """
        Embed all rows and build FAISS index.
        Handles 40k rows in batches to avoid OOM.
        """
        model = get_model()
        texts = df_to_texts(df, col_roles)
        self.texts   = texts
        self.row_ids = list(df.index)

        total   = len(texts)
        all_emb = np.zeros((total, self.dim), dtype="float32")

        for start in range(0, total, batch_size):
            end   = min(start + batch_size, total)
            batch = texts[start:end]
            embs  = model.encode(batch, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
            all_emb[start:end] = embs
            if progress_cb:
                progress_cb(end / total)
            gc.collect()

        # Inner-product index (cosine on normalized vecs)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.index)
        ids = np.arange(total, dtype="int64")
        self.index.add_with_ids(all_emb, ids)

        self._save()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 15) -> list[dict]:
        """
        Semantic search. Returns list of {rank, row_id, text, score}.
        """
        if self.index is None:
            self._load()
        model = get_model()
        q_emb = model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(q_emb, k)
        results = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({
                "rank":   rank + 1,
                "row_id": int(self.row_ids[idx]),
                "text":   self.texts[idx],
                "score":  float(score),
            })
        return results

    def retrieve_rows(self, query: str, df: pd.DataFrame, k: int = 15) -> pd.DataFrame:
        """Returns actual DataFrame rows matching the query."""
        hits = self.retrieve(query, k)
        if not hits:
            return pd.DataFrame()
        row_ids = [h["row_id"] for h in hits]
        valid   = [r for r in row_ids if r in df.index]
        return df.loc[valid].copy()

    # ── Persist ───────────────────────────────────────────────────────────────
    def _save(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump({"texts": self.texts, "row_ids": self.row_ids}, f)

    def _load(self):
        if not self._index_path.exists():
            raise FileNotFoundError("FAISS index not found. Re-upload CSV.")
        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            meta = pickle.load(f)
        self.texts   = meta["texts"]
        self.row_ids = meta["row_ids"]

    def is_cached(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()

    def load_if_cached(self) -> bool:
        if self.is_cached():
            self._load()
            return True
        return False