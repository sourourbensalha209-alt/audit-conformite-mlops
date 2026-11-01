"""Index de recherche des articles : BM25, dense, ou hybride (fusion RRF).

Mise a jour incrementale : quand une loi change, seuls les articles dont le texte a
change sont re-encodes. Le reste vient du cache d'embeddings. C'est ce qui permet a
l'application de "prendre en compte une nouvelle loi automatiquement" sans
reentrainer quoi que ce soit : la connaissance juridique vit dans l'index, pas dans les poids.

    index = HybridIndex(embedder=SentenceTransformerEmbedder(), cache_path="data/corpus/emb.npz")
    index.ensure_fresh(store)                       # a appeler au demarrage ET avant chaque audit
    hits = index.search("liability shall not exceed fees paid", k=5)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Article, CorpusStore, corpus_version

_STOP = {
    "les", "des", "une", "dans", "pour", "par", "sur", "aux", "est", "sont", "qui", "que", "ces",
    "son", "ses", "leur", "sans", "avec", "tout", "toute", "tous", "the", "and", "for", "with",
    "shall", "this", "that", "which", "are", "from", "such", "any", "may", "not", "has", "have",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    s = strip_accents(text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", s) if len(t) > 2 and t not in _STOP]


# --------------------------------------------------------------------------- #
# Encodeurs
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """E5 multilingue : la clause est en anglais, les articles de loi en francais."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer  # import tardif : lourd

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)   # BGE-M3 accepte 8192 : inutile ici
        # E5 exige les préfixes "query: " / "passage: " ; BGE-M3 et la plupart des autres n'en veulent pas.
        self.use_prefix = "e5" in model_name.lower()

    def _enc(self, prefix: str, texts: list[str]) -> np.ndarray:
        if self.use_prefix:
            texts = [f"{prefix}: {t}" for t in texts]
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False),
            dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc("passage", texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc("query", texts)


class HashingEmbedder:
    """Encodeur de secours, sans modele ni reseau (tests, demo hors ligne).
    Ce n'est PAS un encodeur semantique : il ne remplace pas E5/BGE dans les mesures."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_passages


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


@dataclass
class Hit:
    article: Article
    score: float
    via: str  # "bm25" | "dense" | "hybrid"


def doc_text(a: Article) -> str:
    return f"{a.law_title} - art. {a.number}. {a.keywords} {a.text}".strip()


class HybridIndex:
    RRF_K = 60

    def __init__(self, embedder: Embedder | None = None, cache_path: Path | str | None = None):
        self.embedder = embedder
        self.cache_path = Path(cache_path) if cache_path else None
        self.version: str | None = None
        self.n_embedded_last_refresh = 0
        self._cache: dict[str, np.ndarray] = {}
        self._articles: list[Article] = []
        self._bm25: BM25Okapi | None = None
        self._matrix: np.ndarray | None = None
        self._load_cache()

    # -- cache d'embeddings (cle = empreinte du texte indexe)
    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            self._cache = dict(zip(data["keys"].tolist(), data["vectors"]))

    def _save_cache(self, live_keys: set[str]) -> None:
        if not self.cache_path:
            return
        self._cache = {k: v for k, v in self._cache.items() if k in live_keys}  # purge des textes disparus
        if self._cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path,
                                keys=np.array(list(self._cache.keys())),
                                vectors=np.stack(list(self._cache.values())))

    # -- construction / mise a jour
    def refresh(self, articles: dict[str, Article]) -> int:
        """Reconstruit l'index. Retourne le nombre d'articles reellement re-encodes."""
        live = sorted((a for a in articles.values() if a.status == "in_force"), key=lambda a: a.article_id)
        self._articles = live
        texts = [doc_text(a) for a in live]
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

        tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokens) if any(tokens) else None

        embedded = 0
        if self.embedder and live:
            missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
            if missing:
                vecs = self.embedder.embed_passages([t for _, t in missing])
                for (k, _), v in zip(missing, vecs):
                    self._cache[k] = v
                embedded = len(missing)
            self._matrix = np.stack([self._cache[k] for k in keys])
        else:
            self._matrix = None

        self._save_cache(set(keys))
        self.version = corpus_version(articles)
        self.n_embedded_last_refresh = embedded
        return embedded

    def ensure_fresh(self, store: CorpusStore) -> bool:
        """A appeler avant un audit : si la veille a modifie le corpus, l'index se recharge seul."""
        articles = store.load()
        if corpus_version(articles) != self.version:
            self.refresh(articles)
            return True
        return False

    # -- recherche
    def search(
        self,
        query: str,
        k: int = 5,
        *,
        mode: str = "hybrid",
        jurisdictions: set[str] | None = None,
        validated_only: bool = False,
    ) -> list[Hit]:
        if not self._articles:
            return []
        allowed = np.array([
            (jurisdictions is None or a.jurisdiction in jurisdictions) and (a.validated or not validated_only)
            for a in self._articles
        ])
        if mode == "hybrid" and self._matrix is None:
            mode = "bm25"
        if mode == "dense" and self._matrix is None:
            raise RuntimeError("Aucun encodeur configure : utiliser mode='bm25'.")

        rankings: list[tuple[str, list[int]]] = []
        if mode in ("bm25", "hybrid") and self._bm25 is not None:
            scores = np.asarray(self._bm25.get_scores(tokenize(query)))
            rankings.append(("bm25", self._rank(scores, allowed, positive_only=True)))
        if mode in ("dense", "hybrid"):
            q = self.embedder.embed_queries([query])[0]
            rankings.append(("dense", self._rank(self._matrix @ q, allowed)))

        fused: dict[int, float] = {}
        for _, ranked in rankings:
            for pos, idx in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.RRF_K + pos + 1)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self._articles[i], s, mode) for i, s in best]

    @staticmethod
    def _rank(scores: np.ndarray, allowed: np.ndarray, positive_only: bool = False, top: int = 50) -> list[int]:
        masked = np.where(allowed, scores, -np.inf)
        order = np.argsort(-masked)[:top]
        return [int(i) for i in order if np.isfinite(masked[i]) and (not positive_only or masked[i] > 0)]
