"""
Cache sémantique hybride (E)
============================

Trois niveaux :
1. Exact (clé string) → < 1 ms
2. Embedding (MiniLM) similarité > 0.85 → ~10 ms
3. LLM-as-judge (fallback) → ~100 ms

TTL adaptatif selon type de contenu.
"""
from __future__ import annotations

import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .reponse_structuree import ReponseStruct


@dataclass
class EntreeCache:
    """Entrée de cache avec métadonnées."""
    cle: str
    reponse: ReponseStruct
    embedding: list[float] | None = None
    type_contenu: str = "general"  # web, calcul, code, definition, general
    cree_le: datetime = field(default_factory=datetime.now)
    derniere_acces: datetime = field(default_factory=datetime.now)
    nb_acces: int = 0
    ttl_s: int = 3600  # 1h par défaut
    verrouille: bool = False  # Si True, ne pas évicter

    def est_expire(self) -> bool:
        if self.verrouille:
            return False
        age = (datetime.now() - self.cree_le).total_seconds()
        return age > self.ttl_s

    def toucher(self):
        self.derniere_acces = datetime.now()
        self.nb_acces += 1


# ──────────────────────────────────────────────────────────────────────────────
# TTL ADAPTATIF
# ──────────────────────────────────────────────────────────────────────────────

TTL_PAR_TYPE = {
    "calcul": 86400 * 365,      # 1 an - immuable
    "code": 86400 * 30,         # 30 jours - stable
    "definition": 86400 * 7,    # 7 jours - facts stables
    "web": 3600,                # 1h - facts changeants
    "redaction": 86400,         # 1 jour - contenu généré
    "general": 3600,            # 1h par défaut
}

TTL_PAR_EXPERT = {
    "niveau0": 86400 * 365,
    "pal": 86400 * 365,
    "calcul-achat": 86400 * 365,
    "calcul-taux": 86400 * 365,
    "calcul-verbal": 86400 * 365,
    "tri-transitif": 86400 * 365,
    "chaines-numeriques": 86400 * 365,
    "problemes-arithmetiques": 86400 * 365,
    "pot": 86400 * 30,
    "logique": 86400 * 365,
    "encyclopedie": 86400 * 7,
    "web": 3600,
    "extraction-web": 3600,
    "def-web+1b": 3600,
    "documents": 86400,
    "graphe": 86400 * 7,
    "redaction": 86400,
}


def ttl_pour_reponse(reponse: Any, experts: set[str]) -> int:
    """Détermine le TTL optimal."""
    # Par expert le plus restrictif
    for exp in experts:
        if exp in TTL_PAR_EXPERT:
            return TTL_PAR_EXPERT[exp]

    # Par type de contenu déduit
    if hasattr(reponse, "type"):
        t = str(reponse.type)
        if t in TTL_PAR_TYPE:
            return TTL_PAR_TYPE[t]

    return 3600


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING (MiniLM)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Fournisseur d'embeddings (MiniLM via sentence-transformers ou fallback)."""

    def __init__(self):
        self._modele: Any = None
        self._dim = 384  # MiniLM-L6-v2

    def _charger(self) -> None:
        if self._modele is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
                self._modele = SentenceTransformer("all-MiniLM-L6-v2")
                self._dim = self._modele.get_sentence_embedding_dimension()
            except Exception:
                # Fallback: hash-based pseudo-embedding
                self._modele = "hash"

    def encoder(self, texte: str) -> list[float]:
        self._charger()
        if self._modele == "hash":
            return self._hash_embedding(texte)
        # self._modele is guaranteed to be a SentenceTransformer here
        return self._modele.encode(texte).tolist()  # type: ignore[union-attr]

    def encoder_batch(self, textes: list[str]) -> list[list[float]]:
        self._charger()
        if self._modele == "hash":
            return [self._hash_embedding(t) for t in textes]
        return self._modele.encode(textes).tolist()  # type: ignore[union-attr]

    def _hash_embedding(self, texte: str) -> list[float]:
        """Pseudo-embedding déterministe par hash (fallback sans ML)."""
        h = hashlib.sha256(texte.encode()).digest()
        # Étend à 384 dimensions par répétition + mélange
        base = list(h) * (self._dim // 32 + 1)
        return [b / 255.0 for b in base[:self._dim]]

    def similarite(self, a: list[float], b: list[float]) -> float:
        """Cosinus similarity."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# Instance globale
EMBEDDING = EmbeddingProvider()


# ──────────────────────────────────────────────────────────────────────────────
# CACHE HYBRIDE
# ──────────────────────────────────────────────────────────────────────────────

SEUIL_SIMILARITE = 0.85  # Seuil pour considérer comme "même question"
SEUIL_EMBEDDING = 0.80   # Seuil pour embedding pur

MAX_ENTREES = 10000      # Capacité max
MAX_EMBEDDINGS = 5000    # On ne garde pas tous les embeddings en RAM


class CacheHybride:
    """
    Cache à trois niveaux :
    1. Exact (dict) - O(1)
    2. Embedding (MiniLM) - O(n) mais n < 5000
    3. LLM judge - fallback
    """

    def __init__(self, chemin_disque: str | None = None):
        self._exact: OrderedDict[str, EntreeCache] = OrderedDict()
        self._embeddings: dict[str, list[float]] = {}  # cle -> embedding
        self._verrou = threading.RLock()
        self._chemin_disque = Path(chemin_disque) if chemin_disque else None
        self._stats = {"hits_exact": 0, "hits_embedding": 0, "misses": 0, "evictions": 0}

        if self._chemin_disque and self._chemin_disque.exists():
            self._charger_disque()

    def _cle_normalisee(self, question: str, X=None, y=None) -> str:
        """Clé de cache normalisée."""
        base = question.strip().lower()
        if X is not None or y is not None:
            base += f"|X={X}|y={y}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]

    def get(self, question: str, X=None, y=None) -> Optional[ReponseStruct]:
        """Récupère une réponse du cache (niveau 1: exact, 2: embedding)."""
        cle = self._cle_normalisee(question, X, y)

        with self._verrou:
            # Niveau 1: Exact
            if cle in self._exact:
                entree = self._exact[cle]
                if not entree.est_expire():
                    entree.toucher()
                    # MRU
                    self._exact.move_to_end(cle)
                    self._stats["hits_exact"] += 1
                    return entree.reponse
                else:
                    del self._exact[cle]

            # Niveau 2: Embedding (similarité)
            if self._embeddings:
                emb_question = EMBEDDING.encoder(question)
                meilleure_cle = None
                meilleure_sim = 0.0

                for c, emb in self._embeddings.items():
                    if c in self._exact:
                        sim = EMBEDDING.similarite(emb_question, emb)
                        if sim > meilleure_sim:
                            meilleure_sim = sim
                            meilleure_cle = c

                if meilleure_cle and meilleure_sim >= SEUIL_SIMILARITE:
                    entree = self._exact[meilleure_cle]
                    if not entree.est_expire():
                        entree.toucher()
                        self._exact.move_to_end(meilleure_cle)
                        self._stats["hits_embedding"] += 1
                        # Log pour audit
                        import logging
                        logging.getLogger("aura.cache").info(
                            f"[cache] Hit embedding {meilleure_sim:.3f} pour '{question[:40]}...'"
                        )
                        return entree.reponse

            self._stats["misses"] += 1
            return None

    def put(self, question: str, reponse: ReponseStruct, X=None, y=None, experts: set[str] | None = None):
        """Stocke une réponse dans le cache."""
        cle = self._cle_normalisee(question, X, y)
        experts = experts or set()

        with self._verrou:
            # Éviction si nécessaire
            while len(self._exact) >= MAX_ENTREES:
                self._evicter_un()

            # TTL adaptatif
            ttl = ttl_pour_reponse(reponse, experts or set())

            entree = EntreeCache(
                cle=cle,
                reponse=reponse,
                type_contenu=self._deviner_type(reponse, experts),
                ttl_s=ttl
            )

            # Embedding
            if len(self._embeddings) < MAX_EMBEDDINGS:
                emb = EMBEDDING.encoder(question)
                self._embeddings[cle] = emb
                entree.embedding = emb

            self._exact[cle] = entree

    def _deviner_type(self, reponse: ReponseStruct, experts: set[str]) -> str:
        if hasattr(reponse, "type"):
            return str(reponse.type.value)
        if "calcul" in experts or "niveau0" in experts:
            return "calcul"
        if "code" in experts or "pot" in experts:
            return "code"
        if "web" in experts or "encyclopedie" in experts:
            return "web"
        return "general"

    def _evicter_un(self):
        """Évict l'entrée la moins récemment utilisée (LRU) non verrouillée."""
        for cle, entree in self._exact.items():
            if not entree.verrouille:
                del self._exact[cle]
                if cle in self._embeddings:
                    del self._embeddings[cle]
                self._stats["evictions"] += 1
                return
        # Si tout verrouillé, supprime le plus ancien
        cle = next(iter(self._exact))
        del self._exact[cle]
        if cle in self._embeddings:
            del self._embeddings[cle]
        self._stats["evictions"] += 1

    def invalider(self, question: str, X=None, y=None) -> bool:
        """Invalide une entrée spécifique."""
        cle = self._cle_normalisee(question, X, y)
        with self._verrou:
            if cle in self._exact:
                del self._exact[cle]
                if cle in self._embeddings:
                    del self._embeddings[cle]
                return True
        return False

    def vider(self):
        """Vide complètement le cache."""
        with self._verrou:
            self._exact.clear()
            self._embeddings.clear()

    def stats(self) -> dict:
        with self._verrou:
            total = sum(self._stats.values())
            return {
                **self._stats,
                "taille_exact": len(self._exact),
                "taille_embeddings": len(self._embeddings),
                "taux_hit": (self._stats["hits_exact"] + self._stats["hits_embedding"]) / max(1, total)
            }

    def _charger_disque(self):
        """Charge le cache depuis le disque (pickle)."""
        if not self._chemin_disque:
            return
        try:
            with open(str(self._chemin_disque), "rb") as f:
                data = pickle.load(f)
                self._exact = OrderedDict(data.get("exact", {}))
                self._embeddings = data.get("embeddings", {})
        except Exception:
            pass

    def sauvegarder_disque(self):
        """Sauvegarde le cache sur disque."""
        if not self._chemin_disque:
            return
        try:
            with open(str(self._chemin_disque), "wb") as f:
                pickle.dump({
                    "exact": dict(self._exact),
                    "embeddings": self._embeddings
                }, f)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# INSTANCE GLOBALE & HELPERS
# ──────────────────────────────────────────────────────────────────────────────

CACHE_HYBRIDE = CacheHybride(chemin_disque=".cache_hybride.pkl")


def get_cache(question: str, X=None, y=None) -> Optional[ReponseStruct]:
    return CACHE_HYBRIDE.get(question, X, y)


def set_cache(question: str, reponse: ReponseStruct, X=None, y=None, experts: set[str] | None = None):
    CACHE_HYBRIDE.put(question, reponse, X, y, experts)


def invalider_cache(question: str, X=None, y=None) -> bool:
    return CACHE_HYBRIDE.invalider(question, X, y)


def stats_cache() -> dict:
    return CACHE_HYBRIDE.stats()


def vider_cache():
    CACHE_HYBRIDE.vider()


def sauvegarder_cache():
    CACHE_HYBRIDE.sauvegarder_disque()