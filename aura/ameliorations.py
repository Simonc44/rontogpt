"""
Intégration A+B+C+D+E dans le pipeline
======================================

Point d'entrée unique pour activer toutes les améliorations.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Set, TYPE_CHECKING

from . import pipeline_rontogpt
from .reponse_structuree import (
    ReponseStruct, pipeline_vers_structure, structure_vers_dict,
    GUARDRAILS_DEFAUT, GENERATEUR_DEFAUT
)
from .streaming import ReponseStreamer, TUIStreamAdapter, PieceStream, TypePiece
from .reasoning import ReasoningOrchestrator, ajouter_reasoning_au_pipeline
from .cache_hybride import CACHE_HYBRIDE, get_cache, set_cache as _set_cache, stats_cache

if TYPE_CHECKING:
    from .pipeline_rontogpt import PipelineMixin


# ──────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES (définies AVANT pour éviter les erreurs de portée)
# ──────────────────────────────────────────────────────────────────────────────

def pipeline_vers_dict_structure(reponse: ReponseStruct) -> dict:
    """Convertit réponse structurée → format pipeline (compatibilité)."""
    base = structure_vers_dict(reponse)
    base.setdefault("question", "")
    base.setdefault("experts", set())
    base.setdefault("analyse", {})
    base.setdefault("cerveau_choisi", "inconnu")
    base.setdefault("contexte_web", "")
    base.setdefault("contexte_docs", "")
    base.setdefault("formule", "")
    base.setdefault("erreur_pgs", None)
    base.setdefault("pensee", "")
    base.setdefault("provenance", {})
    base.setdefault("repli_propose", None)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def activer_ameliorations_pipeline(pipeline: "PipelineMixin"):
    """Active toutes les améliorations A+B+C+D+E sur une instance de pipeline."""

    # A+B: Structured output + Guardrails (déjà dans reponse_structuree)
    # Le pipeline existant continue de fonctionner, mais on ajoute
    # la conversion structurée à la fin de executer_detaille

    original_executer = pipeline.executer_detaille

    # Wrapper qui capture original_executer via closure
    def executer_avec_structure_wrapped(self: "PipelineMixin", question: str, X=None, y=None) -> dict:
        # 1. Vérification cache hybride (E)
        if hasattr(self, "_cle_reponse"):
            cached = get_cache(question, X, y)
            if cached is not None:
                logging.getLogger("aura.pipeline").info("[cache] Hit hybride")
                return pipeline_vers_dict_structure(cached)

        # 2. Exécution pipeline normal (original_executer est déjà bindé à self)
        resultat = original_executer(question, X, y)

        # 3. Conversion structurée (A)
        reponse_struct = pipeline_vers_structure(resultat)

        # 4. Guardrails (B) - validation + correction
        reponse_struct, erreurs = GUARDRAILS_DEFAUT.valider_et_corriger(reponse_struct)
        if erreurs:
            logging.getLogger("aura.guardrails").warning(f"Guardrails: {erreurs}")

        # 5. Stockage cache (E)
        experts: Set[str] = set(resultat.get("experts", []))
        _set_cache(question, reponse_struct, X, y, experts)

        # 6. Retour format pipeline standard (compatibilité)
        return resultat

    # Monkey-patch: on attache le wrapper à l'instance
    pipeline.executer_detaille = executer_avec_structure_wrapped  # type: ignore[assignment]

    # C: Streaming - ajout d'un streamer (stocké sur l'instance)
    object.__setattr__(pipeline, "_streamer", ReponseStreamer(pipeline))

    # D: Reasoning multi-étapes
    ajouter_reasoning_au_pipeline(pipeline)

    return pipeline


def get_streamer(pipeline: "PipelineMixin") -> ReponseStreamer:
    """Récupère le streamer pour le TUI."""
    if not hasattr(pipeline, "_streamer"):
        object.__setattr__(pipeline, "_streamer", ReponseStreamer(pipeline))
    return pipeline._streamer  # type: ignore[attr-defined]


def activer_tout(pipeline: "PipelineMixin") -> "PipelineMixin":
    """One-liner pour tout activer."""
    return activer_ameliorations_pipeline(pipeline)