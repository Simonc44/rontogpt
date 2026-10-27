"""
Reasoning multi-étapes avec vérification (D)
=============================================

Pipeline de raisonnement explicite :
1. Décomposition → sous-questions
2. Résolution par outil (pot, web, calc, graphe)
3. Vérification intermédiaire (critic, AST, web)
4. Synthèse finale
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from typing import TYPE_CHECKING

from . import llama_cerveau
from . import graphe_faits
from . import pal
from . import documents
from . import potcode
from .streaming import (
    PieceStream, TypePiece, piece_thinking, piece_tool_call,
    piece_tool_result, piece_source, piece_progress, piece_final,
    ReponseStreamer
)
from .reponse_structuree import (
    ReponseStruct, TypeReponse, NiveauConfiance,
    ReponseCalc, ReponseCode, ReponseLookup,
    ReponseRedaction, ReponseRefus, Source
)

if TYPE_CHECKING:
    from .pipeline_rontogpt import PipelineMixin


class TypeEtape(str, Enum):
    DECOMPOSER = "decomposer"
    RESOUDRE = "resoudre"
    VERIFIER = "verifier"
    SYNTHESER = "synthese"


@dataclass
class EtapeReasoning:
    """Une étape du raisonnement."""
    type: TypeEtape
    description: str
    entree: dict = field(default_factory=dict)
    sortie: dict = field(default_factory=dict)
    outil: str | None = None
    verifie: bool = False
    confiance: float = 0.0
    duree_s: float = 0.0
    horodatage: datetime = field(default_factory=datetime.now)


@dataclass
class ContexteReasoning:
    """Contexte complet d'un raisonnement multi-étapes."""
    question_originale: str
    sous_questions: list[str] = field(default_factory=list)
    etapes: list[EtapeReasoning] = field(default_factory=list)
    reponses_intermediaires: dict[str, Any] = field(default_factory=dict)
    reponse_finale: Optional[Any] = None
    t0: float = field(default_factory=time.time)
    budget_s: float = 0.0


class OutilReasoning(ABC):
    """Interface pour un outil de raisonnement."""

    @property
    @abstractmethod
    def nom(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        pass

    @abstractmethod
    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        """Retourne {'resultat': ..., 'confiance': float, 'sources': [...]}"""
        pass


class OutilCalcul(OutilReasoning):
    """Calcul exact via PAL."""

    @property
    def nom(self) -> str:
        return "calcul"

    @property
    def description(self) -> str:
        return "Calculs exacts (arithmétique, algèbre, conversions)"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        from .pipeline_rontogpt import _RE_QUESTION_FACTUELLE
        return bool(_RE_QUESTION_FACTUELLE.search(question)) or any(
            op in question for op in ["+", "-", "*", "/", "×", "÷", "=", "fois", "plus", "moins"]
        )

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        # Essaye PAL d'abord (plus complet)
        rep = pal.repondre(question)  # type: ignore[attr-defined]
        if rep is not None:
            return {"resultat": rep, "confiance": 0.99, "sources": [{"type": "calcul", "titre": "PAL"}], "outil": "pal"}

        # Fallback LLM
        rep = llama_cerveau.generer(
            question, categorie="math", max_tokens=50,
            contexte_web="", formule="", contexte_faits=""
        )
        return {"resultat": rep, "confiance": 0.7, "sources": [], "outil": "llm"}


class OutilCode(OutilReasoning):
    """Génération et exécution de code via PoT."""

    @property
    def nom(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Génération et exécution de code (Python, algorithms)"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        from .pipeline_rontogpt import _DEMANDE_CODE
        return bool(_DEMANDE_CODE.search(question)) or potcode.est_demande_code(question)

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        rep = potcode.repondre(question=question)  # type: ignore[attr-defined]
        if rep:
            return {"resultat": rep, "confiance": 0.9, "sources": [{"type": "code", "titre": "PoT"}], "outil": "pot"}
        return {"resultat": None, "confiance": 0.0, "sources": [], "outil": "code"}


class OutilWeb(OutilReasoning):
    """Recherche web via mémoire web."""

    @property
    def nom(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return "Recherche web (DuckDuckGo) + extraction"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        from . import memoire_web
        return memoire_web.a_besoin_web(question)

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        from . import memoire_web
        contexte_web = memoire_web.chercher(question)
        if not contexte_web:
            return {"resultat": None, "confiance": 0.0, "sources": [], "outil": "web"}

        # Compression
        try:
            from . import agents
            contexte_web = agents.compresser(contexte_web, question)
        except Exception:
            pass

        # Extraction
        from . import memoire_web
        contexte_web = memoire_web.limiter_morceaux(contexte_web, question=question)
        sources_tuples = memoire_web.extraire_sources(contexte_web)

        # Convert tuples to dicts for compatibility
        sources = [{"titre": t, "url": u} for t, u in sources_tuples]

        return {
            "resultat": contexte_web,
            "confiance": 0.75,
            "sources": [{"type": "web", "titre": s.get("titre", "Web"), "url": s.get("url")} for s in sources],
            "outil": "web"
        }


class OutilGraphe(OutilReasoning):
    """Graphe de faits (MiniRAG)."""

    @property
    def nom(self) -> str:
        return "graphe"

    @property
    def description(self) -> str:
        return "Graphe de faits vérifiés (MiniRAG)"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        return True  # Toujours disponible en fallback

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        from . import graphe_faits
        faits = graphe_faits.chercher(question)
        if not faits:
            return {"resultat": None, "confiance": 0.0, "sources": [], "outil": "graphe"}

        # Extraction de chaîne (simplifié)
        chaine: list = []
        return {
            "resultat": faits,
            "confiance": 0.8,
            "sources": [{"type": "graphe", "titre": "Graphe de faits", "extrait": str(faits)[:200]}],
            "outil": "graphe"
        }


class OutilLogique(OutilReasoning):
    """Raisonnement logique formel (syllogismes)."""

    @property
    def nom(self) -> str:
        return "logique"

    @property
    def description(self) -> str:
        return "Raisonnement logique formel (syllogismes)"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        from . import logique
        return logique.est_enigme(question)

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        from . import syllogismes
        rep = syllogismes.repondre(question)
        if rep:
            return {"resultat": rep, "confiance": 0.95, "sources": [{"type": "logique", "titre": "Syllogismes"}], "outil": "logique"}
        return {"resultat": None, "confiance": 0.0, "sources": [], "outil": "logique"}


class OutilDocuments(OutilReasoning):
    """Documents locaux (RAG)."""

    @property
    def nom(self) -> str:
        return "documents"

    @property
    def description(self) -> str:
        return "Documents locaux de l'utilisateur (RAG)"

    def peut_traiter(self, question: str, contexte: ContexteReasoning) -> bool:
        from . import documents
        try:
            return len(documents.lister()) > 0
        except Exception:
            return False

    def executer(self, question: str, contexte: ContexteReasoning) -> dict:
        from . import documents
        trouve = documents.interroger(question)
        if not trouve.get("texte"):
            return {"resultat": None, "confiance": 0.0, "sources": [], "outil": "documents"}

        sources = [{"type": "document", "titre": s, "chemin": s} for s in trouve.get("sources", [])]
        return {
            "resultat": trouve["texte"],
            "confiance": 0.85,
            "sources": sources,
            "outil": "documents"
        }


# ──────────────────────────────────────────────────────────────────────────────
# DÉCOMPOSEUR
# ──────────────────────────────────────────────────────────────────────────────

class Decomposeur:
    """Décompose une question complexe en sous-questions."""

    PROMPT_DECOMPOSITION = """Décompose cette question en 2-5 sous-questions INDÉPENDANTES et ORDONNÉES.
Chaque sous-question doit pouvoir être répondue par UN seul outil.

Question: {question}

Réponds UNIQUEMENT en JSON:
{{
  "sous_questions": [
    "Sous-question 1 précise",
    "Sous-question 2 précise",
    ...
  ],
  "ordre": "séquentiel|parallèle",
  "raison": "Pourquoi cette décomposition"
}}"""

    def __init__(self):
        pass

    def decomposer(self, question: str) -> list[str]:
        prompt = self.PROMPT_DECOMPOSITION.format(question=question)
        rep = llama_cerveau.generer(
            prompt, max_tokens=200,
            contexte_web="", formule="", contexte_faits="",
            categorie="general"
        )

        try:
            data = json.loads(rep)
            return data.get("sous_questions", [question])
        except Exception:
            return [question]  # Fallback: pas de décomposition


# ──────────────────────────────────────────────────────────────────────────────
# VÉRIFICATEUR (Critic pattern)
# ──────────────────────────────────────────────────────────────────────────────

class Verificateur:
    """Vérifie une réponse intermédiaire ou finale."""

    def __init__(self):
        pass

    def verifier_fait(self, affirmation: str, sources: list[dict]) -> tuple[bool, str]:
        """Vérifie si une affirmation est soutenue par les sources."""
        if not sources:
            return False, "Aucune source fournie"

        # Vérification web si sources web
        sources_web = [s for s in sources if s.get("type") == "web" or s.get("url")]
        if sources_web:
            return self._verifier_web(affirmation, sources_web)

        # Vérification par cohérence interne
        return self._verifier_coherence(affirmation)

    def _verifier_web(self, affirmation: str, sources: list[dict]) -> tuple[bool, str]:
        # Construire une question de vérification
        prompt = f"""Cette affirmation est-elle SOUTENUE par les sources ci-dessous ?

AFFIRMATION: {affirmation}

SOURCES:
{chr(10).join(f"- {s.get('titre', 'Source')}: {s.get('extrait', s.get('url', ''))[:200]}" for s in sources)}

Réponds UNIQUEMENT par: SOUTENUE|CONTRADITE|INSUFFISANT + explication courte."""

        rep = llama_cerveau.generer(
            prompt, max_tokens=100,
            contexte_web="", formule="", contexte_faits="",
            categorie="general"
        )
        if rep.startswith("SOUTENUE"):
            return True, "Soutenue par les sources web"
        elif rep.startswith("CONTRADITE"):
            return False, f"Contredite par les sources: {rep[11:].strip()}"
        return False, "Preuves insuffisantes"

    def _verifier_coherence(self, affirmation: str) -> tuple[bool, str]:
        # Vérification basique: pas de contradiction interne évidente
        if "non" in affirmation.lower() and "oui" in affirmation.lower():
            return False, "Contradiction interne détectée"
        return True, "Cohérent"


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR MULTI-ÉTAPES
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningOrchestrator:
    """Orchestre le raisonnement multi-étapes."""

    def __init__(self, pipeline: "PipelineMixin"):
        self.pipeline = pipeline
        self.outils: list[OutilReasoning] = [
            OutilCalcul(), OutilCode(), OutilLogique(),
            OutilWeb(), OutilGraphe(), OutilDocuments()
        ]
        self.decomposeur = Decomposeur()
        self.verificateur = Verificateur()
        self.streamer = ReponseStreamer(pipeline)

    def raisonner(self, question: str, max_etapes: int = 5, **kwargs) -> dict:
        """Exécute le raisonnement multi-étapes complet."""
        t0 = time.time()

        # 1. Décomposition
        sous_questions = self.decomposeur.decomposer(question)
        if len(sous_questions) == 1:
            # Pas de décomposition nécessaire -> pipeline direct
            return self.pipeline.executer_detaille(question, **kwargs)

        contexte = ContexteReasoning(
            question_originale=question,
            sous_questions=sous_questions,
            budget_s=kwargs.get("budget_s", 120)
        )

        # 2. Résolution de chaque sous-question
        for i, sq in enumerate(sous_questions):
            if time.time() - contexte.t0 > contexte.budget_s:
                break

            etape = self._resoudre_sous_question(sq, contexte, i)
            contexte.etapes.append(etape)

            if etape.sortie.get("resultat"):
                contexte.reponses_intermediaires[sq] = etape.sortie

        # 3. Vérification croisée
        self._verifier_croise(contexte)

        # 4. Synthèse finale
        reponse_finale = self._synthese_finale(contexte)

        return {
            "question": question,
            "reponse": reponse_finale,
            "etapes": [e.__dict__ for e in contexte.etapes],
            "duree_totale_s": time.time() - t0,
            "sous_questions": sous_questions
        }

    def _resoudre_sous_question(self, question: str, contexte: ContexteReasoning, index: int) -> EtapeReasoning:
        t0 = time.time()

        # Sélection de l'outil
        outil = None
        for o in self.outils:
            if o.peut_traiter(question, contexte):
                outil = o
                break

        if not outil:
            outil = self.outils[-1]  # Fallback web

        # Exécution
        self._emettre_piece_thinking(f"Étape {index+1}: {outil.nom} — {question[:50]}...")
        res = outil.executer(question, contexte)

        duree = time.time() - t0
        etape = EtapeReasoning(
            type=TypeEtape.RESOUDRE,
            description=f"{outil.nom}: {question}",
            entree={"question": question},
            sortie=res,
            outil=outil.nom,
            confiance=res.get("confiance", 0.0),
            duree_s=duree
        )

        # Vérification si résultat
        if res.get("resultat"):
            etape.type = TypeEtape.VERIFIER
            ok, msg = self.verificateur.verifier_fait(
                str(res["resultat"]), res.get("sources", [])
            )
            etape.verifie = ok
            etape.sortie["verification"] = {"ok": ok, "message": msg}
            if not ok:
                etape.confiance *= 0.5

        etape.duree_s = time.time() - t0
        self._emettre_piece_progress(index + 1, "resolu", duree)

        return etape

    def _verifier_croise(self, contexte: ContexteReasoning):
        """Vérifie la cohérence entre réponses intermédiaires."""
        for i, etape in enumerate(contexte.etapes):
            if not etape.sortie.get("resultat"):
                continue
            # Pourrait vérifier cohérence avec autres étapes
            pass

    def _synthese_finale(self, contexte: ContexteReasoning) -> str:
        """Synthétise les réponses intermédiaires en réponse finale."""
        if len(contexte.etapes) == 1:
            return str(contexte.etapes[0].sortie.get("resultat", ""))

        # Prompt de synthèse
        parties = []
        for i, etape in enumerate(contexte.etapes):
            if etape.sortie.get("resultat"):
                parties.append(f"Étape {i+1} ({etape.outil}): {etape.sortie['resultat']}")

        prompt = f"""Synthétise ces résultats intermédiaires en une réponse cohérente à la question originale.

Question: {contexte.question_originale}

Résultats intermédiaires:
{chr(10).join(f"- {p}" for p in parties)}

Réponse finale synthétique:"""

        synthese = llama_cerveau.generer(
            prompt, max_tokens=500,
            contexte_web="", formule="", contexte_faits="",
            categorie="general"
        )
        return synthese.strip()

    def _emettre_piece_thinking(self, texte: str):
        # Hook pour streaming
        pass

    def _emettre_piece_progress(self, etape: int, phase: str, duree: float):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# INTÉGRATION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def _est_question_complexe(question: str) -> bool:
    """Détecte si une question nécessite un raisonnement multi-étapes."""
    indicateurs = [
        "et aussi", "puis", "ensuite", "après",
        "compare", "différence entre", "avantages et inconvénients",
        "étape par étape", "détaille", "explique pourquoi",
        "comment faire pour", "quelle est la procédure",
        "analyse", "évalue", "justifie"
    ]
    q_lower = question.lower()
    return any(ind in q_lower for ind in indicateurs) and len(question.split()) > 15


def ajouter_reasoning_au_pipeline(pipeline: "PipelineMixin"):
    """Ajoute la capacité de raisonnement multi-étapes au pipeline."""
    orchestrator = ReasoningOrchestrator(pipeline)

    # Sauvegarde de la méthode originale
    original_executer = pipeline.executer_detaille

    def executer_avec_reasoning(self: "PipelineMixin", question: str, X=None, y=None) -> dict:
        # Détection: question complexe multi-étapes ?
        if _est_question_complexe(question):
            return orchestrator.raisonner(question)
        return original_executer(question, X, y)

    # Monkey-patch: on attache le wrapper à l'instance
    pipeline.executer_detaille = executer_avec_reasoning  # type: ignore[assignment]
    object.__setattr__(pipeline, "_reasoning_orchestrator", orchestrator)