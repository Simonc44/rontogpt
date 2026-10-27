"""
Streaming typé pour TUI (C)
===========================

Pièces typées remplaçant le texte brut :
- thinking, tool_call, source, final, error, progress
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Literal, Optional, Union
from typing import AsyncGenerator

from .reponse_structuree import (
    ReponseStruct, TypeReponse, NiveauConfiance,
    Source, ReponseCalc, ReponseCode, ReponseLookup,
    ReponseRedaction, ReponseRefus, ReponseClarification, ReponseErreur,
    pipeline_vers_structure
)


class TypePiece(str, Enum):
    THINKING = "thinking"           # Raisonnement interne
    TOOL_CALL = "tool_call"         # Appel d'outil (web, calc, code)
    TOOL_RESULT = "tool_result"     # Résultat d'outil
    SOURCE = "source"               # Source citée
    PROGRESS = "progress"           # Avancement (morceaux)
    FINAL = "final"                 # Réponse finale structurée
    ERROR = "error"                 # Erreur
    CLARIFICATION = "clarification" # Question de clarification
    METADATA = "metadata"           # Métadonnées (experts, budget, etc.)


@dataclass
class PieceStream:
    """Une pièce du flux de réponse."""
    type: TypePiece
    contenu: Any
    horodatage: datetime = field(default_factory=datetime.now)
    sequence: int = 0
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "contenu": self.contenu,
            "horodatage": self.horodatage.isoformat(),
            "sequence": self.sequence,
            "meta": self.meta
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "PieceStream":
        data = json.loads(s)
        return cls(
            type=TypePiece(data["type"]),
            contenu=data["contenu"],
            horodatage=datetime.fromisoformat(data["horodatage"]),
            sequence=data.get("sequence", 0),
            meta=data.get("meta", {})
        )


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCTEURS DE PIÈCES
# ──────────────────────────────────────────────────────────────────────────────

def piece_thinking(texte: str, phase: str = "raisonnement", seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.THINKING,
        contenu={"texte": texte, "phase": phase},
        sequence=seq
    )


def piece_tool_call(nom: str, args: dict, seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.TOOL_CALL,
        contenu={"outil": nom, "arguments": args},
        sequence=seq
    )


def piece_tool_result(nom: str, resultat: Any, succes: bool = True, seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.TOOL_RESULT,
        contenu={"outil": nom, "resultat": resultat, "succes": succes},
        sequence=seq
    )


def piece_source(source: "Source", seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.SOURCE,
        contenu=source.to_dict(),
        sequence=seq
    )


def piece_progress(morceaux: int, phase: str, secondes: float, seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.PROGRESS,
        contenu={"morceaux": morceaux, "phase": phase, "secondes": secondes},
        sequence=seq
    )


def piece_final(reponse: "ReponseStruct", seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.FINAL,
        contenu=reponse.model_dump(mode="json"),
        sequence=seq,
        meta={"type_reponse": reponse.type.value}
    )


def piece_error(message: str, code: str | None = None, seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.ERROR,
        contenu={"message": message, "code": code},
        sequence=seq
    )


def piece_clarification(questions: list[str], seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.CLARIFICATION,
        contenu={"questions": questions},
        sequence=seq
    )


def piece_metadata(experts: set[str], budget_restant: float | None, seq: int = 0) -> PieceStream:
    return PieceStream(
        type=TypePiece.METADATA,
        contenu={"experts": list(experts), "budget_restant_s": budget_restant},
        sequence=seq
    )


# ──────────────────────────────────────────────────────────────────────────────
# STREAMER ASYNCHRONE
# ──────────────────────────────────────────────────────────────────────────────

class ReponseStreamer:
    """Génère un flux de pièces typées depuis le pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def streamer(self, question: str, **kwargs) -> AsyncGenerator[PieceStream, None]:
        """Point d'entrée principal - génère le flux complet."""
        t0 = time.time()
        morceaux = 0

        # 1. Analyse initiale
        yield piece_metadata(
            experts=set(), budget_restant=None, seq=self._next_seq()
        )

        # 2. Si clarification nécessaire
        # (le pipeline gère ça en interne)

        # 3. Exécution pipeline avec yield progressif
        # On simule le streaming en découpant la réponse finale
        # TODO: brancher sur le vrai streaming du pipeline quand dispo
        resultat = self.pipeline.executer_detaille(question, **kwargs)

        # Conversion structurée
        reponse = pipeline_vers_structure(resultat)

        # 3. Yield sources si présentes
        if hasattr(reponse, "sources") and reponse.sources:
            for s in reponse.sources:
                yield piece_source(s, seq=self._next_seq())

        # 4. Yield final
        yield piece_final(reponse, seq=self._next_seq())

    def streamer_sync(self, question: str, **kwargs) -> list[PieceStream]:
        """Version synchrone pour compatibilité."""
        pieces = []
        resultat = self.pipeline.executer_detaille(question, **kwargs)
        reponse = pipeline_vers_structure(resultat)

        pieces.append(piece_metadata(
            experts=resultat.get("experts", set()),
            budget_restant=None, seq=self._next_seq()
        ))

        if hasattr(reponse, "sources") and reponse.sources:
            for s in reponse.sources:
                pieces.append(piece_source(s, seq=self._next_seq()))

        pieces.append(piece_final(reponse, seq=self._next_seq()))
        return pieces


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTATEUR TUI
# ──────────────────────────────────────────────────────────────────────────────

class TUIStreamAdapter:
    """Adapte le flux de pièces pour l'affichage TUI existant."""

    def __init__(self, streamer: ReponseStreamer):
        self.streamer = streamer
        self._buffer = ""
        self._morceaux = 0
        self._phase = "initialisation"
        self._t0 = time.time()

    def traiter_piece(self, piece: PieceStream) -> Optional[str]:
        """Traite une pièce et retourne le texte à afficher (ou None)."""
        if piece.type == TypePiece.THINKING:
            self._phase = piece.contenu.get("phase", "raisonnement")
            return None  # Affiché via puce clignotante

        elif piece.type == TypePiece.PROGRESS:
            self._morceaux = piece.contenu.get("morceaux", 0)
            self._phase = piece.contenu.get("phase", self._phase)
            return None  # Affiché via label d'attente

        elif piece.type == TypePiece.TOOL_CALL:
            outil = piece.contenu.get("outil")
            args = piece.contenu.get("arguments", {})
            return f"🔧 {outil}({json.dumps(args, ensure_ascii=False)})"

        elif piece.type == TypePiece.TOOL_RESULT:
            outil = piece.contenu.get("outil")
            succes = piece.contenu.get("succes", True)
            icone = "✅" if succes else "❌"
            return f"{icone} {outil} terminé"

        elif piece.type == TypePiece.SOURCE:
            src = piece.contenu
            titre = src.get("titre", "Source")
            url = src.get("url") or src.get("chemin") or "interne"
            return f"🔗 {titre} — {url}"

        elif piece.type == TypePiece.FINAL:
            # Retourne la réponse structurée pour affichage riche
            return piece.contenu

        elif piece.type == TypePiece.ERROR:
            return f"❌ {piece.contenu.get('message')}"

        elif piece.type == TypePiece.CLARIFICATION:
            q = piece.contenu.get("questions", [])
            return "❓ " + "\n".join(f"  • {qq}" for qq in q)

        elif piece.type == TypePiece.METADATA:
            return None  # Interne

        return None

    def reset(self):
        self._buffer = ""
        self._morceaux = 0
        self._phase = "initialisation"
        self._t0 = time.time()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER SYNC POUR INTÉGRATION PIPELINE EXISTANT
# ──────────────────────────────────────────────────────────────────────────────

def executer_avec_stream(pipeline: Any, question: str, **kwargs) -> tuple[list[PieceStream], Any]:
    """Exécute et retourne (pièces, résultat_structuré)."""
    streamer = ReponseStreamer(pipeline)
    pieces = streamer.streamer_sync(question, **kwargs)

    # Dernière pièce = réponse finale
    reponse_finale = None
    for p in reversed(pieces):
        if p.type == TypePiece.FINAL:
            reponse_finale = p.contenu
            break

    return pieces, reponse_finale