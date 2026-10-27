"""
Réponses structurées & Guardrails amont (A + B)
================================================

Remplace le post-processing lourd par des contraintes amont :
- Schémas Pydantic pour validation amont
- Génération contrainte (Outlines si dispo, fallback regex)
- Sérialisation/désérialisation unifiée
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Literal, TypeVar, Union
from typing import get_type_hints, get_origin, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from . import llama_cerveau
from .provenance import provenance as _provenance


# ──────────────────────────────────────────────────────────────────────────────
# TYPES DE BASE
# ──────────────────────────────────────────────────────────────────────────────

class TypeReponse(str, Enum):
    """Types de réponse reconnus par le pipeline."""
    CALC = "calc"              # Résultat numérique exact
    CODE = "code"              # Bloc de code exécutable
    LOOKUP = "lookup"          # Fait vérifiable (définition, fait)
    REDACTION = "redaction"    # Texte long (analyse, dissertation)
    REFUS = "refus"            # Impossible / hors périmètre
    CLARIFICATION = "clarification"  # Question en retour
    ERREUR = "erreur"          # Erreur technique


class NiveauConfiance(str, Enum):
    HAUT = "haut"      # ≥ 0.9 : source vérifiée, calcul exact
    MOYEN = "moyen"    # 0.6-0.9 : source probable, raisonnement
    FAIBLE = "faible"  # < 0.6 : incertain, hallucination possible


class Source(BaseModel):
    """Source citable (web, doc, calcul, code)."""
    type: Literal["web", "document", "calcul", "code", "wikipedia", "graphe", "interne"]
    titre: str
    url: str | None = None
    chemin: str | None = None
    extrait: str | None = None
    score: float | None = None  # Pertinence 0-1
    date_acces: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


# ──────────────────────────────────────────────────────────────────────────────
# MODÈLES DE RÉPONSE PAR TYPE
# ──────────────────────────────────────────────────────────────────────────────

class ReponseCalc(BaseModel):
    """Résultat de calcul exact (niveau0, PAL, etc.)."""
    type: Literal[TypeReponse.CALC] = TypeReponse.CALC
    valeur: Union[int, float, str]  # "42", "3.14", "1/3"
    unite: str | None = None
    formule: str | None = None
    etapes: list[str] = Field(default_factory=list)
    confiance: NiveauConfiance = NiveauConfiance.HAUT
    sources: list[Source] = Field(default_factory=list)


class ReponseCode(BaseModel):
    """Bloc de code validé (syntaxe + exécution optionnelle)."""
    type: Literal[TypeReponse.CODE] = TypeReponse.CODE
    langage: str
    code: str
    description: str | None = None
    valide_syntaxiquement: bool = True
    execute: bool = False
    sortie: str | None = None
    confiance: NiveauConfiance = NiveauConfiance.HAUT
    sources: list[Source] = Field(default_factory=list)

    @field_validator("code")
    @classmethod
    def code_non_vide(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Code vide")
        return v


class ReponseLookup(BaseModel):
    """Fait vérifiable avec citation."""
    type: Literal[TypeReponse.LOOKUP] = TypeReponse.LOOKUP
    reponse: str
    sources: list[Source] = Field(min_length=1)
    extrait_source: str | None = None
    confiance: NiveauConfiance = Field(default=NiveauConfiance.MOYEN)

    @model_validator(mode="after")
    def verifier_sources(self) -> "ReponseLookup":
        if not self.sources:
            raise ValueError("Lookup nécessite au moins une source")
        return self


class ReponseRedaction(BaseModel):
    """Texte long structuré (analyse, dissertation, comparaison)."""
    type: Literal[TypeReponse.REDACTION] = TypeReponse.REDACTION
    titre: str | None = None
    introduction: str | None = None
    sections: list[dict] = Field(default_factory=list)  # [{"titre": "...", "contenu": "..."}]
    conclusion: str | None = None
    sources: list[Source] = Field(default_factory=list)
    mots_cles: list[str] = Field(default_factory=list)
    nb_mots_approx: int | None = None
    confiance: NiveauConfiance = NiveauConfiance.MOYEN

    def to_markdown(self) -> str:
        parties = []
        if self.titre:
            parties.append(f"# {self.titre}\n")
        if self.introduction:
            parties.append(self.introduction + "\n")
        for s in self.sections:
            parties.append(f"## {s.get('titre', '')}\n{s.get('contenu', '')}\n")
        if self.conclusion:
            parties.append(f"## Conclusion\n{self.conclusion}\n")
        if self.sources:
            parties.append("## Sources\n" + "\n".join(
                f"- [{s.titre}]({s.url or s.chemin or 'interne'})" for s in self.sources
            ))
        return "\n".join(parties)


class ReponseRefus(BaseModel):
    """Refus explicite avec raison."""
    type: Literal[TypeReponse.REFUS] = TypeReponse.REFUS
    raison: str
    suggestion: str | None = None
    type_refus: Literal["hors_perimetre", "ambigu", "dangereux", "technique", "politique"] = "hors_perimetre"
    confiance: NiveauConfiance = NiveauConfiance.HAUT


class ReponseClarification(BaseModel):
    """Question de retour à l'utilisateur."""
    type: Literal[TypeReponse.CLARIFICATION] = TypeReponse.CLARIFICATION
    questions: list[str] = Field(min_length=1, max_length=3)
    contexte: str | None = None
    confiance: NiveauConfiance = NiveauConfiance.HAUT


class ReponseErreur(BaseModel):
    """Erreur technique."""
    type: Literal[TypeReponse.ERREUR] = TypeReponse.ERREUR
    message: str
    code: str | None = None
    recuperable: bool = True
    confiance: NiveauConfiance = NiveauConfiance.FAIBLE


# Union de tous les types
ReponseStruct = Union[
    ReponseCalc, ReponseCode, ReponseLookup,
    ReponseRedaction, ReponseRefus,
    ReponseClarification, ReponseErreur
]


# ──────────────────────────────────────────────────────────────────────────────
# GARDIEN DE CONTRAINTES (Guardrails amont - B)
# ──────────────────────────────────────────────────────────────────────────────

class Contrainte(ABC):
    """Contrainte de validation sur la réponse générée."""

    @abstractmethod
    def verifier(self, reponse: ReponseStruct) -> tuple[bool, str | None]:
        """Retourne (ok, message_erreur)."""
        pass

    @abstractmethod
    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        """Tente de corriger automatiquement."""
        return reponse


class InterdirePhrases(Contrainte):
    """Interdit certaines phrases types (hallucinations fréquentes)."""

    PHRASES_INTERDITES = [
        "je suis un modèle de langage",
        "en tant qu'ia",
        "je ne peux pas",
        "je suis désolé",
        "en tant que modèle",
        "comme modèle de langage",
    ]

    def verifier(self, reponse: ReponseStruct) -> tuple[bool, str | None]:
        texte = self._extraire_texte(reponse).lower()
        for p in self.PHRASES_INTERDITES:
            if p in texte:
                return False, f"Phrase interdite détectée: '{p}'"
        return True, None

    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        texte = self._extraire_texte(reponse)
        for p in self.PHRASES_INTERDITES:
            texte = re.sub(re.escape(p), "", texte, flags=re.IGNORECASE)
        return self._reinjecter_texte(reponse, texte.strip())

    def _extraire_texte(self, r: ReponseStruct) -> str:
        if isinstance(r, ReponseRedaction):
            return r.to_markdown()
        return getattr(r, "reponse", getattr(r, "valeur", getattr(r, "code", str(r))))

    def _reinjecter_texte(self, r: ReponseStruct, texte: str) -> ReponseStruct:
        if isinstance(r, ReponseRedaction):
            # Reconstruire grossièrement
            return ReponseRedaction(
                type=r.type, titre=r.titre, introduction=texte[:200],
                sections=[{"titre": "Réponse", "contenu": texte[200:]}],
                sources=r.sources, confiance=r.confiance
            )
        return r


class ExigerSources(Contrainte):
    """Exige des sources pour les lookups et rédactions factuelles."""

    def verifier(self, reponse: ReponseStruct) -> tuple[bool, str | None]:
        if isinstance(reponse, (ReponseLookup, ReponseRedaction)):
            if not reponse.sources:
                return False, "Sources manquantes pour un contenu factuel"
        return True, None

    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        # Ne peut pas inventer de sources
        return reponse


class FormatJSONValide(Contrainte):
    """Vérifie que le code JSON est valide."""

    def verifier(self, reponse: ReponseStruct) -> tuple[bool, str | None]:
        if isinstance(reponse, ReponseCode) and reponse.langage == "json":
            try:
                json.loads(reponse.code)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"JSON invalide: {e}"
        return True, None

    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        if isinstance(reponse, ReponseCode) and reponse.langage == "json":
            try:
                obj = json.loads(reponse.code)
                return ReponseCode(
                    **reponse.model_dump(),
                    code=json.dumps(obj, indent=2, ensure_ascii=False)
                )
            except Exception:
                pass
        return reponse


class LimiterLongueur(Contrainte):
    """Limite la longueur selon le type."""

    LIMITES = {
        ReponseLookup: 500,
        ReponseCalc: 200,
        ReponseRefus: 300,
        ReponseClarification: 400,
    }

    def verifier(self, reponse: ReponseStruct) -> tuple[bool, str | None]:
        limite = self.LIMITES.get(type(reponse))
        if limite:
            texte = self._extraire_texte(reponse)
            if len(texte) > limite:
                return False, f"Trop long ({len(texte)} > {limite})"
        return True, None

    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        """Tronque la réponse si elle dépasse la limite."""
        limite = self.LIMITES.get(type(reponse))
        if limite:
            texte = self._extraire_texte(reponse)
            if len(texte) > limite:
                # Tronquer
                if isinstance(reponse, ReponseRedaction):
                    return ReponseRedaction(
                        **reponse.model_dump(),
                        introduction=reponse.introduction[:limite] if reponse.introduction else "",
                        sections=reponse.sections,
                        conclusion=reponse.conclusion
                    )
                # Pour les autres types, on ne peut pas facilement tronquer
        return reponse

    def _extraire_texte(self, r: ReponseStruct) -> str:
        if isinstance(r, ReponseRedaction):
            return r.to_markdown()
        return getattr(r, "reponse", getattr(r, "valeur", getattr(r, "code", str(r))))


class GuardrailManager:
    """Gestionnaire de guardrails - validation + correction amont."""

    def __init__(self, contraintes: list[Contrainte] | None = None):
        self.contraintes = contraintes or [
            InterdirePhrases(),
            ExigerSources(),
            FormatJSONValide(),
            LimiterLongueur(),
        ]

    def valider(self, reponse: ReponseStruct) -> tuple[bool, list[str]]:
        """Valide toutes les contraintes. Retourne (ok, erreurs)."""
        erreurs = []
        for c in self.contraintes:
            ok, msg = c.verifier(reponse)
            if not ok:
                erreurs.append(msg or "Contrainte échouée")
        return len(erreurs) == 0, erreurs

    def corriger(self, reponse: ReponseStruct) -> ReponseStruct:
        """Applique les corrections en cascade."""
        for c in self.contraintes:
            reponse = c.corriger(reponse)
        return reponse

    def valider_et_corriger(self, reponse: ReponseStruct) -> tuple[ReponseStruct, list[str]]:
        """Valide, corrige si besoin, re-valide."""
        ok, erreurs = self.valider(reponse)
        if not ok:
            reponse = self.corriger(reponse)
            ok, erreurs = self.valider(reponse)
        return reponse, erreurs


# Instance globale par défaut
GUARDRAILS_DEFAUT = GuardrailManager()


# ──────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION CONTRAINTE (Outlines si dispo, fallback regex)
# ──────────────────────────────────────────────────────────────────────────────

class GenerateurContraint(ABC):
    """Interface pour génération contrainte."""

    @abstractmethod
    def generer(self, prompt: str, schema: type[ReponseStruct], **kwargs) -> ReponseStruct:
        pass


class GenerateurFallback(GenerateurContraint):
    """Fallback sans Outlines : prompt engineering + validation post-hoc."""

    def __init__(self, guardrails: GuardrailManager = GUARDRAILS_DEFAUT):
        self.guardrails = guardrails

    def _construire_prompt(self, prompt: str, schema: type[ReponseStruct]) -> str:
        schema_json = schema.model_json_schema()
        return f"""{prompt}

RÉPONDS OBLIGATOIREMENT EN JSON VALIDE selon ce schéma :
```json
{json.dumps(schema_json, indent=2, ensure_ascii=False)}
```

RÈGLES STRICTES :
- AUCUN texte hors JSON
- Aucune phrase "je suis un modèle", "je ne peux pas", "je suis désolé"
- Sources obligatoires pour faits/rédactions
- JSON valide si type code
- Longueur adaptée au type"""

    def generer(self, prompt: str, schema: type[ReponseStruct], **kwargs) -> ReponseStruct:
        prompt_complet = self._construire_prompt(prompt, schema)
        brut = llama_cerveau.generer(prompt_complet, **kwargs)

        # Extraction JSON
        try:
            # Cherche le premier objet JSON complet
            match = re.search(r"\{.*\}", brut, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(brut)
        except json.JSONDecodeError:
            # Fallback : construction minimale
            return self._fallback_minimal(schema, prompt)

        try:
            reponse = schema(**data)
        except ValidationError as e:
            return self._fallback_minimal(schema, prompt)

        # Guardrails
        reponse, erreurs = self.guardrails.valider_et_corriger(reponse)
        if erreurs:
            LOG.warning(f"Guardrails non résolus: {erreurs}")
        return reponse

    def _fallback_minimal(self, schema: type[ReponseStruct], prompt: str) -> ReponseStruct:
        """Réponse minimale valide en cas d'échec."""
        if schema == ReponseRefus:
            return ReponseRefus(type=TypeReponse.REFUS, raison="Génération échouée")
        if schema == ReponseClarification:
            return ReponseClarification(type=TypeReponse.CLARIFICATION,
                                        questions=["Pouvez-vous préciser votre demande ?"])
        if schema == ReponseCalc:
            return ReponseCalc(type=TypeReponse.CALC, valeur="0", confiance=NiveauConfiance.FAIBLE)
        return ReponseErreur(type=TypeReponse.ERREUR, message="Génération structurée échouée")


# Instance par défaut
GENERATEUR_DEFAUT = GenerateurFallback()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE CONVERSION PIPELINE → STRUCTURÉ
# ──────────────────────────────────────────────────────────────────────────────

def pipeline_vers_structure(resultat: dict) -> ReponseStruct:
    """Convertit un résultat du pipeline (dict) en réponse structurée."""
    experts = set(resultat.get("experts", []))
    reponse = resultat.get("reponse", "")
    contexte_web = resultat.get("contexte_web", "")
    provenance = resultat.get("provenance", {})

    # Sources
    sources = []
    if provenance.get("niveau") == "verifié":
        for s in provenance.get("sources", []):
            sources.append(Source(
                type="web", titre=s.get("titre", "Source web"),
                url=s.get("url"), score=s.get("score")
            ))

    # Déduction du type selon experts + contenu
    if "niveau0" in experts or "calcul" in experts:
        return ReponseCalc(
            type=TypeReponse.CALC, valeur=str(resultat.get("reponse", "")),
            formule=resultat.get("formule"), sources=sources,
            confiance=NiveauConfiance.HAUT
        )
    if "code" in experts or "pot" in experts:
        return ReponseCode(
            type=TypeReponse.CODE, langage="python", code=resultat.get("reponse", ""),
            sources=sources, confiance=NiveauConfiance.HAUT
        )
    if "web" in experts or "encyclopedie" in experts:
        return ReponseLookup(
            type=TypeReponse.LOOKUP, reponse=resultat.get("reponse", ""),
            sources=sources, confiance=NiveauConfiance.MOYEN
        )
    if any(e in experts for e in ["web", "encyclopedie", "documents", "graphe"]):
        # Construire une ReponseRedaction à partir du texte
        texte = str(resultat.get("reponse", ""))
        return ReponseRedaction(
            type=TypeReponse.REDACTION,
            introduction=texte[:500] if len(texte) > 500 else texte,
            sections=[{"titre": "Réponse", "contenu": texte[500:]}] if len(texte) > 500 else [{"titre": "Réponse", "contenu": texte}],
            sources=sources, confiance=NiveauConfiance.MOYEN
        )

    # Défaut
    return ReponseLookup(
        type=TypeReponse.LOOKUP, reponse=str(resultat.get("reponse", "")),
        sources=sources, confiance=NiveauConfiance.FAIBLE
    )


def structure_vers_dict(reponse: ReponseStruct) -> dict:
    """Sérialise pour compatibilité pipeline/API existante."""
    base = {"type": reponse.type.value, "confiance": reponse.confiance.value}
    base.update(reponse.model_dump(exclude={"type", "confiance"}, mode="json"))
    return base


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

import logging
LOG = logging.getLogger("aura.reponse_structuree")