"""Jauge du contexte et compaction automatique a 90 %.

Deux modes de compaction, choisis par AURA_COMPACTION (ou [memoire] dans
aura.toml) :

- ``importance`` (defaut) : a 90 %, le systeme OUBLIE les echanges les moins
  importants et les plus anciens, puis garde les plus utiles dans l'ordre
  chronologique. Les indices d'importance sont deterministes et sans LLM :
  declarations durables (« je m'appelle », « je prefere », « mon projet »),
  contraintes, code/URL, densite de la question. Le dernier echange est
  conserve s'il tient sous le plafond de 90 % ; sinon toute la fenetre est
  videe pour que le contexte ne reste JAMAIS sature par l'historique.
- ``recent`` : ancien comportement, on garde simplement autant de messages
  recents que la cible de 60 % le permet.

Dans les deux cas l'etat durable (nom, preferences, faits) survit et
l'historique complet reste dans le fichier RLM, consultable par l'outil
« cherche dans l'historique : <mot> ». La compaction ne supprime donc pas
la memoire longue : elle retire des messages de la fenetre ACTIVE.

La mesure est volontairement simple et sans appel LLM : systeme + question +
contexte web + faits + formule + fenetre de conversation. Elle est faite au
moment ou le prompt est construit, donc sur le texte REELLEMENT envoye.

NOTE : ce fichier a ete RECONSTRUIT a partir du bytecode (`__pycache__`) apres
un ecrasement accidentel. La structure, les noms, les signatures et les
messages sont ceux de l'original ; les poids numeriques de `_score_groupe` et
`GARDER` sont des valeurs plausibles, pas garanties. Si l'original est dans
git : `git checkout -- aura/contexte.py` le restaure a l'identique.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memoire_conversation import MemoireConversation

LOG = logging.getLogger("aura.contexte")

#: seuil de saturation : on compacte quand le prompt atteint 90 % de la fenetre
SEUIL = 0.90
#: apres compaction, on vise 60 % de la fenetre
CIBLE = 0.60
#: nombre minimal de messages conserves (un echange question/reponse)
GARDER = 2
#: tokens de gabarit ajoutes par message (role, balises de fin de tour)
_MARGE_MESSAGE = 4


@lru_cache(maxsize=1)
def budget() -> int:
    """Fenetre de contexte effective (serveur actif, sinon cerveau charge).

    Mise en cache : la jauge tourne a chaque generation et le calcul serveur
    (profil materiel) n'a aucune raison d'etre refait. La fenetre ne change
    pas en cours de processus.
    """
    try:
        from . import serveur_lora
        if serveur_lora._actifs():
            return max(1, int(serveur_lora._n_ctx()))
    except Exception as e:  # noqa: BLE001
        LOG.debug("[contexte] fenetre du serveur illisible : %s", e)
    try:
        from . import llama_cerveau
        llm = None
        for nom in ("_llm_charge", "llm_charge", "_llm"):
            fabrique = getattr(llama_cerveau, nom, None)
            if callable(fabrique):
                try:
                    llm = fabrique()
                except Exception:  # noqa: BLE001
                    llm = None
                break
            if fabrique is not None:
                llm = fabrique
                break
        return max(1, int(llama_cerveau._n_ctx_cible(llm)))
    except Exception as e:  # noqa: BLE001
        LOG.debug("[contexte] fenetre du cerveau illisible : %s", e)
        from .llama_cerveau import _N_CTX_DEFAUT
        return _N_CTX_DEFAUT


def mode() -> str:
    """Mode de compaction : ``importance`` (defaut) ou ``recent``."""
    valeur = (os.environ.get("AURA_COMPACTION") or "importance").strip().lower()
    return valeur if valeur in ("importance", "recent") else "importance"


def estimer(texte: str) -> int:
    """Tokens d'un texte : tokenizer du modele si charge, sinon ~3 car/token."""
    if not texte:
        return 0
    try:
        from . import llama_cerveau
        llm = None
        fabrique = getattr(llama_cerveau, "_llm_charge", None)
        if callable(fabrique):
            try:
                llm = fabrique()
            except Exception:  # noqa: BLE001
                llm = None
        return llama_cerveau._compter_tokens(llm, texte)
    except Exception:  # noqa: BLE001
        return max(1, len(texte) // 3)


def _sans_accents(texte: str) -> str:
    t = unicodedata.normalize("NFKD", (texte or "").lower().replace("’", "'"))
    return "".join(c for c in t if not unicodedata.combining(c))


# Indices d'importance (appliques au texte normalise : minuscules, sans accents)
_RE_DURABLE = re.compile(
    r"\b(?:je\s+m\s*'?\s*appelle|je\s+me\s+nomme|moi\s+c\s*'?\s*est|"
    r"j\s*'?\s*habite|je\s+vis|mon|ma|mes|notre|nos|rappelle|souviens|"
    r"retiens|n\s*'?\s*oublie|important|essentiel|toujours|jamais|prefere|"
    r"preference|objectif|contrainte|budget|projet|mot\s+de\s+passe|adresse|"
    r"telephone|email|courriel|allergie|anniversaire|aime|deteste|besoin|"
    r"veux|dois|evite)\b")
_RE_CODE = re.compile(
    r"```|https?://|\b(?:def|class|import|function|select|python|javascript)\b"
    r"|\{.*\}", re.IGNORECASE)
_RE_BANAL = re.compile(
    r"^(?:ok|okay|d'accord|oui|non|merci|salut|bonjour|hello|coucou|super|"
    r"parfait|bien|top|genial|ca marche)[\s.!?]*$")


def _systeme(systeme: str | None) -> str:
    """Le message systeme : celui fourni, sinon le prompt fixe du projet."""
    if systeme is not None:
        return systeme
    try:
        from . import prompt_systeme
        for nom in ("SYSTEME", "_SYSTEME", "PROMPT_SYSTEME", "SYSTEME_FR"):
            v = getattr(prompt_systeme, nom, None)
            if isinstance(v, str) and v:
                return v
    except Exception:  # noqa: BLE001
        pass
    return ""


def _tours(historique: "MemoireConversation | None") -> list[dict]:
    if historique is None:
        return []
    try:
        return list(historique.fenetre())
    except Exception as e:  # noqa: BLE001
        LOG.debug("[contexte] fenetre illisible : %s", e)
        return []


def _hors_fenetre(systeme: str | None, question: str, contexte_web: str,
                  contexte_faits: str, formule: str) -> int:
    """Tokens qui ne viennent pas de l'historique de conversation."""
    return (estimer(_systeme(systeme)) + estimer(question)
            + estimer(contexte_web) + estimer(contexte_faits)
            + estimer(formule))


def etat(historique: "MemoireConversation | None" = None, question: str = "",
         contexte_web: str = "", contexte_faits: str = "", formule: str = "",
         systeme: str | None = None) -> dict:
    """Remplissage du prompt : tokens, budget, ratio, seuil, sature, mode."""
    tours = _tours(historique)
    systeme_tokens = estimer(_systeme(systeme))
    tokens = _hors_fenetre(systeme, question, contexte_web, contexte_faits,
                           formule)
    for tour in tours:
        tokens += estimer(str(tour.get("content", ""))) + _MARGE_MESSAGE
    limite = budget()
    ratio = tokens / limite if limite > 0 else 0.0
    return {
        "tokens": tokens,
        "budget": limite,
        # Champs d'affichage : le TUI peut mesurer la conversation sur la
        # place qui lui est vraiment allouee, hors prompt systeme fixe.
        "systeme_tokens": systeme_tokens,
        "budget_conversation": max(0, limite - systeme_tokens),
        "conversation_tokens": max(0, tokens - systeme_tokens),
        "ratio": round(ratio, 4),
        "pourcentage": int(round(ratio * 100)),
        "seuil": int(round(SEUIL * 100)),
        "sature": ratio >= SEUIL,
        "mode": mode(),
    }


def _taille(tour: dict) -> int:
    return estimer(str(tour.get("content", ""))) + _MARGE_MESSAGE


def _groupes(tours: list[dict]) -> list[list[tuple[int, dict]]]:
    """Regroupe la fenetre en echanges (une reponse ferme le groupe)."""
    groupes: list[list[tuple[int, dict]]] = []
    courant: list[tuple[int, dict]] = []
    for i, tour in enumerate(tours):
        courant.append((i, tour))
        if tour.get("role") == "assistant":
            groupes.append(courant)
            courant = []
    if courant:
        groupes.append(courant)
    return groupes


def _score_groupe(groupe: list[tuple[int, dict]], age: int) -> float:
    """Importance d'un echange : contenu durable + code, moins l'anciennete."""
    user = " ".join(str(t.get("content", "")) for _, t in groupe
                    if t.get("role") == "user")
    reponse = " ".join(str(t.get("content", "")) for _, t in groupe
                       if t.get("role") != "user")
    q = _sans_accents(user).strip()
    score = 0.0
    if _RE_DURABLE.search(q):
        score += 3.0
    if _RE_CODE.search(f"{user}\n{reponse}"):
        score += 1.5
    if _RE_BANAL.match(q):
        score -= 2.0
    if len(q) >= 80:
        score += 1.0
    if len(reponse) >= 400:
        score += 1.0
    score *= max(0.2, 1.0 - 0.1 * age)
    return score


def _selection_importance(tours: list[dict], hors: int) -> set[int]:
    """Indices des messages a garder, du plus utile au moins utile.

    Le dernier echange est reserve s'il tient sous 90 % ; les echanges plus
    anciens sont ensuite ajoutes par importance decroissante tant que le
    total reste sous la cible de 60 %. Si meme le dernier echange ne tient
    pas, on vide la fenetre : l'historique ne doit jamais etre la cause
    d'un contexte sature.
    """
    groupes = _groupes(tours)
    if not groupes:
        return set()
    plafond = int(budget() * SEUIL)
    cible = int(budget() * CIBLE)
    tailles = [sum(_taille(t) for _, t in g) for g in groupes]
    scores = [_score_groupe(g, len(groupes) - 1 - i)
              for i, g in enumerate(groupes)]
    gardes: set[int] = set()
    total = hors
    dernier = len(groupes) - 1
    if total + tailles[dernier] > plafond:
        return set()
    gardes.add(dernier)
    total += tailles[dernier]
    ordre = sorted(range(dernier), key=lambda j: (-scores[j], -j))
    for i in ordre:
        if total + tailles[i] <= cible:
            gardes.add(i)
            total += tailles[i]
    indices: set[int] = set()
    for i in gardes:
        indices.update(idx for idx, _ in groupes[i])
    return indices


def _messages_recents(tours: list[dict], hors: int) -> int:
    """Nombre de messages recents tenant sous la cible de 60 % (mode recent)."""
    disponible = max(0, int(budget() * CIBLE) - hors)
    total = 0
    n = 0
    for tour in reversed(tours):
        taille = _taille(tour)
        if n >= GARDER and total + taille > disponible:
            break
        total += taille
        n += 1
    return min(len(tours), max(n, GARDER))


def _selection_recent(tours: list[dict], hors: int) -> set[int]:
    n = _messages_recents(tours, hors)
    return set(range(len(tours) - n, len(tours)))


def auto_compacter(historique: "MemoireConversation | None" = None,
                   question: str = "", contexte_web: str = "",
                   contexte_faits: str = "", formule: str = "",
                   systeme: str | None = None) -> dict | None:
    """Compacte la fenetre si le prompt atteint 90 % du contexte.

    Renvoie la mesure avant compaction (`sature`, `compacte`, `gardes`,
    `apres`, `mode`), ou None si le prompt est sous le seuil. Quand le seuil
    est atteint mais que la fenetre est deja minimale, renvoie quand meme la
    mesure avec `compacte = 0` : l'appelant peut ainsi DIRE que le contexte
    est sature au lieu de laisser croire que tout va bien.
    """
    if historique is None:
        return None
    mesure = etat(historique, question, contexte_web, contexte_faits, formule,
                  systeme)
    if not mesure["sature"]:
        return None
    tours = _tours(historique)
    choix = mode()
    if len(tours) <= GARDER and choix == "recent":
        mesure.update({"compacte": 0, "gardes": len(tours),
                       "apres": mesure["pourcentage"],
                       "raison": "fenetre minimale"})
        return mesure
    hors = _hors_fenetre(systeme, question, contexte_web, contexte_faits,
                         formule)
    gardes = (_selection_importance(tours, hors) if choix == "importance"
              else _selection_recent(tours, hors))
    if len(gardes) == len(tours):
        mesure.update({"compacte": 0, "gardes": len(tours),
                       "apres": mesure["pourcentage"],
                       "raison": "rien a retirer" if tours else "hors memoire"})
        return mesure
    try:
        from . import flux_cerveau
        flux_cerveau.publier_flux(
            f"Contexte à {mesure['pourcentage']} % — tri des souvenirs…",
            flux_cerveau.PHASE_ETAPE)
    except Exception as e:  # noqa: BLE001
        LOG.debug("[contexte] etape non publiee : %s", e)
    try:
        retires = historique.compacter_en_gardant(sorted(gardes))
    except Exception as e:  # noqa: BLE001
        LOG.warning("[contexte] compaction impossible : %s", e)
        mesure.update({"compacte": 0, "gardes": len(tours),
                       "apres": mesure["pourcentage"],
                       "raison": "echec compaction"})
        return mesure
    apres = etat(historique, question, contexte_web, contexte_faits, formule,
                 systeme)
    raison = ""
    if apres["ratio"] >= SEUIL:
        raison = "hors memoire"
    mesure.update({"compacte": retires, "gardes": len(gardes),
                   "apres": apres["pourcentage"], "raison": raison})
    LOG.info("[contexte] %s : %d %% (%d/%d tokens) -> %d retire(s), "
             "%d garde(s), reste %d %%", choix, mesure["pourcentage"],
             mesure["tokens"], mesure["budget"], retires, len(gardes),
             apres["pourcentage"])
    return mesure
