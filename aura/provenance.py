"""Provenance : d'OU vient une reponse, dite en un badge.

La promesse du produit est « l'assistant local en francais qui prouve ses
reponses ». Une promesse invisible ne vaut rien : chaque reponse porte donc un
BADGE qui dit comment elle a ete obtenue, et ce que cela garantit (ou pas).

Ce module est PUR : un dict de resultat (celui de `executer_detaille`) entre,
un dict `{niveau, badge, detail, verifie, quitte_la_machine}` sort. Il ne
juge pas la reponse, il decrit son chemin. Aucune dependance, aucun reseau.

Niveaux, du plus sur au moins sur :
    exact      calcul / solveur deterministe : prouve, pas devine
    document   extrait de VOS fichiers, cite
    source     fait lu sur une source (Wikipedia, web, API) et cite
    base       base de faits locale relue, ou horloge de la machine
    memoire    reponse deja donnee, retrouvee par similarite (approximatif)
    local      redige par le petit modele : NON verifie
    cloud      reponse de Gemini : la question a quitte l'ordinateur
    precision  l'assistant demande une precision au lieu de repondre
    inconnu    l'assistant dit qu'il ne sait pas / ne peut pas verifier

`verifie` est vrai pour exact / document / source / base ; faux pour le reste.
`quitte_la_machine` n'est vrai que pour `cloud`.
"""
from __future__ import annotations

import re
import unicodedata

# cerveau_choisi -> (niveau, detail). Les noms viennent de
# registre_experts.py et de pipeline_rontogpt.py.
_TABLE: dict[str, tuple[str, str]] = {
    # calcul et solveurs : exacts
    "niveau0-instantane": ("exact", "calcule exactement"),
    "fast-lang": ("exact", "base de langue locale"),
    "calcul-achat": ("exact", "calcule exactement"),
    "calcul-verbal": ("exact", "calcule exactement"),
    "calcul-taux": ("exact", "calcule exactement"),
    "chaines-numeriques": ("exact", "calcule exactement"),
    "problemes-arithmetiques": ("exact", "calcule exactement"),
    "tri-transitif": ("exact", "deduit par un solveur logique"),
    "logique": ("exact", "deduit par un solveur logique"),
    "potcode": ("exact", "code execute et verifie par des tests"),
    "pot-1b+ast": ("exact", "chaque etape de calcul a ete verifiee"),
    # skills deterministes (cf. skills.toml)
    "skill-identite": ("base", "reponse fixe"),
    "skill-algebre": ("exact", "equation resolue exactement"),
    "skill-pgcd": ("exact", "calcule exactement"),
    "skill-memoire": ("base", "lu dans la memoire de la conversation"),
    "skill-resume": ("memoire", "resume de la conversation"),
    # vos fichiers
    "documents": ("document", "d'apres vos documents"),
    # sources externes citees
    "wikipedia": ("source", "source : Wikipedia"),
    "titulaires": ("source", "source : Wikidata"),
    "extraction-api": ("source", "source : service public"),
    "extraction-web": ("source", "source : web"),
    "def-web+1b": ("source", "reformule a partir d'extraits web"),
    # base locale
    "micro-faits": ("base", "base de faits locale"),
    "rag-binaire": ("base", "base de faits locale relue"),
    "constante": ("base", "reponse fixe"),
    "constante-fonctionnement": ("base", "reponse fixe"),
    "constante-capacites": ("base", "reponse fixe"),
    "constante-capacite": ("base", "reponse fixe"),
    "horloge": ("base", "horloge de votre ordinateur"),
    # approximatif
    "cache-semantique": ("memoire", "reponse deja donnee, retrouvee par similarite"),
    # autres
    "gemini": ("cloud", "reponse de Gemini : la question a quitte votre ordinateur"),
    "clarification": ("precision", "precision demandee"),
    "web-indisponible": ("inconnu", "internet injoignable : non verifie"),
    "logique-non-formalisee": ("inconnu", "enonce non compris : pas de reponse prouvee"),
}

_BADGES: dict[str, str] = {
    "exact": "Calculé exactement",
    "document": "D'après vos documents",
    "source": "Source web",
    "base": "Base vérifiée",
    "memoire": "Réponse mémorisée",
    "local": "Modèle local, non vérifié",
    "cloud": "Envoyé à Gemini",
    "precision": "Précision demandée",
    "inconnu": "Je ne sais pas",
}

_VERIFIES = frozenset({"exact", "document", "source", "base"})

# un cerveau generatif s'appelle « <fichier>-<n>b » (cf. aiguilleur._generatif)
_RE_GENERATIF = re.compile(r"-\d+(?:\.\d+)?b$")

# aveux d'ignorance que le pipeline ecrit lui-meme
_RE_ABSTENTION = re.compile(
    r"\bje ne peux pas verifier\b|\bje prefere m'abstenir\b|\bje ne sais pas\b"
    r"|\bje n'ai pas (?:trouve|d'information)\b|\baucune source\b")


def _sans_accents(texte: str) -> str:
    t = unicodedata.normalize("NFKD", (texte or "").lower().replace("’", "'"))
    return "".join(c for c in t if not unicodedata.combining(c))


def provenance(resultat: dict) -> dict:
    """Decrit le chemin d'une reponse. Ne leve jamais."""
    try:
        cerveau = str(resultat.get("cerveau_choisi") or "")
        reponse = str(resultat.get("reponse") or "")
        sans_accent = _sans_accents(reponse)
        if resultat.get("en_attente") or cerveau == "clarification":
            niveau, detail = _TABLE["clarification"]
        elif reponse.lstrip().startswith(("[Code]", "[Logique]")):
            niveau, detail = "exact", "verifie par un solveur ou des tests"
        elif cerveau in _TABLE:
            niveau, detail = _TABLE[cerveau]
        elif _RE_GENERATIF.search(cerveau) or cerveau:
            # cerveau generatif (ou etiquette inconnue) : jamais presente
            # comme verifiee. Avec des extraits web, on le dit.
            if str(resultat.get("contexte_web") or "").strip():
                niveau, detail = "local", "redige par le modele local a partir d'extraits web"
            else:
                niveau, detail = "local", "redige par le modele local, sans source"
        else:
            niveau, detail = "local", "origine inconnue"
        # un aveu d'ignorance n'est jamais habille en reponse
        if niveau in ("local", "source") and _RE_ABSTENTION.search(sans_accent) \
                and len(reponse) < 500:
            niveau, detail = "inconnu", "l'assistant prefere ne pas deviner"
    except Exception:  # noqa: BLE001 — un badge ne doit jamais casser une reponse
        niveau, detail = "local", "origine inconnue"
    return {
        "niveau": niveau,
        "badge": _BADGES[niveau],
        "detail": detail,
        "verifie": niveau in _VERIFIES,
        "quitte_la_machine": niveau == "cloud",
    }
