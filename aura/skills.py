"""Skills ciblés : corriger les lacunes mesurées d'un petit modèle local.

Deux familles, volontairement séparées :

1. **Déterministes** (`reponse` ou `handler`) : la réponse ne passe PAS par
   le LLM. C'est le seul remède fiable à un 1B qui se trompe sur une règle
   précise (identité, équation linéaire, prénom retenu).
2. **Consignes** (`instructions`) : un texte COURT ajouté au prompt système
   quand la question correspond. Un 1B recopie les consignes longues ; on se
   limite donc à quelques skills et à quelques phrases.

Le registre vit dans `skills.toml` (racine du projet), pour qu'ajouter une
compétence ne demande pas de toucher au pipeline. L'identité détaillée vit
dans `identite.toml` (10 fiches, 100 phrases). Un skill qui lève une
exception est ignoré : il ne doit jamais casser un tour.
"""
from __future__ import annotations

import math
import os
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

_RACINE = Path(__file__).resolve().parent.parent
_FICHIER = _RACINE / "skills.toml"
_FICHIER_IDENTITE = _RACINE / "identite.toml"

#: Identité de secours si `skills.toml` est absent ou illisible. Le fichier
#: reste la source de vérité ; cette constante évite un crash au démarrage.
IDENTITE_DEFAUT = (
    "Je suis RontoGPT, un assistant francophone qui tourne en local sur ta "
    "machine. Mon cerveau est Gemma 3 1B, un modèle de langage créé par "
    "Google DeepMind. Le système RontoGPT qui m'entoure est créé par Simon "
    "Chusseau. Gemma apporte le langage ; RontoGPT apporte les solveurs, les "
    "experts, la mémoire et la recherche. Je ne suis pas Gemini : Gemma est "
    "mon modèle de base, pas mon identité complète. Je fonctionne sans "
    "envoyer la conversation au cloud, sauf si un service distant est "
    "explicitement configuré."
)


@dataclass(frozen=True)
class Skill:
    id: str
    titre: str
    priorite: int = 0
    quand: tuple[str, ...] = ()
    motif: re.Pattern[str] | None = None
    contexte: str | None = None
    handler: str | None = None
    reponse: str | None = None
    instructions: str | None = None
    #: remplacer une reponse approchante deja en cache (cf. `doit_reconsolider`)
    reconsolider: bool = False


@dataclass(frozen=True)
class FicheIdentite:
    """Une fiche thématique de `identite.toml` (10 phrases)."""

    theme: str
    titre: str
    mots: tuple[str, ...]
    phrases: tuple[str, ...]


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, espaces compactés (les motifs l'utilisent)."""
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.replace("’", "'").replace("-", " ")
    return re.sub(r"\s+", " ", n).strip()


def _lire(chemin: str | Path | None = None) -> tuple[Skill, ...]:
    chemin = Path(chemin) if chemin is not None else Path(
        os.environ.get("AURA_SKILLS", str(_FICHIER)))
    try:
        donnees = tomllib.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    skills: list[Skill] = []
    for brut in donnees.get("skill", []):
        if not isinstance(brut, dict):
            continue
        sid = str(brut.get("id") or "").strip()
        if not sid:
            continue
        quand = tuple(
            _normaliser(str(q)) for q in brut.get("quand", [])
            if str(q).strip()
        )
        motif: re.Pattern[str] | None = None
        motif_brut = brut.get("motif")
        if motif_brut:
            try:
                motif = re.compile(str(motif_brut))
            except re.error:
                motif = None
        try:
            priorite = int(brut.get("priorite", 0))
        except (TypeError, ValueError):
            priorite = 0
        skills.append(Skill(
            id=sid,
            titre=str(brut.get("titre") or sid),
            priorite=priorite,
            quand=quand,
            motif=motif,
            contexte=str(brut["contexte"]) if brut.get("contexte") else None,
            handler=str(brut["handler"]) if brut.get("handler") else None,
            reponse=str(brut["reponse"]) if brut.get("reponse") else None,
            instructions=(str(brut["instructions"])
                          if brut.get("instructions") else None),
            reconsolider=bool(brut.get("reconsolider", False)),
        ))
    return tuple(skills)


_CACHE: tuple[Skill, ...] | None = None


def charger(chemin: str | Path | None = None) -> tuple[Skill, ...]:
    """Le registre (cache pour le fichier par défaut)."""
    global _CACHE
    if chemin is not None:
        return _lire(chemin)
    if _CACHE is None:
        _CACHE = _lire(None)
    return _CACHE


def recharger(chemin: str | Path | None = None) -> tuple[Skill, ...]:
    """Relit le registre (tests, ou édition de skills.toml à chaud)."""
    global _CACHE
    _CACHE = _lire(chemin)
    return _CACHE


def reponse_fixe(skill_id: str) -> str | None:
    """La réponse fixe d'un skill (`reponse = "..."`), ou None."""
    for skill in charger():
        if skill.id == skill_id and skill.reponse:
            return skill.reponse
    return None


#: Évaluée au chargement : `regles_reponse` s'en sert comme constante unique.
IDENTITE = reponse_fixe("identite") or IDENTITE_DEFAUT


def _lire_identite(chemin: str | Path | None = None
                   ) -> tuple[FicheIdentite, ...]:
    chemin = Path(chemin) if chemin is not None else Path(
        os.environ.get("AURA_IDENTITE", str(_FICHIER_IDENTITE)))
    try:
        donnees = tomllib.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    fiches: list[FicheIdentite] = []
    for brut in donnees.get("fiche", []):
        if not isinstance(brut, dict):
            continue
        theme = _normaliser(str(brut.get("theme") or ""))
        phrases = tuple(str(p).strip() for p in brut.get("phrases", [])
                        if str(p).strip())
        if not theme or not phrases:
            continue
        fiches.append(FicheIdentite(
            theme=theme,
            titre=str(brut.get("titre") or theme),
            mots=tuple(_normaliser(str(m)) for m in brut.get("mots", [])
                       if str(m).strip()),
            phrases=phrases,
        ))
    return tuple(fiches)


_CACHE_IDENTITE: tuple[FicheIdentite, ...] | None = None


def charger_identite(chemin: str | Path | None = None
                     ) -> tuple[FicheIdentite, ...]:
    """Les fiches d'identité (cache pour le fichier par défaut)."""
    global _CACHE_IDENTITE
    if chemin is not None:
        return _lire_identite(chemin)
    if _CACHE_IDENTITE is None:
        _CACHE_IDENTITE = _lire_identite(None)
    return _CACHE_IDENTITE


def doit_reconsolider(skill_id: str) -> bool:
    """Le skill remplace-t-il une reponse approchante deja en cache ?

    Sans cela, une ancienne hallucination (« Je suis Gemma… ») reste servie
    par le cache semantique a une question voisine, meme apres correction.
    """
    for skill in charger():
        if skill.id == skill_id:
            return skill.reconsolider
    return False


def _actifs(question: str, avec_contexte: bool = False) -> list[Skill]:
    """Skills qui correspondent à la question, les plus prioritaires d'abord."""
    q = _normaliser(question)
    retenus: list[tuple[int, Skill]] = []
    for skill in charger():
        if skill.contexte == "web":
            if not avec_contexte:
                continue
        else:
            par_motif = bool(skill.motif and skill.motif.search(q))
            par_phrase = any(p in q for p in skill.quand)
            if not par_motif and not par_phrase:
                continue
        bonus = 20 if skill.motif is not None and skill.motif.search(q) else 0
        retenus.append((skill.priorite + bonus, skill))
    retenus.sort(key=lambda paire: (-paire[0], paire[1].id))
    return [skill for _, skill in retenus]


# -- algèbre linéaire -------------------------------------------------------

_RE_EQ = re.compile(r"([0-9xX+\-*/.\s]+)=([0-9xX+\-*/.\s]+)")
_RE_PUISSANCE = re.compile(r"\^|\*\*|[²³]")
_RE_TERME = re.compile(r"[+-]?[^+-]+")


def _cote(texte: str) -> tuple[Fraction, Fraction] | None:
    """Un côté d'équation -> (coefficient de x, constante), ou None."""
    cote = (texte or "").replace(" ", "")
    if not cote:
        return None
    a, b = Fraction(0), Fraction(0)
    for terme in _RE_TERME.findall(cote):
        bas = terme.lower()
        if "x" in bas:
            if bas.count("x") != 1:
                return None
            coeff = bas.replace("x", "").replace("*", "")
            if coeff in ("", "+"):
                coeff = "1"
            elif coeff == "-":
                coeff = "-1"
            elif coeff.startswith("/"):
                coeff = "1" + coeff
            elif coeff.startswith("-/"):
                coeff = "-1" + coeff[1:]
            try:
                a += Fraction(coeff)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            if "*" in terme:
                return None
            try:
                b += Fraction(terme)
            except (ValueError, ZeroDivisionError):
                return None
    return a, b


def _fraction(texte: Fraction) -> str:
    if texte.denominator == 1:
        return str(texte.numerator)
    return f"{texte.numerator}/{texte.denominator} ≈ {float(texte):.4g}"


def _algebre(question: str, historique: Any = None) -> str | None:
    """Résout ax + b = cx + d, avec x d'un seul côté ou des deux.

    S'abstient (None) dès que l'équation sort du linéaire : puissances,
    parenthèses, plusieurs variables, plusieurs « = ». Un solveur qui devine
    serait pire que pas de solveur.
    """
    texte = (question or "")
    for avant, apres in (("−", "-"), ("–", "-"), ("×", "*"), ("÷", "/")):
        texte = texte.replace(avant, apres)
    texte = re.sub(r"(?<=\d),(?=\d)", ".", texte)
    # GARDE-FOU : une affectation dans une question de code n'est pas une
    # equation. On ne résout que sur intention explicite, ou si la question
    # EST l'equation (aucun mot autour).
    q = _normaliser(question)
    if any(mot in q for mot in ("python", "javascript", "java", "code",
                                "programme", "fonction", "affectation",
                                "variable")):
        return None
    intention = any(p in q for p in (
        "resous", "resoudre", "equation", "trouve x", "valeur de x",
        "combien vaut x", "solve"))
    if (not intention
            and re.fullmatch(r"[0-9xX+\-*/.\s=,]+",
                             texte.strip(" .;:!?")) is None):
        return None
    if texte.count("=") != 1 or _RE_PUISSANCE.search(texte):
        return None
    if "(" in texte or ")" in texte:
        return None
    m = _RE_EQ.search(texte)
    if m is None:
        return None
    gauche = m.group(1).strip(" .;:!?")
    droite = m.group(2).strip(" .;:!?")
    if "x" not in (gauche + droite).lower():
        return None
    if set(re.findall(r"[a-z]", (gauche + droite).lower())) - {"x"}:
        return None
    cote_gauche = _cote(gauche)
    cote_droit = _cote(droite)
    if cote_gauche is None or cote_droit is None:
        return None
    a1, b1 = cote_gauche
    a2, b2 = cote_droit
    a = a1 - a2
    b = b2 - b1
    if a == 0:
        if b == 0:
            return ("Toute valeur de x convient : les deux membres sont "
                    "identiques.")
        return "Aucune solution : l'équation est impossible."
    x = b / a
    verif_gauche = a1 * x + b1
    verif_droite = a2 * x + b2
    pas = abs(a)
    if pas == 0:
        bis = ""
    else:
        bis = f"\n\nÉtapes :\nx = {b} / {a}\n"
        bis += f"  {a1}×{x} + {b1} = {a2}×{x} + {b2}\n"
        bis += f"  {a1}×{x} = {a2}×{x} + ({b2} - {b1})\n"
        bis += f"  ({a1 - a2})×{x} = {b2 - b1}\n"
        bis += f"  {a}×{x} = {b}\n"
        bis += f"  {x} = {b} / {a}"
    return (f"x = {_fraction(x)}"
            + (f"\n{bis}" if bis else "")
            + f"\n\nVérification : {_fraction(verif_gauche)} = "
            f"{_fraction(verif_droite)}.")


# -- arithmétique entière ---------------------------------------------------


def _pgcd(question: str, historique: Any = None) -> str | None:
    """PGCD / PPCM exacts, avec la vérification qui rend le résultat lisible.

    Corrige un défaut mesuré : le 1B annonçait « PGCD de 84 et 126 » (le
    deuxième nombre n'était même pas celui de la question), puis un facteur
    commun faux. Ici le calcul est fait, pas raconté.
    """
    q = _normaliser(question)
    ppcm = any(p in q for p in ("ppcm", "plus petit commun multiple",
                                "plus petit multiple commun"))
    if not ppcm and not any(p in q for p in ("pgcd",
                                             "plus grand commun diviseur")):
        return None
    nombres = [int(n) for n in re.findall(r"\d+", question or "")]
    if not 2 <= len(nombres) <= 4 or any(n > 10**9 for n in nombres):
        return None
    if ppcm:
        if 0 in nombres:
            return (f"PPCM({', '.join(map(str, nombres))}) = 0 : "
                    "zéro est multiple de tout nombre.")
        valeur = math.lcm(*nombres)
        verif = " et ".join(f"{valeur} = {n} × {valeur // n}"
                            for n in nombres)
        return (f"PPCM({', '.join(map(str, nombres))}) = {valeur}\n\n"
                f"Vérification : {verif}.")
    valeur = math.gcd(*nombres)
    if valeur == 0:
        return (f"PGCD({', '.join(map(str, nombres))}) = 0 "
                "(convention pour des nombres tous nuls).")
    # algorithme d'Euclide, pour les deux nombres d'une question ordinaire
    etapes: list[str] = []
    if len(nombres) == 2:
        a, b = max(nombres), min(nombres)
        while b:
            quotient, reste = divmod(a, b)
            etapes.append(f"{a} = {b} × {quotient} + {reste}")
            a, b = b, reste
    verif = " ; ".join(f"{n} = {valeur} × {n // valeur}" for n in nombres)
    quotients = [str(n // valeur) for n in nombres]
    if valeur == 1:
        fin = "Les nombres sont premiers entre eux."
    elif len(quotients) == 2:
        fin = f"{quotients[0]} et {quotients[1]} sont premiers entre eux."
    else:
        fin = (", ".join(quotients[:-1]) + " et " + quotients[-1]
               + " sont premiers entre eux.")
    return (f"PGCD({', '.join(map(str, nombres))}) = {valeur}\n\n"
            + ("\n".join(etapes) + "\n\n" if etapes else "")
            + f"Vérification : {verif}. {fin}")


# -- honnêteté sur les prémisses impossibles -------------------------------

_RE_CAPITALE_ASTRE = re.compile(
    r"capitale\s+(?:de\s+la\s+|du\s+|de\s+)?"
    r"(lune|soleil|mars|jupiter|saturne|venus|mercure|neptune|uranus|pluton)\b")

_ASTRES = {
    "lune": "la Lune", "soleil": "le Soleil", "mars": "Mars",
    "jupiter": "Jupiter", "saturne": "Saturne", "venus": "Vénus",
    "mercure": "Mercure", "neptune": "Neptune", "uranus": "Uranus",
    "pluton": "Pluton",
}


def _honnetete(question: str, historique: Any = None) -> str | None:
    """Refuse une prémisse impossible au lieu d'inventer une réponse.

    Défaut mesuré : « Quelle est la capitale de la Lune ? » a reçu « Touhou
    Wiki », c'est-à-dire une page de fan-fiction présentée comme un fait.
    """
    m = _RE_CAPITALE_ASTRE.search(_normaliser(question))
    if m is None:
        return None
    astre = _ASTRES[m.group(1)]
    return (f"Aucune : {astre} n'est pas un pays et n'a pas de capitale. "
            "Aucune source sérieuse n'en cite ; une page de fan-fiction "
            "n'est pas une source.")


# -- mémoire de l'utilisateur ----------------------------------------------

_RE_PROJET = re.compile(
    r"(?:je\s+(?:travaille|bosse|d[ée]veloppe|construis|con[çc]ois)\s+"
    r"(?:sur|dans|a|à)|mon\s+projet\s+(?:est|c'est|porte\s+sur))\s+"
    r"(.+?)(?:[.!?]|$)", re.IGNORECASE)


def _memoire(question: str, historique: Any = None) -> str | None:
    """Répond aux rappels personnels à partir de l'état et de la fenêtre.

    L'état durable (`utilisateur`) et les tours bruts vivaient déjà dans
    `MemoireConversation`, mais n'étaient jamais relus par la synthèse : le
    1B répondait « je n'ai pas accès à tes données » alors que le prénom
    était connu. Ici, la réponse est lue, pas générée.
    """
    if historique is None:
        return None
    try:
        etat = historique.etat()
    except Exception:                              # noqa: BLE001
        return None
    q = _normaliser(question)
    nom = etat.get("utilisateur")
    demande_nom = ("mon prenom" in q
                   or "comment je m'appelle" in q
                   or re.search(r"\bquel est mon nom\b(?!\s+(?:de|du|des)\b)",
                                q) is not None)
    if demande_nom:
        return f"Tu t'appelles {nom}." if nom else None
    if any(p in q for p in ("sur quoi je travaille", "mon projet",
                            "sur quel projet")):
        try:
            tours = historique.fenetre()
        except Exception:                          # noqa: BLE001
            return None
        for tour in reversed(tours):
            if tour.get("role") != "user":
                continue
            m = _RE_PROJET.search(str(tour.get("content") or ""))
            if m is not None:
                projet = m.group(1).strip(" .,;:")
                return f"Tu travailles sur {projet}."
    return None


def _resume(question: str, historique: Any = None) -> str | None:
    """Résumé HONNÊTE de la fenêtre : les derniers sujets, pas d'invention.

    Après `/compact`, la fenêtre est vide : le dire vaut mieux que la réponse
    hors-sujet en anglais observée le 25 oct. 2026. L'état durable (prénom)
    reste disponible, lui.
    """
    if historique is None:
        return None
    try:
        tours = historique.fenetre()
        etat = historique.etat()
    except Exception:                              # noqa: BLE001
        return None
    nom = etat.get("utilisateur")
    if not tours:
        texte = "Je n'ai plus le détail des tours depuis la compaction."
        if nom:
            texte += f" Je me souviens que tu t'appelles {nom}."
        return texte
    questions = [re.sub(r"\s+", " ", str(t.get("content") or "")).strip()
                 for t in tours if t.get("role") == "user"]
    if not questions:
        return None
    puces = []
    for q in questions[-3:]:
        court = q[:120] + ("…" if len(q) > 120 else "")
        puces.append(f"• {court}")
    return "Derniers sujets de la conversation :\n" + "\n".join(puces)


def _demande_detail(q: str) -> bool:
    return any(m in q for m in ("en detail", "explique", "developpe",
                                "decris", "precisement"))


def _identite(question: str, historique: Any = None) -> str | None:
    """Réponse d'identité choisie dans `identite.toml`, sans passer par le 1B.

    Dix fiches thématiques : présentation, créateurs, architecture, solveurs,
    mémoire, sources, limites, confidentialité, langue, matériel. La question
    « qui es-tu » garde la présentation courte ; une question sur le créateur
    distingue explicitement Google (Gemma) de Simon Chusseau (RontoGPT).
    """
    q = _normaliser(question)
    if "nom de domaine" in q:            # « ton nom » n'est pas l'identité
        return None
    fiches = charger_identite()
    if not fiches:
        return IDENTITE
    detail = _demande_detail(q)
    # Question sur l'origine : ne jamais laisser « ton modèle » répondre
    # avant « qui a créé Gemma ». C'est la distinction centrale.
    if (("gemma" in q or "rontogpt" in q)
            and any(m in q for m in ("createur", "cree", "google",
                                     "deepmind", "simon"))):
        for fiche in fiches:
            if fiche.theme == "createurs":
                phrases = list(fiche.phrases)
                # Une question sur Gemma commence par le créateur de Gemma ;
                # une question sur RontoGPT commence par Simon Chusseau.
                if "gemma" in q and "rontogpt" not in q:
                    ordre = (1, 2, 3, 4, 0, 5, 6, 7, 8, 9)
                    phrases = [phrases[i] for i in ordre
                               if i < len(phrases)]
                return " ".join(phrases if detail else phrases[:4])
    meilleure: FicheIdentite | None = None
    meilleur_score = 0
    meilleure_longueur = 0
    for fiche in fiches:
        longueurs = [len(m) for m in fiche.mots if m in q]
        score = len(longueurs)
        longueur = max(longueurs) if longueurs else 0
        if (score, longueur) > (meilleur_score, meilleure_longueur):
            meilleure, meilleur_score, meilleure_longueur = fiche, score, longueur
    if meilleure is None:
        return IDENTITE
    if meilleure.theme == "presentation" and not detail:
        return IDENTITE
    return " ".join(meilleure.phrases if detail else meilleure.phrases[:4])


Handler = Callable[[str, Any], str | None]

_HANDLERS: dict[str, Handler] = {
    "identite": _identite,
    "algebre": _algebre,
    "pgcd": _pgcd,
    "honnetete": _honnetete,
    "memoire": _memoire,
    "resume": _resume,
}


def repondre(question: str, historique: Any = None,
             production: bool = False) -> tuple[str, str] | None:
    """Réponse déterministe (id du skill, texte), ou None.

    `production=True` (rédaction, code) écarte identité et mémoire : une
    question qui parle DE l'assistant dans un texte à produire n'est pas une
    question sur l'assistant.
    """
    for skill in _actifs(question):
        if skill.contexte == "web":
            continue
        if production and skill.id in ("identite", "memoire"):
            continue
        if skill.handler is not None:
            handler = _HANDLERS.get(skill.handler)
            if handler is None:
                continue
            try:
                reponse = handler(question, historique)
            except Exception:                      # noqa: BLE001
                reponse = None
            if reponse:
                return skill.id, reponse
        elif skill.reponse:
            return skill.id, skill.reponse
    return None


def bloc_pour(question: str, avec_contexte: bool = False) -> str:
    """Les consignes courtes des skills actifs, ou une chaîne vide."""
    choisies = [s for s in _actifs(question, avec_contexte)
                if s.instructions]
    morceaux: list[str] = []
    total = 0
    for skill in choisies[:2]:
        texte = f"COMPÉTENCE — {skill.titre} : {skill.instructions}"
        if total + len(texte) > 700:
            continue
        morceaux.append(texte)
        total += len(texte)
    return "\n\n".join(morceaux)


def completer(systeme: str, question: str,
             avec_contexte: bool = False) -> str:
    """Ajoute les consignes actives au prompt système, sans le remplacer."""
    bloc = bloc_pour(question, avec_contexte)
    return f"{systeme}\n\n{bloc}" if bloc else systeme
