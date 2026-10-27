"""Aiguilleur : le petit modele local, ou Gemini.

POURQUOI CE MODULE
------------------
Le 1B local est rapide sur ce qu'il sait faire, et il ignore ce qu'il ne sait
pas faire. L'aiguilleur lit DEUX choses et DECIDE, avant qu'un appel distant ne
soit engage :

1. la TACHE demandee (la question) — une dissertation, un essai, un rapport,
   « N pages », une traduction : des competences hors de portee d'un 1B. A qui
   on demande une dissertation, un 1B rend 300 mots a cote ; le detecter sur la
   QUESTION evite de livrer cette reponse.
2. la REPONSE rendue (le resultat du pipeline) — vide, erreur, refus, aveu
   d'ignorance, boucle, echo de la question, reponse trop courte, mauvaise
   langue, probleme chiffre non calcule, fait precis devine sans aucune source,
   ou domaine que tu designes.

CE QUI N'EST JAMAIS REJUGE
--------------------------
Une reponse d'un tier EXACT ou VERIFIE (calcul, PAL, RAG, micro-faits,
Wikipedia, PoT+AST, constantes, horloge, clarification, [Code], [Logique])
n'est jamais rejugee : la remplacer par une reponse probable serait une
regression. Seules les reponses GENERATIVES du cerveau local sont candidates.
Exception : une reponse VIDE ou « [RontoGPT] ... », quel que soit son tier — il
n'y a alors rien a conserver.

CE MODULE NE PARLE A PERSONNE
-----------------------------
`decider()` rend une `Decision` (escalader ou non, avec la raison et le moment).
L'appel HTTP vit dans aura/gemini.py (`gemini.repondre`) ; le pipeline
(aura/pipeline_rontogpt.py) fait le lien. Sans cible disponible — pas de cle,
`AURA_GEMINI=0`, ou pause apres un echec — `decider()` renvoie toujours
« garder le petit modele ».

ORDRE D'EVALUATION
------------------
Les regles de REPONSE d'abord (elles nomment la faute precisement : « refus » se
lit mieux que « redaction »), puis les regles de TACHE, puis « domaine ».
La premiere regle qui s'applique gagne.

Regles (`AURA_AIGUILLEUR_REGLES` = sous-ensemble, defaut : toutes) :
    vide              la reponse locale est vide
    erreur            la reponse est « [RontoGPT] ... »
    refus             la reponse commence par un refus
    aveu              la reponse avoue son ignorance (courte)
    boucle            la reponse se repete
    echo              la reponse recopie la question
    court             question ouverte, reponse de moins de 25 mots
    redaction         demande longue, reponse de 900 caracteres ou moins
    langue            question francaise, reponse en anglais
    probleme          probleme chiffre en mots, sans formule exacte ni code
    factuel           fait precis devine sans aucune source web
    tache-redaction   la question demande une dissertation / un essai / un
                      rapport / « N pages » / « N mots » — long-form, la
                      faiblesse mesuree du 1B (cf. README, « honest limits »)
    tache-traduction  la question demande une traduction
    domaine           categorie routee listee dans `AURA_AIGUILLEUR_DOMAINES`

Variables :
    AURA_AIGUILLEUR_REGLES     sous-ensemble des regles (defaut : toutes)
    AURA_AIGUILLEUR_DOMAINES   categories routees escaladees d'office, ex.
                               « code,math » (defaut : aucune)
    (les anciens noms AURA_GEMINI_REGLES / AURA_GEMINI_DOMAINES restent lus)
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

#: Toutes les regles de jugement, dans l'ordre d'evaluation.
REGLES = ("vide", "erreur", "refus", "aveu", "boucle", "echo", "court",
          "redaction", "langue", "probleme", "factuel", "tache-redaction",
          "tache-traduction", "domaine")


@dataclass(frozen=True)
class Decision:
    """Ce que l'aiguilleur decide. `raison` est None quand rien ne declenche."""

    escalader: bool
    raison: str | None = None
    moment: str = ""            # "reponse" | "tache" | ""
    fournisseur: str = "local"  # "gemini" | "nemotron" | "local"


_LOCAL = Decision(False)


def _escalader(raison: str, moment: str, fournisseur: str = "gemini") -> Decision:
    return Decision(True, raison, moment, fournisseur)


def _env(nom: str, defaut: str = "") -> str:
    return (os.environ.get(nom) or defaut).strip()


def _liste(principal: str, ancien: str = "") -> set[str]:
    """Liste « a,b » de la variable principale, ou de l'ancien nom s'il est seul."""
    brut = _env(principal) or (_env(ancien) if ancien else "")
    return {x.strip() for x in brut.split(",") if x.strip()}


def _regles_actives() -> set[str]:
    choisies = _liste("AURA_AIGUILLEUR_REGLES", "AURA_GEMINI_REGLES")
    return (choisies & set(REGLES)) if choisies else set(REGLES)


def _mode_cloud() -> str:
    """Quel mode cloud utiliser : 'gemini', 'nemotron', 'demander', 'non'."""
    v = os.environ.get("AURA_CLOUD", "demander").strip().lower()
    return v if v in ("gemini", "nemotron", "demander", "non", "oui") else "demander"


def _mode_cloud_nemotron() -> bool:
    """Vrai si on doit utiliser Nemotron 3 Ultra via NVIDIA NIM."""
    return _mode_cloud() == "nemotron"


def disponible() -> bool:
    """Vrai si l'escalade peut aboutir maintenant (cible configuree, pas en pause)."""
    from . import gemini
    # Import optionnel de nemotron (nécessite openai)
    try:
        from . import nemotron
        nemotron_ok = True
    except Exception:
        nemotron_ok = False
    
    mode = _mode_cloud()
    if mode == "gemini":
        return gemini.disponible()
    if mode == "nemotron":
        if nemotron_ok:
            return nemotron.nemotron_disponible("default")
        return False
    # mode "demander" ou "oui" -> au moins un des deux
    nemotron_dispo = nemotron_ok and nemotron.nemotron_disponible("default")
    return gemini.disponible() or nemotron_dispo


# -- le jugement : DETERMINISTE ----------------------------------------------

def _sans_accents(texte: str) -> str:
    t = unicodedata.normalize("NFKD", (texte or "").lower().replace("’", "'"))
    return "".join(c for c in t if not unicodedata.combining(c))


def _mots(texte: str) -> int:
    return len((texte or "").split())


#: Reponses du cerveau local qu'on a le droit de rejuger. Tout le reste est
#: exact ou verifie (cf. docstring du module).
_ESCALADABLES = frozenset({"def-web+1b", "extraction-web"})
_RE_CERVEAU_GENERATIF = re.compile(r"-\d+(?:\.\d+)?b$")
_PREFIXES_VERIFIES = ("[Logique]", "[Code]")


def _generatif(cerveau: str | None) -> bool:
    c = cerveau or ""
    return c in _ESCALADABLES or bool(_RE_CERVEAU_GENERATIF.search(c))


# Refus et aveux d'impuissance. Seulement sur une reponse COURTE : un long
# developpement qui contient un « je ne sais pas » en passant reste une reponse.
_RE_REFUS = re.compile(
    r"^\W*(?:je suis desole|desole|pardon|je ne (?:peux|puis) pas|"
    r"je ne suis pas en mesure)\b")
_RE_AVEU = re.compile(
    r"\bje ne (?:peux|puis) pas\b|\bje ne suis pas en mesure\b|"
    r"\bje ne sais pas\b|\bje ne connais pas\b|\bje ne dispose pas\b|"
    r"\bje ne comprends pas\b|\bil m'est impossible\b|"
    r"\bje n'ai pas (?:reussi|acces|d'information|la possibilite|les moyens)|"
    r"\baucune source ne definit\b")
_AVEU_MAX_CAR = 500

_RE_OUVERTE = re.compile(
    r"\b(?:explique\w*|pourquoi|comment|compare\w*|analyse\w*|redige\w*|"
    r"decri\w*|resume\w*|developpe\w*|argumente\w*|discute\w*|presente\w*|"
    r"avantages|inconvenients|difference\w*)\b")
_COURT_MIN_MOTS = 25

# Demande de production LONGUE. Seuil : celui du banc du projet
# (« court » = 900 caracteres ou moins, cf. scripts/bench_qualite.py).
_RE_LONGUE = re.compile(
    r"\b(?:ecri\w*|redig\w*|compos\w*|produi\w*|gener\w*|developp\w*)\b"
    r".{0,80}\b(?:dissertation|essai|article|discours|rapport|synthese)\b"
    r"|\b(?:[2-9]|\d{2,})\s*(?:pages?|paragraphes?)\b|\b[1-9]\d{2,}\s*mots\b")
_REDACTION_MIN_CAR = 900

# Probleme en mots avec des nombres : la famille que le 1B local rate le plus
# (GSM8K : 0,12 via le pipeline, cf. resultats/BILAN_HONNETE_20261011.md) et
# que les solveurs exacts n'ont pas pris (sinon on ne serait pas ici).
_RE_NOMBRE = re.compile(r"\d+(?:[.,]\d+)?")
_RE_CODE = re.compile(r"\b(?:fonction|python|code|script|programme|def|assert)\b")
_PROBLEME_MIN_MOTS = 12

# Fait precis (qui / quand / ou / combien / capitale...) : sans AUCUNE source,
# le 1B le « devine » — c'est la que son aplomb est le plus trompeur.
_RE_FACTUEL = re.compile(
    r"^\W*(?:"
    r"qui (?:est|etait|sont|etaient|ont|a (?:ecrit|invente|decouvert|fonde|"
    r"realise|compose|peint|dirige|gagne|marque))\b"
    r"|quand (?:est|a|ont|etait|sont|fut|se|s'est)\b"
    r"|en quelle annee\b"
    r"|combien (?:de|d')\b"
    r"|ou (?:est|se trouve|se situe|sont|habite|vit|est ne|est mort)\b"
    r"|quel(?:le)?s? (?:est|etait|sont|etaient) (?:le|la|l'|les) "
    r"(?:capitale|population|superficie|surface|date|distance|hauteur|"
    r"altitude|taille|devise|monnaie|langue|president|premier ministre|"
    r"fondateur|auteur|nombre|montant|prix|cout)\b)")

# Langue : un 1B qui glisse en anglais sur une question francaise.
_MOTS_EN = frozenset(
    "the of and to is are was were this that with for you your in by be have "
    "has not but from which what".split())
_MOTS_FR = frozenset(
    "le la les des une un et est que qui dans pour pas sur avec du de ce cette "
    "sont nous vous il elle ne se je tu mais ou comment pourquoi quel quelle".split())
_RE_LANGUE_DEMANDEE = re.compile(
    r"\b(?:anglais|english|traduis\w*|traduct\w*|translate|allemand|espagnol|"
    r"italien|latin)\b")


# -- regles de TACHE : ce que le 1B ne sait pas faire ------------------------

#: Traduction : le 1B local traduit mal, et le refuse souvent. Formes verbales
#: (« traduis », « traduit », « traduire », « translate ») et le nom suivi d'une
#: cible (« traduction de/en ... ») — pas le simple mot « traduction », qui
#: apparait aussi dans une question theorique.
_RE_TRADUCTION = re.compile(
    r"\b(?:tradui(?:s|t|re|sez)|translate\w*)\b"
    r"|\btraduction\s+(?:du|de|d'|en)\b")


def _tache_faible(question: str, actives: set[str]) -> str | None:
    """Une competence que le 1B local n'a pas, detectee sur la QUESTION seule."""
    q = _sans_accents(question)
    if "tache-redaction" in actives and _RE_LONGUE.search(q):
        return "tache-redaction"
    if "tache-traduction" in actives and _RE_TRADUCTION.search(q):
        return "tache-traduction"
    return None


# -- regles de REPONSE : ce que le 1B vient de rendre ------------------------

def _norme_courte(texte: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _sans_accents(texte)).strip()


def _en_boucle(reponse: str) -> bool:
    phrases = [p.strip().lower() for p in re.split(r"(?<=[.!?])\s+|\n+", reponse)
               if len(p.strip()) >= 15]
    return len(phrases) >= 4 and len(set(phrases)) / len(phrases) < 0.6


def _est_echo(question: str, reponse: str) -> bool:
    q, r = _norme_courte(question), _norme_courte(reponse)
    if len(q) < 15 or not r:
        return False
    return r == q or (q in r and len(r) <= len(q) * 1.25)


def _mauvaise_langue(question: str, reponse: str) -> bool:
    q = _sans_accents(question)
    if _RE_LANGUE_DEMANDEE.search(q) or "```" in reponse:
        return False
    mots_q = re.findall(r"[a-z']+", q)
    francais = (sum(m in _MOTS_FR for m in mots_q) >= 2
                or any(c in question.lower() for c in "éèêàùçôîï"))
    mots_r = re.findall(r"[a-z']+", _sans_accents(reponse))
    if not francais or len(mots_r) < 12:
        return False
    anglais = sum(m in _MOTS_EN for m in mots_r) / len(mots_r)
    fr = sum(m in _MOTS_FR for m in mots_r) / len(mots_r)
    return anglais >= 0.22 and fr < 0.08


def decider(question: str, resultat: dict,
            categorie: str | None = None) -> Decision:
    """Escalader vers la cible distante, ou garder le petit modele local.

    `resultat` est le dict de l'orchestrateur (cles lues : reponse,
    cerveau_choisi, en_attente, contexte_web, formule).
    """
    if not disponible():
        return _LOCAL
    actives = _regles_actives()
    reponse = str(resultat.get("reponse") or "")
    fournisseur = _mode_cloud() if _mode_cloud() in ("gemini", "nemotron") else "gemini"

    def _escalader_mode(raison: str, moment: str) -> Decision:
        return _escalader(raison, moment, fournisseur)

    # Rien a conserver : quel que soit le tier.
    if not reponse.strip():
        return _escalader_mode("vide", "reponse") if "vide" in actives else _LOCAL
    if reponse.startswith("[RontoGPT]"):
        return _escalader_mode("erreur", "reponse") if "erreur" in actives else _LOCAL

    # Une relance de clarification n'est pas une reponse faible. Un tier exact
    # ou verifie n'est jamais rejuge, pas plus qu'un resultat [Code]/[Logique]
    # (que le pipeline verifie avant de le livrer).
    if (resultat.get("en_attente")
            or not _generatif(resultat.get("cerveau_choisi"))
            or reponse.lstrip().startswith(_PREFIXES_VERIFIES)):
        return _LOCAL

    q = _sans_accents(question)
    r = _sans_accents(reponse)
    sans_source = not str(resultat.get("contexte_web") or "").strip()
    sans_formule = not resultat.get("formule")

    if len(reponse) < _AVEU_MAX_CAR:
        if "refus" in actives and _RE_REFUS.search(r):
            return _escalader_mode("refus", "reponse")
        if "aveu" in actives and _RE_AVEU.search(r):
            return _escalader_mode("aveu", "reponse")
    if "boucle" in actives and _en_boucle(reponse):
        return _escalader_mode("boucle", "reponse")
    if "echo" in actives and _est_echo(question, reponse):
        return _escalader_mode("echo", "reponse")
    if ("court" in actives and _RE_OUVERTE.search(q)
            and _mots(reponse) < _COURT_MIN_MOTS):
        return _escalader_mode("court", "reponse")
    if ("redaction" in actives and _RE_LONGUE.search(q)
            and len(reponse) <= _REDACTION_MIN_CAR):
        return _escalader_mode("redaction", "reponse")
    if "langue" in actives and _mauvaise_langue(question, reponse):
        return _escalader_mode("langue", "reponse")
    if ("probleme" in actives and sans_formule
            and not _RE_CODE.search(q)
            and _mots(question) >= _PROBLEME_MIN_MOTS
            and len(_RE_NOMBRE.findall(question)) >= 2):
        return _escalader_mode("probleme", "reponse")
    if ("factuel" in actives and sans_source and sans_formule
            and _RE_FACTUEL.search(q)):
        return _escalader_mode("factuel", "reponse")

    # La TACHE demandee excede-t-elle ce que le petit modele sait faire ?
    tache = _tache_faible(question, actives)
    if tache:
        return _escalader_mode(tache, "tache")

    if ("domaine" in actives and categorie
            and categorie in _liste("AURA_AIGUILLEUR_DOMAINES",
                                    "AURA_GEMINI_DOMAINES")):
        return _escalader_mode("domaine", "reponse")
    return _LOCAL


def raison(question: str, resultat: dict,
           categorie: str | None = None) -> str | None:
    """Raison d'escalader (raccourci de `decider`), ou None."""
    return decider(question, resultat, categorie).raison
