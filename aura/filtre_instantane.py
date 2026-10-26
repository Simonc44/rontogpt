"""Niveau 0 : reponse instantanee (0.1s) — le coeur de l'architecture CPU-native.

Philosophie : sur un PC sans GPU, le composant le plus lent est le LLM
(~14 tok/s). La reponse la plus rapide est celle qu'on n'a pas besoin de
generer. Deux detecteurs, du plus rapide au plus cher :

1. **Calcul direct** (< 1 ms) — regex + eval arithmetique securise (AST) :
   « carre de 12 », « 2 puissance 10 », « 45*12 », « 15% de 200 » et les
   expressions dictees en mots (« 15 divise par 3 plus 4 puissance 2 »).
   Resultat EXACT, zero hallucination, zero token genere.
1bis. **PAL** (< 1 ms) — dates (« dans 45 jours », « combien de jours
   jusqu'au 25 decembre »), unites (« 5 miles en km », « 100 f en c »),
   pourcentages composes (« 15% de 200 plus 30% de 100 ») : voir pal.py.
2. **Cache semantique** (~4 ms) — deux lectures des reponses deja
   donnees : vecteurs MiniLM (cosinus >= 0.70 = vraie reformulation
   reconnue, index dense NumPy persiste en .npy) puis, a defaut ou en
   repli, TF-IDF char n-grams (>= 0.85). Une question similaire renvoie
   la reponse enregistree : le systeme « apprend » ses reponses.

Si rien ne matche → None → le routeur et le Llama prennent la main.

TROIS FONCTIONS, ET L'ORDRE COMPTE
----------------------------------
`repondre`        etages 1 et 1bis, EXACTS, < 1 ms
`repondre_cache`  etage 2, APPROXIMATIF, ~1 ms
`repondre_tout`   exact puis flou — pour les bancs, PAS pour la cascade

La cascade (`orchestrateur`) appelle les deux etages SEPAREMENT : elle
place tous les solveurs exacts (calcul_achat, calcul_verbal, calcul_taux,
tri-transitif, SAT, potcode) ENTRE les deux. Un seul appel combine ferait
repasser le cache — approximation a seuil — devant des solveurs exacts.
"""
import ast
import json
import logging
import operator
import os
import re
import time
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np

LOG = logging.getLogger("aura.filtre0")

# Cache de reponses (jsonl) : AURA_FILTRE_CACHE permet d'en pointer un
# AUTRE — indispensable pour les bancs (scripts/bench_qualite.py) qui ne
# doivent ni polluer la memoire reelle du projet ni les tests (un hit de
# cache y ferait echouer le routage testé).
_FICHIER = Path(os.environ.get("AURA_FILTRE_CACHE")
                or (Path(__file__).parent / ".cache_reponses.jsonl"))
_SEUIL_SIMILARITE = 0.85   # repli TF-IDF (formes)
# Seuil du vecteur MiniLM (sens) : calibre sur le cache reel — les paires
# a risque (« conjugue X » vs « conjugue Y », enonces de logique) scorent
# 0.72-0.93 mais sont filtrees en aval par _hit_douteux (jetons porteurs),
# tandis que les reformulations vraies (0.70+) deviennent des hits.
_SEUIL_VECTEUR = 0.70

# reponses contextuelles (liées a l'utilisateur ou a la conversation) :
# a ne JAMAIS mettre en cache, la reponse serait fausse hors contexte.
_MOTS_CONTEXTUELS = ("répète", "repete", "redis ", "recommence", "refais",
                     "encore", "mon ", "ma ", "mes ", "je ", "j'", "moi", "tu ", "ton ",
                     "ta ", "tes ", "nous ", "votre ", "vos ", "prenom",
                     "mon nom", "mon age")

# Une reponse redigee depend fortement de la consigne exacte. La similarite
# vectorielle suffit pour retrouver un fait court, pas pour reutiliser une
# dissertation ou une explication sur une autre question.
_RE_QUESTION_OUVERTE = re.compile(
    r"\b(expliquez?|pourquoi|comment|analysez?|comparez?|redigez?|"
    r"dissertation|developpez?|argumentez?|discutez?|decrivez?|"
    r"presentez?|quels? sont les avantages|quelles? sont les limites)\b",
    re.IGNORECASE,
)
_REPONSE_CACHE_MAX = 700


def _cache_reutilisable(question: str, reponse: str) -> bool:
    """Garde uniquement les reponses courtes a des demandes fermees."""
    q = (question or "").strip()
    r = (reponse or "").strip()
    if not q or not r or len(r) > _REPONSE_CACHE_MAX:
        return False
    q_normalisee = unicodedata.normalize("NFKD", q.lower())
    q_normalisee = "".join(c for c in q_normalisee
                           if not unicodedata.combining(c))
    if _RE_QUESTION_OUVERTE.search(q_normalisee):
        return False
    return True

# -- eval arithmetique securise (AST, jamais eval()) ------------------------

# Annoté : sans le type, mypy (qui verifie desormais les corps non annotes,
# `check_untyped_defs`) ne sait pas QUELLES fonctions sont ces valeurs et
# refuse de les appeler — « Cannot call function of unknown type ».
_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_PUISSANCE_MAX = 1e15  # garde-fou : 2**10000 ferait planter le CPU

# -- expressions dictees en mots ---------------------------------------------
# « 15 divise par 3 plus 4 puissance 2 » -> « 15 / 3 + 4 ** 2 »
_MOTS_OPERATEURS = (
    (re.compile(r"\bdivis\w*\s+par\b"), "/"),
    (re.compile(r"\bmultipli\w*\s+par\b"), "*"),
    (re.compile(r"\bfois\b"), "*"),
    (re.compile(r"\bau\s+carre\b"), "**2"),
    (re.compile(r"\bau\s+cube\b"), "**3"),
    (re.compile(r"\bpuissance\b"), "**"),
    (re.compile(r"\bplus\b"), "+"),
    (re.compile(r"\bmoins\b"), "-"),
)
# operateurs « forts » : leur presence seule autorise la conversion
# (plus/moins seuls sont ambigus : « il fait moins 5 degres »)
_OP_FORTE = re.compile(
    r"\b(divis\w*\s+par|multipli\w*\s+par|fois|puissance|au\s+carre|au\s+cube)\b")
_DEMANDE_MATH = re.compile(
    r"\b(combien fait|combien font|combien vaut|calcule|resultat de|resous)\b")
_RESIDUS = re.compile(r"[\d.+\-*/()\s]+")
_NOMBRE = re.compile(r"\d+(?:\.\d+)?")


def _expression_en_mots(q: str) -> str | None:
    """Convertit une expression dictee en arithmetique, ou None.

    « 15 divise par 3 plus 4 puissance 2 » -> « 15 / 3 + 4 ** 2 ».
    Les mots non convertis sont jetes ; il faut >= 2 nombres et >= 1
    operateur binaire pour tenter l'evaluation.
    """
    expr = q
    for motif, op in _MOTS_OPERATEURS:
        expr = motif.sub(f" {op} ", expr)
    candidat = " ".join("".join(_RESIDUS.findall(expr)).split())
    if len(_NOMBRE.findall(candidat)) < 2:
        return None
    if not re.search(r"[+\-*/]", candidat.replace("**", "")):
        return None
    return candidat


def _evaluer(noeud):
    if isinstance(noeud, ast.Expression):
        return _evaluer(noeud.body)
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)):
        return noeud.value
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _OPS:
        gauche, droite = _evaluer(noeud.left), _evaluer(noeud.right)
        if isinstance(noeud.op, ast.Pow):
            if abs(droite) > 64 or abs(gauche) ** min(abs(droite), 64) > _PUISSANCE_MAX:
                raise ValueError("puissance hors limites")
        return _OPS[type(noeud.op)](gauche, droite)
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPS:
        return _OPS[type(noeud.op)](_evaluer(noeud.operand))
    raise ValueError("expression non autorisee")


# -- contre-verification independante (fix bug 2, 12 oct. 2026) -----------
# « Combien font (4583 * 921) + 14582 » a ete servi 1887823 au lieu de
# 4235525. Un resultat de calcul n'est desormais livre que si DEUX
# evaluateurs independants s'accordent : l'AST garde ci-dessus et un
# parseur a descente recursive en fractions exactes (priorites : ** >
# unaire > * / // % > + -). Desaccord = abstention (None), jamais un chiffre
# faux servi avec l'aplomb d'un resultat prouve.
_JETON_CALCUL = re.compile(r"\s*(\d+(?:\.\d+)?|\*\*|//|[-+*/%()])")


def _evaluer_independant(expr: str) -> Fraction:
    """Second evaluateur, SANS ast : descente recursive en fractions exactes."""
    jetons, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _JETON_CALCUL.match(expr, pos)
        if not m:
            raise ValueError("jeton inattendu")
        jetons.append(m.group(1))
        pos = m.end()
    i = 0

    def voir():
        return jetons[i] if i < len(jetons) else ""

    def prendre():
        nonlocal i
        j = voir()
        i += 1
        return j

    def somme():
        v = produit()
        while voir() in ("+", "-"):
            op = prendre()
            d = produit()
            v = v + d if op == "+" else v - d
        return v

    def produit():
        v = unaire()
        while voir() in ("*", "/", "//", "%"):
            op = prendre()
            d = unaire()
            if op == "*":
                v = v * d
            elif d == 0:
                raise ZeroDivisionError
            elif op == "/":
                v = v / d
            elif op == "//":
                v = Fraction(v // d)
            else:
                v = v % d
        return v

    def unaire():
        if voir() in ("+", "-"):
            op = prendre()
            v = unaire()
            return -v if op == "-" else v
        return puissance()

    def puissance():
        base = atome()
        if voir() == "**":
            prendre()
            exposant = unaire()
            if exposant.denominator != 1 or abs(exposant) > 64:
                raise ValueError("puissance hors limites")
            return base ** int(exposant)
        return base

    def atome():
        j = prendre()
        if j == "(":
            v = somme()
            if prendre() != ")":
                raise ValueError("parenthese fermante attendue")
            return v
        if not j or not j[0].isdigit():
            raise ValueError("operande attendu")
        return Fraction(j)

    resultat = somme()
    if i != len(jetons):
        raise ValueError("jetons en surplus")
    return resultat


def _calculer(expr: str):
    """AST garde + contre-verification independante (cf. ci-dessus)."""
    v_ast = _evaluer(ast.parse(expr, mode="eval"))
    v_ind = _evaluer_independant(expr)
    if isinstance(v_ast, int):
        accord = v_ind == v_ast
    else:
        accord = abs(float(v_ind) - float(v_ast)) <= 1e-9 * max(1.0, abs(float(v_ind)))
    if not accord:
        raise ValueError("evaluateurs en desaccord")
    return v_ast


def _calcul_direct(question: str) -> str | None:
    """Resout les maths directes par regex + AST. Aucun LLM, aucune erreur."""
    q = unicodedata.normalize("NFKD", question.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    # Une consigne de format apres la question (« ? Reponds en un seul
    # chiffre ») ne fait pas partie de l'expression. Sans cette coupure,
    # le 1 de « un » est ramasse comme un operande et fait echouer l'AST.
    q = re.sub(r"\?\s*(?:reponds?|repondre|donne|ecris|affiche)\b.*$",
               "?", q)
    # « x » n'est un signe de multiplication QUE entre deux operandes
    # (« 4583 x 921 », « 3x4 ») : remplace partout, il transformait « 2x+3 »
    # en « 2*+3 » (= 6) et corrompait les mots contenant un x.
    q = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", " * ", q)
    q = q.replace(",", ".").replace("×", "*").replace("÷", "/") \
         .replace("^", "**")

    # expression dictee en mots, AVANT les motifs simples : sur une
    # expression composee, « 4 puissance 2 » tout seul donnerait 16
    # au lieu de 21 (le bug du 15/3 + 4^2)
    if _OP_FORTE.search(q) or _DEMANDE_MATH.search(q):
        expr = _expression_en_mots(q)
        if expr:
            try:
                return _fmt(_calculer(expr))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                pass

    # « carre de 12 » / « cubé de 3 »
    m = re.search(r"carre de \(?(-?\d+(?:\.\d+)?)\)?", q)
    if m:
        n = float(m.group(1))
        return _fmt(n * n)

    # « cube de 3 » / « cubé de 3 » — meme famille que « carre de », mais
    # le motif manquait : au banc 100, les 3 questions « Quel est le cube
    # de N ? » partaient au LLM (~40 s) alors que « le carre de N » voisin
    # etait deja exact. Un mot de vocabulaire separe 5 ms de 40 s.
    m = re.search(r"\bcube de \(?(-?\d+(?:\.\d+)?)\)?", q)
    if m:
        n = float(m.group(1))
        return _fmt(n * n * n)

    # « racine carree de 144 »
    m = re.search(r"racine carree de \(?(\d+(?:\.\d+)?)\)?", q)
    if m:
        return _fmt(float(m.group(1)) ** 0.5)

    # « 2 puissance 10 »
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*puissance\s*(-?\d+)", q)
    if m and abs(int(m.group(2))) <= 64:
        return _fmt(float(m.group(1)) ** int(m.group(2)))

    # « 15% de 200 » / « 15 pourcent de 200 »
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|pourcent)\s*de\s*(\d+(?:\.\d+)?)", q)
    if m:
        return _fmt(float(m.group(1)) / 100 * float(m.group(2)))

    # expression arithmetique brute : « calcule 45*12+3 », ou juste « 2+2 ».
    # GARDE (10 oct. 2026) : ce motif cherchait n'importe ou dans la phrase,
    # donc il attrapait « 3/4 » dans « 3/4 of the building is occupied » et
    # rendait 0,75 — une reponse a une AUTRE question, servie en 0,02 s avec
    # l'aplomb d'un resultat prouve. Mesure au banc GSM8K : 7 cas sur 100,
    # tous faux (cf. ARCHITECTURE.md § 3sexies). Une expression nue ne vaut
    # que si la question EST l'expression : il ne doit rester aucune lettre
    # une fois l'expression retiree. Les formes dictees (« combien fait »,
    # « calcule », « 3 fois 7 ») partent plus haut et ne dependent pas d'ici.
    #
    # FIX (12 oct. 2026) : le motif ne matchait pas les expressions commencant
    # par une parenthese ouvrante comme « (4583 * 921) + 14582 ». Ajout d'un
    # motif elargi acceptant le prefixe «(» et la verification du residus
    # hors-expression (aucune lettre = la question EST l'expression).
    m = re.search(
        r"(\(?-?\d+(?:\.\d+)?\)?(?:\s*[\+\-\*/%]\s*\(?-?\d+(?:\.\d+)?\)?)+)", q)
    if m:
        avant, apres = q[:m.start()], q[m.end():]
        if not re.search(r"[a-z]", avant + apres):
            try:
                return _fmt(_calculer(m.group(1)))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                pass
    return None


def _fmt(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else f"{v:.6g}"


# -- cache semantique --------------------------------------------------------

_vectoriseur = None
_matrice = None
_entrees: list[dict] = []
_vecteurs = None          # np.ndarray (n, 384) float32 | None : index MiniLM


def _fichier_vecteurs() -> Path:
    """Index dense colle au cache jsonl (suit AURA_FILTRE_CACHE)."""
    return Path(str(_FICHIER) + ".vecs.npy")


def _encoder(textes: list[str]) -> np.ndarray | None:
    """Vecteurs MiniLM, ou None (coupe, modele absent, erreur)."""
    try:
        from . import embeddings
        if not embeddings.actif():
            return None
        return embeddings.encoder(textes)
    except Exception:  # noqa: BLE001
        return None


def _minilm_actif() -> bool:
    """Coupe generale : AURA_MINILM=0 desactive l'index semantique."""
    try:
        from . import embeddings
        return embeddings.actif()
    except Exception:  # noqa: BLE001
        return False


def _construire_vecteurs() -> None:
    """Charge l'index semantique (.npy) ou le reconstruit (~4 ms/entree).

    Sans MiniLM : index vide, le cache retombe sur le TF-IDF historique
    (comportement inchange).
    """
    global _vecteurs
    if not _entrees or not _minilm_actif():
        _vecteurs = None
        return
    fich = _fichier_vecteurs()
    if fich.exists():
        try:
            v = np.load(fich)
            if v.shape == (len(_entrees), 384):
                _vecteurs = v.astype(np.float32, copy=False)
                return
        except (OSError, ValueError):
            pass
    v = _encoder([e["q"] for e in _entrees])
    if v is None:
        _vecteurs = None
        try:
            fich.unlink(missing_ok=True)   # npy orphelin (MiniLM coupee)
        except OSError:
            pass
        return
    _vecteurs = np.asarray(v, dtype=np.float32)
    _sauver_vecteurs()


def _sauver_vecteurs() -> None:
    if _vecteurs is None:
        return
    try:
        with _fichier_vecteurs().open("wb") as f:
            np.save(f, _vecteurs)
    except OSError as e:
        LOG.debug("[niveau0] index vectoriel non ecrit : %s", e)


def _charger_cache():
    global _vectoriseur, _matrice, _entrees
    if _vectoriseur is not None:
        return
    if _FICHIER.exists():
        try:
            _entrees = [json.loads(ligne) for ligne in
                        _FICHIER.read_text(encoding="utf-8").splitlines()
                        if ligne.strip()]
        except (OSError, json.JSONDecodeError):
            _entrees = []
    # PURGE DES CALCULS (fix bug 2) : une question que le calcul exact sait
    # resoudre n'a rien a faire en memoire approximative. Une entree ancienne
    # et FAUSSE (ex : 1887823 pour (4583 * 921) + 14582) etait relue a vie.
    if _entrees:
        gardees = [e for e in _entrees
                   if _calcul_direct(str(e.get("q", ""))) is None]
        if len(gardees) != len(_entrees):
            LOG.info("[niveau0] cache : %d entree(s) de calcul retiree(s)",
                     len(_entrees) - len(gardees))
            _entrees = gardees
            try:
                with _FICHIER.open("w", encoding="utf-8") as f:
                    for e in _entrees:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                _fichier_vecteurs().unlink(missing_ok=True)
            except OSError as e:
                LOG.debug("[niveau0] purge des calculs non ecrite : %s", e)
    if _entrees:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _vectoriseur = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                       max_features=2000)
        _matrice = _vectoriseur.fit_transform([e["q"] for e in _entrees])
    _construire_vecteurs()


def _similar(question: str) -> int | None:
    """Index de l'entree la plus proche si >= seuil, sinon None.

    1. Sens (MiniLM) : cosinus >= _SEUIL_VECTEUR sur l'index dense —
       une reformulation est reconnue, pas seulement sa copie de forme.
    2. Repli TF-IDF char n-grams (>= _SEUIL_SIMILARITE) : present sans
       MiniLM, et conserve comme second avis.
    """
    q = question.lower().strip()
    if _vecteurs is not None:
        v = _encoder([q])
        if v is not None:
            sims = _vecteurs @ np.asarray(v[0], dtype=np.float32)
            i = int(np.argmax(sims))
            if float(sims[i]) >= _SEUIL_VECTEUR:
                return i
    from sklearn.metrics.pairwise import cosine_similarity
    if _vectoriseur is None or _matrice is None:
        return None
    vec = _vectoriseur.transform([q])
    sims = cosine_similarity(vec, _matrice)[0]
    i = int(np.argmax(sims))
    return i if sims[i] >= _SEUIL_SIMILARITE else None


_STOP_MOTS = frozenset((
    "conjugue", "combien", "font", "fait", "quel", "quelle", "quels",
    "quelles", "est", "cette", "dans", "pour", "avec", "sans", "plus",
    "moins", "tous", "tout", "toute", "toutes", "autre", "autres",
    "personne", "premiere", "deuxieme", "troisieme", "singulier",
    "pluriel", "peut", "etre", "sont", "mais", "comme", "aussi",
))


# Nombres en toutes lettres -> chiffres, pour comparer « cube de sept » et
# « cube de 7 » comme la MEME question.
_NOMBRES_MOTS = {
    "zero": "0", "un": "1", "une": "1", "deux": "2", "trois": "3",
    "quatre": "4", "cinq": "5", "six": "6", "sept": "7", "huit": "8",
    "neuf": "9", "dix": "10", "onze": "11", "douze": "12",
    "treize": "13", "quatorze": "14", "quinze": "15", "seize": "16",
    "vingt": "20", "trente": "30", "quarante": "40", "cinquante": "50",
    "soixante": "60", "cent": "100", "mille": "1000",
}


def _nombres(texte: str) -> list[str]:
    """Nombres d'une question, normalises en chiffres.

    Les nombres sont les jetons les PLUS decideants d'une question de
    calcul : « le cube de 7 » et « le cube de 3 » ne different QUE par lui.
    Or _hit_douteux ne regardait que les mots de plus de 3 lettres, donc le
    nombre etait ignore et le cache servait « 3^3 = 27 » pour « le cube de
    7 » (bug reel trouve par scripts/banc_100.py : 2 questions sur 30).
    """
    n = unicodedata.normalize("NFKD", texte.lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    trouves = re.findall(r"\d+(?:[.,]\d+)?", n)
    trouves += [_NOMBRES_MOTS[m] for m in re.findall(r"[a-z]+", n)
                if m in _NOMBRES_MOTS]
    return sorted(trouves)


def _hit_douteux(question: str, entree: dict) -> bool:
    """Garde-fou anti-contamination du cache semantique : refuse un hit
    dont la structure ressemble mais dont des mots porteurs de sens
    (verbe, nom, nombre) sont absents de la question ET de la reponse
    en cache. Exemple reel : « conjugue POUVOIR au present » recevait la
    table de « conjugue FAIRE au futur » (similarite TF-IDF >= seuil)."""
    import unicodedata

    def jetons(s: str) -> set[str]:
        n = unicodedata.normalize("NFKD", s.lower())
        n = "".join(c for c in n if not unicodedata.combining(c))
        return set(re.findall(r"[a-z0-9]+", n))

    # Les reponses longues/ouvertes sont trop dependantes de la consigne pour
    # etre reutilisees par similarite, y compris dans un cache deja existant.
    if not _cache_reutilisable(question, str(entree.get("r", ""))) \
            or not _cache_reutilisable(str(entree.get("q", "")),
                                       str(entree.get("r", ""))):
        return True

    # NOMBRES DECISIFS : deux questions qui ne different que par un nombre
    # sont deux questions DIFFERENTES. On exige donc la MEME liste de
    # nombres (comparee en multiset, donc insensible a l'ordre).
    if _nombres(question) != _nombres(str(entree.get("q", ""))):
        return True
    q_jetons = jetons(question)
    cache_jetons = jetons(str(entree.get("q", "")) + " " +
                          str(entree.get("r", "")))
    for w in q_jetons:
        if len(w) > 3 and w not in _STOP_MOTS and w not in cache_jetons:
            return True
    return False


def repondre(question: str) -> str | None:
    """Etage DETERMINISTE du niveau 0 (< 1 ms) : calcul direct puis PAL.

    N'inclut PAS le cache semantique, et c'est une propriete d'ARCHITECTURE,
    pas un detail : le cache est approximatif (cosinus MiniLM >= 0,70 sur
    384 dimensions) alors que ces deux etages sont exacts. Un flou ne doit
    jamais passer devant un exact, sinon on ne peut plus savoir ce que le
    systeme sait calculer (observe : le cache a servi 240 a une question de
    taux avant que `calcul_taux` ne la calcule).

    L'autre etage est `repondre_cache`, a consulter APRES les solveurs.
    """
    t0 = time.time()

    # 1. calcul direct (< 1 ms)
    r = _calcul_direct(question)
    if r is not None:
        LOG.info("[niveau0] calcul direct en %.1f ms", (time.time() - t0) * 1000)
        return r

    # 1bis. PAL (< 1 ms) : dates, unites, pourcentages composes —
    # import paresseux (pal importe filtre_instantane pour l'AST garde)
    from . import pal
    r = pal.repondre(question)
    if r is not None:
        LOG.info("[niveau0] PAL en %.1f ms", (time.time() - t0) * 1000)
        return r
    return None


def repondre_cache(question: str) -> str | None:
    """Cache semantique SEUL (~1 ms). A consulter APRES tout solveur exact.

    JAMAIS pour une question contextuelle : sa reponse depend de la
    conversation en cours, pas du texte seul.
    """
    t0 = time.time()
    if any(m in question.lower() for m in _MOTS_CONTEXTUELS):
        return None
    _charger_cache()
    if _entrees:
        i = _similar(question)
        if i is not None:
            if _hit_douteux(question, _entrees[i]):
                LOG.info("[niveau0] hit de cache REFUSE (jetons divergents) "
                         "pour : %s", question[:60])
                return None
            LOG.info("[niveau0] cache semantique en %.1f ms",
                     (time.time() - t0) * 1000)
            return _entrees[i]["r"]
    return None


def repondre_tout(question: str) -> str | None:
    """Exact PUIS flou.

    Reserve aux outils qui veulent « le niveau 0 complet » d'un seul appel
    (bancs, mesures de latence). La CASCADE, elle, ne doit PAS utiliser
    cette fonction : elle appelle les deux etages SEPAREMENT, pour placer
    tous les solveurs exacts entre les deux.
    """
    r = repondre(question)
    return r if r is not None else repondre_cache(question)


def enregistrer(question: str, reponse: str):
    """Memorise une reponse (apres generation LLM) pour les futures questions.

    Les questions contextuelles (mon prenom, mon age...) sont exclues : leur
    reponse depend de la conversation, pas du texte de la question.
    """
    if (not question or not reponse or reponse.startswith("[RontoGPT]")
            or not _cache_reutilisable(question, reponse)):
        return
    q = question.lower().strip()
    if any(m in q for m in _MOTS_CONTEXTUELS):
        return  # reponse contextuelle : jamais en cache
    if _calcul_direct(question) is not None:
        return  # calcul exact : recalcule a chaque fois, jamais memorise
    _charger_cache()
    if _entrees and _similar(question) is not None:
        return  # deja connu
    entree = {"q": question.lower().strip(), "r": reponse,
              "t": time.strftime("%Y-%m-%d")}
    _entrees.append(entree)
    try:
        with _FICHIER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        from sklearn.feature_extraction.text import TfidfVectorizer
        global _vectoriseur, _matrice, _vecteurs
        if _vectoriseur is None:
            _vectoriseur = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                           max_features=2000)
            _matrice = _vectoriseur.fit_transform([e["q"] for e in _entrees])
        else:
            nouvelle = _vectoriseur.transform([entree["q"]])
            from scipy.sparse import vstack
            _matrice = vstack([_matrice, nouvelle])
    except OSError as e:
        LOG.warning("[niveau0] cache non ecrit : %s", e)
    # memoire semantique : garde l'index aligne (1er enregistrement inclus)
    if _vecteurs is None or _vecteurs.shape[0] not in (
            len(_entrees), len(_entrees) - 1):
        _construire_vecteurs()          # (re)construit l'index complet
    if _vecteurs is not None and _vecteurs.shape[0] == len(_entrees) - 1:
        v = _encoder([entree["q"]])
        if v is not None:
            _vecteurs = np.vstack([_vecteurs,
                                   np.asarray(v, dtype=np.float32)])
            _sauver_vecteurs()


def _cle_exacte(question: str) -> str:
    """Forme comparable d'une question : minuscules, sans accents, sans
    ponctuation. « Qui es-tu ? » et « qui es tu » rendent la MEME cle."""
    n = unicodedata.normalize("NFKD", question.lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", n))


def _entree_exacte(question: str) -> int | None:
    """Index de l'entree dont la question est la meme une fois normalisee."""
    cible = _cle_exacte(question)
    if not cible:
        return None
    for i, e in enumerate(_entrees):
        if _cle_exacte(str(e.get("q", ""))) == cible:
            return i
    return None


def mettre_a_jour(question: str, reponse: str) -> bool:
    """RECONSOLIDATION : remplace la reponse (similaire) existante par la nouvelle.

    Comme le cerveau : un souvenir contredit par une preuve plus forte est
    reecrit, pas double. Sans cela, une reponse fausse mise en cache reste
    collée pour toujours (enregistrer refuse les doublons).
    Renvoie True si une entree a ete remplacee.
    """
    global _vectoriseur, _matrice
    if not question or not reponse or reponse.startswith("[RontoGPT]"):
        return False
    _charger_cache()
    # EXACT D'ABORD, FLOU ENSUITE (meme doctrine que `repondre_tout`). Sans ce
    # passage, la reconsolidation dependait ENTIEREMENT de l'index MiniLM :
    # sans lui (machine neuve, checkout CI sans `modeles/minilm`), « Qui es-tu ?
    # » ne matchait pas l'entree « qui es tu » — le TF-IDF exige 0.85 et la
    # ponctuation suffit a l'abaisser — et la correction du cache ne faisait
    # RIEN, sans le dire. Le flou reste utile pour les vraies reformulations.
    i = _entree_exacte(question)
    if i is None and _vectoriseur is not None and _matrice is not None:
        i = _similar(question.lower().strip())
    if i is None:
        return False
    # GARDE-FOU : ne JAMAIS reecrire la reponse d'une AUTRE question. Sans
    # ce test, la reconsolidation ecrasait l'entree d'une question voisine :
    # « cube de 8 » (meme vecteur que « cube de 3 ») reecrivait l'entree de
    # « cube de 3 », et le cache servait ensuite « 8^3 = 512 » a la question
    # « le cube de 3 ? » — le garde-fou de repondre() etait donc contourne
    # par l'ECRITURE (bug reel trouve par scripts/banc_100.py).
    # Aucune perte : la reponse verifiee est enregistree juste apres par
    # enregistrer(), qui cree l'entree manquante au lieu d'en ecraser une.
    if _hit_douteux(question, _entrees[i]):
        LOG.info("[niveau0] reconsolidation REFUSEE (question "
                 "differente) : %s", question[:50])
        return False
    _entrees[i]["r"] = reponse
    _vectoriseur, _matrice = None, None      # recalcule au prochain acces
    try:
        with _FICHIER.open("w", encoding="utf-8") as f:
            for e in _entrees:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        LOG.info("[niveau0] reconsolidation : reponse remplacee pour « %s »",
                 question[:50])
        return True
    except OSError as e:
        LOG.warning("[niveau0] reconsolidation non ecrite : %s", e)
        return False


def purger_contextuelles():
    """Retire du cache les entrees contextuelles (mises en cache avant le filtre)."""
    global _vectoriseur, _matrice, _entrees, _vecteurs
    _charger_cache()
    avant = len(_entrees)
    garder = [e for e in _entrees
              if not any(m in e["q"] for m in _MOTS_CONTEXTUELS)]
    if len(garder) != avant:
        _entrees = garder
        _vectoriseur, _matrice = None, None   # recalcule au prochain acces
        _vecteurs = None                      # index reconstruit au prochain acces
        try:
            _fichier_vecteurs().unlink(missing_ok=True)
        except OSError:
            pass
        try:
            with _FICHIER.open("w", encoding="utf-8") as f:
                for e in _entrees:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except OSError as e:
            LOG.warning("[niveau0] purge non ecrite : %s", e)
        LOG.info("[niveau0] purge : %d entrees contextuelles retirees",
                 avant - len(garder))
    return avant - len(_entrees)


def stats() -> dict:
    _charger_cache()
    return {"entrees_cache": len(_entrees), "seuil": _SEUIL_SIMILARITE,
            "fichier": str(_FICHIER)}
