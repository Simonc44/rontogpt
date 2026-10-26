"""Expert « PoT-code » : le 1B ecrit une fonction + des tests, la sandbox verifie.

Inspiré de Program-of-Thoughts et du self-debug : sur une demande de code,
le modele ecrit une fonction `resoudre` accompagnee d'asserts DANS LE MEME
BLOC. La sandbox execute tout : un assert qui echoue = code refuse — le
code rate ne sort JAMAIS, le pipeline retombe sur la generation normale.

Securite (honnete) :
- builtins restrictifs : pas d'import, pas d'open/eval/exec/compile/getattr ;
- **barriere AST** : tout nom ou attribut dunder (`__class__`,
  `__bases__`, `__globals__`...) est refuse AVANT l'execution. Les builtins
  seuls ne suffisaient pas : `().__class__.__bases__[0].__subclasses__()`
  n'en utilise aucun. Le docstring promettait cette barriere, le code ne
  l'avait pas — c'est desormais vrai, et teste par les evasions connues ;
- budget d'instructions par sys.settrace : toute boucle infinie est coupee
  (kill-switch garanti, pas un simple timeout de thread) ;
- extrait UNIQUEMENT du code entre balises ```python ... ``` ou ``` ... ```.
"""
import ast
import logging
import re
import sys
import unicodedata

LOG = logging.getLogger("aura.potcode")

_BUDGET_INSTRUCTIONS = 200_000    # ≈ 2-5 s de pur Python, largement assez

_SYSTEME_CODE = (
    "Tu ecris du code Python. Reponds UNIQUEMENT avec un bloc de code :\n"
    "```python\n"
    "def resoudre(donnees):\n"
    "    \"\"\"Resout le probleme demande.\"\"\"\n"
    "    ...ton code...\n"
    "    return resultat\n"
    "\n"
    "# tests : au moins 3 asserts sur resoudre avec des exemples concrets\n"
    "assert resoudre(...) == ...\n"
    "```\n"
    "Regles :\n"
    "- la fonction s'appelle resoudre et prend les donnees du probleme\n"
    "- pas d'import, pas d'entrees/sorties, pas d'acces a des fichiers\n"
    "- les asserts doivent couvrir les cas du probleme, cas limites inclus\n"
    "- rien hors du bloc de code"
)

_BLOC_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# builtins autorises : arithmetique, structures de donnees, exceptions —
# tout ce dont un algorithme a besoin, rien de ce qui touche au systeme
_SANDBOX_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "IndexError": IndexError, "KeyError": KeyError,
    "ZeroDivisionError": ZeroDivisionError,
}


def extraire_code(texte: str) -> str | None:
    """Extrait le premier bloc de code balise, ou None."""
    m = _BLOC_CODE.search(texte)
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


class _BudgetEpuise(Exception):
    pass


def _refuser_evasions(arbre: ast.AST) -> None:
    """Refuse les constructions qui menent hors de la sandbox, AVANT `exec`.

    Les builtins restreints ne suffisent pas : `().__class__.__bases__[0]`
    n'appelle aucun builtin et donne acces a `object`, puis a toutes les
    sous-classes chargees — donc a `os` et a `system`. On refuse donc, dans
    l'ARBRE :

    - tout attribut dunder (`obj.__class__`, `f.__globals__`...) ;
    - tout nom dunder (sauf ce que l'interpreteur ajoute lui-meme, qu'on ne
      peut pas ecrire : `__name__` est donc interdit aussi) ;
    - `import` / `from ... import` (l'import d'un module atteint le systeme) ;
    - une chaine qui melange `{` et `__` : `"{0.__class__}".format(x)` fait
      passer l'acces dunder par un FORMAT, pas par l'AST.
    """
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Attribute) and noeud.attr.startswith("__"):
            raise ValueError(f"evasion interdite : attribut {noeud.attr} "
                             f"(ligne {noeud.lineno})")
        if isinstance(noeud, ast.Name) and noeud.id.startswith("__"):
            raise ValueError(f"evasion interdite : nom {noeud.id} "
                             f"(ligne {noeud.lineno})")
        if isinstance(noeud, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"import interdit dans la sandbox "
                             f"(ligne {noeud.lineno})")
        if (isinstance(noeud, ast.Constant) and isinstance(noeud.value, str)
                and "__" in noeud.value and "{" in noeud.value):
            raise ValueError("evasion interdite : acces dunder dans une "
                             f"chaine de formatage (ligne {noeud.lineno})")


def _executer_sandbox(code: str,
                      budget: int = _BUDGET_INSTRUCTIONS) -> dict:
    """Compile et execute le code (definition + asserts) dans la sandbox.

    Renvoie l'environnement d'execution. Leve ValueError si le code est
    refuse : syntaxe, instruction illicite, budget depasse, assert rate.
    """
    try:
        # `ast.parse` d'abord : la barriere inspecte l'ARBRE. `compile()` rend
        # un objet code (pas un AST), et `ast.walk` y voit des `node._fields`
        # inexistants — l'erreur que ce detour a deja produite.
        syntaxe = ast.parse(code, filename="<aura-sandbox>", mode="exec")
        arbre = compile(syntaxe, "<aura-sandbox>", "exec")
    except SyntaxError as e:
        raise ValueError(f"syntaxe invalide : {e}") from e
    _refuser_evasions(syntaxe)

    env: dict = {"__builtins__": dict(_SANDBOX_BUILTINS)}
    compteur = {"n": 0}

    def _trace(frame, event, arg):
        if event == "line":
            compteur["n"] += 1
            if compteur["n"] > budget:
                raise _BudgetEpuise()
        return _trace

    # LE TRACEUR PRECEDENT EST RESTAURE, jamais efface. `sys.settrace(None)`
    # supprimait celui de l'HOTE : un debuggeur, un profileur ou la couverture
    # de code attaches au meme processus se taisaient apres le premier code
    # execute en sandbox. Mesure (16 oct. 2026) : sur la MEME suite, `pytest
    # --cov` rendait 20 % quand le traceur etait efface et 58 % quand il etait
    # restaure — les modules exercises APRES le premier sandbox (routeur,
    # pipeline, llama_cerveau) tombaient a 7-20 % contre 59-71 % en isolation.
    precedent = sys.gettrace()
    sys.settrace(_trace)
    try:
        exec(arbre, env)            # noqa: S102 — sandbox construite pour
    except _BudgetEpuise:
        raise ValueError("budget d'instructions depasse (boucle infinie ?)")
    except AssertionError as e:
        raise ValueError(f"assert echoue : {e or '(sans message)'}")
    except Exception as e:
        raise ValueError(f"execution illicite : {type(e).__name__}: {e}")
    finally:
        sys.settrace(precedent)
    return env


def verifier_code(code: str, budget: int = _BUDGET_INSTRUCTIONS) -> dict:
    """Chaine complete : extraction deja faite -> sandbox + asserts.

    Un assert qui echoue leve ValueError : le code rate ne sort jamais.
    Renvoie un rapport avec la fonction validee.
    """
    env = _executer_sandbox(code, budget)
    fn = env.get("resoudre")
    if not callable(fn):
        raise ValueError("fonction resoudre absente")
    nb_asserts = sum(1 for ligne in code.splitlines()
                     if ligne.strip().startswith("assert "))
    if nb_asserts == 0:
        raise ValueError("aucun assert fourni")
    LOG.info("[potcode] code valide : %d asserts passes", nb_asserts)
    return {"fonction": fn, "nb_asserts": nb_asserts, "statut": "valide"}


# ---------------------------------------------------------------------------
# AIGUILLAGE DU CODE (fix bug 3, 12 oct. 2026)
# ---------------------------------------------------------------------------
# Une demande d'ecriture de code ne doit JAMAIS partir en recherche internet :
# l'IA cherchait sa propre fonction sur DuckDuckGo. Ce detecteur est PUR
# (regex, zero dependance, < 0,1 ms) ; le routeur et l'orchestrateur s'en
# servent pour VERROUILLER la demande sur la sandbox locale et couper
# memoire_web. Un faux positif coute peu (pas de web pour une question qui
# parle de code) ; un faux negatif etait le bug.
#
# Volontairement PAS un simple mot-cle : « quelle est la derniere version de
# Python ? » ou « qu'est-ce que Python ? » sont des questions web legitimes.

def _norm(texte: str) -> str:
    t = unicodedata.normalize("NFKD", (texte or "").lower().replace("’", "'"))
    return "".join(c for c in t if not unicodedata.combining(c))


# « code postal », « code civil », « code promo »... : le mot code n'est pas du code
_FAUX_CODE = re.compile(
    r"\bcodes?\s+(?:postal|postaux|penal|civil|du travail|de la route|"
    r"de la consommation|promo|promotion|secret|pin|barre|iban|couleur|ape|"
    r"naf|insee|morse|erreur)\b")
# syntaxe Python directe : def, assert, import, class, bloc ```
_SYNTAXE_CODE = re.compile(
    r"(?m)\bdef\s+\w+|\bassert\b|^\s*(?:import|from)\s+[\w.]+(?:\s+import\b|\s*$)|"
    r"\bclass\s+\w+\s*[:(]|```|\blambda\s+\w*\s*:|\bprint\s*\(")
_VERBES_CODE = (
    r"(?:ecris|ecrire|ecrivez|ecris-moi|redige|rediger|cree|creer|code|coder|"
    r"programme|programmer|implemente|implementer|genere|generer|fais|faire|"
    r"fournis|donne|donne-moi|montre|montre-moi|propose|write|implement|"
    r"generate|create|build|make)")
_OBJETS_CODE = (
    r"(?:fonction|function|script|algorithme|algorithm|classe|class|methode|"
    r"snippet|code|python|javascript|typescript|sql|html|css|bash|regex|"
    r"expression reguliere|test unitaire|tests unitaires|unittest|pytest|"
    r"programme\s+(?:qui|python|en|pour|capable))")
_ECRIRE_CODE = re.compile(
    rf"\b{_VERBES_CODE}\b[^.?!\n]{{0,80}}?\b{_OBJETS_CODE}\b")
_LANGAGES = (r"(?:python|javascript|typescript|js|rust|java|php|ruby|kotlin|"
             r"swift|sql|bash|powershell|golang|html|css|c\+\+|c#)")
_CODE_LANGAGE = re.compile(
    rf"\b(?:code|script|fonction|function|snippet|programme|exemple)\b"
    rf"[^.?!\n]{{0,40}}?\b{_LANGAGES}(?![\w+#])|"
    rf"\bcomment\b[^?\n]{{0,80}}?\b(?:en|avec|sous|in)\s+{_LANGAGES}(?![\w+#])")
_AUTRES_LANGAGES = re.compile(
    r"\b(?:javascript|typescript|js|rust|java|php|ruby|kotlin|swift|sql|bash|"
    r"powershell|golang|html|css)\b|c\+\+|c#")
# hors contrat de la sandbox (pas d'import, pas d'E/S, pas de reseau)
_HORS_SANDBOX = re.compile(
    r"\b(?:fichier|fichiers|api|requete|requetes|http|https|reseau|socket|"
    r"pandas|numpy|input|scrap\w*|tkinter|flask|django|fastapi|interface|gui|"
    r"base de donnees|selenium)\b")
_CONTRAT_SANDBOX = re.compile(r"\b(?:fonction|function|algorithme|algorithm|def|assert)\b")


def est_demande_code(question: str) -> bool:
    """Vrai si la question demande d'ECRIRE/corriger du code.

    Exemples vrais : « Ecris une fonction Python qui trie une liste »,
    « def somme(a, b)... corrige », « assert resoudre([1]) == [1] »,
    « Comment inverser une liste en Python ? ».
    Exemples faux : « derniere version de Python ? », « code postal de
    Nantes ? », « programme TV de ce soir ».
    """
    q = _FAUX_CODE.sub(" ", _norm(question))
    return bool(_SYNTAXE_CODE.search(q) or _ECRIRE_CODE.search(q)
                or _CODE_LANGAGE.search(q))


def demande_sandbox_python(question: str) -> bool:
    """Vrai si la demande tient dans le contrat de la sandbox (fonction
    Python pure `resoudre`, sans import ni E/S) : on lance alors le PoT-code.

    Les autres demandes de code (JavaScript, fichiers, reseau...) restent
    verrouillees HORS WEB mais sont rendues au cerveau, sans sandbox.
    """
    if not est_demande_code(question):
        return False
    q = _norm(question)
    if _AUTRES_LANGAGES.search(q) and "python" not in q:
        return False
    return bool(_CONTRAT_SANDBOX.search(q)) and not _HORS_SANDBOX.search(q)
