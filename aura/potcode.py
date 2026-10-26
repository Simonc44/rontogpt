"""Expert « PoT-code » : le 1B ecrit une fonction + des tests, la sandbox verifie.

Inspiré de Program-of-Thoughts et du self-debug : sur une demande de code,
le modele ecrit une fonction `resoudre` accompagnee d'asserts DANS LE MEME
BLOC. La sandbox execute tout : un assert qui echoue = code refuse — le
code rate ne sort JAMAIS, le pipeline retombe sur la generation normale.

Securite (honnete) :
- **processus separe** : le code s'execute dans un interpreteur ENFANT
  (`python -I -S -c ...`), pas dans le processus d'RontoGPT. Une fuite de
  la sandbox ne donne donc ni la memoire du cerveau, ni ses fichiers
  ouverts, ni la possibilite de tuer l'API ; le parent impose un delai et
  tue l'enfant ;
- **limite memoire** : RLIMIT_AS 256 Mio (POSIX) et Job Object Windows
  (limite de memoire du processus + mort avec le job) : un `bytearray`
  geant echoue dans l'enfant au lieu de faire tomber la machine ;
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
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any

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


#: Memoire maximale du processus de la sandbox.
_MEMOIRE_MAX = 256 << 20
#: Delai maximal d'un code (secondes) : le budget d'instructions arrete les
#: boucles, celui-ci attrape le reste (attente bloquante, calcul lourd).
_DELAI_MAX = 10.0

#: Programme de l'ENFANT. Volontairement autonome : il n'importe PAS `aura`
#: (l'import du paquet charge le cerveau et tout le pipeline), il recoit le
#: code, le budget et la liste des builtins autorises en JSON sur stdin.
_RUNNER = '''
import builtins, json, sys

demande = json.loads(sys.stdin.read())
code, budget, noms = demande["code"], demande["budget"], demande["builtins"]

# LIMITES DU PROCESSUS, posees par l'enfant lui-meme (POSIX). Windows est
# borne par un Job Object cote parent. Best-effort : un systeme sans
# `resource` n'empeche pas le code de tourner, il retire juste la borne.
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except Exception:
    pass

env = {"__builtins__": {n: getattr(builtins, n) for n in noms
                        if hasattr(builtins, n)}}
compteur = [0]


def repondre(**kw):
    print(json.dumps(kw))
    raise SystemExit(0)


def trace(frame, event, arg):
    if event == "line":
        compteur[0] += 1
        if compteur[0] > budget:
            raise SystemExit("budget")
    return trace


sys.settrace(trace)
try:
    exec(compile(code, "<aura-sandbox>", "exec"), env)
except AssertionError as e:
    repondre(ok=False, erreur="assert echoue : %s" % (e or "(sans message)"))
except SystemExit as e:
    if str(e) == "budget":
        repondre(ok=False, erreur="budget d'instructions depasse "
                                  "(boucle infinie ?)")
    repondre(ok=False, erreur="sortie interdite dans la sandbox")
except MemoryError:
    repondre(ok=False, erreur="memoire du bac a sable depassee (256 Mio)")
except Exception as e:
    repondre(ok=False, erreur="execution illicite : %s: %s"
                              % (type(e).__name__, e))

if not callable(env.get("resoudre")):
    repondre(ok=False, erreur="fonction resoudre absente")
repondre(ok=True, asserts=sum(1 for ligne in code.splitlines()
                              if ligne.strip().startswith("assert ")))
'''


def _env_enfant() -> dict[str, str]:
    """Environnement minimal de l'enfant : ni AURA_*, ni identifiants."""
    # Liste BLANCHE : ce dont un interpreteur a besoin pour demarrer, et rien
    # qui ressemble a un secret (`AURA_*`, `*_KEY`, `*_TOKEN` n'y sont pas par
    # construction ; une liste noire laisserait passer `DB_PASSWORD`).
    gardes = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC",
              "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
              "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV")
    env = {k: v for k, v in os.environ.items() if k in gardes}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _kernel32() -> Any:
    """La DLL noyau, sans nommer `ctypes.WinDLL` dans le code.

    Les stubs de `ctypes` ne declarent `WinDLL` que sous Windows : un appel
    direct passe sous mypy sur la machine de developpement et **casse la CI
    Linux** (`Module has no attribute "WinDLL"`). Le `getattr` dit la verite :
    cette branche ne s'execute que sous Windows, et le typage n'a pas a la
    resoudre ailleurs.
    """
    import ctypes
    return getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)


def _job_memoire_windows(proc: subprocess.Popen) -> "object | None":
    """Job Object Windows : memoire bornee et mort avec le job.

    `resource` n'existe pas sous Windows ; c'est le SEUL endroit ou la limite
    memoire peut etre posee la-bas. Best-effort assume : si l'API refuse, on
    le dit dans le journal et le reste de la sandbox (processus separe,
    delai, AST, builtins) reste en place.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Ioc(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class Basique(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", ctypes.c_ulong),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", ctypes.c_ulong),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", ctypes.c_ulong),
                        ("SchedulingClass", ctypes.c_ulong)]

        class Etendu(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basique),
                        ("IoInfo", Ioc),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        noyau = _kernel32()
        noyau.CloseHandle.argtypes = [wintypes.HANDLE]
        noyau.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,
                                                   wintypes.HANDLE]
        noyau.CreateJobObjectW.restype = wintypes.HANDLE
        job = noyau.CreateJobObjectW(None, None)
        if not job:
            return None
        infos = Etendu()
        # PROCESS_MEMORY | KILL_ON_JOB_CLOSE : 256 Mio, et le processus meurt
        # avec le job meme si le parent disparait.
        infos.BasicLimitInformation.LimitFlags = 0x00000100 | 0x00002000
        infos.ProcessMemoryLimit = _MEMOIRE_MAX
        if not noyau.SetInformationJobObject(job, 9, ctypes.byref(infos),
                                             ctypes.sizeof(infos)):
            noyau.CloseHandle(job)
            return None
        handle = getattr(proc, "_handle", None)
        if handle is None or not noyau.AssignProcessToJobObject(job, handle):
            noyau.CloseHandle(job)
            return None
        return job
    except Exception as e:  # noqa: BLE001 — la sandbox reste utilisable
        LOG.warning("[potcode] limite memoire Windows indisponible : %s", e)
        return None


def _executer_isole(code: str, budget: int,
                    timeout: float = _DELAI_MAX) -> dict:
    """Execute le code dans un processus ENFANT jetable et rend son rapport.

    Le parent ne partage ni memoire, ni fichiers ouverts, ni verrous avec
    l'enfant : une evasion de la barriere AST atteindrait un interpreteur
    qui ne connait ni le cerveau, ni l'API, ni la memoire de conversation.
    """
    demande = json.dumps({"code": code, "budget": budget,
                          "builtins": sorted(_SANDBOX_BUILTINS)})
    with tempfile.TemporaryDirectory(prefix="aura-sandbox-") as dossier:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _RUNNER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=dossier,
            env=_env_enfant(),
            creationflags=0x08000000 if os.name == "nt" else 0)
        job = _job_memoire_windows(proc)
        try:
            sortie, erreur = proc.communicate(demande, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise ValueError("delai de la sandbox depasse (code bloque)")
        finally:
            if job is not None:
                _kernel32().CloseHandle(job)

    # Le rapport est la DERNIERE ligne : une sortie parasite de l'enfant
    # (avertissement, trace) ne doit pas transformer un refus en panne.
    lignes = [ligne for ligne in (sortie or "").splitlines() if ligne.strip()]
    try:
        rapport = json.loads(lignes[-1] if lignes else "{}")
    except ValueError as e:
        detail = (erreur or "").strip()[:200] or "aucune sortie"
        raise ValueError(f"sandbox muette ({detail})") from e
    if not rapport.get("ok"):
        raise ValueError(str(rapport.get("erreur") or "code refuse"))
    return rapport


def verifier_code(code: str, budget: int = _BUDGET_INSTRUCTIONS) -> dict:
    """Chaine complete : extraction deja faite -> barriere AST -> enfant.

    Un assert qui echoue leve ValueError : le code rate ne sort jamais.
    Rend ``{"nb_asserts", "statut"}`` — plus la fonction : elle vit dans
    l'enfant, qui meurt apres verification. Aucun appelant ne s'en servait,
    et la garder aurait oblige a re-executer le code dans CE processus.
    """
    try:
        syntaxe = ast.parse(code, filename="<aura-sandbox>", mode="exec")
    except SyntaxError as e:
        raise ValueError(f"syntaxe invalide : {e}") from e
    _refuser_evasions(syntaxe)

    rapport = _executer_isole(code, budget)
    nb_asserts = int(rapport.get("asserts") or 0)
    if nb_asserts == 0:
        raise ValueError("aucun assert fourni")
    LOG.info("[potcode] code valide : %d asserts passes", nb_asserts)
    return {"nb_asserts": nb_asserts, "statut": "valide"}


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
