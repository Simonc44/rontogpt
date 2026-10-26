"""Pont RontoGPT -> ile Coucou (Mochi).

Comment ca marche
-----------------
Coucou expose un *relais local* sous forme de named pipe Windows :
    \\\\.\\pipe\\coucou-<SID de l'utilisateur>
Ce pipe accepte UNE ligne JSON dont le champ ``hook_event_name`` decrit
l'evenement. Tout evenement different de ``PermissionRequest`` est pousse
tel quel vers l'ile, qui le rend (pastille "agent", cloche, carte...).

C'est exactement le meme canal que celui utilise par ``coucou-hook.exe``
pour Claude Code : RontoGPT peut donc apparaitre dans l'ile comme un agent,
au meme titre que Claude — avec sa propre pastille « aura ».

Ce que ce module envoie
-----------------------
    activite  -> UserPromptSubmit   (etat "reflexion" + le texte demande)
    termine   -> Stop               (etat "termine", son + badge)
    erreur    -> StopFailure        (etat "erreur", son)
    notification -> Notification    (message d'information)

Tout est *best effort* : si Coucou ne tourne pas, l'appel renvoie
``envoye: False`` et ne leve jamais. Le pont se coupe avec AURA_COUCOU=0.

Hors Windows (pipe inexistant), les fonctions se contentent de dire que
le pont n'est pas disponible : aucun import lourd, aucune exception.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

__all__ = [
    "EVENEMENTS",
    "notifier",
    "activite",
    "termine",
    "erreur",
    "est_disponible",
    "nom_pipe",
]

# Interrupteur global : AURA_COUCOU=0 desactive tout le pont.
_ENV = "AURA_COUCOU"
# On n'attend pas indefiniment un pipe occupe : au-dela, tant pis.
_TIMEOUT_DEFAUT = 0.6

# Nom "public" -> hook_event_name compris par l'ile.
_MAP: dict[str, str] = {
    "activite": "UserPromptSubmit",
    "termine": "Stop",
    "erreur": "StopFailure",
    "notification": "Notification",
}
EVENEMENTS = tuple(_MAP)


def _sur_windows() -> bool:
    return os.name == "nt"


# ── Nom du pipe ────────────────────────────────────────────────────────

def _sid_utilisateur() -> str | None:
    """SID du compte courant, ex. ``S-1-5-21-...`` (Windows seulement).

    Doit correspondre EXACTEMENT a ``platform::current_user_sid()`` cote
    Coucou, sinon le nom du pipe ne tombe pas juste. On interroge donc le
    jeton de processus via l'API Win32, sans dependance externe.
    """
    if not _sur_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        # argtypes/restypes EXPLICITES obligatoires : sans eux, ctypes
        # tronque les pointeurs 64 bits en int 32, et ConvertSidToStringSidW
        # echoue. (Bug observe : repli silencieux sur USERNAME -> mauvais pipe.)
        # typeshed n'expose `WinDLL` que sous `sys.platform == "win32"` : mypy
        # tourne sous Linux en CI et le declare donc absent. L'attribut EXISTE
        # a l'execution ici (nous sommes deja dans la branche `_sur_windows()`),
        # mais mypy ne sait pas le deduire de `os.name`. Un seul ignore, sur
        # l'alias, plutot qu'un par appel.
        WinDLL = ctypes.WinDLL  # type: ignore[attr-defined]
        advapi32 = WinDLL("advapi32", use_last_error=True)
        kernel32 = WinDLL("kernel32", use_last_error=True)
        TOKEN_QUERY = 0x0008
        TokenUser = 1

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p

        jeton = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
                kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(jeton)):
            return None
        try:
            # 1er appel : taille du tampon (echoue volontairement, code 122).
            taille = wintypes.DWORD(0)
            advapi32.GetTokenInformation(
                jeton, TokenUser, None, 0, ctypes.byref(taille))
            if taille.value == 0:
                return None
            tampon = ctypes.create_string_buffer(taille.value)
            if not advapi32.GetTokenInformation(
                    jeton, TokenUser, tampon, taille.value, ctypes.byref(taille)):
                return None
            # TOKEN_USER commence par un SID_AND_ATTRIBUTES : le SID est le
            # premier pointeur de la structure.
            sid = ctypes.cast(tampon, ctypes.POINTER(ctypes.c_void_p))[0]
            chaine = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(chaine)):
                return None
            try:
                return chaine.value
            finally:
                kernel32.LocalFree(ctypes.cast(chaine, ctypes.c_void_p))
        finally:
            kernel32.CloseHandle(jeton)
    except Exception:                     # jamais bloquant : repli USERNAME
        return None


def _cle_utilisateur() -> str:
    """Cle du pipe : SID si possible, sinon le compte (comme Coucou)."""
    return _sid_utilisateur() or os.environ.get("USERNAME") or "user"


def _nom_pipe_pour(cle: str) -> str:
    return "\\\\.\\pipe\\coucou-" + (cle or "user")


def nom_pipe() -> str:
    """Chemin du pipe du relais Coucou pour l'utilisateur courant."""
    return _nom_pipe_pour(_cle_utilisateur())


# ── Construction du message ────────────────────────────────────────────

def _agent_valide(agent: str | None) -> str | None:
    """Meme regle que l'ile : minuscules/chiffres/tirets, <= 24, jamais
    'claude' (reserve a Claude Code)."""
    if not agent or len(agent) > 24 or agent == "claude":
        return None
    if not all(c.islower() or c.isdigit() or c == "-" for c in agent):
        return None
    return agent


def _payload(message: str | None, evenement: str = "notification",
             agent: str | None = "aura") -> dict:
    """Objet JSON a ecrire sur le pipe. Leve ValueError si l'evenement est
    inconnu : mieux vaut une erreur claire qu'un evenement silencieusement
    ignore par l'ile."""
    evenement = (evenement or "notification").strip().lower()
    nom_hook = _MAP.get(evenement)
    if nom_hook is None:
        raise ValueError(
            f"evenement inconnu : {evenement!r} (attends {', '.join(EVENEMENTS)})")
    corps: dict = {"hook_event_name": nom_hook}
    valide = _agent_valide(agent)
    if valide:
        corps["coucou_agent"] = valide
    texte = (message or "").strip()
    if texte:
        corps["message"] = texte
    return corps


# ── Ecriture sur le pipe ───────────────────────────────────────────────

def _envoyer(nom: str, ligne: str, timeout: float) -> tuple[bool, str | None]:
    """Ouvre le pipe et ecrit la ligne, avec un delai maximum.

    L'ouverture d'un named pipe Windows *bloque* tant qu'aucune instance
    serveur n'est libre : on la fait donc dans un fil demon, pour ne jamais
    figer l'appelant (GUI, API, CLI)."""
    resultat: dict = {"ok": False, "erreur": None}

    def _run() -> None:
        try:
            # newline="" : on ecrit notre propre '\n', sans qu'il devienne
            # '\r\n' (le relais coupe la ligne sur b'\n').
            with open(nom, "w", encoding="utf-8", newline="") as f:
                f.write(ligne)
            resultat["ok"] = True
        except OSError as e:
            resultat["erreur"] = e.strerror or str(e)

    fil = threading.Thread(target=_run, daemon=True)
    fil.start()
    fil.join(timeout)
    if fil.is_alive():
        return False, "delai depasse (pipe occupe)"
    return bool(resultat["ok"]), resultat["erreur"]


def _vider(resultat: dict, nom: str, evenement: str, raison: str) -> dict:
    resultat.setdefault("pipe", nom)
    resultat.setdefault("evenement", evenement)
    resultat["raison"] = raison
    return resultat


def notifier(message: str | None, *, evenement: str = "notification",
             agent: str | None = "aura",
             timeout: float = _TIMEOUT_DEFAUT) -> dict:
    """Pousse un evenement vers l'ile Coucou.

    Renvoie toujours un dict : ``{"envoye": bool, "evenement": ..., ...}``.
    Ne leve jamais, sauf ``ValueError`` pour un evenement mal nomme (erreur
    de programmation, pas panne d'execution).
    """
    if os.environ.get(_ENV) == "0":
        return {"envoye": False, "raison": "pont desactive (AURA_COUCOU=0)"}

    corps = _payload(message, evenement, agent)     # peut lever ValueError
    if not _sur_windows():
        return _vider({"envoye": False}, nom_pipe(),
                      corps["hook_event_name"], "pipe Coucou : Windows uniquement")

    nom = nom_pipe()
    ligne = json.dumps(corps, ensure_ascii=False) + "\n"
    ok, erreur = _envoyer(nom, ligne, timeout)
    if ok:
        return _vider({"envoye": True}, nom, corps["hook_event_name"], "envoye")
    return _vider({"envoye": False}, nom, corps["hook_event_name"],
                  erreur or "Coucou injoignable (pipe absent)")


def est_disponible() -> bool:
    """Vrai si le relais Coucou expose son pipe (non bloquant)."""
    if not _sur_windows() or os.environ.get(_ENV) == "0":
        return False
    try:
        return os.path.basename(nom_pipe()) in os.listdir("\\\\.\\pipe\\")
    except OSError:
        return False


# ── Raccourcis lisibles ────────────────────────────────────────────────

def activite(message: str | None = None, *, agent: str | None = "aura") -> dict:
    return notifier(message, evenement="activite", agent=agent)


def termine(message: str | None = None, *, agent: str | None = "aura") -> dict:
    return notifier(message, evenement="termine", agent=agent)


def erreur(message: str | None = None, *, agent: str | None = "aura") -> dict:
    return notifier(message, evenement="erreur", agent=agent)


# ── Ligne de commande : `python -m aura.coucou "message"` ───────────────
#
# Pensé pour etre appele par N'IMPORTE QUEL outil qui sait lancer une
# commande : un agent d'IDE (Antigravity...), un script de build, un hook.
# C'est ce qui relie un agent a l'ile Mochi sans integration dediee.

def principal(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="aura.coucou",
        description="Pousse un evenement vers l'ile Coucou (Mochi).")
    ap.add_argument("message", nargs="?", default="",
                    help="texte a afficher dans l'ile")
    ap.add_argument("--evenement", default="notification", choices=EVENEMENTS,
                    help="notification (defaut), activite, termine, erreur")
    ap.add_argument("--agent", default="aura",
                    help="nom de la pastille (minuscules/chiffres/tirets)")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    args = ap.parse_args(argv)

    if not args.message.strip():
        ap.error("un message est requis")
    try:
        resultat = notifier(args.message, evenement=args.evenement, agent=args.agent)
    except ValueError as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(resultat, ensure_ascii=False))
    else:
        etat = "envoye" if resultat.get("envoye") else "non envoye"
        print(f"{etat} : {resultat.get('raison')}")
    return 0 if resultat.get("envoye") else 1


if __name__ == "__main__":
    raise SystemExit(principal())
