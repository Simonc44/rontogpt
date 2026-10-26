"""Serveur OpenAI local dédié (atelier) pour RontoGPT.

Ce module lance un serveur HTTP léger compatible OpenAI sur un port dédié
(par défaut 8761) et expose des fonctions d'état, de démarrage et d'arrêt
utilisées par l'Aura API (`/v1/api/*`).

Le but : offrir un point d'entrée "OpenAI" séparé de l'Aura principal,
même clé d'authentification, même machine, mais processus à part.

Usage (depuis la ligne de commande) :
    python -m aura.openai_atelier           # lance le serveur
    python -m aura.openai_atelier --smoke   # auto-test puis exit 0 (CI)
"""
from __future__ import annotations

import argparse
import hmac
import os
import sys
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ------------------------------------------------------------
# Configuration par défaut
# ------------------------------------------------------------
DEFAULT_PORT = 8761          # port dédié, distinct de l'Aura (8760)
DEFAULT_HOST = "127.0.0.1"   # bind en local seulement
API_KEY_ENV = "AURA_API_CLE"  # same cle que l'Aura principale
MAX_TOKENS = 4096            # contexte fenetre par defaut


# ------------------------------------------------------------
# Etat partagé (thread-safe)
# ------------------------------------------------------------
class AtelierEtat:
    """Etat global du atelier OpenAI (unique par processus)."""

    def __init__(self) -> None:
        self.actif: bool = False
        self.serveur_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        self.version: str = "unknown"
        self.server_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def basculer(self) -> dict:
        """Retourne l'etat bascule (on <-> off) et renvoie le dict nouveau etat."""
        with self.lock:
            self.actif = not self.actif
            return {
                "actif": self.actif,
                "serveur": self.serveur_url,
                "version": self.version,
            }


etat = AtelierEtat()


# ------------------------------------------------------------
# FastAPI / uvicorn bootstrap
# ------------------------------------------------------------
def _refus_cle(entetes) -> Optional[JSONResponse]:
    """401 si une cle est configuree et que la requete ne la porte pas.

    La cle etait acceptee par `--cle` mais JAMAIS verifiee : n'importe qui
    joignant le port parlait au cerveau. Meme comparaison en temps constant
    que `aura.api`.
    """
    attendue = os.environ.get(API_KEY_ENV, "").strip()
    if not attendue:
        return None
    fournie = str(entetes.get("x-rontogpt-cle")
                  or entetes.get("x-aura-cle") or "")
    if hmac.compare_digest(fournie, attendue):
        return None
    return JSONResponse({"ok": False, "erreur": "cle invalide"},
                        status_code=401)


def create_app() -> FastAPI:
    """Retourne l'instance FastAPI avec les routes /v1/*."""

    app = FastAPI(title="RontoGPT Atelier OpenAI")

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "etat": "sain"})

    @app.get("/status")
    def status(request: Request) -> JSONResponse:
        refus = _refus_cle(request.headers)
        if refus is not None:
            return refus
        with etat.lock:
            return JSONResponse(
                {
                    "ok": True,
                    "actif": etat.actif,
                    "serveur": etat.serveur_url,
                    "version": etat.version,
                }
            )

    @app.post("/v1/ask")
    async def ask(request: Request) -> JSONResponse:
        refus = _refus_cle(request.headers)
        if refus is not None:
            return refus
        # `Request.json()` est une COROUTINE (Starlette) : sans `await` elle
        # rend un objet coroutine, et `.get()` sur cet objet leve
        # AttributeError -> 500 sur chaque appel. Le handler doit donc etre
        # `async` lui aussi.
        corps = await request.json()
        question = corps.get("question", "").strip()
        if not question:
            return JSONResponse({"ok": False, "erreur": "champ 'question' requis"}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "reponse": "(réponse de l'atelier — delegation vers le cerveau local non implementee ici)",
                "experts": [],
                "cerveau": "atelier",
                "latence": 0.0,
            }
        )

    @app.post("/v1/ask/stream")
    async def ask_stream(request: Request) -> JSONResponse:  # simplifie pour l'exemple
        refus = _refus_cle(request.headers)
        if refus is not None:
            return refus
        corps = await request.json()
        question = corps.get("question", "").strip()
        if not question:
            return JSONResponse({"ok": False, "erreur": "champ 'question' requis"}, status_code=400)

        def generator():
            yield {"type": "final", "reponse": "(stream non implemente)", "experts": [], "cerveau": "atelier", "latence": 0.0}
            yield {"type": "fin"}

        return JSONResponse({"ok": True, "stream": list(generator())})

    return app


# ------------------------------------------------------------
# Lancement du serveur
# ------------------------------------------------------------
def demarrer_serveur(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST, cle: Optional[str] = None) -> None:
    """Démarre le serveur uvicorn en thread autonome.

    Refuse de demarrer hors loopback sans cle : meme regle que `aura.api`
    (`aura.api._verifier_exposition`), importe ICI et non en tete de module
    parce que `aura.api` applique la configuration des son import.
    """
    if cle:
        os.environ[API_KEY_ENV] = cle      # sinon `--cle` ne serait pas verifiee
    from .api import _verifier_exposition

    souci = _verifier_exposition(host)
    if souci:
        raise ValueError(souci)
    app = create_app()
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def run():
        server.run()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    etat.server_thread = thread
    etat.actif = True
    etat.serveur_url = f"http://{host}:{port}"

    print(f"[atelier] Serveur OpenAI local demarre sur {etat.serveur_url}")
    print("[atelier] Presse la touche 'q' ou Ctrl+C pour arreter.")


def arreter_serveur() -> dict:
    """Demande l'arret propre du serveur et renvoie l'etat final."""
    etat.actif = False
    return etat.basculer()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Serveur OpenAI local pour RontoGPT")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port du serveur (defaut {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Host d'ecoute (defaut {DEFAULT_HOST})")
    parser.add_argument("--cle", type=str, default=os.environ.get(API_KEY_ENV), help="Cle API AURA (optionnelle, lu aussi de AURA_API_CLE)")
    parser.add_argument("--no-preload", action="store_true", help="Ne pas precharger le cerveau au demarrage")
    parser.add_argument("--smoke", action="store_true", help="Mode auto-test : verifie que le module se charge")
    args = parser.parse_args(argv)

    # Si --smoke, on teste juste que le module se charge et que les routes existent
    if args.smoke:
        import importlib
        mod = importlib.import_module("aura.openai_atelier")
        assert hasattr(mod, "create_app"), "create_app manquant"
        assert hasattr(mod, "demarrer_serveur"), "demarrer_serveur manquant"
        assert hasattr(mod, "arreter_serveur"), "arreter_serveur manquant"
        print("[smoke] Module aura.openai_atelier charge OK avec create_app, demarrer_serveur, arreter_serveur")
        return

    try:
        demarrer_serveur(port=args.port, host=args.host, cle=args.cle)
    except ValueError as e:
        print(f"[atelier] REFUS : {e}", file=sys.stderr)
        raise SystemExit(2) from e

    # Boucle principale : attend le signal d'arret (Ctrl+C)
    try:
        while etat.actif:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[atelier] Arret demande par l'utilisateur.")
    finally:
        arreter_serveur()


if __name__ == "__main__":
    main()