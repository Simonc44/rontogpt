"""API locale HTTP : le coeur RontoGPT expose /v1/* pour le TUI, le vocal, les scripts.

Le but de l'industrialisation : UN seul process porte le cerveau, et tous les
front-ends (tui, RontoGPT-Vocal, scripts, tierces) lui parlent par JSON au
lieu de charger chacun leur propre Llama. Standard decompat : stdlib only
(aucune dependance lourde de plus), boucle d'evenements served en threads.

Demarrage :
    python -m aura.api               # port AURA_API_PORT (8760), preload du cerveau
    python -m aura.api --no-preload  # cerveau charge a la 1re question
    python -m aura.api --smoke       # auto-test puis exit 0 (CI / verification)

Routes :
    GET  /v1/health  -> {"status": "ok", "version": "..."}
    GET  /v1/status  -> etat du coeur (prete / chargement / erreur / serveur / gpu)
    GET  /v1/config  -> configuration effective (AURA_*)
    GET  /v1/cloud   -> etat de Gemini (cle, modele, mode de repli, redaction)
    GET  /v1/coucou  -> le pont vers l'ile Coucou est-il disponible ?
    GET  /v1/memoire -> ce que le cerveau garde en tete (fenetre + etat civil)
    POST /v1/ask     {"question": "..."} -> {ok, reponse, experts, cerveau, latence,
                     en_attente, clarification}
    POST /v1/ask/stream {"question": "..."} -> SSE : data {"type":"morceau",
                     texte, phase} ou phase = "pensee" | "reponse" (la
                     reflexion ET la reponse sont poussees, chacune avec sa
                     phase : le client les affiche comme il veut)
                     puis data {"type":"final", reponse, experts, latence}
                     puis data [DONE]

`en_attente` = la reponse n'est PAS une reponse : la porte de clarification
(aura/clarification.py) a juge la question trop ouverte et demande une
precision. Le tour SUIVANT porte la precision ; le client n'a rien a faire
de special (c'est le meme couple question/reponse), mais il peut afficher
un etat distinct (« question en attente ») et `clarification` en donne la
raison (`famille`, `manque`, `questions`).
    POST /v1/notify  {"message": "...", "evenement": "termine"} -> {ok, envoye}
                     (pousse un evenement vers l'ile Coucou / Mochi)
    POST /v1/learn   {"sujet","relation","objet"} -> edite un FAIT verifie
                     ou {"question","correcte"} -> enregistre une correction
                     (edition de connaissance SANS reentrainement)
    POST /v1/reset   -> oublie la conversation en cours {ok, oublie, raison}
                     (pour le TUI : le `reset` du CLI, sans relancer le processus)
    POST /v1/compact -> vide la fenetre brute, conserve l'etat civil {ok, compacte}
    POST /v1/init    -> charge le cerveau maintenant (prechauffage) {ok, prete}
    POST /v1/cloud   {cle, actif, mode, modele, redaction} -> applique les
                     reglages Gemini au processus et rend le nouvel etat

Securite : bind 127.0.0.1 par defaut (jamais expose sur le reseau). Si
AURA_API_CLE est posee, chaque appel doit fournir l'entete X-RontoGPT-Cle.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, cast

from . import config as _config

if TYPE_CHECKING:
    # Le cerveau n'est importe qu'au PREMIER appel (`charger`), jamais a
    # l'import du serveur : la declaration sert au type de `_Etat.ia` et ne
    # coute rien au runtime (`from __future__ import annotations` ci-dessus).
    from .orchestrateur import RontoGPT

_config.appliquer()

_LOG_COURT = 60_000          # corps de requete acceptes (60 Ko)


def build() -> str:
    """Empreinte du CODE charge dans ce processus (noms, tailles, dates).

    Sert a detecter une API ANCIENNE restee ouverte sur le port : le TUI se
    connectait a elle en silence et on testait des correctifs qui n'etaient
    pas charges (symptome : le code partait toujours sur le web).
    """
    import hashlib
    from pathlib import Path
    h = hashlib.sha1()
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        try:
            s = f.stat()
        except OSError:
            continue
        h.update(f"{f.name}:{s.st_size}:{s.st_mtime_ns}".encode())
    return h.hexdigest()[:10]


_BUILD_DEMARRAGE = build()   # figee au demarrage : ce que CE processus execute

# Le flux de tokens est GLOBAL (un seul cerveau) : une seule requete streamee
# a la fois, sinon les morceaux de deux questions se melangeraient.
_FLUX_LIBRE = threading.Lock()
_VERROU_FEEDBACK = threading.Lock()
_FICHIER_FEEDBACK = os.path.join(os.path.dirname(__file__), "feedback.jsonl")


#: routes publiques, telles que la commande `/api` du TUI les affiche
_ROUTES_PUBLIQUES = (
    "GET  /v1/health      sante, version, build",
    "GET  /v1/status      cerveau pret ou non, GPU, budget",
    "GET  /v1/memoire     fenetre de conversation + jauge de contexte",
    "GET  /v1/documents   documents locaux indexes",
    "GET  /v1/langue      langue de reponse active",
    "GET  /v1/cloud       etat de Gemini",
    "POST /v1/ask         question -> reponse + preuve",
    "POST /v1/ask/stream  meme chose, en flux (pensee puis reponse)",
    "POST /v1/ask/cloud   envoi explicite a Gemini (confirme: true)",
    "POST /v1/reset       oublier la conversation",
    "POST /v1/compact     vider la fenetre brute, garder l'etat civil",
    "POST /v1/memoire     editer un fait memorise (set/delete)",
    "POST /v1/documents   ajouter ou retirer un document local",
    "POST /v1/langue      choisir fr, en ou auto",
    "POST /v1/chat/completions  compatibilite OpenAI locale",
)


def _etat_atelier() -> dict:
    """Etat de l'atelier OpenAI, sans jamais lever (module optionnel)."""
    try:
        from . import openai_atelier as oa
        return {"disponible": True, "actif": bool(oa.etat.actif),
                "serveur": oa.etat.serveur_url}
    except Exception as e:                           # noqa: BLE001
        return {"disponible": False, "erreur": str(e)}


def _reponse_detaille(ia, question: str) -> dict:
    """Le resultat complet d'une question, au format de /v1/ask."""
    t0 = time.time()
    r = ia.executer_detaille(question)
    return {
        "ok": True,
        "reponse": str(r.get("reponse", "")),
        "experts": sorted(r.get("experts") or set()),
        "cerveau": str(r.get("cerveau_choisi", "-")),
        "latence": round(time.time() - t0, 2),
        # PORTE DE CLARIFICATION : la reponse est une QUESTION de relance.
        # Un client qui l'ignore fonctionne (c'est du texte), mais un client
        # qui la lit peut marquer l'etat au lieu de presenter une relance
        # comme une reponse.
        "en_attente": bool(r.get("en_attente")),
        "clarification": r.get("clarification"),
        # PREUVE : comment la reponse a ete obtenue (aura/provenance.py) —
        # {niveau, badge, detail, verifie, quitte_la_machine}.
        "provenance": r.get("provenance"),
        # SEMI-CLOUD : le modele local est juge insuffisant ; rien n'est parti.
        # Si ce champ est present, le client peut proposer « Demander a
        # Gemini » (POST /v1/ask/cloud, avec confirme=true).
        "repli_propose": r.get("repli_propose"),
        "repli_echec": r.get("repli_echec"),
        # JAUGE DE CONTEXTE : saturation et compaction du tour, pour que le
        # client puisse DIRE « contexte plein » au lieu de le taire.
        "contexte": getattr(ia, "_dernier_contexte", None),
    }


class _Etat:
    """Coeur partage : IA lazy (un seul chargement), thread-safe."""

    def __init__(self, smoke: bool = False):
        self.smoke = smoke
        # ANNOTE : sans le type, mypy deduit `None` de l'initialisation et
        # refuse `self.ia = RontoGPT()` plus bas.
        self.ia: RontoGPT | None = None
        self.erreur: str | None = None
        self.chargement = False
        self.verrou = threading.Lock()

    def charger(self):
        """Charge RontoGPT une seule fois (le premier appel attend)."""
        if self.smoke:
            self.erreur = "mode smoke : cerveau non charge"
            return None
        with self.verrou:
            if self.ia is not None or self.erreur is not None:
                return self.ia
            self.chargement = True
            try:
                from .orchestrateur import RontoGPT
                self.ia = RontoGPT()
                # Prechauffage MiniLM : sa session ONNX (~1 s) est payee ici,
                # hors de la premiere question de l'utilisateur.
                from . import embeddings
                embeddings.encoder(["prechauffage"])
            except Exception as e:               # jamais de 500 silencieux
                self.erreur = str(e)
            finally:
                self.chargement = False
            return self.ia


def _verifier_cle(entetes) -> bool:
    attendue = os.environ.get("AURA_API_CLE")
    return (not attendue or entetes.get("X-RontoGPT-Cle") == attendue
            or entetes.get("X-Aura-Cle") == attendue)


class _Handler(BaseHTTPRequestHandler):
    etat: _Etat = _Etat()          # remplace par _creer_handler()

    # -- utilitaires --------------------------------------------------
    def log_message(self, format, *args):        # journalisation calmee
        pass

    def _json(self, donnees, code: int = 200) -> None:
        corps = json.dumps(donnees, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _corps(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > _LOG_COURT:
                return {}
            return json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            return {}

    # -- GET ----------------------------------------------------------
    def do_GET(self):                                # noqa: N802 (stdlib)
        if not _verifier_cle(self.headers):
            return self._json({"ok": False, "erreur": "cle invalide"}, 401)
        if self.path == "/v1/api":
            return self._api_status()
        if self.path == "/v1/api/disable":
            return self._api_desactiver()
        if self.path == "/v1/api/deploy":
            return self._api_deployer()
        if self.path == "/v1/health":
            return self._json({"status": "ok",
                               "version": _config.version(),
                               "build": _BUILD_DEMARRAGE})
        if self.path == "/v1/status":
            from . import serveur_lora
            e = self.etat
            return self._json({
                "version": _config.version(),
                "build": _BUILD_DEMARRAGE,
                "profil": os.environ.get("AURA_PROFIL", "equilibre"),
                "prete": e.ia is not None,
                "chargement": e.chargement,
                "erreur": e.erreur,
                "serveur_llama": serveur_lora._actifs(),
                "gpu": os.environ.get("AURA_GPU", "auto"),
                "budget_s": os.environ.get("AURA_BUDGET_S", "45"),
            })
        if self.path == "/v1/config":
            return self._json({"version": _config.version(),
                               "environnement": _config.efficace()})
        if self.path == "/v1/cloud":
            from . import gemini
            return self._json(gemini.etat())
        if self.path == "/v1/memoire":
            return self._memoire()
        if self.path == "/v1/documents":
            return self._documents()
        if self.path == "/v1/langue":
            return self._langue()
        if self.path == "/v1/coucou":
            from . import coucou
            return self._json({
                "ok": True,
                "disponible": coucou.est_disponible(),
                "evenements": list(coucou.EVENEMENTS),
            })
        return self._json({"ok": False, "erreur": "route inconnue"}, 404)

    # -- POST ---------------------------------------------------------
    def do_POST(self):                               # noqa: N802 (stdlib)
        if not _verifier_cle(self.headers):
            return self._json({"ok": False, "erreur": "cle invalide"}, 401)
        if self.path == "/v1/notify":
            return self._notify()
        if self.path == "/v1/learn":
            return self._learn()
        if self.path == "/v1/reset":
            return self._reset()
        if self.path == "/v1/compact":
            return self._compact()
        if self.path == "/v1/cloud":
            return self._cloud()
        if self.path == "/v1/memoire":
            return self._editer_memoire()
        if self.path == "/v1/documents":
            return self._editer_documents()
        if self.path == "/v1/langue":
            return self._regler_langue()
        if self.path == "/v1/chat/completions":
            return self._chat_completions()
        if self.path == "/v1/init":
            return self._init()
        if self.path == "/v1/ask/stream":
            return self._ask_stream()
        if self.path == "/v1/feedback":
            return self._feedback()
        if self.path == "/v1/ask/cloud":
            return self._ask_cloud()
        if self.path != "/v1/ask":
            return self._json({"ok": False, "erreur": "route inconnue"}, 404)
        return self._ask()

    # -- OpenAI local (/v1/api/*) : compromis du pc ------------------------
    def _api_status(self) -> None:
        """Etat de l'API locale (`/api` du TUI) : adresse, version, cle, routes.

        Toujours REELLE : elle decrit CE serveur (celui qui repond), pas un
        serveur annexe. L'etat de l'atelier OpenAI (`aura/openai_atelier.py`)
        est ajoute a part, et son absence ne fait jamais echouer la route.
        """
        # `socketserver` type `server_address` en `tuple | str | Buffer` : ce
        # serveur est TOUJOURS un socket AF_INET, donc toujours un couple
        # (hote, port). Sans ce `cast`, mypy refuse l'indexation — et il a
        # raison de le faire pour le type declare, pas pour ce serveur.
        hote, port = cast(tuple[str, int], self.server.server_address)
        return self._json({
            "ok": True,
            "actif": True,
            "serveur": f"http://{hote}:{port}",
            "version": _config.version(),
            "build": _BUILD_DEMARRAGE,
            "cle_requise": bool(os.environ.get("AURA_API_CLE")),
            "routes": list(_ROUTES_PUBLIQUES),
            "atelier": _etat_atelier(),
        })

    def _api_desactiver(self) -> None:
        """Arrete l'atelier OpenAI s'il tourne ; sinon le dit sans planter."""
        try:
            from . import openai_atelier as oa
            oa.arreter_serveur()
        except Exception as e:                       # noqa: BLE001
            return self._json({"ok": False, "erreur": str(e)}, 503)
        return self._api_status()

    def _api_deployer(self) -> None:
        """Refuse de lancer l'atelier tant que ses routes sont factices.

        `openai_atelier.create_app` repond par un TEXTE DE REMPLISSAGE (« reponse
        de l'atelier — delegation non implementee ») : le presenter comme un
        serveur RontoGPT ferait croire a une vraie reponse.
        """
        return self._json(
            {"ok": False,
             "erreur": "atelier OpenAI non finalise : ses routes renvoient un "
                       "texte factice, pas la reponse du cerveau"}, 501)

    def _feedback(self) -> None:
        """Vote utilisateur (pouce haut/bas) : consigne seulement, sans effet
        sur le cache ni le routage (a exploiter plus tard, apres mesure)."""
        corps = self._corps()
        vote = str(corps.get("vote") or "")
        if vote not in ("up", "down"):
            return self._json({"ok": False,
                               "erreur": "vote attendu : 'up' ou 'down'"}, 400)
        entree = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "vote": vote,
                  "question": str(corps.get("question") or "")[:500],
                  "reponse": str(corps.get("reponse") or "")[:2000]}
        with _VERROU_FEEDBACK:
            with open(_FICHIER_FEEDBACK, "a", encoding="utf-8") as f:
                f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        return self._json({"ok": True})

    def _cloud(self) -> None:
        """Reglages Gemini demandes par le client (`/cloud` du TUI).

        La cle n'est jamais renvoyee en clair : l'etat ne dit que si elle est
        presente et d'ou elle vient. Un echec de validation ne modifie rien
        et rend 400.
        """
        from . import gemini
        corps = self._corps()

        def _drapeau(nom: str) -> bool:
            valeur = corps.get(nom)
            if isinstance(valeur, bool):
                return valeur
            return str(valeur).strip().lower() in (
                "1", "true", "oui", "on", "yes")

        try:
            etat = gemini.regler(
                cle=(str(corps["cle"]) if "cle" in corps else None),
                escalade=(_drapeau("actif") if "actif" in corps else None),
                mode=(str(corps["mode"]) if "mode" in corps else None),
                modele=(str(corps["modele"]) if "modele" in corps else None),
                redaction=(_drapeau("redaction")
                           if "redaction" in corps else None),
            )
        except ValueError as e:
            return self._json({"ok": False, "erreur": str(e)}, 400)
        except OSError as e:
            return self._json({"ok": False,
                               "erreur": f"ecriture de .env impossible : {e}"},
                              500)
        return self._json(etat)

    def _ask(self) -> None:
        question = str(self._corps().get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        try:
            return self._json(_reponse_detaille(ia, question))
        except Exception as e:                       # l'IA vit encore
            return self._json({"ok": False, "erreur": str(e)}, 500)

    def _langue(self) -> None:
        ia = self.etat.ia
        return self._json({"ok": True,
                           "langue": getattr(ia, "_langue_reponse", "auto")})

    def _regler_langue(self) -> None:
        langue = str(self._corps().get("langue") or "").strip().lower()
        if langue not in ("auto", "fr", "en"):
            return self._json({"ok": False, "erreur": "langue attendue : fr, en ou auto"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False, "erreur": self.etat.erreur or "IA non chargee"}, 503)
        ia._langue_reponse = langue
        return self._json({"ok": True, "langue": langue})

    def _chat_completions(self) -> None:
        """Adaptateur OpenAI minimal, reel et sans FastAPI.

        Il est volontairement non-streaming : les clients OpenAI usuels
        (editeurs, Open WebUI) peuvent deja converser avec le cerveau unique,
        sans lancer l'ancien atelier factice.
        """
        corps = self._corps()
        messages = corps.get("messages")
        if not isinstance(messages, list):
            return self._json({"error": {"message": "messages requis"}}, 400)
        question = next((str(m.get("content") or "").strip()
                         for m in reversed(messages) if isinstance(m, dict)
                         and m.get("role") == "user"), "")
        if not question:
            return self._json({"error": {"message": "message utilisateur requis"}}, 400)
        if corps.get("stream") is True:
            return self._json({"error": {"message": "stream non pris en charge ; utilisez /v1/ask/stream"}}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"error": {"message": self.etat.erreur or "IA non chargee"}}, 503)
        try:
            r = _reponse_detaille(ia, question)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": {"message": str(e)}}, 500)
        return self._json({
            "id": f"chatcmpl-ronto-{int(time.time() * 1000)}",
            "object": "chat.completion", "created": int(time.time()),
            "model": str(corps.get("model") or "rontogpt-local"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": r["reponse"]}}],
            "rontogpt": {"provenance": r.get("provenance"), "latence": r["latence"]},
        })

    def _ask_cloud(self) -> None:
        """Envoi EXPLICITE d'une question a Gemini : le client l'a demande.

        `confirme` doit valoir true : c'est la trace que l'utilisateur a vu
        l'avertissement « cette question va quitter votre ordinateur ». Sans
        lui, rien ne part.
        """
        corps = self._corps()
        question = str(corps.get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        if corps.get("confirme") is not True:
            return self._json(
                {"ok": False,
                 "erreur": "confirmation requise (confirme=true) : la question "
                           "va quitter l'ordinateur"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        t0 = time.time()
        try:
            r = ia.envoyer_a_gemini(question)
        except Exception as e:                       # noqa: BLE001
            return self._json({"ok": False, "erreur": str(e)}, 500)
        r["latence"] = round(time.time() - t0, 2)
        return self._json(r, 200 if r.get("ok") else 502)

    # -- streaming SSE : les tokens partent au fil de l'eau -----------
    def _sse(self, donnees: dict) -> None:
        """Un evenement Server-Sent Events ; un client parti n'est pas une erreur."""
        try:
            charge = json.dumps(donnees, ensure_ascii=False)
            self.wfile.write(f"data: {charge}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _ask_stream(self) -> None:
        corps = self._corps()
        question = str(corps.get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        if not _FLUX_LIBRE.acquire(blocking=False):
            return self._json(
                {"ok": False, "erreur": "un flux est deja en cours — reessaie"},
                409)

        from . import llama_cerveau
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Le generateur publie chaque morceau depuis CE thread : l'abonne
        # ecrit donc directement dans la reponse, sans file intermediaire.
        # LES DEUX PHASES, et chacune ANNONCEE : c'est la demande produit
        # « on voit la pensee ET la reponse ». Le client recoit son `phase`
        # et peut les afficher differemment ; un client qui ignore le champ
        # se contente de concatener, comme avant.
        desabonner = llama_cerveau.abonner_flux_phase(
            lambda texte, phase: self._sse({"type": "morceau",
                                            "texte": texte, "phase": phase}))
        # ETAPES (« Recherche sur le web… ») : canal a part, jamais melange
        # aux morceaux de la reponse.
        desabonner_etapes = llama_cerveau.abonner_flux(
            lambda libelle: self._sse({"type": "etape", "libelle": libelle}),
            phases=(llama_cerveau.PHASE_ETAPE,))
        try:
            resultat = _reponse_detaille(ia, question)
            self._sse({"type": "final", **resultat})
        except Exception as e:                       # noqa: BLE001
            self._sse({"type": "erreur", "erreur": str(e)})
        finally:
            desabonner()
            desabonner_etapes()
            _FLUX_LIBRE.release()
            self._sse({"type": "fin"})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    # -- remise a zero de la conversation (/v1/reset) ----------------
    def _reset(self) -> None:
        """Oublie la conversation en cours (equivalent du `reset` en CLI).

        Ajoute pour le TUI (tui/) : sans cette route, la seule facon d'oublier
        la conversation etait de relancer le processus.
        """
        ia = self.etat.ia
        if ia is None:
            return self._json({"ok": True, "oublie": False,
                               "raison": "aucune conversation en memoire"})
        try:
            ia.reinitialiser_conversation()
        except Exception as e:                       # noqa: BLE001
            return self._json({"ok": False, "erreur": str(e)}, 500)
        return self._json({"ok": True, "oublie": True,
                           "raison": "conversation oubliee"})

    # -- memoire de conversation (/v1/memoire, /v1/compact) -----------
    def _memoire(self) -> None:
        """Ce que le cerveau garde en tete : fenetre brute + etat civil.

        Volontairement en LECTURE SEULE et borne : le TUI affiche un resume,
        pas l'historique complet (qui vit dans le fichier RLM).
        """
        ia = self.etat.ia
        historique = getattr(ia, "_historique", None) if ia is not None else None
        if historique is None:
            return self._json({"ok": True, "messages": 0, "echanges": 0,
                               "etat": {}, "fenetre": [], "contexte": None})
        from . import contexte
        tours = historique.fenetre()
        return self._json({
            "ok": True,
            "messages": len(tours),
            "echanges": (len(tours) + 1) // 2,
            "etat": historique.etat(),
            # Remplissage AU REPOS (fenetre + systeme, sans question ni web) :
            # une borne basse, suffisante pour voir la saturation arriver.
            "contexte": contexte.etat(historique),
            "fenetre": [{"role": str(t.get("role", "")),
                         "contenu": str(t.get("content", ""))[:200]}
                        for t in tours],
        })

    def _editer_memoire(self) -> None:
        corps = self._corps()
        ia = self.etat.ia
        historique = getattr(ia, "_historique", None) if ia is not None else None
        if historique is None:
            return self._json({"ok": False, "erreur": "initialisez le cerveau avant d'editer la memoire"}, 409)
        cle = str(corps.get("cle") or "").strip()
        if corps.get("supprimer") is True:
            if not cle:
                return self._json({"ok": False, "erreur": "cle requise"}, 400)
            return self._json({"ok": True, "supprime": historique.oublier_etat(cle),
                               "etat": historique.etat()})
        try:
            cle, valeur = historique.definir_etat(cle, str(corps.get("valeur") or ""))
        except ValueError as e:
            return self._json({"ok": False, "erreur": str(e)}, 400)
        return self._json({"ok": True, "cle": cle, "valeur": valeur,
                           "etat": historique.etat()})

    def _documents(self) -> None:
        from . import documents
        try:
            return self._json({"ok": True, "fichiers": documents.lister(),
                               "statistiques": documents.statistiques()})
        except (OSError, ValueError, RuntimeError) as e:
            return self._json({"ok": False, "erreur": str(e)}, 500)

    def _editer_documents(self) -> None:
        from . import documents
        corps = self._corps()
        try:
            supprimer = str(corps.get("supprimer") or "").strip()
            if supprimer:
                return self._json({"ok": True, "supprimes": documents.supprimer(supprimer),
                                   "fichiers": documents.lister()})
            chemins = corps.get("ajouter")
            if isinstance(chemins, str):
                chemins = [chemins]
            if not isinstance(chemins, list) or not chemins:
                return self._json({"ok": False, "erreur": "ajouter (liste de chemins) ou supprimer requis"}, 400)
            bilans, erreurs = [], []
            for chemin in chemins:
                try:
                    bilans.append(documents.ajouter(str(chemin)))
                except (OSError, ValueError, RuntimeError) as e:
                    erreurs.append({"chemin": str(chemin), "erreur": str(e)})
            return self._json({"ok": bool(bilans), "ajoutes": len(bilans),
                               "bilans": bilans, "erreurs": erreurs,
                               "fichiers": documents.lister()}, 200 if bilans else 400)
        except (OSError, ValueError, RuntimeError) as e:
            return self._json({"ok": False, "erreur": str(e)}, 500)

    def _compact(self) -> None:
        """Compacte la fenetre : tours bruts jetes, etat civil conserve."""
        ia = self.etat.ia
        historique = getattr(ia, "_historique", None) if ia is not None else None
        if historique is None:
            return self._json({"ok": True, "compacte": 0,
                               "raison": "aucune conversation en memoire"})
        try:
            n = historique.compacter()
        except Exception as e:                       # noqa: BLE001
            return self._json({"ok": False, "erreur": str(e)}, 500)
        return self._json({
            "ok": True,
            "compacte": n,
            "raison": ("fenetre videe, etat utilisateur conserve" if n
                       else "aucune conversation en memoire"),
        })

    def _init(self) -> None:
        """Charge le cerveau maintenant (le TUI evite d'attendre au 1er tour)."""
        if self.etat.ia is not None:
            return self._json({"ok": True, "prete": True, "deja": True})
        ia = self.etat.charger()
        if ia is None:
            return self._json({"ok": False, "prete": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        return self._json({"ok": True, "prete": True, "deja": False})

    # -- edition de connaissance (/v1/learn) --------------------------
    def _learn(self) -> None:
        """Edite la connaissance SANS reentrainement (cf. note MEMIT plus bas)."""
        corps = self._corps()
        sujet = str(corps.get("sujet") or "").strip()
        relation = str(corps.get("relation") or "").strip()
        objet = str(corps.get("objet") or "").strip()
        if sujet and relation and objet:
            from . import graphe_faits
            ajoute = graphe_faits.ajouter(
                sujet, relation, objet, str(corps.get("source") or "api"))
            return self._json({"ok": True, "type": "fait",
                               "ajoute": bool(ajoute)})
        question = str(corps.get("question") or "").strip()
        correcte = str(corps.get("correcte") or "").strip()
        if question and correcte:
            from . import autoamelioration
            autoamelioration.enregistrer_correction(
                question, str(corps.get("mauvaise") or ""), correcte,
                str(corps.get("categorie") or "general"))
            return self._json({"ok": True, "type": "correction",
                               "enregistre": True})
        return self._json(
            {"ok": False,
             "erreur": "faut (sujet, relation, objet) ou (question, correcte)"},
            400)

    # -- pont Coucou (ile Mochi) --------------------------------------
    def _notify(self) -> None:
        corps = self._corps()
        message = str(corps.get("message") or "").strip()
        if not message:
            return self._json({"ok": False,
                               "erreur": "champ 'message' requis"}, 400)
        evenement = str(corps.get("evenement") or "notification")
        agent = str(corps.get("agent") or "aura")
        from . import coucou
        try:
            resultat = coucou.notifier(message, evenement=evenement, agent=agent)
        except ValueError as e:                      # evenement mal nomme
            return self._json({"ok": False, "erreur": str(e)}, 400)
        return self._json({"ok": True, **resultat})


def _creer_handler(etat: _Etat):
    return type("HandlerAvecEtat", (_Handler,), {"etat": etat})


# -- auto-test (CI, verification d'installation) ------------------------
def _smoke(port: int) -> int:
    base = f"http://127.0.0.1:{port}"

    def _get(route: str, attendu: int = 200) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(base + route, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def _post(route: str, corps: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            base + route, data=json.dumps(corps).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    checks: list[tuple[str, bool]] = []
    checks.append(("health", _get("/v1/health")[1].get("status") == "ok"))
    checks.append(("status : non prete", _get("/v1/status")[1].get("prete") is False))
    checks.append(("config", "AURA_PROFIL" in _get("/v1/config")[1]["environnement"]))
    checks.append(("question vide -> 400", _post("/v1/ask", {"question": ""})[0] == 400))
    code, rep = _post("/v1/ask", {"question": "bonjour"})
    checks.append(("smoke sans cerveau -> 503 ok:false",
                   code == 503 and rep.get("ok") is False))
    checks.append(("route inconnue -> 404", _get("/v1/inconnu")[0] == 404))
    ok = all(v for _, v in checks)
    for nom, v in checks:
        print(f"  [{'OK' if v else 'KO'}] {nom}")
    return 0 if ok else 1


def demarrer_en_tache(host: str = "127.0.0.1", port: int | None = None,
                      precharger: bool = True):
    """Demarre l'API dans un fil daemon : rend (serveur, fil, port).

    Utilise par l'interface terminal (`aura/tui.py`) : un SEUL processus
    RontoGPT porte le cerveau ET l'API que le TUI interroge, au lieu de
    lancer un second interpreteur Python qui rechargerait le modele.
    `OSError` si le port est deja pris — un autre RontoGPT tourne alors :
    le TUI s'y connectera, ce n'est pas une erreur.
    """
    if port is None:
        port = int(os.environ.get("AURA_API_PORT", "8760"))
    etat = _Etat()
    serveur = ThreadingHTTPServer((host, port), _creer_handler(etat))
    serveur.daemon_threads = True
    fil = threading.Thread(target=serveur.serve_forever, daemon=True)
    fil.start()
    if precharger:
        threading.Thread(target=etat.charger, daemon=True).start()
    return serveur, fil, port


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aura.api")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("AURA_API_PORT", "8760")))
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 par defaut : ne jamais exposer sur le reseau")
    ap.add_argument("--no-preload", action="store_true",
                    help="ne pas charger le cerveau au demarrage")
    ap.add_argument("--smoke", action="store_true",
                    help="auto-test des routes puis exit 0")
    args = ap.parse_args(argv)

    etat = _Etat(smoke=args.smoke)
    serveur = ThreadingHTTPServer((args.host, args.port), _creer_handler(etat))
    serveur.daemon_threads = True        # un client lent ne bloque pas l'arret
    fil = threading.Thread(target=serveur.serve_forever, daemon=True)
    fil.start()
    print(f"[api] http://{args.host}:{args.port}/v1/health "
          f"(profil {os.environ.get('AURA_PROFIL', 'equilibre')})")

    try:
        if args.smoke:
            code = _smoke(args.port)
            return code
        # rechauffage du serveur Vulkan en tache de fond : la premiere
        # question ne paie pas les 5-15 s de boot llama-server
        try:
            from . import serveur_lora
            serveur_lora.rechauffer()
        except Exception:              # jamais bloquant
            pass
        if not args.no_preload:
            threading.Thread(target=etat.charger, daemon=True).start()
            print("[api] chargement du cerveau en tache de fond...")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[api] arret")
    finally:
        serveur.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
