"""Interface RontoGPT basée sur pywebview : le rendu (fenêtre macOS, bulles de
chat, panneau Aperçu) est fait entièrement en HTML/CSS/JS (voir web/index.html,
repris de ta maquette). Ce fichier Python ne fait plus que :

  - charger le cerveau RontoGPT en tâche de fond (comme avant),
  - exposer une petite API JS <-> Python via pywebview (classe Api),
  - piloter la fenêtre (déplacement, réduction, agrandissement, fermeture),
    puisque la fenêtre est "frameless" : la barre de titre custom est en HTML,
    donc plus de bouton système, il faut recréer ce comportement côté Python.

Installation : pip install pywebview
Usage        : python -m aura.gui_web   (--smoke : ferme seul apres 2,5 s)
               python aura/gui_web.py   (marche aussi, depuis n'importe quel
               dossier : la racine du projet est ajoutee a sys.path)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:      # uniquement pour les annotations (import relou en runtime)
    import webview

_ICI = Path(__file__).resolve().parent
_RACINE = _ICI.parent                      # racine du projet (contient aura/)
# Lancement en script direct (python aura/gui_web.py, double-clic...) :
# sys.path[0] est alors aura/, pas la racine -> import aura impossible.
if str(_RACINE) not in sys.path:
    sys.path.insert(0, str(_RACINE))

# NB : config/maj sont importes LOCALEMENT (dans main/_verifier_maj) :
# un import top-level de `aura.*` execute aura/__init__.py -> orchestrateur
# -> gplearn/llama_cpp, absents du Python global AVANT la relance venv.

_INDEX_HTML = _ICI / "web" / "index.html"

# Dependencies du projet : si l'interpreteur courant n'en a pas (double-clic
# sur le .py avec le Python global de Windows, qui n'a ni gplearn ni
# llama_cpp), on se relance automatiquement avec .venv, sinon message clair.
_DEPS = ("webview", "numpy", "sklearn", "gplearn", "llama_cpp")


def _python_venv() -> Path | None:
    for cand in (_RACINE / ".venv" / "Scripts" / "python.exe",
                 _RACINE / ".venv" / "bin" / "python"):
        if cand.is_file():
            return cand
    return None


def _manquantes() -> list[str]:
    import importlib.util
    return [m for m in _DEPS if importlib.util.find_spec(m) is None]


def _relancer_si_besoin() -> None:
    """Relance avec .venv si des modules du projet manquent (aucune boucle :
    dans la venv on n'est deja plus 'hors venv')."""
    manquantes = _manquantes()
    if not manquantes:
        return
    deja_venv = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    py = _python_venv()
    if py is not None and not deja_venv:
        import subprocess
        print(f"[gui_web] {', '.join(manquantes)} absent(s) de "
              f"{sys.executable} -> relance avec {py}")
        raise SystemExit(subprocess.call(
            [str(py), str(Path(__file__).resolve()), *sys.argv[1:]]))
    raise SystemExit(
        "[gui_web] modules manquants : " + ", ".join(manquantes) + "\n"
        "Installe-les avec :  uv sync\n"
        "Puis lance avec    :  .venv\\Scripts\\python.exe -m aura.gui_web")


class Api:
    """Pont exposé à JS sous `window.pywebview.api`.

    Chaque méthode publique (sans _ devant) est appelable depuis le JS via
    `pywebview.api.nom_methode(args)` et renvoie une Promise côté JS.
    pywebview exécute chaque appel dans son propre thread : un appel bloquant
    (le temps qu'RontoGPT réfléchisse) ne fige donc pas la fenêtre.
    """

    def __init__(self) -> None:
        self._ia: Any = None          # RontoGPT charge en tache de fond
        self._erreur: str | None = None
        self._fenetre: "webview.Window | None" = None

    # ---------------- cycle de vie de la fenêtre (barre custom) ----------------
    def attacher_fenetre(self, fenetre: "webview.Window") -> None:
        self._fenetre = fenetre

    def fermer(self) -> None:
        if self._fenetre:
            self._fenetre.destroy()

    def reduire(self) -> None:
        if self._fenetre:
            self._fenetre.minimize()

    def agrandir(self) -> None:
        if self._fenetre:
            self._fenetre.toggle_fullscreen()

    def deplacer(self, dx: int, dy: int) -> None:
        """Déplace la fenêtre de (dx, dy) pixels — appelé en continu pendant
        le drag de la barre de titre HTML (voir web/index.html)."""
        if not self._fenetre:
            return
        try:
            self._fenetre.move(self._fenetre.x + int(dx), self._fenetre.y + int(dy))
        except Exception:  # noqa: BLE001
            pass

    # ---------------- chargement du cerveau ----------------
    def charger_ia(self) -> None:
        threading.Thread(target=self._charger_ia_thread, daemon=True).start()

    def _charger_ia_thread(self) -> None:
        t0 = time.time()
        try:
            # RECHAUFFAGE SERVEUR (aura.toml [serveur] actif = true) :
            # llama-server Vulkan demarre EN PARALLELE du chargement du
            # cerveau — la premiere question n'attend pas les 5-15 s de
            # boot. AURA_SERVEUR=0 ou binaire absent -> no-op silencieux,
            # le chemin in-process prend le relais.
            from aura import serveur_lora as _serveur_lora
            _serveur_lora.rechauffer()
        except Exception:              # jamais bloquant
            pass
        try:
            from aura.orchestrateur import RontoGPT

            ia = RontoGPT()
            try:  # pré-charge la réponse de démarrage, cache froid absorbé
                ia.executer_detaille("Bonjour")
            except Exception:  # noqa: BLE001
                pass
            self._ia = ia
            self._pousser("aura_pret", {"latence": round(time.time() - t0, 1)})
            self._verifier_maj()
        except Exception as e:  # noqa: BLE001
            self._erreur = str(e)
            self._pousser("aura_erreur", {"message": str(e)})

    def etat(self) -> dict[str, Any]:
        """État du chargement, interrogeable par le JS.

        Un évènement poussé (`aura_pret`) peut partir AVANT que la page ait
        défini `window.auraEvenement` : il serait alors perdu et l'interface
        resterait sur « chargement... ». Le JS interroge donc cette méthode
        jusqu'à ce que le cerveau soit prêt (ou en erreur).
        """
        return {"prete": self._ia is not None, "erreur": self._erreur}

    def _verifier_maj(self) -> None:
        """Pousse 'maj_disponible' vers le JS si une release plus recente
        existe (3 s max, silencieux hors-ligne — AURA_MAJ=0 pour couper)."""
        try:
            try:
                from . import maj
            except ImportError:           # lancement en script direct
                from aura import maj
            rap = maj.verifier(timeout=3.0)
            if rap.get("dispo"):
                self._pousser("maj_disponible",
                              {"version": rap.get("version"),
                               "nouvelle": rap.get("nouvelle"),
                               "url": rap.get("url")})
        except Exception:  # noqa: BLE001
            pass

    # ---------------- question / réponse ----------------
    def envoyer(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"ok": False, "erreur": "question vide"}
        if self._ia is None:
            return {"ok": False, "erreur": "IA non prête"}
        t0 = time.time()
        # FLUX EN DIRECT : chaque morceau genere est pousse au JS
        # (« morceau ») pendant l'appel — la bulle se remplit au fur et a
        # mesure au lieu d'attendre la fin. Desabonnement garanti (finally).
        desabonner = None
        try:
            # LES DEUX PHASES : la reflexion et la reponse arrivent
            # separement, chacune avec sa `phase`. Le JS les met dans deux
            # bulles — sans quoi la pensee du modele et sa reponse
            # s'ecriraient dans le meme fil et l'utilisateur ne saurait plus
            # laquelle est sa reponse.
            from aura.llama_cerveau import abonner_flux_phase
            desabonner = abonner_flux_phase(
                lambda morceau, phase: self._pousser(
                    "morceau", {"texte": morceau, "phase": phase}))
        except Exception:  # noqa: BLE001   # jamais bloquant
            desabonner = None
        try:
            r = self._ia.executer_detaille(question)
            return {
                "ok": True,
                "reponse": str(r.get("reponse", "")) or "...",
                "experts": sorted(r.get("experts") or set()),
                "cerveau": str(r.get("cerveau_choisi", "-")),
                "latence": round(time.time() - t0, 1),
                # Relance : « reponse » est une QUESTION (question trop
                # ouverte) — le front peut le signaler au lieu d'afficher
                # une reponse qui n'en est pas une.
                "en_attente": bool(r.get("en_attente")),
                "clarification": r.get("clarification"),
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "erreur": str(e)}
        finally:
            if desabonner is not None:
                try:
                    desabonner()
                except Exception:  # noqa: BLE001
                    pass

    # ---------------- documents locaux (RAG) ----------------
    # Les tests appellent ces méthodes sans pywebview : les imports sont donc
    # locaux, et le type de dialogue a une valeur de repli (OPEN_DIALOG = 10).
    @staticmethod
    def _documents():
        try:
            from . import documents
        except ImportError:               # lancement en script direct
            from aura import documents
        return documents

    def documents_liste(self) -> dict[str, Any]:
        try:
            return {"ok": True, "fichiers": self._documents().lister()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "erreur": str(e), "fichiers": []}

    def documents_ajouter(self) -> dict[str, Any]:
        """Ouvre le sélecteur de fichiers, indexe ce qui a été choisi."""
        try:
            import webview
            type_dialogue = webview.OPEN_DIALOG
        except Exception:                 # pywebview absent (tests)
            type_dialogue = 10            # webview.OPEN_DIALOG
        if self._fenetre is None:
            return {"ok": False, "erreur": "fenêtre indisponible", "fichiers": []}
        try:
            choix = self._fenetre.create_file_dialog(
                type_dialogue, allow_multiple=True,
                file_types=("Documents (*.pdf;*.txt;*.md;*.json;*.csv)",
                            "Tous les fichiers (*.*)"))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "erreur": str(e), "fichiers": []}
        if not choix:
            return {"ok": False, "erreur": "annule", "fichiers": []}
        documents = self._documents()
        ajoutes, erreurs = 0, []
        for chemin in choix:
            try:
                documents.ajouter(chemin)
                ajoutes += 1
            except Exception as e:  # noqa: BLE001
                erreurs.append(f"{Path(chemin).name} : {e}")
        try:
            fichiers = documents.lister()
        except Exception:  # noqa: BLE001
            fichiers = []
        return {"ok": ajoutes > 0, "ajoutes": ajoutes,
                "erreurs": erreurs, "fichiers": fichiers}

    def documents_supprimer(self, nom: str) -> dict[str, Any]:
        try:
            documents = self._documents()
            supprimes = documents.supprimer(nom)
            return {"ok": supprimes > 0, "supprimes": supprimes,
                    "fichiers": documents.lister()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "erreur": str(e), "supprimes": 0, "fichiers": []}

    # ---------------- pousser un évènement vers JS depuis un thread Python ----------------
    def _pousser(self, evenement: str, donnees: dict) -> None:
        if not self._fenetre:
            return
        js = (
            "window.auraEvenement && window.auraEvenement("
            f"{json.dumps(evenement)}, {json.dumps(donnees)})"
        )
        try:
            self._fenetre.evaluate_js(js)
        except Exception:  # noqa: BLE001
            pass


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="aura.gui_web")
    ap.add_argument("--smoke", action="store_true", help="ferme seul apres 2,5 s (test)")
    args = ap.parse_args(argv)

    _relancer_si_besoin()          # bons modules ? sinon relance .venv
    # config APRES la verification des deps (voir commentaire module)
    try:
        from . import config as _config
    except ImportError:             # lancement en script direct
        from aura import config as _config
    _config.appliquer()            # aura.toml + profils, avant l'orchestrateur
    import webview  # pip install pywebview (apres verification des deps)

    api = Api()
    fenetre = webview.create_window(
        "RontoGPT",
        url=str(_INDEX_HTML),
        js_api=api,
        width=980,
        height=680,
        frameless=True,
        easy_drag=False,  # le drag est géré en JS via api.deplacer (voir index.html)
        resizable=True,
        background_color="#E6E6E6",
    )
    assert fenetre is not None, "create_window a renvoye None"
    api.attacher_fenetre(fenetre)

    def au_demarrage() -> None:
        api.charger_ia()
        if args.smoke:
            def _fermer_apres() -> None:
                time.sleep(2.5)
                fenetre.destroy()

            threading.Thread(target=_fermer_apres, daemon=True).start()

    webview.start(au_demarrage, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
