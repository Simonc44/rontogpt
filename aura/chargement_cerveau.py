"""Chargement du cerveau (llama.cpp) : initialisation, prechauffage, dechargement.

Ce module isole tout ce qui touche a llama.cpp (construction du Llama,
gestion du cache KV, mlock, draft_model) pour que `llama_cerveau.py`
reste centre sur la generation et le flux.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama

from . import flux_cerveau  # noqa: F401 — assure l'initialisation des symbols re-exportes
from . import llama_cerveau as _llama_cerveau  # pour les valeurs par defaut

LOG = logging.getLogger("aura.chargement")

# ── CERVEAU GLOBAL ──────────────────────────────────────────────────────────
# Un seul Llama a la fois. Le verrou garantit qu'un appel concurrent
# attend la fin du chargement et recoit le MEME objet.
_llm: "Llama | None" = None
_verrou_chargement = threading.Lock()


def _draft_model():
    """Speculative decoding in-process : prompt-lookup / n-gram.

    Le brouillon est extrait de l'historique lui-meme (recherche de n-grammes
    deja vus) — AUCUN second modele, donc aucun cout de RAM de poids.

    PAR DEFAUT OFF (AURA_SPEC=1 pour activer) et ce n'est pas un choix
    esthetique : llama_cpp force logits_all=True des qu'un draft_model est
    passe (~500 Mo de scores pour Llama-3.2-1B), et le bench mesure un
    NET RECUL sur ce CPU :
        AURA_SPEC=1 : 2.1 / 3.4 tok/s   (repétitif / neuf)
        AURA_SPEC=0 : 6.0 / 8.8 tok/s
    Le boosting reel passe par le chemin SERVEUR (--spec-type ngram-mod,
    voir serveur_lora._spec_type_ok : +170 % sur texte repete, cout nul
    ailleurs) qui n'a pas ce defaut. Ici l'interrupteur reste pour
    comparer/debuguer.

    Medusa / EAGLE-3 : impossibles sur Llama-3.2-1B (aucun checkpoint draft
    converti n'existe pour cette cible) — d'ou le prompt-lookup qui, lui,
    marche avec n'importe quel modele.
    """
    if os.environ.get("AURA_SPEC", "0") == "0":
        return None
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        return LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=10)
    except Exception as e:  # noqa: BLE001  (build sans llama_speculative)
        LOG.warning("Speculative decoding indisponible : %s", e)
        return None


# PRECHAUFFAGE (AURA_PRECHUFFAGE=0 pour couper) : la TOUTE PREMIERE
# generation qui suit le chargement d'un modele sort fausse. Mesure sur le
# cerveau par defaut (Gemma 3 1B) : « Quelle est la capitale de la France ? »
# renvoie une auto-description au 1er appel, puis « Paris » a tous les
# suivants — 5 processus sur 5. Le cache KV (SWA) n'est pas encore amorce.
# On paie donc une generation minuscule, jetee, pour que la PREMIERE question
# de l'utilisateur tombe deja sur un modele chaud. Le chemin serveur fait
# deja ce prechauffage (serveur_lora.rechauffer, appele par l'API et la GUI).
def _prechauffer(modele, t0: float) -> None:
    """Une generation minuscule et jetee, pour amorcer le cache du modele."""
    if os.environ.get("AURA_PRECHUFFAGE", "1") == "0":
        return
    t1 = time.time()
    try:
        modele.create_chat_completion(
            messages=[{"role": "user", "content": "Bonjour"}],
            max_tokens=1, temperature=1.0)
    except Exception as e:
        # Un prechauffage qui echoue ne doit JAMAIS empecher le chargement.
        LOG.debug("prechauffage ignore : %s", e)
        return
    LOG.info("prechauffage du cerveau en %.1fs (chargement %.1fs)",
             time.time() - t1, t1 - t0)


def _charger(chemin_gguf: str | None = None,
              reglages: dict | None = None,
              familles: str | None = None) -> "Llama":
    """Charge le modele llama.cpp (thread-safe, idempotent).

    Args:
        chemin_gguf: chemin absolu du fichier GGUF (defaut: llama_cerveau._CHEMIN_GGUF).
        reglages: dict avec n_ctx, n_batch, n_threads, n_threads_batch,
            flash_attn, type_kv, n_gpu_layers, use_mlock
            (defaut: llama_cerveau._reglages_materiel()).
        familles: nom de famille lu dans l'en-tete GGUF (defaut: llama_cerveau.famille()).

    Returns:
        L'instance Llama partagee.
    """
    if chemin_gguf is None:
        chemin_gguf = _llama_cerveau._CHEMIN_GGUF
    if reglages is None:
        reglages = _llama_cerveau._reglages_materiel()
    if familles is None:
        familles = _llama_cerveau.famille()

    global _llm
    with _verrou_chargement:
        if _llm is not None:
            return _llm
        if not os.path.exists(chemin_gguf):
            raise FileNotFoundError(
                f"GGUF introuvable : {chemin_gguf}\n"
                "Telecharge-le avec : python scripts/telecharger_llama.py\n"
                "ou boote le .aef dont le cerveau est en cache : "
                "python -m aura.kernel aura_system.aef")

        # Preflight memoire (best effort)
        try:
            from . import profil_materiel as _pm
            ok, diagnostic = _pm.verifier_place(os.path.getsize(chemin_gguf))
            if not ok:
                raise MemoryError(diagnostic)
            LOG.info("[profil] memoire du cerveau : %s", diagnostic)
        except MemoryError:
            raise
        except Exception as e:  # profil indisponible : le chargement essaie
            LOG.debug("[profil] preflight memoire ignore : %s", e)

        # INTEGRITE : un GGUF substitue est un cerveau substitue.
        from . import integrite
        integrite.verifier(chemin_gguf)

        from llama_cpp import Llama
        # Garde cache disque : refuse l'ouverture si le dossier n'est pas safe.
        from . import garde_cache_disque
        garde_cache_disque.installer(Llama)

        t0 = time.time()
        try:
            modele = Llama(
                model_path=chemin_gguf,
                n_ctx=reglages["n_ctx"],
                n_batch=reglages["n_batch"],
                n_threads=reglages["n_threads"],
                n_threads_batch=reglages["n_threads_batch"],
                flash_attn=reglages["flash_attn"],
                type_k=reglages["type_kv"],
                type_v=reglages["type_kv"],
                n_gpu_layers=reglages["n_gpu_layers"],
                use_mlock=reglages["use_mlock"],
                draft_model=_draft_model(),
                verbose=False,
            )
        except Exception as e:
            nom_famille = familles or "famille inconnue"
            raise RuntimeError(
                f"chargement du cerveau impossible ({nom_famille} : "
                f"{os.path.basename(chemin_gguf)}) : {e}\n"
                "Si l'architecture est recente (gemma3), la version de "
                "llama-cpp-python est trop ancienne :\n"
                "    pip install --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu "
                "llama-cpp-python==0.3.36\n"
                "Le chemin SERVEUR (AURA_SERVEUR=1) utilise son propre "
                "binaire llama-server et n'est pas concerne.") from e
        _llm = modele
        LOG.info("cerveau %s charge en %.1fs (%s, gpu_layers=%d, spec=%s)",
                 familles or "?", time.time() - t0,
                 os.path.basename(chemin_gguf), reglages["n_gpu_layers"],
                 "on" if modele.draft_model is not None else "off")
        _prechauffer(modele, t0)
        return modele


def decharger() -> None:
    """Libere le cerveau et rend la RAM tout de suite (rechargement possible).

    Appele automatiquement a la sortie du process : fermer explicitement la
    session llama.cpp evite la cascade « Exception ignored in: Llama.__del__ »
    au teardown de l'interpreteur et rend ~600 Mo au systeme immediatement.
    """
    global _llm
    with _verrou_chargement:
        modele, _llm = _llm, None
    if modele is None:
        return
    try:
        modele.close()
    except Exception:          # deja ferme / backend deja demonte
        LOG.debug("fermeture du cerveau")


def llm_charge():
    """Llama charge, ou None si le modele n'est pas (encore) en memoire."""
    return _llm


atexit.register(decharger)