"""Voix locale : ecouter (STT) et dire (TTS), sans qu'un octet quitte la machine.

Le TUI appelle `GET /v1/voix` (qu'est-ce qui est installe ?) puis
`POST /v1/voix/ecouter` (enregistre au micro, transcrit, rend le texte).
La lecture a voix haute reste faite par le TUI (synthese Windows locale).

DEPENDANCES OPTIONNELLES — `uv sync --extra voix` :
    faster-whisper   transcription locale (modele telecharge au 1er usage)
    sounddevice      capture du micro

Aucune n'est importee au chargement du module : sans elles, `etat()` le dit
en une phrase et `ecouter()` leve `VoixIndisponible` avec la commande a lancer.

Reglages :
    AURA_WHISPER=small      taille du modele (tiny, base, small, medium)
    AURA_WHISPER_DEVICE=cpu cpu par defaut ; "cuda" si disponible
"""
from __future__ import annotations

import importlib.util
import os
import threading

#: echantillonnage attendu par Whisper
FREQUENCE = 16_000
#: bornes de la duree d'ecoute (secondes) : jamais un micro ouvert sans fin
DUREE_MIN = 1.0
DUREE_MAX = 30.0
DUREE_DEFAUT = 6.0

_COMMANDE = "uv sync --extra voix"

_VERROU = threading.Lock()
_modele = None


class VoixIndisponible(RuntimeError):
    """Une dependance manque, ou aucun micro n'est utilisable."""


def _present(nom: str) -> bool:
    try:
        return importlib.util.find_spec(nom) is not None
    except (ImportError, ValueError):
        return False


def etat() -> dict:
    """Ce que la voix peut faire ici. Ne leve jamais, ne charge aucun modele."""
    stt = _present("faster_whisper")
    micro = _present("sounddevice")
    manque = [n for n, ok in (("faster-whisper", stt), ("sounddevice", micro)) if not ok]
    return {
        "ok": True,
        "ecoute": stt and micro,
        "modele": os.environ.get("AURA_WHISPER", "small"),
        "manque": manque,
        "installer": _COMMANDE if manque else "",
        "local": True,
    }


def borner_duree(valeur) -> float:
    """Duree demandee ramenee dans [DUREE_MIN, DUREE_MAX] ; defaut si illisible."""
    try:
        duree = float(valeur)
    except (TypeError, ValueError):
        return DUREE_DEFAUT
    if duree != duree:                      # NaN
        return DUREE_DEFAUT
    return max(DUREE_MIN, min(DUREE_MAX, duree))


def _charger_modele():
    global _modele
    with _VERROU:
        if _modele is None:
            from faster_whisper import WhisperModel
            _modele = WhisperModel(
                os.environ.get("AURA_WHISPER", "small"),
                device=os.environ.get("AURA_WHISPER_DEVICE", "cpu"),
                compute_type="int8",
            )
        return _modele


def transcrire(audio, langue: str | None = "fr") -> str:
    """Texte d'un signal mono float32 a 16 kHz."""
    modele = _charger_modele()
    segments, _info = modele.transcribe(
        audio, language=langue if langue in ("fr", "en") else None,
        vad_filter=True, beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


def ecouter(duree=DUREE_DEFAUT, langue: str | None = "fr") -> dict:
    """Enregistre `duree` secondes au micro puis transcrit.

    Rend {ok, texte, duree}. Leve `VoixIndisponible` si une dependance ou le
    micro manque : l'appelant affiche la phrase telle quelle.
    """
    info = etat()
    if not info["ecoute"]:
        raise VoixIndisponible(
            f"voix non installee ({', '.join(info['manque'])}) : lancer {_COMMANDE}")
    duree = borner_duree(duree)
    try:
        import sounddevice as sd
        brut = sd.rec(int(duree * FREQUENCE), samplerate=FREQUENCE,
                      channels=1, dtype="float32")
        sd.wait()
    except Exception as e:  # noqa: BLE001 — PortAudio leve des types varies
        raise VoixIndisponible(f"micro inutilisable : {e}") from e
    texte = transcrire(brut.reshape(-1), langue)
    return {"ok": True, "texte": texte, "duree": duree}
