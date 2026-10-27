"""Tests du module de voix (aura/voix.py) : dependances optionnelles, duree,
modele Whisper et transcription. Aucun reseau, aucun micro reel.
"""
import sys
import types

import pytest

from aura import voix


class _ModeleFaux:
    """Pose d'un modele factice : transcribe renvoie des segments."""

    def __init__(self, segments=()):
        self.segments = list(segments)
        self.appels = []

    def transcribe(self, audio, language=None, vad_filter=True, beam_size=1):
        self.appels.append((audio, language, vad_filter, beam_size))
        return list(self.segments), {"duree": 1.0}


class _BrutFaux:
    """Pose d'un signal : reshape(-1) -> une liste d'echantillons."""

    def __init__(self, n=10):
        self.n = n

    def reshape(self, n):
        return list(range(self.n))


def _mon_module_fictif(nom, **attrs):
    mod = types.ModuleType(nom)
    for cle, val in attrs.items():
        setattr(mod, cle, val)
    return mod


# ── etat() : les dependances optionnelles (STT + micro) sont-elles la/les bonnes ?


def test_etat_voix_ok_si_stt_et_micro_presents(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: True)
    r = voix.etat()
    assert r["ok"] is True
    assert r["ecoute"] is True
    assert r["manque"] == []
    assert r["installer"] == ""


def test_etat_voix_distingue_stt_micro(monkeypatch):
    # STT present, micro absent : pas de lecture vocale possible.
    monkeypatch.setattr(voix, "_present", lambda nom: nom == "sounddevice")
    r = voix.etat()
    assert r["ok"] is True
    assert r["ecoute"] is False
    assert r["manque"] == ["faster-whisper"]


def test_etat_voix_manque_tout(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: False)
    r = voix.etat()
    assert r["ok"] is True
    assert r["ecoute"] is False
    assert r["manque"] == ["faster-whisper", "sounddevice"]
    assert r["installer"] == voix._COMMANDE


def test_etat_voix_modele_depuis_variable_environnement(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: True)
    monkeypatch.setenv("AURA_WHISPER", "medium")
    assert voix.etat()["modele"] == "medium"


# ── borner_duree : bornes + defauts


def test_borner_duree_valeurs_valides():
    assert voix.borner_duree(2.0) == pytest.approx(2.0)
    assert voix.borner_duree(10.0) == pytest.approx(10.0)


def test_borner_duree_defaut_sur_invalide():
    assert voix.borner_duree(None) == voix.DUREE_DEFAUT
    assert voix.borner_duree("pas-un-nombre") == voix.DUREE_DEFAUT
    assert voix.borner_duree(float("nan")) == voix.DUREE_DEFAUT


def test_borner_duree_applique_bornes():
    assert voix.borner_duree(0) == voix.DUREE_MIN          # bornee en bas
    assert voix.borner_duree(1e9) == voix.DUREE_MAX        # bornee en haut


# ── _charger_modele : une seule instance, en cache


def test_charger_modele_cree_une_fois_et_la_met_en_cache(monkeypatch):
    monkeypatch.setattr(voix, "_modele", None)
    instances = []

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            instances.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "faster_whisper",
                       _mon_module_fictif("faster_whisper", WhisperModel=WhisperModel))
    monkeypatch.setenv("AURA_WHISPER", "tiny")
    monkeypatch.setenv("AURA_WHISPER_DEVICE", "cpu")

    modele = voix._charger_modele()
    modele2 = voix._charger_modele()

    assert len(instances) == 1          # mis en cache : pas de 2e instanciation
    assert modele is modele2
    args, kwargs = instances[0]
    assert args == ("tiny",)
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"


# ── transcrire : joint les segments, filtre la langue


def test_transcrire_jointe_les_segments(monkeypatch):
    modele = _ModeleFaux([types.SimpleNamespace(text="  bonjour  "),
                         types.SimpleNamespace(text="monde ")])
    monkeypatch.setattr(voix, "_modele", modele)
    assert voix.transcrire(_BrutFaux(16), langue="fr") == "bonjour monde"


def test_transcrire_filtre_la_langue(monkeypatch):
    modele = _ModeleFaux()
    monkeypatch.setattr(voix, "_modele", modele)
    voix.transcrire(_BrutFaux(16), langue="en")
    assert modele.appels[-1][1] == "en"


def test_transcrire_accepte_none_pour_langue_ouverte(monkeypatch):
    modele = _ModeleFaux()
    monkeypatch.setattr(voix, "_modele", modele)
    voix.transcrire(_BrutFaux(16), langue="de")
    assert modele.appels[-1][1] is None


# ── ecouter : refuse sans dependances, sinon renvoie le texte


def test_ecouter_refuse_si_dependances_absentes(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: False)
    with pytest.raises(voix.VoixIndisponible) as exc:
        voix.ecouter(duree=2.0)
    assert voix._COMMANDE in str(exc.value)


def test_ecouter_refuse_si_micro_en_erreur(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: True)

    class _SdFaux:
        @staticmethod
        def rec(*args, **kwargs):
            raise RuntimeError("portaudio indisponible")

        @staticmethod
        def wait():
            return None

    monkeypatch.setitem(sys.modules, "sounddevice", _SdFaux)
    with pytest.raises(voix.VoixIndisponible):
        voix.ecouter(duree=2.0)


def test_ecouter_recupere_le_texte_du_modele(monkeypatch):
    monkeypatch.setattr(voix, "_present", lambda nom: True)
    modele = _ModeleFaux([types.SimpleNamespace(text="le micro marche")])
    monkeypatch.setattr(voix, "_modele", modele)

    class _SdFaux:
        @staticmethod
        def rec(*args, **kwargs):
            return _BrutFaux(32)

        @staticmethod
        def wait():
            return None

    monkeypatch.setitem(sys.modules, "sounddevice", _SdFaux)

    r = voix.ecouter(duree=2.0, langue="fr")
    assert r["ok"] is True
    assert r["texte"] == "le micro marche"
    assert r["duree"] == pytest.approx(2.0)
    assert modele.appels[-1][0] is not None      # le brut a traverse transcribe
