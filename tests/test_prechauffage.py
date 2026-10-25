"""Prechauffage du cerveau : la 1re generation apres chargement etait FAUSSE.

Mesure sur le cerveau par defaut (Gemma 3 1B) : « Quelle est la capitale de la
France ? » rendait une auto-description au TOUT PREMIER appel, puis « Paris »
a tous les suivants — 5 processus sur 5, et le test `test_reponse_fr` passait
ou echouait selon ce qui avait tourne avant lui. Le cache KV (SWA) n'est pas
encore amorce a ce moment-la.

`_charger` paie donc une generation minuscule et jetee avant de rendre le
modele. Ce fichier verrouille trois choses : le prechauffage a bien lieu, il
a lieu AVANT que le modele soit rendu a l'appelant, et son echec eventuel ne
peut pas empecher le chargement.
"""
import sys
import types

import pytest

from aura import llama_cerveau as lc
from aura import profil_materiel


class FauxModele:
    """Modele factice : compte les appels, et peut echouer au premier."""

    def __init__(self, echoue: bool = False):
        self.appels: list[dict] = []
        self.echoue = echoue
        self.draft_model = None      # lu par le log de _charger

    def create_chat_completion(self, **kw):
        self.appels.append(kw)
        if self.echoue:
            raise RuntimeError("modele casse")
        return {"choices": [{"message": {"content": "Bonjour"}}]}


class TestPrechauffer:
    def test_une_generation_minuscule_et_jetee(self, monkeypatch):
        monkeypatch.delenv("AURA_PRECHUFFAGE", raising=False)
        faux = FauxModele()
        lc._prechauffer(faux, 0.0)
        assert len(faux.appels) == 1
        assert faux.appels[0]["max_tokens"] == 1     # minuscule, jetee
        assert faux.appels[0]["temperature"] == 1.0

    def test_interrupteur_AURA_PRECHUFFAGE(self, monkeypatch):
        monkeypatch.setenv("AURA_PRECHUFFAGE", "0")
        faux = FauxModele()
        lc._prechauffer(faux, 0.0)
        assert faux.appels == []

    def test_un_echec_de_prechauffage_n_empeche_pas_le_chargement(
            self, monkeypatch):
        """Le prechauffage est un CONFORT : il ne doit jamais lever."""
        monkeypatch.delenv("AURA_PRECHUFFAGE", raising=False)
        lc._prechauffer(FauxModele(echoue=True), 0.0)


class TestOrdreDansCharger:
    def test_prechauffe_apres_publication_et_avant_le_retour(self, monkeypatch):
        """Le point du bug : le 1er appel UTILISATEUR doit trouver chaud.

        On ne teste pas llama.cpp : on remplace le modele et on verifie
        l'ORDRE — `_llm` est publie, PUIS le prechauffage tourne, PUIS le
        modele est rendu. Le prechauffage voit donc toujours le modele
        complet, et l'appelant ne peut pas le recevoir avant.
        """
        reglages = {"n_ctx": 4096, "n_batch": 768, "n_threads": 4,
                    "n_threads_batch": 4, "flash_attn": True, "type_kv": 8}
        faux = FauxModele()
        vu: list[str] = []

        monkeypatch.setattr(lc, "_llm", None)
        monkeypatch.setattr(lc, "_CHEMIN_GGUF", __file__)   # existe
        monkeypatch.setattr(lc, "_reglages_materiel", lambda: dict(reglages))
        monkeypatch.setattr(lc, "_couches_gpu", lambda: 0)
        monkeypatch.setattr(lc, "_draft_model", lambda: None)
        monkeypatch.setattr(profil_materiel, "verifier_place",
                            lambda taille: (True, "test"))

        def faux_llama(**kw):
            vu.append("chargement")
            return faux

        sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=faux_llama)
        monkeypatch.delenv("AURA_PRECHUFFAGE", raising=False)

        def noter_prechauffage(modele, t0):
            # a ce moment precis, le modele doit deja etre publie
            assert lc._llm is modele
            vu.append("prechauffage")

        monkeypatch.setattr(lc, "_prechauffer", noter_prechauffage)
        try:
            rendu = lc._charger()
        finally:
            del sys.modules["llama_cpp"]

        assert vu == ["chargement", "prechauffage"]
        assert rendu is faux


@pytest.mark.parametrize("valeur", ["1", ""])
def test_prechauffage_actif_par_defaut(monkeypatch, valeur):
    monkeypatch.setenv("AURA_PRECHUFFAGE", valeur)
    faux = FauxModele()
    lc._prechauffer(faux, 0.0)
    assert len(faux.appels) == 1
