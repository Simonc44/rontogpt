"""Routage MoE Top-2 : deux experts par requete, poids renormalises.

Le grain PAR TOKEN n'est pas testable ici, et ne doit pas l'etre : un
adaptateur LoRA est un DELTA DE POIDS LINEAIRE que llama-server applique pour
toute la generation (`POST /lora-adapters`). Un vrai routage par token suppose
des poids de routeur DANS le modele. Ces tests verrouillent ce que le runtime
permet reellement : la selection Top-2 par requete, le rejet du bruit et la
renormalisation du budget de specialisation.
"""
import json

import pytest

from aura import serveur_lora


class TestRouteurTop2:
    def test_garde_les_deux_meilleurs(self):
        poids = serveur_lora.routeur_top2(
            {"math": 0.5, "web": 0.3, "code": 0.15, "general": 0.05})
        assert set(poids) == {"math", "web"}
        assert poids["math"] == 0.625
        assert poids["web"] == 0.375

    def test_les_poids_somment_a_un(self):
        for scores in ({"math": 0.9, "web": 0.4},
                       {"math": 0.34, "code": 0.33, "web": 0.33},
                       {"general": 1.0}):
            poids = serveur_lora.routeur_top2(scores)
            assert abs(sum(poids.values()) - 1.0) < 0.01

    def test_ecarte_le_bruit_sous_le_seuil(self):
        """Un expert a 10 % n'est pas un expert : aucun mix."""
        assert serveur_lora.routeur_top2({"math": 0.10, "web": 0.05}) == {}

    def test_un_seul_expert_au_dessus_du_seuil(self):
        assert serveur_lora.routeur_top2(
            {"math": 0.20, "web": 0.01}) == {"math": 1.0}

    def test_egalite_tranchee_par_l_ordre_de_boot(self):
        """Meme question, meme mix : l'ordre de _IDS est deterministe."""
        a = serveur_lora.routeur_top2({"web": 0.5, "math": 0.5})
        b = serveur_lora.routeur_top2({"math": 0.5, "web": 0.5})
        assert a == b == {"math": 0.5, "web": 0.5}
        assert list(a) == ["math", "web"]

    def test_categories_sans_adaptateur_ignorees(self):
        """Les scores non-LoRA (pot, rag, titulaires) ne montent rien."""
        assert serveur_lora.routeur_top2({"pot": 0.9, "rag": 0.8}) == {}

    def test_score_non_numerique_ignore(self):
        assert serveur_lora.routeur_top2(
            {"math": "beaucoup", "web": 0.6}) == {"web": 1.0}

    def test_nan_et_infini_ignores(self):
        assert serveur_lora.routeur_top2(
            {"math": float("nan"), "web": float("inf"),
             "code": 0.4}) == {"code": 1.0}

    def test_negatif_ignore(self):
        assert serveur_lora.routeur_top2({"math": -0.5, "web": 0.6}) == {
            "web": 1.0}

    def test_top_k_est_deux(self):
        assert serveur_lora.TOP_K == 2


class TestActivationTop2:
    def test_repli_sur_la_categorie_routee(self, monkeypatch):
        vus = {}

        def faux_mixte(poids):
            vus["poids"] = poids
            return True

        monkeypatch.setattr(serveur_lora, "activer_mixte", faux_mixte)
        assert serveur_lora.activer_top2({}, defaut="code") is True
        assert vus["poids"] == {"code": 1.0}

    def test_sans_score_ni_defaut_refuse(self, monkeypatch):
        monkeypatch.setattr(serveur_lora, "activer_mixte",
                            lambda poids: pytest.fail("aucun mix possible"))
        assert serveur_lora.activer_top2({}) is False

    def test_deux_echelles_reelles_envoyees(self, monkeypatch):
        """Le corps POST ne contient que DEUX echelles non nulles."""
        envoyes = []

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"success": true}'

        def faux_urlopen(req, timeout):
            envoyes.append(json.loads(req.data.decode()))
            return R()

        monkeypatch.setattr(serveur_lora, "_assurer", lambda: True)
        monkeypatch.setattr(serveur_lora.urllib.request, "urlopen",
                            faux_urlopen)
        monkeypatch.setattr(serveur_lora, "_actif", None)
        try:
            assert serveur_lora.activer_top2(
                {"math": 0.5, "web": 0.3, "code": 0.2, "general": 0.0}) is True
        finally:
            serveur_lora._actif = None
        echelles = {c["id"]: c["scale"] for c in envoyes[-1]}
        assert echelles == {0: 0.625, 1: 0.375, 2: 0.0, 3: 0.0}
