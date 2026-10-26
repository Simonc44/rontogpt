"""Exposition de l'API locale : qui peut entrer, et ce qui fuit.

Ce qui est verrouille ici :

- ecouter hors loopback sans cle est un REFUS au demarrage (pas un message) ;
- les routes qui modifient un etat durable ne repondent qu'a 127.0.0.1,
  meme avec une cle valide ;
- la cle se compare en temps constant, par les deux entetes acceptes ;
- `/v1/config` ne livre plus la valeur des variables qui portent un secret ;
- l'atelier OpenAI verifie enfin la cle qu'il acceptait sans la regarder.
"""
from pathlib import Path

import pytest

from aura import api, config

RACINE = Path(__file__).resolve().parent.parent
CLE = "une-cle-de-32-caracteres-ok-000"


class TestLoopback:
    @pytest.mark.parametrize("adresse",
                             ["127.0.0.1", "127.0.0.5", "::1", "localhost"])
    def test_local(self, adresse):
        assert api._est_local(adresse) is True

    @pytest.mark.parametrize("adresse",
                             ["0.0.0.0", "::", "192.168.1.10", "10.0.0.1",
                              "", "example.com"])
    def test_distant(self, adresse):
        assert api._est_local(adresse) is False


class TestExposition:
    def test_loopback_sans_cle_est_permis(self, monkeypatch):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        assert api._verifier_exposition("127.0.0.1") is None
        assert api._verifier_exposition("localhost") is None

    def test_ouverte_sans_cle_est_refusee(self, monkeypatch):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        message = api._verifier_exposition("0.0.0.0")
        assert message and "AURA_API_CLE" in message

    def test_cle_courte_est_refusee(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", "court")
        assert api._verifier_exposition("0.0.0.0")

    def test_cle_suffisante_est_acceptee(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", CLE)
        assert api._verifier_exposition("0.0.0.0") is None

    def test_la_cle_explicite_prime_sur_l_environnement(self, monkeypatch):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        assert api._verifier_exposition("0.0.0.0", CLE) is None
        assert api._verifier_exposition("0.0.0.0", "court")

    def test_demarrer_en_tache_refuse_lexposition(self, monkeypatch):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        with pytest.raises(ValueError, match="AURA_API_CLE"):
            api.demarrer_en_tache(host="0.0.0.0", port=0)

    def test_la_cli_refuse_avant_d_ecouter(self, monkeypatch, capsys):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        assert api.main(["--host", "0.0.0.0", "--port", "0"]) == 2
        assert "REFUS" in capsys.readouterr().err


class TestCle:
    def test_sans_cle_configuree_tout_passe(self, monkeypatch):
        monkeypatch.delenv("AURA_API_CLE", raising=False)
        assert api._verifier_cle({}) is True

    def test_cle_absente_ou_fausse_refusee(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", CLE)
        assert api._verifier_cle({}) is False
        assert api._verifier_cle({"X-RontoGPT-Cle": "faux"}) is False

    def test_les_deux_entetes_acceptes(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", CLE)
        assert api._verifier_cle({"X-RontoGPT-Cle": CLE}) is True
        assert api._verifier_cle({"X-Aura-Cle": CLE}) is True


class TestRoutesLocales:
    @pytest.mark.parametrize("chemin", [
        "/v1/cloud", "/v1/learn", "/v1/documents",
        "/v1/api", "/v1/api/deploy", "/v1/api/disable",
        "/v1/cloud?actif=1",
    ])
    def test_reservees_a_la_machine(self, chemin):
        assert api._route_locale_seulement(chemin) is True

    @pytest.mark.parametrize("chemin", [
        "/v1/ask", "/v1/ask/stream", "/v1/health", "/v1/config",
        "/v1/ask/cloud",
    ])
    def test_ouvertes(self, chemin):
        assert api._route_locale_seulement(chemin) is False


class TestMasquage:
    def test_les_secrets_sont_masques(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", "secret-abc")
        monkeypatch.setenv("AURA_GEMINI_API_KEY", "AIza-secret")
        monkeypatch.setenv("AURA_GEMINI_MAX_TOKENS", "4096")
        monkeypatch.setenv("AURA_PROFIL", "equilibre")
        vues = config.efficace(masquer=True)
        assert vues["AURA_API_CLE"] == "***"
        assert vues["AURA_GEMINI_API_KEY"] == "***"
        # `MAX_TOKENS` contient « TOKEN » sans etre un secret : il doit rester.
        assert vues["AURA_GEMINI_MAX_TOKENS"] == "4096"
        assert vues["AURA_PROFIL"] == "equilibre"

    def test_sans_masquage_la_valeur_reste_lisible(self, monkeypatch):
        monkeypatch.setenv("AURA_API_CLE", "secret-abc")
        assert config.efficace()["AURA_API_CLE"] == "secret-abc"

    def test_la_route_config_demande_le_masquage(self):
        """`efficace()` seule ne suffit pas : c'est l'APPEL qui doit masquer."""
        source = (RACINE / "aura" / "api.py").read_text(encoding="utf-8")
        assert "efficace(masquer=True)" in source


class TestAtelier:
    def _client(self, monkeypatch, cle=None):
        from aura import openai_atelier as oa
        if cle is None:
            monkeypatch.delenv(oa.API_KEY_ENV, raising=False)
        else:
            monkeypatch.setenv(oa.API_KEY_ENV, cle)
        from fastapi.testclient import TestClient
        return oa, TestClient(oa.create_app())

    def test_la_cle_est_enfin_verifiee(self, monkeypatch):
        oa, client = self._client(monkeypatch, CLE)
        assert client.get("/health").status_code == 200      # vivacite : ouverte
        assert client.get("/status").status_code == 401
        assert client.post("/v1/ask", json={"question": "x"}).status_code == 401
        assert client.post("/v1/ask/stream",
                           json={"question": "x"}).status_code == 401
        entete = {"X-RontoGPT-Cle": CLE}
        assert client.get("/status", headers=entete).status_code == 200
        assert client.post("/v1/ask", json={"question": "x"},
                           headers=entete).status_code == 200

    def test_sans_cle_configuree_l_atelier_repond(self, monkeypatch):
        _oa, client = self._client(monkeypatch)
        assert client.get("/status").status_code == 200

    def test_demarrage_distant_sans_cle_refuse(self, monkeypatch):
        from aura import openai_atelier as oa
        monkeypatch.delenv(oa.API_KEY_ENV, raising=False)
        with pytest.raises(ValueError, match="AURA_API_CLE"):
            oa.demarrer_serveur(host="0.0.0.0", port=0)
