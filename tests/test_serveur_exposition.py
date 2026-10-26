"""llama-server : pas d'écoute réseau sans jeton.

`llama-server` n'a aucune authentification par défaut. Une simple variable
d'environnement (`AURA_SERVEUR_ADRESSE=0.0.0.0:8757`) publiait donc le modèle
et ses adaptateurs à tout le réseau. Ce qui est verrouillé ici :

- l'écoute hors loopback sans `AURA_SERVEUR_CLE` (>= 16 car.) est refusée ;
- le jeton part au binaire (`--api-key`) ET dans nos propres requêtes ;
- le repli est le chemin in-process, qui ne publie rien.
"""
import pytest

from aura import serveur_lora as sl

CLE = "un-jeton-de-32-caracteres-000000"


@pytest.fixture
def adresse(monkeypatch):
    def poser(valeur: str) -> None:
        monkeypatch.setattr(sl, "_ADRESSE", valeur)
    monkeypatch.delenv("AURA_SERVEUR_CLE", raising=False)
    return poser


class TestExposition:
    def test_loopback_sans_jeton_est_permis(self, adresse):
        adresse("127.0.0.1:8757")
        assert sl._verifier_exposition() is None

    def test_reseau_sans_jeton_est_refuse(self, adresse):
        adresse("0.0.0.0:8757")
        message = sl._verifier_exposition()
        assert message and "AURA_SERVEUR_CLE" in message

    def test_jeton_court_refuse(self, adresse, monkeypatch):
        adresse("0.0.0.0:8757")
        monkeypatch.setenv("AURA_SERVEUR_CLE", "court")
        assert sl._verifier_exposition()

    def test_jeton_suffisant_accepte(self, adresse, monkeypatch):
        adresse("0.0.0.0:8757")
        monkeypatch.setenv("AURA_SERVEUR_CLE", CLE)
        assert sl._verifier_exposition() is None

    def test_lan_explicite_refuse(self, adresse):
        adresse("192.168.1.11:8757")
        assert sl._verifier_exposition()


class TestEntetes:
    def test_le_jeton_part_dans_les_requetes(self, monkeypatch):
        monkeypatch.setenv("AURA_SERVEUR_CLE", CLE)
        assert sl._entetes()["Authorization"] == f"Bearer {CLE}"

    def test_sans_jeton_aucune_autorisation(self, monkeypatch):
        monkeypatch.delenv("AURA_SERVEUR_CLE", raising=False)
        assert "Authorization" not in sl._entetes()


class TestDemarrage:
    def test_refuse_avant_de_lancer_le_binaire(self, adresse, monkeypatch):
        """Le refus doit tomber AVANT d'écouter, pas après."""
        adresse("0.0.0.0:8757")
        appels = []
        monkeypatch.setattr(sl, "_binaire", lambda: appels.append(1))
        monkeypatch.setattr(sl, "_indispo", None)
        assert sl._demarrer() is False
        assert appels == []
        assert "AURA_SERVEUR_CLE" in (sl._indispo or "")

    def test_binaire_sans_api_key_renonce_au_serveur(self, adresse, monkeypatch,
                                                     tmp_path):
        """Un llama-server qui ne sait pas authentifier ne doit pas écouter."""
        monkeypatch.setenv("AURA_SERVEUR_CLE", CLE)
        adresse("127.0.0.1:8757")
        faux = tmp_path / "llama-server.exe"
        faux.write_text("x")
        monkeypatch.setattr(sl, "_binaire", lambda: faux)
        monkeypatch.setattr(sl, "_gguf_cerveau", lambda: tmp_path / "c.gguf")
        monkeypatch.setattr(sl, "_aide_binaire", lambda _b: "--help sans auth")
        monkeypatch.setattr(sl, "_indispo", None)
        lance = []
        monkeypatch.setattr(sl.subprocess, "Popen",
                            lambda *a, **k: lance.append(1))
        assert sl._demarrer() is False
        assert lance == []
        assert "api-key" in (sl._indispo or "")
