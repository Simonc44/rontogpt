"""Verrouillage RAM et mode demon : ce qui est REELLEMENT garanti.

Deux confusions a empecher :

1. `mlock` ne raccourcit PAS le chargement a froid ; il empeche seulement la
   pagination des poids. Les tests verifient l'option posee sur le binaire et
   sur le chemin in-process, pas un gain invente.
2. `--demon` doit DETACHER le processus, pas lancer un serveur au premier plan
   qui garderait le terminal.
"""
import subprocess

import pytest

from aura import serveur_lora, superviseur


class TestMlock:
    def test_actif_par_defaut(self, monkeypatch):
        monkeypatch.delenv("AURA_MLOCK", raising=False)
        assert serveur_lora._mlock() is True

    def test_coupe_par_zero(self, monkeypatch):
        monkeypatch.setenv("AURA_MLOCK", "0")
        assert serveur_lora._mlock() is False

    def test_b11111_utilise_load_mode(self, monkeypatch):
        """b11111 a remplace --mlock par --load-mode (mlock/mmap+mlock)."""
        monkeypatch.delenv("AURA_MLOCK", raising=False)
        aide = "-lm, --load-mode MODE   - mlock: force ... - mmap+mlock: ..."
        assert serveur_lora._args_mlock(aide) == ["--load-mode",
                                                  "mmap+mlock"]

    def test_ancien_binaire_utilise_mlock(self, monkeypatch):
        monkeypatch.delenv("AURA_MLOCK", raising=False)
        assert serveur_lora._args_mlock("--mlock force ...") == ["--mlock"]

    def test_binaire_sans_option_ne_pose_rien(self, monkeypatch):
        """Option inconnue = boot mort : on ne devine jamais."""
        monkeypatch.delenv("AURA_MLOCK", raising=False)
        assert serveur_lora._args_mlock("--help sans verrou") == []

    def test_coupe_aucune_option(self, monkeypatch):
        monkeypatch.setenv("AURA_MLOCK", "0")
        assert serveur_lora._args_mlock("--load-mode") == []

    def test_la_cle_de_config_gouverne_les_deux_chemins(self):
        from aura import config
        assert config._CLES[("llama", "mlock")] == "AURA_MLOCK"


class TestDemon:
    def test_commande_supervisee_api_seule(self):
        cmd = superviseur.commander_demon()
        assert cmd[1:] == ["-m", "aura", "--superviser", "--api"]

    def test_api_deja_prete_ne_lance_rien(self, monkeypatch):
        monkeypatch.setattr(superviseur, "sante_ok", lambda: True)
        monkeypatch.setattr(superviseur.subprocess, "Popen",
                            lambda *a, **k: pytest.fail(
                                "une API vivante ne doit pas etre doublee"))
        assert superviseur.lancer_demon() == 0

    def test_detache_et_rend_la_main(self, monkeypatch, tmp_path):
        vus = {}

        class _P:
            pid = 4242

        def faux_popen(cmd, **kw):
            vus["cmd"] = cmd
            vus["kw"] = kw
            return _P()

        journal = tmp_path / "demon.log"
        monkeypatch.setattr(superviseur, "sante_ok", lambda: False)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: True)
        monkeypatch.setattr(superviseur.subprocess, "Popen", faux_popen)
        monkeypatch.setattr(superviseur, "JOURNAL_DEMON", journal)
        assert superviseur.lancer_demon() == 0
        assert vus["cmd"] == superviseur.commander_demon()
        assert vus["kw"]["cwd"] == str(superviseur._RACINE)
        assert vus["kw"]["env"]["AURA_API_PORT"] == str(superviseur.PORT)
        if hasattr(subprocess, "DETACHED_PROCESS"):
            assert vus["kw"]["creationflags"] & subprocess.DETACHED_PROCESS
        else:
            assert vus["kw"]["start_new_session"] is True
        assert journal.exists()

    def test_echec_de_sante_renvoie_un(self, monkeypatch, tmp_path):
        class _P:
            pid = 4242

        monkeypatch.setattr(superviseur, "sante_ok", lambda: False)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: False)
        monkeypatch.setattr(superviseur.subprocess, "Popen",
                            lambda *a, **k: _P())
        monkeypatch.setattr(superviseur, "JOURNAL_DEMON",
                            tmp_path / "demon.log")
        assert superviseur.lancer_demon() == 1

    def test_cli_demon_ne_bloque_pas(self, monkeypatch):
        from aura.__main__ import main as cli_main

        vus = []
        monkeypatch.setattr(superviseur, "lancer_demon",
                            lambda: vus.append("demon") or 0)
        monkeypatch.setattr(superviseur, "lancer",
                            lambda **k: pytest.fail(
                                "--demon ne doit pas bloquer au premier plan"))
        assert cli_main(["--superviser", "--demon"]) == 0
        assert vus == ["demon"]
