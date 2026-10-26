"""Portails devant le cache disque de llama.cpp (avis PYSEC-2026-2447).

Ce qui est verrouille ici :

- chaque portail REFUSE a lui seul (activation, coffre, marqueur, permissions) ;
- le garde laisse passer le cache memoire sans condition, et n'ouvre le cache
  disque que si tous les portails sont ouverts ;
- aucune autre ligne de `aura/` ne peut ouvrir le chemin (portail 0).
"""
import ast
import os
from pathlib import Path

import pytest

from aura import garde_cache_disque as garde

RACINE = Path(__file__).resolve().parent.parent
_POSIX = pytest.mark.skipif(not hasattr(os, "getuid"),
                            reason="modes POSIX absents (Windows)")


@pytest.fixture
def coffre(tmp_path):
    """Coffre prepare, sans variable d'environnement posee."""
    return garde.preparer(tmp_path / "coffre")


@pytest.fixture
def ouvert(monkeypatch, coffre):
    """Les quatre portails ouverts."""
    monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
    monkeypatch.setenv(garde.PORTE_COFFRE, str(coffre))
    return coffre


class TestPortails:
    def test_activation_fermee_par_defaut(self, monkeypatch, coffre):
        monkeypatch.delenv(garde.PORTE_ACTIVATION, raising=False)
        monkeypatch.setenv(garde.PORTE_COFFRE, str(coffre))
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "activation"

    @pytest.mark.parametrize("valeur", ["", "0", "oui", "true", "2"])
    def test_activation_stricte(self, monkeypatch, coffre, valeur):
        """Une valeur « presque 1 » ne doit pas ouvrir le portail."""
        monkeypatch.setenv(garde.PORTE_ACTIVATION, valeur)
        monkeypatch.setenv(garde.PORTE_COFFRE, str(coffre))
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "activation"

    def test_aucun_coffre_designe(self, monkeypatch):
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        monkeypatch.delenv(garde.PORTE_COFFRE, raising=False)
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "coffre"

    def test_coffre_relatif_refuse(self, monkeypatch, tmp_path):
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(garde.Refus) as e:
            garde.verifier("coffre")
        assert e.value.portail == "coffre"

    def test_coffre_inexistant_refuse(self, monkeypatch, tmp_path):
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        with pytest.raises(garde.Refus) as e:
            garde.verifier(tmp_path / "absent")
        assert e.value.portail == "coffre"

    def test_lien_symbolique_refuse(self, monkeypatch, coffre, tmp_path):
        lien = tmp_path / "lien"
        try:
            lien.symlink_to(coffre, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("creation de lien symbolique impossible ici")
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        with pytest.raises(garde.Refus) as e:
            garde.verifier(lien)
        assert e.value.portail == "coffre"

    def test_coffre_non_prepare_refuse(self, monkeypatch, tmp_path):
        brut = tmp_path / "brut"
        brut.mkdir()
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        with pytest.raises(garde.Refus) as e:
            garde.verifier(brut)
        assert e.value.portail == "marqueur"

    def test_coffre_deplace_refuse(self, monkeypatch, coffre, tmp_path):
        """Le marqueur porte le chemin : un coffre recopie ne passe plus."""
        (coffre / garde.MARQUEUR).write_text(str(tmp_path / "ailleurs") + "\n",
                                             encoding="utf-8")
        monkeypatch.setenv(garde.PORTE_ACTIVATION, "1")
        monkeypatch.setenv(garde.PORTE_COFFRE, str(coffre))
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "marqueur"

    @_POSIX
    def test_dossier_accessible_aux_autres_refuse(self, ouvert):
        os.chmod(ouvert, 0o755)
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "permissions"

    @_POSIX
    def test_marqueur_accessible_aux_autres_refuse(self, ouvert):
        os.chmod(ouvert / garde.MARQUEUR, 0o644)
        with pytest.raises(garde.Refus) as e:
            garde.verifier()
        assert e.value.portail == "permissions"

    @_POSIX
    def test_preparer_pose_les_modes(self, coffre):
        assert oct(ouvert_mode(coffre)) == "0o700"
        assert oct(ouvert_mode(coffre / garde.MARQUEUR)) == "0o600"

    def test_tous_les_portails_ouverts(self, ouvert):
        assert garde.verifier() == ouvert
        assert garde.autorise() is True

    def test_autorise_dit_non_sans_ouvrir(self, monkeypatch, coffre):
        monkeypatch.delenv(garde.PORTE_ACTIVATION, raising=False)
        assert garde.autorise(coffre) is False


def ouvert_mode(chemin: Path) -> int:
    return chemin.stat().st_mode & 0o777


class _CacheRam:
    """Meme nom de classe que le cache memoire de llama.cpp."""


_CacheRam.__name__ = "LlamaRAMCache"


class _CacheDisque:
    """Meme nom de classe que le cache disque de llama.cpp."""


_CacheDisque.__name__ = "LlamaDiskCache"


def _classe_factice():
    class FauxLlama:
        def __init__(self):
            self.vus = []

        def set_cache(self, cache):
            self.vus.append(cache)

    return FauxLlama


class TestGarde:
    def test_cache_memoire_passe_sans_portail(self, monkeypatch):
        monkeypatch.delenv(garde.PORTE_ACTIVATION, raising=False)
        classe = _classe_factice()
        assert garde.installer(classe) is True
        modele = classe()
        memoire = _CacheRam()
        modele.set_cache(memoire)
        assert modele.vus == [memoire]

    def test_cache_disque_refuse_sans_activation(self, monkeypatch):
        monkeypatch.delenv(garde.PORTE_ACTIVATION, raising=False)
        classe = _classe_factice()
        garde.installer(classe)
        modele = classe()
        with pytest.raises(garde.Refus) as e:
            modele.set_cache(_CacheDisque())
        assert e.value.portail == "activation"
        assert modele.vus == [], "le cache disque ne doit meme pas etre pose"

    def test_cache_disque_refuse_si_un_portail_est_ferme(self, ouvert,
                                                         monkeypatch):
        os.chmod(ouvert, 0o700)
        monkeypatch.setenv(garde.PORTE_COFFRE, str(ouvert / "absent"))
        classe = _classe_factice()
        garde.installer(classe)
        with pytest.raises(garde.Refus) as e:
            classe().set_cache(_CacheDisque())
        assert e.value.portail == "coffre"

    def test_cache_disque_ouvert_par_les_portails(self, ouvert):
        classe = _classe_factice()
        garde.installer(classe)
        modele = classe()
        disque = _CacheDisque()
        modele.set_cache(disque)
        assert modele.vus == [disque]

    def test_installer_est_idempotent(self):
        classe = _classe_factice()
        original = classe.set_cache
        garde.installer(classe)
        garde.installer(classe)
        assert classe.set_cache.__wrapped__ is original

    def test_installer_sans_set_cache_ne_leve_pas(self):
        """Un double de test incomplet ne doit pas casser le chargement."""
        class SansCache:
            pass

        assert garde.installer(SansCache) is False

    def test_installer_sans_llama_cpp_ne_leve_pas(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "llama_cpp", None)
        assert garde.installer() is False


class TestPortailZero:
    """Aucun autre module ne doit pouvoir ouvrir le chemin vulnerable."""

    def test_aucune_autre_ligne_de_aura_n_ouvre_le_cache_disque(self):
        coupables = {}
        for fichier in sorted((RACINE / "aura").glob("*.py")):
            if fichier.name == "garde_cache_disque.py":
                continue
            lignes = _references(fichier)
            if lignes:
                coupables[fichier.name] = lignes
        assert not coupables, (
            "le cache disque de llama.cpp n'est atteignable que derriere "
            f"aura/garde_cache_disque.py : {coupables}")


def _references(fichier: Path) -> list[int]:
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    lignes = []
    for noeud in ast.walk(arbre):
        nom = getattr(noeud, "attr", None) or getattr(noeud, "id", None)
        if nom in ("set_cache", "LlamaDiskCache"):
            lignes.append(noeud.lineno)
    return lignes
