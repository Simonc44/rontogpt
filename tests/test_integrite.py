"""Scellement SHA-256 des fichiers charges : un fichier substitue est REFUSE.

Le point qui compte : `joblib.load` et `llama_cpp` executent du code a la
deserialisation. Le sceau ne protege pas d'un attaquant qui tourne deja sous
le compte (il reecrit le sceau), il refuse un fichier modifie ou remplace.
"""
import pytest

from aura import integrite


class TestSceau:
    def test_absent_puis_ok(self, tmp_path):
        fichier = tmp_path / "modele.bin"
        fichier.write_bytes(b"poids" * 100)
        assert integrite.verifier(fichier, strict=False) == "absent"
        sceau = integrite.sceller(fichier)
        assert sceau == tmp_path / "modele.bin.sha256"
        assert sceau.read_text(encoding="utf-8").startswith(
            integrite.empreinte(fichier))
        assert integrite.verifier(fichier) == "ok"

    def test_fichier_modifie_refuse(self, tmp_path):
        fichier = tmp_path / "modele.bin"
        fichier.write_bytes(b"a" * 16)
        integrite.sceller(fichier)
        fichier.write_bytes(b"b" * 16)
        with pytest.raises(integrite.IntegriteRompue, match="ne correspond plus"):
            integrite.verifier(fichier)

    def test_meme_taille_ne_suffit_pas(self, tmp_path):
        """Le remplacement le plus discret garde la taille : le sceau voit."""
        fichier = tmp_path / "modele.bin"
        fichier.write_bytes(b"\x00" * 1024)
        integrite.sceller(fichier)
        fichier.write_bytes(b"\x01" * 1024)
        with pytest.raises(integrite.IntegriteRompue):
            integrite.verifier(fichier)

    def test_strict_refuse_un_fichier_non_scelle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_INTEGRITE", "strict")
        fichier = tmp_path / "x.bin"
        fichier.write_bytes(b"x")
        with pytest.raises(integrite.IntegriteRompue, match="pas scelle"):
            integrite.verifier(fichier)

    def test_desactive_ne_verifie_rien(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_INTEGRITE", "0")
        fichier = tmp_path / "x.bin"
        fichier.write_bytes(b"x")
        integrite.sceller(fichier)
        fichier.write_bytes(b"y")
        assert integrite.verifier(fichier) == "desactive"

    def test_empreinte_lit_par_blocs(self, tmp_path, monkeypatch):
        """Un fichier plus gros qu'un bloc doit rendre le memeSHA-256."""
        fichier = tmp_path / "gros.bin"
        fichier.write_bytes(b"z" * (integrite.BLOC * 2 + 7))
        import hashlib
        attendu = hashlib.sha256(fichier.read_bytes()).hexdigest()
        assert integrite.empreinte(fichier) == attendu


class TestJoblib:
    def test_charge_un_fichier_scelle(self, tmp_path):
        joblib = pytest.importorskip("joblib")
        fichier = tmp_path / "routeur.joblib"
        joblib.dump({"classes": ["math"]}, fichier)
        integrite.sceller(fichier)
        assert integrite.charger_joblib(fichier) == {"classes": ["math"]}

    def test_refuse_un_fichier_altere(self, tmp_path):
        joblib = pytest.importorskip("joblib")
        fichier = tmp_path / "routeur.joblib"
        joblib.dump({"a": 1}, fichier)
        integrite.sceller(fichier)
        fichier.write_bytes(fichier.read_bytes() + b"\x00")
        with pytest.raises(integrite.IntegriteRompue):
            integrite.charger_joblib(fichier)


class TestLigneDeCommande:
    def test_sceller_puis_verifier(self, tmp_path, capsys):
        fichier = tmp_path / "m.bin"
        fichier.write_bytes(b"abc")
        assert integrite.main(["--sceller", str(fichier)]) == 0
        assert integrite.main(["--verifier", str(fichier)]) == 0
        assert "ok" in capsys.readouterr().out

    def test_verifier_signale_un_fichier_rompu(self, tmp_path, capsys):
        fichier = tmp_path / "m.bin"
        fichier.write_bytes(b"abc")
        integrite.sceller(fichier)
        fichier.write_bytes(b"abd")
        assert integrite.main(["--verifier", str(fichier)]) == 1
        assert "ROMPU" in capsys.readouterr().err
