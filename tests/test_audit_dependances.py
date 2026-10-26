"""Audit des dependances : annonce un venv etranger au verrou, epingle pip-audit.

Le script existe parce que `pip-audit --no-deps -r` ne passe pas sur Windows
pour ce projet (voir l'en-tete de `scripts/audit_dependances.py` : empreinte
manquante pour `llama-cpp-python`, puis chemin trop long dans son sdist). Ce
qui est verrouille ici :

  - un venv ABSENT est refuse avec le chemin regarde (jamais d'audit vert sur
    un dossier qui n'existe pas) ;
  - un venv en retard est ANNONCE avant les avis, avec les deux versions : un
    venv reste en `urllib3 2.7.0` a deja annonce deux avis alors que le verrou
    figeait `2.8.0`, donc deux avis qui n'existaient pas ;
  - la commande lancee est celle de la CI : meme pip-audit epingle, meme avis
    ignore, et un `--path` calcule (jamais recopie).
"""
import sys
import types
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

import audit_dependances as ad  # noqa: E402


def _faux_venv(racine: Path) -> Path:
    """Un venv minimal : le fichier de l'interpreteur suffit a le reconnaitre."""
    p = racine / "Scripts" / "python.exe"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _audit_espionne(monkeypatch, code: int) -> list:
    """Remplace le lancement de pip-audit et rend la liste des commandes."""
    lancees: list = []

    def faux_run(commande, *_a, **_k):
        lancees.append(commande)
        return types.SimpleNamespace(returncode=code)

    monkeypatch.setattr(ad.subprocess, "run", faux_run)
    return lancees


def test_refuse_un_venv_absent(tmp_path, capsys):
    assert ad.main(["--venv", str(tmp_path / "inexistant")]) == 2
    err = capsys.readouterr().err
    assert "interpreteur" in err and "uv sync" in err


def test_un_venv_en_retard_est_annonce_avant_les_avis(tmp_path, capsys,
                                                      monkeypatch):
    _faux_venv(tmp_path / ".venv")
    monkeypatch.setattr(ad, "site_packages", lambda _p: tmp_path)
    monkeypatch.setattr(ad, "versions_verrou",
                        lambda _c: {"urllib3": {"2.8.0"}})
    monkeypatch.setattr(ad, "versions_installees",
                        lambda _sp: {"urllib3": "2.7.0", "pytest": "9.0.2"})
    lancees = _audit_espionne(monkeypatch, code=1)      # des avis -> code 1

    assert ad.main(["--venv", str(tmp_path / ".venv")]) == 1
    assert len(lancees) == 1, "l'audit a bien lieu malgre l'ecart"
    err = capsys.readouterr().err
    assert "urllib3" in err
    assert "2.7.0" in err and "2.8.0" in err, "les deux versions, pas une"
    assert "uv sync --frozen" in err


def test_la_commande_est_celle_de_la_ci(tmp_path, capsys, monkeypatch):
    _faux_venv(tmp_path / ".venv")
    monkeypatch.setattr(ad, "site_packages", lambda _p: tmp_path)
    monkeypatch.setattr(ad, "versions_verrou",
                        lambda _c: {"urllib3": {"2.8.0"}})
    monkeypatch.setattr(ad, "versions_installees",
                        lambda _sp: {"urllib3": "2.8.0"})
    lancees = _audit_espionne(monkeypatch, code=0)

    assert ad.main(["--venv", str(tmp_path / ".venv")]) == 0
    assert "conforme au verrou" in capsys.readouterr().out
    commande = lancees[0]
    assert commande[:4] == ["uvx", "--from", "pip-audit==2.10.1", "pip-audit"]
    # Un `--path` CALCULE : pip-audit sur un dossier inexistant rend un audit
    # vert sans avoir rien regarde.
    assert commande[commande.index("--path") + 1] == str(tmp_path)
    # TOUS les avis, dans l'ordre : `commande.index("--ignore-vuln")` ne rend
    # que la PREMIERE occurrence, donc une boucle dessus comparait chaque avis
    # au premier. Le test ne le voyait pas tant qu'il n'y avait qu'un avis.
    ignores = [commande[i + 1] for i, jeton in enumerate(commande)
               if jeton == "--ignore-vuln"]
    assert ignores == list(ad._AVIS_IGNORES)


def test_versions_verrou_garde_toutes_les_versions_d_un_nom(tmp_path):
    """`numpy` figure deux fois dans le verrou (un jeu de marqueurs par
    entree) : retenir la derniere lue ferait un faux ecart sur une machine
    ou c'est l'autre qui est installee."""
    verrou = tmp_path / "uv.lock"
    verrou.write_text('version = 1\n\n'
                      '[[package]]\nname = "numpy"\nversion = "2.3.4"\n\n'
                      '[[package]]\nname = "numpy"\nversion = "2.4.0"\n\n'
                      '[[package]]\nname = "AltGraph"\nversion = "0.17.5"\n',
                      encoding="utf-8")
    versions = ad.versions_verrou(verrou)
    assert versions["numpy"] == {"2.3.4", "2.4.0"}
    assert versions["altgraph"] == {"0.17.5"}, "PEP 503 : AltGraph == altgraph"


def test_seuls_les_paquets_presents_des_deux_cotes_sont_compares():
    """Une entree du verrou visant une autre version de Python n'est pas
    installee ici : ce n'est pas un ecart, c'est une entree inactive."""
    figees = {"numpy": {"2.3.4", "2.4.0"}, "scipy": {"1.17.0"}}
    assert ad.ecarts(figees, {"numpy": "2.4.0", "inconnu": "1.0"}) == []
    assert ad.ecarts(figees, {"numpy": "1.0"}) == [
        ("numpy", ["2.3.4", "2.4.0"], "1.0")]
