"""Detecteur de concepts (brique 1) : MiniLM -> concept -> bloc LEXIQUE.

Calibration (micro-eval 12 paires requete->concept, scripts/_probe) :
- descriptions COURTES (1 phrase) avec tournure d'exemple : 11/12 vs 7/12
  avec des definitions longues ;
- seuil 0.40 (hors-sujet ~0.15-0.35, concerne >= 0.43) ;
- garde-fou _MOTS_MIN = 7 : tous les faux positifs mesures (meteo, blague,
  calcul) font <= 6 mots.
"""
import os

import pytest

from aura import concepts, embeddings

# une question >= 7 mots, calibree 0.695 sur « determinisme-liberte »
_Q_CONCEPT = ("est ce que le libre arbitre existe si tout est "
              "determine par nos genes")


class TestCoupe:
    def test_actif_par_defaut(self):
        assert concepts.actif() is True

    def test_aura_concepts_zero_coupe_tout(self, monkeypatch):
        monkeypatch.setenv("AURA_CONCEPTS", "0")
        assert concepts.actif() is False
        assert concepts.detecter(_Q_CONCEPT) is None
        assert concepts.ancrage(_Q_CONCEPT) is None

    def test_seuil_lisible(self, monkeypatch):
        monkeypatch.setenv("AURA_CONCEPTS_SEUIL", "0.55")
        assert concepts.seuil() == 0.55
        monkeypatch.setenv("AURA_CONCEPTS_SEUIL", "pas un nombre")
        assert concepts.seuil() == concepts._SEUIL_DEFAUT


class TestGardeFous:
    def test_texte_vide(self):
        assert concepts.detecter("") is None
        assert concepts.detecter("   ") is None

    def test_question_courte_rejetee(self):
        # 2 mots < _MOTS_MIN : bloque AVANT tout encodage, meme si le
        # mot « libre arbitre » pointe vers un concept
        assert concepts.detecter("libre arbitre ?") is None

    def test_encodeur_indisponible_sans_erreur(self, monkeypatch):
        # modele absent / coupe : inerte, jamais d'exception
        monkeypatch.setattr(embeddings, "encoder", lambda textes: None)
        monkeypatch.setattr(embeddings, "encoder1", lambda texte: None)
        assert concepts.detecter(_Q_CONCEPT) is None
        assert concepts.ancrage(_Q_CONCEPT) is None


_moder = pytest.mark.skipif(
    not embeddings.modele_present(),
    reason="modeles/minilm absent : scripts/telecharger_minilm.py")


@_moder
class TestDetection:
    PAIRES = [
        (_Q_CONCEPT, "determinisme-liberte"),
        ("si tous les A sont B et que x est A alors x est B est ce valide",
         "syllogisme"),
        ("deux variables bougent ensemble est ce que l une cause l autre",
         "causalite-vs-correlation"),
        ("est ce que je devrais toujours maximiser le bonheur du plus grand "
         "nombre meme si ca coute une personne", "utilitarisme"),
    ]

    @pytest.mark.parametrize("question,attendu", PAIRES)
    def test_concept_reconnu(self, question, attendu):
        hit = concepts.detecter(question)
        assert hit is not None, question
        assert hit["concept"] == attendu
        assert hit["score"] >= concepts.seuil()

    @pytest.mark.parametrize("question", [
        "calcule 2 puissance 10 s il te plait",
        "comment faire du pain maison simplement",
    ])
    def test_hors_sujet_rejetes(self, question):
        assert concepts.detecter(question) is None

    def test_ancrage_format(self):
        bloc = concepts.ancrage(_Q_CONCEPT)
        assert bloc is not None
        assert bloc.startswith("LEXIQUE")
        assert "determinisme-liberte" in bloc
        # le LEXIQUE est un ajout prudent, jamais un ordre dogmatique
        assert "sans forcer le hors-sujet" in bloc


@_moder
class TestMatriceMemoiree:
    def test_construite_puis_reutilisee(self, monkeypatch):
        monkeypatch.setattr(concepts, "_cache_matrice", None)
        monkeypatch.setattr(concepts, "_cache_cle", None)
        monkeypatch.setattr(concepts, "_cache_noms", None)

        concepts.detecter(_Q_CONCEPT)
        assert concepts._cache_matrice is not None
        assert concepts._cache_cle == tuple(concepts._CONCEPTS.items())
        m = concepts._cache_matrice
        # 2e detection : la matrice des 30 descriptions n'est pas recalculee
        concepts.detecter("deux variables bougent ensemble est ce que "
                          "l une cause l autre")
        assert concepts._cache_matrice is m


class TestConfigToml:
    """[minilm] / [concepts] / [flou] de aura.toml -> variables AURA_*."""

    def test_toml_pose_les_variables(self, tmp_path, monkeypatch):
        from aura import config
        for cle in ("AURA_CONCEPTS", "AURA_CONCEPTS_SEUIL",
                    "AURA_MINILM", "AURA_FLOU"):
            monkeypatch.delenv(cle, raising=False)
        avant = {k: v for k, v in os.environ.items()
                 if k.startswith("AURA_")}
        f = tmp_path / "aura.toml"
        f.write_text('[minilm]\nactif = false\n\n'
                     '[concepts]\nactif = false\nseuil = 0.55\n\n'
                     '[flou]\nactif = false\n', encoding="utf-8")
        try:
            config.appliquer(chemin=f)
            assert os.environ["AURA_MINILM"] == "0"
            assert os.environ["AURA_CONCEPTS"] == "0"
            assert os.environ["AURA_CONCEPTS_SEUIL"] == "0.55"
            assert os.environ["AURA_FLOU"] == "0"
        finally:
            # appliquer pose dans os.environ : on restitue l'environnement
            for k in [k for k in os.environ if k.startswith("AURA_")]:
                if k not in avant:
                    del os.environ[k]
            os.environ.update(avant)
