"""Amplification du dataset general : reformulations sures, pas de bruit.

Le point critique : une enveloppe ne doit jamais empiler un prefixe deja pose
(« Dis-moi : Dis-moi : … ») et ne doit pas toucher au coeur de la question.
Les transformations de sens (inversion, voix passive, synonymes du contenu)
sont volontairement absentes : elles rendraient la reponse fausse.
"""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

import amplifier_general as amp  # noqa: E402


class TestNettoyer:
    def test_prefixes_empiles(self):
        assert amp.nettoyer(
            "Dis-moi : Peux-tu m'expliquer : a quoi sert un four ?") == \
            "a quoi sert un four ?"

    def test_ponctuation_finale_ajoutee(self):
        assert amp.nettoyer("quelle est la capitale de la France") == \
            "quelle est la capitale de la France ?"

    def test_question_propre_inchangee(self):
        assert amp.nettoyer("Quelle est la capitale ?") == \
            "Quelle est la capitale ?"

    def test_vide(self):
        assert amp.nettoyer("   ") == ""


class TestVariantes:
    def test_distinctes_et_bornees(self):
        v = amp.variantes("a quoi sert un four ?", 4)
        assert len(v) == 4
        assert len({x.casefold() for x in v}) == 4

    def test_le_coeur_est_preserve(self):
        for variante in amp.variantes("a quoi sert un four ?", 8):
            assert "a quoi sert un four" in variante

    def test_une_seule_variante_demandee(self):
        assert amp.variantes("a quoi sert un four ?", 1) == \
            ["a quoi sert un four ?"]


class TestFiltre:
    def test_rejette_une_url(self):
        assert amp.lisible(
            "Quelle est la capitale de la France ?",
            "La source est ici : https://exemple.fr") is False

    def test_rejette_une_reponse_trop_courte(self):
        assert amp.lisible("Quelle est la capitale de la France ?",
                           "Paris.") is False

    def test_accepte_une_paire_propre(self):
        assert amp.lisible(
            "Quelle est la capitale de la France ?",
            "Paris est la capitale de la France depuis 987.") is True


class TestAmplifier:
    def test_ecrit_des_questions_uniques(self, tmp_path):
        source = tmp_path / "general.jsonl"
        source.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in [
            {"question": "a quoi sert un four ?",
             "reponse": "Un four sert à cuire les aliments."},
            {"question": "a quoi sert un four ?",          # doublon
             "reponse": "Un four sert à cuire les aliments."},
            {"question": "qui a ecrit les miserables ?",
             "reponse": "Victor Hugo a écrit Les Misérables en 1862."},
        ]) + "\n", encoding="utf-8")
        sortie = tmp_path / "general_amplifie.jsonl"
        rapport = amp.amplifier(source, sortie, maximum=50, par_question=3)
        lignes = [json.loads(ligne) for ligne in
                  sortie.read_text(encoding="utf-8").splitlines()]
        assert rapport["exemples"] == len(lignes) == 6
        assert len({e["question"] for e in lignes}) == 6
