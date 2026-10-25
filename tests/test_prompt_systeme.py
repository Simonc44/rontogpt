"""Prompt système : défaut mesuré + variante française Gemini 3.

Le point verrouillé ici est le CHOIX, pas le contenu éditorial : une valeur
inconnue retombe sur le défaut (une faute de frappe dans aura.toml ne doit pas
changer le prompt en silence), et `fr` reste sélectionnable.
"""
import pytest

from aura import config, prompt_systeme


class TestChoix:
    def test_defaut_sans_variable(self, monkeypatch):
        monkeypatch.delenv("AURA_SYSTEME", raising=False)
        assert prompt_systeme.systeme() == prompt_systeme.SYSTEME_DEFAUT

    def test_variable_fr(self, monkeypatch):
        monkeypatch.setenv("AURA_SYSTEME", "fr")
        assert prompt_systeme.systeme() == prompt_systeme.SYSTEME_FR

    def test_argument_bat_la_variable(self, monkeypatch):
        monkeypatch.setenv("AURA_SYSTEME", "fr")
        assert prompt_systeme.systeme("defaut") == \
            prompt_systeme.SYSTEME_DEFAUT

    @pytest.mark.parametrize("valeur", ["", "francais", "inconnu"])
    def test_valeur_inconnue_retombe_sur_le_defaut(self, monkeypatch, valeur):
        monkeypatch.setenv("AURA_SYSTEME", valeur)
        assert prompt_systeme.systeme() == prompt_systeme.SYSTEME_DEFAUT

    def test_casse_et_espaces_ignores(self, monkeypatch):
        monkeypatch.setenv("AURA_SYSTEME", "  FR ")
        assert prompt_systeme.systeme() == prompt_systeme.SYSTEME_FR

    def test_la_cle_de_config_gouverne_le_prompt(self):
        assert config._CLES[("llama", "systeme")] == "AURA_SYSTEME"


class TestContenuFrancais:
    def test_structure_gemini3(self):
        p = prompt_systeme.SYSTEME_FR
        for balise in ("<role>", "</role>", "<priorites>", "<contraintes>",
                       "<format>", "<methode>"):
            assert balise in p

    def test_clauses_qui_comptent(self):
        p = prompt_systeme.SYSTEME_FR
        assert "2026" in p
        assert "français" in p
        assert "CONTEXTE" in p
        assert "N'invente" in p

    def test_ancrage_limite_au_contexte_fourni(self):
        """Sans contexte, le modèle doit pouvoir répondre — pas s'abstenir."""
        p = prompt_systeme.SYSTEME_FR.lower()
        assert "quand un contexte" in p

    def test_le_defaut_reste_l_anglais_court(self):
        """Le 1B recopie les prompts longs (mesure du 24 oct. 2026)."""
        assert prompt_systeme.SYSTEME_DEFAUT == "You are a helpful assistant."
