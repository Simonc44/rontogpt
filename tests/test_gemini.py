"""Appel Gemini (API Google, HTTP direct) : aucun appel reseau, urllib mocke.

Le JUGEMENT (quand escalader) vit dans aura/aiguilleur.py et se teste dans
tests/test_aiguilleur.py. Ici on ne teste que l'APPEL : resolution de la cle,
corps de requete, extraction du texte, echecs non bloquants et pause.
"""
import io
import json
import logging
import urllib.error

import pytest

from aura import gemini

_VARIABLES = ("AURA_GEMINI", "AURA_GEMINI_HISTORIQUE", "AURA_GEMINI_MODELE",
              "AURA_GEMINI_TIMEOUT", "AURA_GEMINI_MAX_TOKENS",
              "AURA_GEMINI_REFLEXION")


@pytest.fixture(autouse=True)
def _isole(monkeypatch, tmp_path):
    """Pas de cle, pas de `.env` reel : l'etat de depart est connu.

    La fixture `_gemini_coupe` du conftest coupe la cible dans toute la suite ;
    ici on la re-active et on neutralise le `.env` de la racine en pointant
    `_RACINE` sur un dossier temporaire vide.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for v in _VARIABLES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(gemini, "_RACINE", tmp_path)
    monkeypatch.setenv("AURA_GEMINI", "1")
    gemini.reinitialiser()
    yield
    gemini.reinitialiser()


@pytest.fixture
def avec_cle(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fausse-cle-test")


Q = "Quelle est la capitale du Chili ?"


# -- configuration et disponibilite -------------------------------------------

def test_sans_cle_jamais_d_appel():
    assert gemini.configure() is False
    assert gemini.disponible() is False
    assert gemini.repondre("bonjour") is None


def test_interrupteur_general(avec_cle, monkeypatch):
    assert gemini.disponible() is True
    monkeypatch.setenv("AURA_GEMINI", "0")
    assert gemini.configure() is False
    assert gemini.disponible() is False
    assert gemini.repondre("bonjour") is None


def test_cle_lue_dans_le_env(tmp_path):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=cle-env\n", encoding="utf-8")
    assert gemini.configure() is True
    assert gemini._cle() == "cle-env"


def test_la_cle_de_l_environnement_prime_sur_le_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=cle-env\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "cle-shell")
    assert gemini._cle() == "cle-shell"


# -- l'appel -------------------------------------------------------------------

def _faux_urlopen(vus, texte="  Santiago.  "):
    def faux(req, timeout):
        vus["url"] = req.full_url
        vus["cle"] = req.get_header("X-goog-api-key")
        vus["corps"] = json.loads(req.data.decode("utf-8"))
        vus["timeout"] = timeout
        corps = json.dumps({"candidates": [{"content": {"parts": [{"text": texte}]}}]})
        return io.BytesIO(corps.encode("utf-8"))
    return faux


def test_appel_mocke_renvoie_le_texte(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    assert gemini.repondre(Q) == "Santiago."
    assert vus["url"] == (f"{gemini.URL_BASE}/{gemini.MODELE_DEFAUT}"
                          ":generateContent")
    assert vus["cle"] == "fausse-cle-test"
    assert vus["timeout"] == 45
    corps = vus["corps"]
    assert corps["generationConfig"]["maxOutputTokens"] == 4096
    assert corps["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert corps["systemInstruction"]["parts"][0]["text"]
    assert corps["contents"][-1] == {"role": "user", "parts": [{"text": Q}]}


def test_modele_et_budget_configurables(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setenv("AURA_GEMINI_MODELE", "gemini-2.0-flash")
    monkeypatch.setenv("AURA_GEMINI_MAX_TOKENS", "512")
    monkeypatch.setenv("AURA_GEMINI_TIMEOUT", "5")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre(Q)
    assert "gemini-2.0-flash:generateContent" in vus["url"]
    assert vus["corps"]["generationConfig"]["maxOutputTokens"] == 512
    # la reflexion n'existe que sur la famille 2.5
    assert "thinkingConfig" not in vus["corps"]["generationConfig"]
    assert vus["timeout"] == 5


def test_prompt_et_faits_personnalises(avec_cle, monkeypatch):
    """Une redaction fournit son prompt et ses faits verifies a Gemini."""
    vus = {}
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre(Q, contexte_web="extrait", contexte_faits="Fait verifie.",
                    systeme="PROMPT REDACTION", max_tokens=777)
    corps = vus["corps"]
    assert corps["systemInstruction"]["parts"][0]["text"] == "PROMPT REDACTION"
    assert corps["generationConfig"]["maxOutputTokens"] == 777
    contenu = corps["contents"][-1]["parts"][0]["text"]
    assert "FAITS VERIFIES" in contenu and "Fait verifie." in contenu


def test_valeur_numerique_invalide_retombe_sur_le_defaut(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setenv("AURA_GEMINI_TIMEOUT", "beaucoup")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre(Q)
    assert vus["timeout"] == 45


def test_la_reflexion_n_est_jamais_livree(avec_cle, monkeypatch):
    corps = json.dumps({"candidates": [{"content": {"parts": [
        {"text": "je reflechis", "thought": True}, {"text": "Santiago."}]}}]})
    monkeypatch.setattr(gemini.urllib.request, "urlopen",
                        lambda req, timeout: io.BytesIO(corps.encode("utf-8")))
    assert gemini.repondre(Q) == "Santiago."


@pytest.mark.parametrize("donnees", [
    {"candidates": []},
    {"candidates": [{"content": {"parts": []}}]},
    {"candidates": [{"content": {"parts": [{"text": "   "}]},
                     "finishReason": "MAX_TOKENS"}]},
    {"promptFeedback": {"blockReason": "SAFETY"}},
])
def test_reponse_sans_texte_renvoie_none(avec_cle, monkeypatch, donnees):
    corps = json.dumps(donnees)
    monkeypatch.setattr(gemini.urllib.request, "urlopen",
                        lambda req, timeout: io.BytesIO(corps.encode("utf-8")))
    assert gemini.repondre(Q) is None
    assert gemini.derniere_erreur()


def test_extraits_web_joints_et_bornes(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre(Q, contexte_web="Santiago : capitale. " + "x" * 5000)
    contenu = vus["corps"]["contents"][-1]["parts"][0]["text"]
    assert contenu.startswith(Q)
    assert "EXTRAITS WEB" in contenu and "Santiago : capitale." in contenu
    assert len(contenu) < len(Q) + 3200


def test_l_historique_ne_part_que_pour_une_question_contextuelle(avec_cle):
    hist = [{"role": "user", "content": "je m'appelle Simon"},
            {"role": "assistant", "content": "Enchante Simon"}]
    long_q = "Donne-moi trois idees de repas equilibres pour toute la semaine"
    contenus = gemini._contenus(long_q, "", hist)
    assert [c["role"] for c in contenus] == ["user"]


def test_l_historique_part_pour_une_relance(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_GEMINI_HISTORIQUE", "2")
    hist = [{"role": "user", "content": "je parle de Paris"},
            {"role": "assistant", "content": "Paris est une ville."}]
    contenus = gemini._contenus("et sa population ?", "", hist)
    assert [c["role"] for c in contenus] == ["user", "model", "user"]
    assert contenus[0]["parts"][0]["text"] == "je parle de Paris"


# -- echecs : jamais bloquants, jamais bavards sur la cle ----------------------

def test_erreur_reseau_non_bloquante_et_cle_absente_des_logs(
        avec_cle, monkeypatch, caplog):
    def boom(req, timeout):
        raise OSError("x-goog-api-key: fausse-cle-test refuse")

    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    with caplog.at_level(logging.WARNING, logger="aura.gemini"):
        assert gemini.repondre(Q) is None
    assert gemini.derniere_erreur()
    assert "fausse-cle-test" not in caplog.text


def test_cle_refusee_met_l_appel_en_pause(avec_cle, monkeypatch):
    def boom(req, timeout):
        corps = json.dumps({"error": {"message": "API key not valid"}}).encode()
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(corps))

    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    assert gemini.repondre(Q) is None
    assert "cle API refusee" in gemini.derniere_erreur()
    # la pause empeche de rappeler Google a chaque question
    assert gemini.disponible() is False
    gemini.reinitialiser()
    assert gemini.disponible() is True
