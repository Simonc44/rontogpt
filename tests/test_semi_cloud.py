"""Semi-cloud : RIEN ne quitte l'ordinateur sans un geste de l'utilisateur.

Le mode par defaut est `demander` (AURA_CLOUD absent) :
- le pipeline rend la reponse LOCALE et propose l'envoi (`repli_propose`) ;
- la route POST /v1/ask/cloud refuse sans `confirme: true` ;
- chaque reponse porte sa provenance, et `quitte_la_machine` n'est vrai que
  pour une reponse de Gemini.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from aura import gemini

# NB : `tests/` n'est pas un paquet, donc on ne peut pas importer les fixtures
# de test_aiguilleur.py : elles sont reecrites ici (memes noms, meme sens).
_VARIABLES = ("AURA_CLOUD", "AURA_GEMINI", "AURA_GEMINI_HISTORIQUE",
              "AURA_AIGUILLEUR_REGLES", "AURA_AIGUILLEUR_DOMAINES",
              "AURA_GEMINI_REGLES", "AURA_GEMINI_DOMAINES")


@pytest.fixture(autouse=True)
def _isole(monkeypatch, tmp_path):
    """Pas de cle, pas de `.env` reel : on part d'un etat connu."""
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
def sans_envoi_auto(monkeypatch):
    """Une cle, et le mode PAR DEFAUT du produit (AURA_CLOUD absent)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fausse-cle-test")
    monkeypatch.delenv("AURA_CLOUD", raising=False)


def _pipeline_factice(monkeypatch, reponse_locale, cerveau="gemma3-1b",
                      contexte_web="extrait web"):
    """RontoGPT sans charger le modele : seul l'aiguillage est reel."""
    from aura import cache_reponses, orchestrateur
    from aura.orchestrateur import RontoGPT

    ia = RontoGPT.__new__(RontoGPT)
    monkeypatch.setattr(cache_reponses, "cle_reponse", lambda q, X, y: None)
    monkeypatch.setattr(cache_reponses, "memoriser", lambda c, r: None)
    monkeypatch.setattr(
        RontoGPT, "_executer_detaille",
        lambda self, q, X=None, y=None, t0=None:
        {"reponse": reponse_locale, "cerveau_choisi": cerveau,
         "contexte_web": contexte_web})
    monkeypatch.setattr(orchestrateur, "_BUDGET_S", 0)
    ia._derniere_categorie = "general"
    return ia


@pytest.fixture
def appels(monkeypatch):
    vus: list[str] = []
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: vus.append(q) or "reponse Gemini")
    return vus


def test_defaut_ne_rien_envoyer_mais_proposer(sans_envoi_auto, appels,
                                              monkeypatch):
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert appels == [], "aucune question ne doit partir sans clic"
    assert r["reponse"] == ""
    assert r["repli_propose"]["fournisseur"] == "gemini"
    assert r["repli_propose"]["raison"] == "vide"
    assert "repli" not in r


def test_mode_non_ne_propose_meme_pas(sans_envoi_auto, appels, monkeypatch):
    monkeypatch.setenv("AURA_CLOUD", "non")
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert appels == [] and "repli_propose" not in r and "repli" not in r


def test_valeur_inconnue_retombe_sur_demander(sans_envoi_auto, appels,
                                              monkeypatch):
    monkeypatch.setenv("AURA_CLOUD", "peut-etre")
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert appels == [] and "repli_propose" in r


def test_mode_oui_garde_l_envoi_automatique(sans_envoi_auto, appels,
                                            monkeypatch):
    monkeypatch.setenv("AURA_CLOUD", "oui")
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert appels and r["reponse"] == "reponse Gemini"
    assert r["provenance"]["niveau"] == "cloud"
    assert r["provenance"]["quitte_la_machine"] is True


def test_reponse_locale_valide_n_est_pas_proposee(sans_envoi_auto, appels,
                                                  monkeypatch):
    ia = _pipeline_factice(monkeypatch, "Paris est la capitale de la France.")
    r = ia.executer_detaille("question de test")
    assert appels == [] and "repli_propose" not in r


def test_la_provenance_accompagne_chaque_reponse(sans_envoi_auto, monkeypatch):
    ia = _pipeline_factice(monkeypatch, "Paris est la capitale de la France.")
    r = ia.executer_detaille("question de test")
    assert r["provenance"]["niveau"] == "local"
    assert r["provenance"]["verifie"] is False
    assert r["provenance"]["quitte_la_machine"] is False


def test_un_tier_exact_est_badge_exact(sans_envoi_auto, monkeypatch):
    ia = _pipeline_factice(monkeypatch, "42", "calcul-verbal")
    r = ia.executer_detaille("combien font 6 fois 7 ?")
    assert r["provenance"]["niveau"] == "exact"
    assert r["provenance"]["verifie"] is True


# -- l'envoi EXPLICITE -----------------------------------------------------

def test_envoyer_a_gemini_explicite(sans_envoi_auto, appels, monkeypatch):
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    r = ia.envoyer_a_gemini("Quelle est la capitale du Chili ?")
    assert r["ok"] is True and r["reponse"] == "reponse Gemini"
    assert r["provenance"]["quitte_la_machine"] is True
    assert appels == ["Quelle est la capitale du Chili ?"]


def test_envoyer_a_gemini_sans_cible_dit_pourquoi(monkeypatch):
    # pas de cle : `_isole` (autouse) l'a retiree
    from aura.orchestrateur import RontoGPT
    ia = RontoGPT.__new__(RontoGPT)
    r = ia.envoyer_a_gemini("une question")
    assert r["ok"] is False and r["erreur"]


# -- la route HTTP ---------------------------------------------------------

@pytest.fixture
def serveur(monkeypatch):
    from aura import api

    class Faux:
        def envoyer_a_gemini(self, question):
            return {"ok": True, "reponse": "S", "question": question}

    etat = api._Etat()
    etat.ia = Faux()
    monkeypatch.delenv("AURA_API_CLE", raising=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api._creer_handler(etat))
    srv.daemon_threads = True
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _post(base, route, corps):
    req = urllib.request.Request(
        base + route, data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def test_route_cloud_refuse_sans_confirmation(serveur):
    code, rep = _post(serveur, "/v1/ask/cloud", {"question": "q"})
    assert code == 400 and rep["ok"] is False
    assert "confirm" in rep["erreur"]


def test_route_cloud_refuse_une_confirmation_non_booleenne(serveur):
    # « true » en chaine ou 1 ne valent pas un clic
    for faux in ("true", 1, "oui"):
        code, _ = _post(serveur, "/v1/ask/cloud",
                        {"question": "q", "confirme": faux})
        assert code == 400


def test_route_cloud_exige_une_question(serveur):
    code, _ = _post(serveur, "/v1/ask/cloud", {"confirme": True})
    assert code == 400


def test_route_cloud_envoie_apres_confirmation(serveur):
    code, rep = _post(serveur, "/v1/ask/cloud",
                      {"question": "q", "confirme": True})
    assert code == 200 and rep["ok"] is True and rep["reponse"] == "S"
