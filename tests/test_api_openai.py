"""La compatibilite OpenAI, pour de vrai : /v1/models et /v1/chat/completions.

Ce que ces tests verrouillent :

- un client OpenAI (Open WebUI, editeur, SDK `openai`) demande d'abord
  `GET /v1/models` : la route repond dans son format, ou il refuse de se
  connecter ;
- `POST /v1/chat/completions` rend `chat.completion` et, avec `stream: true`,
  des `chat.completion.chunk` suivis de `data: [DONE]` ;
- le contenu envoye est la REPONSE, jamais la reflexion du modele ;
- une API protegee par `AURA_API_CLE` accepte `Authorization: Bearer <cle>`,
  le seul en-tete qu'un client OpenAI sait envoyer.

Aucun cerveau reel : une fausse IA publie des morceaux comme le vrai.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from aura import api as api_mod
from aura import llama_cerveau

CLE = "une-cle-de-32-caracteres-ok-000"


@pytest.fixture()
def serveur():
    from http.server import ThreadingHTTPServer
    etat = api_mod._Etat(smoke=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api_mod._creer_handler(etat))
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", etat
    srv.shutdown()
    fil.join(timeout=5)


def _appel(base, route, corps=None, entetes=None):
    donnees = None if corps is None else json.dumps(corps).encode()
    req = urllib.request.Request(
        base + route, data=donnees,
        headers={"Content-Type": "application/json", **(entetes or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


class _FauxIA:
    """Publie des morceaux, dont un de REFLEXION, comme le vrai cerveau."""

    def __init__(self, morceaux=("Bonjour", " !"), pensee=""):
        self.morceaux = morceaux
        self.pensee = pensee

    def executer_detaille(self, question):          # noqa: ARG002
        if self.pensee:
            llama_cerveau.publier_flux(self.pensee, llama_cerveau.PHASE_PENSEE)
        for m in self.morceaux:
            llama_cerveau.publier_flux(m, llama_cerveau.PHASE_REPONSE)
        return {"reponse": "".join(self.morceaux), "experts": {"test"},
                "cerveau_choisi": "faux",
                "provenance": {"niveau": "base", "badge": "Base vérifiée",
                               "detail": "test", "verifie": True}}


def _messages(texte="salut"):
    return {"messages": [{"role": "user", "content": texte}]}


class TestModeles:
    def test_liste_au_format_openai(self, serveur):
        base, _ = serveur
        code, corps = _appel(base, "/v1/models")
        j = json.loads(corps)
        assert code == 200 and j["object"] == "list"
        assert [m["id"] for m in j["data"]] == [api_mod.MODELE_OPENAI]
        assert j["data"][0]["object"] == "model"

    def test_le_nom_du_modele_est_stable(self):
        """Un nom qui change a chaque version casse le choix des clients."""
        assert api_mod.MODELE_OPENAI == "rontogpt-local"


class TestChatCompletions:
    def test_reponse_complete(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA()
        code, corps = _appel(base, "/v1/chat/completions", _messages())
        j = json.loads(corps)
        assert code == 200 and j["object"] == "chat.completion"
        assert j["choices"][0]["message"] == {"role": "assistant",
                                              "content": "Bonjour !"}
        assert j["choices"][0]["finish_reason"] == "stop"
        # La provenance voyage dans un champ a part : un client OpenAI
        # l'ignore, un client RontoGPT l'affiche.
        assert j["rontogpt"]["provenance"]["verifie"] is True

    def test_messages_absents_400(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA()
        code, _ = _appel(base, "/v1/chat/completions", {"model": "x"})
        assert code == 400

    def test_sans_message_utilisateur_400(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA()
        code, _ = _appel(base, "/v1/chat/completions", {
            "messages": [{"role": "system", "content": "tu es un assistant"}]})
        assert code == 400

    def test_sans_cerveau_503(self, serveur):
        base, _ = serveur
        code, _ = _appel(base, "/v1/chat/completions", _messages())
        assert code == 503

    def test_le_dernier_message_utilisateur_gagne(self, serveur):
        base, etat = serveur
        vues = []

        class _Espion(_FauxIA):
            def executer_detaille(self, question):
                vues.append(question)
                return super().executer_detaille(question)

        etat.ia = _Espion()
        _appel(base, "/v1/chat/completions", {"messages": [
            {"role": "user", "content": "premier"},
            {"role": "assistant", "content": "réponse"},
            {"role": "user", "content": "dernier"},
        ]})
        assert vues == ["dernier"]


class TestChatCompletionsFlux:
    def test_trames_chunk_puis_done(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA()
        code, corps = _appel(base, "/v1/chat/completions",
                             {**_messages(), "stream": True})
        assert code == 200
        trames = [ligne[6:] for ligne in corps.splitlines()
                  if ligne.startswith("data: ") and ligne != "data: [DONE]"]
        objets = [json.loads(t) for t in trames]
        assert all(o["object"] == "chat.completion.chunk" for o in objets)
        contenu = "".join(o["choices"][0]["delta"].get("content", "")
                          for o in objets)
        assert contenu == "Bonjour !"
        assert objets[-1]["choices"][0]["finish_reason"] == "stop"
        assert objets[-1]["rontogpt"]["provenance"]["verifie"] is True
        assert corps.rstrip().endswith("data: [DONE]")

    def test_un_tour_sans_morceau_livre_quand_meme_sa_reponse(self, serveur):
        """Un tour deterministe ne publie rien : le contenu doit partir quand meme.

        Mesure avant correction : « 17 x 23 » rendait `role`, `stop`,
        `[DONE]` — et un message VIDE. C'est-a-dire aucune des reponses
        exactes, celles qui font la valeur du produit.
        """
        base, etat = serveur

        class _SansFlux:
            def executer_detaille(self, question):      # noqa: ARG002
                return {"reponse": "391", "experts": {"calcul"},
                        "cerveau_choisi": "calcul", "provenance": None}

        etat.ia = _SansFlux()
        code, corps = _appel(base, "/v1/chat/completions",
                             {**_messages("17 x 23"), "stream": True})
        assert code == 200
        contenu = "".join(
            json.loads(ligne[6:])["choices"][0]["delta"].get("content", "")
            for ligne in corps.splitlines()
            if ligne.startswith("data: ") and ligne != "data: [DONE]")
        assert contenu == "391"

    def test_la_reflexion_n_est_pas_du_contenu(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA(pensee="JE REFLECHIS TRES FORT")
        code, corps = _appel(base, "/v1/chat/completions",
                             {**_messages(), "stream": True})
        assert code == 200
        assert "JE REFLECHIS" not in corps

    def test_le_verrou_est_rendu(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA()
        _appel(base, "/v1/chat/completions", {**_messages(), "stream": True})
        code, _ = _appel(base, "/v1/chat/completions",
                         {**_messages("encore"), "stream": True})
        assert code == 200


class TestCleEnPorteur:
    def test_bearer_accepte(self, serveur, monkeypatch):
        base, etat = serveur
        etat.ia = _FauxIA()
        monkeypatch.setenv("AURA_API_CLE", CLE)
        entete = {"Authorization": f"Bearer {CLE}"}
        code, _ = _appel(base, "/v1/models", entetes=entete)
        assert code == 200, "un client OpenAI n'envoie que cet en-tete"
        code, _ = _appel(base, "/v1/chat/completions", _messages(), entete)
        assert code == 200

    def test_bearer_faux_refuse(self, serveur, monkeypatch):
        base, _ = serveur
        monkeypatch.setenv("AURA_API_CLE", CLE)
        code, _ = _appel(base, "/v1/models",
                         entetes={"Authorization": "Bearer pas-la-bonne"})
        assert code == 401

    def test_l_entete_maison_reste_accepte(self, serveur, monkeypatch):
        base, _ = serveur
        monkeypatch.setenv("AURA_API_CLE", CLE)
        code, _ = _appel(base, "/v1/models", entetes={"X-RontoGPT-Cle": CLE})
        assert code == 200
