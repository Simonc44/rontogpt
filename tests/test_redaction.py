"""Redaction longue : trois blocs, pas une continuation.

Le 1B ne sait pas « poursuivre » un texte : quand on lui redonne ce qui est
deja ecrit, il le commente (« Le texte fourni : que faut-il faire ? »). La
redaction est donc decoupee en introduction / developpement / conclusion,
chacune repartant de la question nue. Ces tests verrouillent les trois appels,
l'absence d'historique et l'assemblage.
"""
from aura import gemini
from aura import orchestrateur as orch
from aura.detection_demande import _SYSTEME_REDACTION
from aura.synthese_rontogpt import _SYSTEME_REDACTION_GEMINI


class TestRedactionTroisBlocs:
    def test_trois_appels_assembles(self, monkeypatch):
        ia = orch.RontoGPT()
        vus = []

        def faux(question, *a, **k):
            vus.append((question, k))
            return f"Bloc {len(vus)} " + ("contenu utile " * 40)

        monkeypatch.setattr(ia, "_llama", faux)
        texte = ia._generer("Faut-il encadrer la technologie ?", "", "",
                            redaction=True)
        assert len(vus) == 3
        assert texte.count("Bloc") == 3
        assert "introduction" in vus[0][0].lower()
        assert "développement" in vus[1][0].lower()
        assert "conclusion" in vus[2][0].lower()

    def test_pas_d_historique_ni_de_plan(self, monkeypatch):
        """L'historique persistant et le plan ont deja contamine une redaction."""
        ia = orch.RontoGPT()
        vus = []

        def faux(question, *a, **k):
            vus.append((question, k))
            return "Bloc " + ("contenu utile " * 40)

        monkeypatch.setattr(ia, "_llama", faux)
        ia._generer("Question\n\nPLAN A SUIVRE :\n1. ecrire du code",
                    "", "", redaction=True)
        for question, kw in vus:
            assert "PLAN A SUIVRE" not in question
            assert kw.get("historique") == []
            assert str(kw.get("systeme")).startswith(_SYSTEME_REDACTION)

    def test_serveur_muet_replie_sur_le_chemin_normal(self, monkeypatch):
        ia = orch.RontoGPT()
        appels = []

        def faux(question, *a, **k):
            appels.append(question)
            # Les trois blocs echouent (serveur muet), le repli normal repond.
            return "" if len(appels) <= 3 else "repli"

        monkeypatch.setattr(ia, "_llama", faux)
        assert ia._generer("Question", "", "", redaction=True) == "repli"
        assert len(appels) == 4        # 3 blocs vides + 1 appel normal


class TestRedactionGemini:
    """Option opt-in : une dissertation part chez Gemini en UN appel."""

    @staticmethod
    def _ia(monkeypatch, reponse_gemini):
        ia = orch.RontoGPT()
        appels_gemini = []

        def faux_gemini(question, **kw):
            appels_gemini.append((question, kw))
            return reponse_gemini

        monkeypatch.setattr(gemini, "disponible", lambda: True)
        monkeypatch.setattr(gemini, "repondre", faux_gemini)
        locaux = []

        def faux_local(question, *a, **k):
            locaux.append((question, k))
            return "Bloc " + ("contenu utile " * 40)

        monkeypatch.setattr(ia, "_llama", faux_local)
        return ia, appels_gemini, locaux

    def test_option_active_envoie_a_gemini(self, monkeypatch):
        monkeypatch.setenv("AURA_GEMINI_REDACTION", "1")
        monkeypatch.setenv("AURA_CLOUD", "demander")
        ia, appels, locaux = self._ia(
            monkeypatch, "Introduction.\n\nDéveloppement.\n\nConclusion.")
        texte = ia._generer("Rédige une dissertation sur la liberté",
                            "extrait web", "", redaction=True)
        assert texte.startswith("Introduction")
        assert len(appels) == 1 and not locaux
        question, kw = appels[0]
        assert question.startswith("Rédige une dissertation")
        assert kw["historique"] == []
        assert kw["contexte_web"] == "extrait web"
        assert kw["systeme"] == _SYSTEME_REDACTION_GEMINI
        assert ia._redaction_gemini is True
        assert ia._etiquette_cerveau() == "gemini"

    def test_echec_gemini_replie_sur_les_trois_blocs(self, monkeypatch):
        monkeypatch.setenv("AURA_GEMINI_REDACTION", "1")
        monkeypatch.setenv("AURA_CLOUD", "demander")
        ia, appels, locaux = self._ia(monkeypatch, None)
        texte = ia._generer("Rédige une dissertation", "", "", redaction=True)
        assert len(appels) == 1 and len(locaux) == 3
        assert texte.count("Bloc") == 3
        assert ia._redaction_gemini is False

    def test_aura_cloud_non_bloque_l_envoi(self, monkeypatch):
        monkeypatch.setenv("AURA_GEMINI_REDACTION", "1")
        monkeypatch.setenv("AURA_CLOUD", "non")
        ia, appels, locaux = self._ia(monkeypatch, "Gemini")
        ia._generer("Rédige une dissertation", "", "", redaction=True)
        assert not appels and len(locaux) == 3

    def test_option_inactive_garde_le_local(self, monkeypatch):
        monkeypatch.setenv("AURA_GEMINI_REDACTION", "0")
        ia, appels, locaux = self._ia(monkeypatch, "Gemini")
        ia._generer("Rédige une dissertation", "", "", redaction=True)
        assert not appels and len(locaux) == 3

    def test_question_normale_n_appelle_pas_gemini(self, monkeypatch):
        monkeypatch.setenv("AURA_GEMINI_REDACTION", "1")
        monkeypatch.setenv("AURA_CLOUD", "demander")
        ia, appels, _ = self._ia(monkeypatch, "Gemini")
        ia._generer("Bonjour", "", "")
        assert not appels
