"""Disjoncteur anti-repetition + routage meta-IA.

Bug reference (production) : « es tu une ia aussi puissn qu'un 8B ou pluas »
— question fermee partie en redaction longue, texte recopie mot pour mot,
phrase revenue 3 fois, 153,3 s de generation. Le mode « dissertation » a
depuis ete supprime ; le DISJONCTEUR, lui, reste : c'est lui qui empeche
une boucle de sortir, quel que soit le chemin de generation pris.
"""
from aura import llama_cerveau as lc
from aura import orchestrateur as orch


PHRASE = ("La notion de puissance est souvent consideree comme une qualite "
          "majeure des systemes, des individus et des choses. Cependant, "
          "cette affirmation est loin d'etre veritablement applicable a "
          "toutes les situations.")

QUESTION_META = "es tu une ia aussi puissn qu'un 8B ou pluas"


# ── detection de boucle ────────────────────────────────────────────────

class TestDetectionBoucle:
    def test_phrase_recopiee_detectee(self):
        # le cas observe : la meme phrase longue apparait 2+ fois
        assert lc._en_boucle(PHRASE + " " + PHRASE) is True

    def test_phrase_unique_pas_de_faux_positif(self):
        assert lc._en_boucle(PHRASE) is False

    def test_texte_normal_sain(self):
        texte = ("Le solaire progresse en Europe. L'eolien reste dependant "
                 "du reseau. Une comparaison suppose de comparer les "
                 "couts marginaux, pas les subventions declarees. Chaque "
                 "technique repond a un profil de charge different.")
        assert lc._en_boucle(texte) is False

    def test_boucle_serree_de_4_mots(self):
        # meme sequence de 4 mots repetee 5 fois -> boucle
        mot = "il ne dispose pas de la meme capacite "
        assert lc._en_boucle(mot * 5) is True

    def test_texte_vide(self):
        assert lc._en_boucle("") is False
        assert lc._en_boucle(None) is False


# ── coupure / anti-copie ───────────────────────────────────────────────

class TestCoupureEtAntiCopie:
    def test_coupe_avant_la_deuxieme_occurrence(self):
        bloque = "Intro. " + PHRASE + " " + PHRASE + " suite."
        coupe = lc._couper_boucle(bloque)
        assert lc._en_boucle(coupe) is False
        assert coupe.count("La notion de puissance") == 1
        assert "suite." not in coupe

    def test_sans_doublons_deduplique_les_extraits(self):
        extraits = PHRASE + " " + PHRASE + " Un deuxieme extrait utile."
        propre = lc._sans_doublons(extraits)
        assert propre.count("La notion de puissance") == 1
        assert "deuxieme extrait utile" in propre

    def test_sans_doublons_garde_le_texte_isole(self):
        # une seule copie : rien a retirer, le texte sort intact
        assert lc._sans_doublons("Une phrase unique et suffisamment longue "
                                "pour depasser le seuil de detection.")


# ── generation streamée : coupure IMMEDIATE sur boucle ─────────────────

class TestGenerationAntiBoucle:
    """Le disjoncteur coupe PENDANT le flux, pas apres.

    `_streamer` est le seul point de passage de la generation in-process :
    y couper la boucle garantit que ni la reponse livree, ni les morceaux
    deja publies au flux ne partent en repetition.
    """

    class FauxLlmBoucle:
        """Rend une boucle DES LE stream (chaque chunk = la meme phrase)."""

        def __init__(self):
            self.appels = 0

        def create_chat_completion(self, **kw):
            self.appels += 1

            def flux():
                for _ in range(6):
                    yield {"choices": [{"delta": {"content": PHRASE + " "}}]}
            return flux()

    def test_coupure_des_la_deuxieme_recopie(self):
        llm = self.FauxLlmBoucle()
        texte, boucle = lc._streamer(
            llm, [{"role": "user", "content": "q"}], 190,
            {"temperature": 0.5, "repeat_penalty": 1.1})
        assert boucle is True            # la boucle est SIGNALEE
        assert lc._en_boucle(texte) is False   # et jamais livree telle quelle
        # le flux a ete abandonne bien avant les 6 recopies produites
        assert len(texte) < len(PHRASE) * 3

    class FauxLlmSain:
        def __init__(self):
            self.appels = 0

        def create_chat_completion(self, **kw):
            self.appels += 1

            def flux():
                for mot in ("Reponse ", "saine ", "et complete."):
                    yield {"choices": [{"delta": {"content": mot}}]}
            return flux()

    def test_sans_boucle_aucune_coupure(self):
        llm = self.FauxLlmSain()
        texte, boucle = lc._streamer(
            llm, [{"role": "user", "content": "q"}], 190,
            {"temperature": 0.5})
        assert boucle is False
        assert texte == "Reponse saine et complete."

    def test_profil_de_relance_plus_punitif(self):
        """La relance change VRAIMENT les parametres (sinon elle ne sert a rien)."""
        from aura import echantillonnage as ech
        normal = ech.pour_inprocess(ech.NORMAL)
        relance = ech.pour_inprocess(ech.RELANCE)
        assert relance["repeat_penalty"] > normal["repeat_penalty"]
        assert normal["temperature"] == relance["temperature"] == 1.0


# ── routage meta-IA (l'anti-mode-dissertation) ─────────────────────────

class TestRoutageMetaIA:
    def test_question_meta_detectee(self):
        assert orch._est_meta_ia(QUESTION_META) is True
        assert orch._est_meta_ia("Qui es-tu ?") is True
        assert orch._est_meta_ia("Tu es aussi intelligent que GPT-4 ?") is True
        assert orch._est_meta_ia("Es-tu un modèle 8B ?") is True

    def test_faux_positif_sur_un_essai(self):
        # une dissertation sur la puissance N'EST PAS une question meta
        assert orch._est_meta_ia(
            "La puissance de la technologie est-elle une menace pour "
            "l'homme moderne ?") is False
        assert orch._est_meta_ia(
            "Explique comment la photosynthese fonctionne en detail") is False

    def test_la_capacite_ne_vole_pas_une_dissertation(self):
        """« puissance » NU n'est pas une question sur le MODELE.

        Defaut mesure (scripts/bench_qualite.py, riche-puissance) : la
        reponse-constante d'identite interceptait une dissertation sur la
        puissance de la technologie, alors que `_est_meta_ia` la classe
        correctement comme NON meta. La possesion (« ta puissance ») reste
        reconnue.
        """
        from aura.regles_reponse import _RE_CAPACITE_MODELE

        dissertation = ("la puissance de la technologie est-elle une menace "
                        "pour l'homme moderne ? redige une dissertation "
                        "structuree")
        assert _RE_CAPACITE_MODELE.search(dissertation) is None
        assert _RE_CAPACITE_MODELE.search("quelle est ta puissance ?")

    def test_comparaison_directe_avec_un_assistant(self):
        """« Plus fort que ChatGPT ? » est une question sur le MODELE.

        Sans ce motif, elle partait au 1B : 7 a 34 s de digression, echec du
        banc meta-capacite (cible < 5 s). Elle doit au contraire recevoir la
        reponse-constante.
        """
        from aura.regles_reponse import _RE_CAPACITE_MODELE

        for question in ("Tu es plus fort que ChatGPT ?",
                         "Es-tu meilleur que GPT-4 ?",
                         "Tu es aussi bon que Claude ?"):
            assert _RE_CAPACITE_MODELE.search(question), question
        assert _RE_CAPACITE_MODELE.search(
            "Explique ce qu'est ChatGPT") is None

    def test_question_meta_longue_garde_un_budget_court(self):
        # C'ETAIT LE BUG : >= 9 mots -> redaction longue -> 153 s.
        # La garde qui reste garantit un BUDGET court a cette question.
        assert len(QUESTION_META.split()) >= 9
        assert orch.llama_cerveau._max_tokens_adaptatif(QUESTION_META) == 120

    def test_reponse_courte_sans_llm(self, monkeypatch):
        # la question meta courte part en constante : 0 s, zero generation
        def _explose(*a, **k):
            raise AssertionError("le LLM ne doit pas etre charge")

        monkeypatch.setattr(lc, "_charger", _explose)
        ia = orch.RontoGPT()
        r = ia.executer_detaille(QUESTION_META)
        assert r["cerveau_choisi"] == "constante-capacite"
        assert r["reponse"] == orch._CAPACITE
        assert "1B" in r["reponse"] and "8B" in r["reponse"]
