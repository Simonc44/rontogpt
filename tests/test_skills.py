"""Skills ciblés : identité, algèbre linéaire, mémoire, consignes de prompt.

Chaque test vérifie un DÉFAUT RÉEL observé le 25 oct. 2026, pas une
abstraction : le 1B se présentait comme Gemma/Google, rendait 28 à
« 4x − 7 = 21 », et répondait « je n'ai pas accès à tes données » alors que
le prénom était en mémoire.
"""
import pytest

from aura import skills
from aura.memoire_conversation import MemoireConversation


def _reponse(question: str, **kw) -> str:
    resultat = skills.repondre(question, **kw)
    return resultat[1] if resultat is not None else ""


class TestIdentite:
    def test_identite_nomme_les_deux_createurs(self):
        assert "RontoGPT" in skills.IDENTITE
        assert "Gemma" in skills.IDENTITE
        assert "Google DeepMind" in skills.IDENTITE
        assert "Simon Chusseau" in skills.IDENTITE
        assert "Je ne suis pas Gemini" in skills.IDENTITE

    def test_createur_du_systeme(self):
        r = _reponse("Qui est ton créateur ?")
        assert "Simon Chusseau" in r and "Google DeepMind" in r
        assert "RontoGPT n'est pas un produit de Google" in r

    def test_createur_de_gemma(self):
        r = _reponse("Qui a créé Gemma ?")
        assert r.startswith(
            "Le créateur du modèle Gemma est Google DeepMind.")

    def test_createur_de_rontogpt(self):
        r = _reponse("Qui a créé RontoGPT ?")
        assert r.startswith(
            "Le créateur du système RontoGPT est Simon Chusseau.")

    def test_themes_d_identite(self):
        assert "pipeline" in _reponse("Comment tu fonctionnes ?")
        assert "solveurs" in _reponse("Quels sont tes solveurs ?")
        assert "petit modèle" in _reponse("Quelles sont tes limites ?")
        assert "reste sur ta machine" in _reponse(
            "Mes données vont sur le cloud ?")
        assert "français" in _reponse("Tu parles quelles langues ?")
        assert "GGUF" in _reponse("Quel matériel utilises-tu ?")

    def test_banque_de_cent_phrases(self):
        fiches = skills.charger_identite()
        phrases = [p for fiche in fiches for p in fiche.phrases]
        assert len(fiches) == 10
        assert len(phrases) == 100
        assert len(set(phrases)) == 100
        assert all(p.endswith(".") for p in phrases)

    def test_nom_de_domaine_n_est_pas_l_identite(self):
        assert skills.repondre("Quel est ton nom de domaine ?") is None

    def test_detail_plus_long_que_la_reponse_courte(self):
        court = _reponse("Comment tu fonctionnes ?")
        long = _reponse("Explique-moi ton architecture en détail")
        assert len(long) > len(court)

    def test_qui_es_tu_repond_par_le_skill(self):
        assert _reponse("qui es-tu ?") == skills.IDENTITE

    def test_variantes_de_presentation(self):
        # Même thème, formulation différente : « présente-toi » ne doit pas
        # être le seul chemin vers l'identité.
        assert _reponse("Peux-tu te présenter ?") == skills.IDENTITE
        assert _reponse("Pourrais-tu te présenter en une phrase ?") == \
            skills.IDENTITE

    def test_identite_personnelle_non_capturee(self):
        # « Tu te souviens de mon prénom ? » doit rester au skill mémoire
        # (qui lit l'historique), pas à la fiche « mémoire » de l'identité.
        assert skills.repondre("Tu te souviens de mon prénom ?") is None

    def test_production_ecarte_l_identite(self):
        # « Écris un dialogue où tu es une IA » n'est pas une question
        # sur l'assistant : elle doit aller au cerveau.
        assert skills.repondre("Écris un dialogue où tu es une IA",
                               production=True) is None


class TestAlgebre:
    @pytest.mark.parametrize("question,attendu", [
        ("Résous 4x − 7 = 21", "x = 7"),
        ("2x + 3 = 11", "x = 4"),
        ("x/2 = 5", "x = 10"),
        ("3x + 1 = 2x + 3", "x = 2"),
        ("-2x = 8", "x = -4"),
        ("4,5x = 9", "x = 2"),
    ])
    def test_equations_lineaires(self, question, attendu):
        assert attendu in _reponse(question)

    def test_equation_impossible(self):
        assert "Aucune solution" in _reponse("x + 1 = x + 2")

    def test_ne_touche_pas_a_l_arithmetique_sans_x(self):
        assert skills.repondre("Combien font 2 * 3 * 4 ?") is None

    def test_equation_nue(self):
        assert "x = 7" in _reponse("4x - 7 = 21")

    def test_affectation_python_n_est_pas_une_equation(self):
        assert skills.repondre("Que fait x = 5 en Python ?") is None

    def test_s_abstient_hors_lineaire(self):
        assert skills.repondre("Résous x² − 1 = 0") is None
        assert skills.repondre("Résous 2(x + 1) = 6") is None
        assert skills.repondre("Résous xy = 6") is None


class TestPgcd:
    @pytest.mark.parametrize("question,attendu", [
        ("Quel est le PGCD de 84 et 120 ?", "PGCD(84, 120) = 12"),
        ("pgcd de 17 et 5", "PGCD(17, 5) = 1"),
        ("PGCD de 12, 18 et 30", "PGCD(12, 18, 30) = 6"),
        ("Quel est le PPCM de 4 et 6 ?", "PPCM(4, 6) = 12"),
    ])
    def test_pgcd_et_ppcm_exacts(self, question, attendu):
        assert attendu in _reponse(question)

    def test_ne_repete_pas_le_nombre_faux_du_modele(self):
        # Défaut mesuré : « PGCD de 84 et 126 » alors que la question disait
        # 120. Le calcul ne doit citer que les nombres posés.
        r = _reponse("Quel est le PGCD de 84 et 120 ?")
        assert "126" not in r and "84 = 12 × 7" in r

    def test_sans_nombres_ne_devine_pas(self):
        assert skills.repondre("Quel est le PGCD ?") is None

    def test_arithmetique_sans_pgcd_intacte(self):
        assert skills.repondre("Combien font 17 × 23 ?") is None


class TestHonnetete:
    def test_capitale_de_la_lune_refusee(self):
        # Défaut mesuré : « Touhou Wiki » présenté comme la capitale de la Lune.
        r = _reponse("Quelle est la capitale de la Lune ?")
        assert "Aucune" in r and "Touhou" not in r
        assert "la Lune n'est pas un pays" in r

    def test_autres_astres(self):
        assert "Mars n'est pas un pays" in _reponse(
            "Quelle est la capitale de Mars ?")
        assert "le Soleil n'est pas un pays" in _reponse(
            "Et la capitale du Soleil ?")

    def test_une_vraie_capitale_n_est_pas_capturee(self):
        assert skills.repondre("Quelle est la capitale de la France ?") is None


class TestReconsolidation:
    def test_les_reponses_exactes_remplacent_le_cache(self):
        for sid in ("identite", "algebre", "pgcd", "honnetete"):
            assert skills.doit_reconsolider(sid), sid

    def test_la_memoire_personnelle_ne_va_pas_au_cache(self):
        assert not skills.doit_reconsolider("memoire")
        assert not skills.doit_reconsolider("resume")

    def test_ancienne_hallucination_remplacee_par_l_identite(self):
        """Le cache ne doit pas continuer à servir « Gemma/Google »."""
        from aura import filtre_instantane
        from aura.orchestrateur import RontoGPT
        filtre_instantane.enregistrer(
            "qui es tu",
            "Je suis Gemma, un grand modèle linguistique formé par Google "
            "DeepMind.")
        assert "Gemma" in filtre_instantane.repondre_cache("qui es tu")
        RontoGPT().executer_detaille("Qui es-tu ?")
        assert filtre_instantane.repondre_cache("qui es tu") == skills.IDENTITE


class TestMemoire:
    @pytest.fixture()
    def memoire(self, tmp_path):
        m = MemoireConversation(chemin=tmp_path / "conv.jsonl")
        m.ajouter("user", "Je m'appelle Pierre et je travaille sur un projet "
                          "de radio logicielle.")
        m.ajouter("assistant", "D'accord, Pierre.")
        return m

    def test_prenom_retenu(self, memoire):
        assert _reponse("Quel est mon prénom ?", historique=memoire) == \
            "Tu t'appelles Pierre."

    def test_projet_retenu(self, memoire):
        assert "radio logicielle" in _reponse("Sur quoi je travaille ?",
                                              historique=memoire)

    def test_projet_autre_formulation(self, tmp_path):
        m = MemoireConversation(chemin=tmp_path / "conv2.jsonl")
        m.ajouter("user", "Je m'appelle Camille et je bosse sur un moteur "
                          "de jeu en Rust.")
        assert "moteur de jeu" in _reponse("Sur quoi je travaille ?",
                                          historique=m)

    def test_resume_variantes(self, memoire):
        for question in ("Fais-moi un résumé de nos échanges.",
                         "Fais un résumé de notre discussion",
                         "De quoi a-t-on parlé ?"):
            assert "Derniers sujets" in _reponse(question, historique=memoire)

    def test_compacter_garde_le_prenom(self, memoire):
        memoire.compacter()
        assert _reponse("Quel est mon prénom ?", historique=memoire) == \
            "Tu t'appelles Pierre."

    def test_sans_memoire_ne_devine_pas(self):
        assert skills.repondre("Quel est mon prénom ?") is None

    def test_nom_de_domaine_n_est_pas_le_prenom(self, memoire):
        assert skills.repondre("Quel est mon nom de domaine ?",
                               historique=memoire) is None

    def test_resume_liste_les_sujets(self, memoire):
        r = _reponse("Résume notre conversation", historique=memoire)
        assert "Derniers sujets" in r and "radio logicielle" in r

    def test_resume_apres_compaction_est_honnete(self, memoire):
        memoire.compacter()
        r = _reponse("De quoi parlions-nous ?", historique=memoire)
        assert "compaction" in r and "Pierre" in r


class TestConsignes:
    def test_code_recoit_sa_consigne(self):
        bloc = skills.bloc_pour("Corrige ce code : for i in range(10) print(i)")
        assert "CODE" in bloc and "exécutable tel quel" in bloc

    def test_sources_actives_avec_contexte_web(self):
        assert "SOURCES" in skills.bloc_pour("Quel temps fait-il ?",
                                             avec_contexte=True)
        assert "SOURCES" not in skills.bloc_pour("Quel temps fait-il ?")

    def test_completer_ne_remplace_pas_le_prompt(self):
        base = "You are a helpful assistant."
        complete = skills.completer(base, "Corrige ce code", False)
        assert complete.startswith(base) and "CODE" in complete
        assert skills.completer(base, "Bonjour", False) == base


class TestInjectionPrompt:
    def test_prompt_standard_recoit_la_consigne(self, monkeypatch):
        from aura import llama_cerveau as lc, serveur_lora
        vus = {}
        monkeypatch.setattr(serveur_lora, "_actifs", lambda: False)
        monkeypatch.setattr(lc, "_charger",
                            lambda: (_ for _ in ()).throw(RuntimeError("sans modele")))
        monkeypatch.setattr(
            lc._skills, "completer",
            lambda base, question, contexte: vus.update(question=question) or base)
        lc.generer("Corrige ce code", systeme=lc._SYSTEME)
        assert vus.get("question") == "Corrige ce code"

    def test_prompt_expert_non_complete(self, monkeypatch):
        from aura import llama_cerveau as lc, serveur_lora
        monkeypatch.setattr(serveur_lora, "_actifs", lambda: False)
        monkeypatch.setattr(lc, "_charger",
                            lambda: (_ for _ in ()).throw(RuntimeError("sans modele")))
        monkeypatch.setattr(
            lc._skills, "completer",
            lambda *a: (_ for _ in ()).throw(AssertionError("prompt expert")))
        lc.generer("question", systeme="PROMPT EXPERT")


class TestPipeline:
    def test_l_equation_ne_part_pas_au_calculateur(self):
        from aura.orchestrateur import RontoGPT
        ia = RontoGPT()
        assert "x = 7" in ia.executer("Résous 4x − 7 = 21")

    def test_le_pgcd_est_calcule_pas_raconte(self):
        from aura.orchestrateur import RontoGPT
        r = RontoGPT().executer_detaille("Quel est le PGCD de 84 et 120 ?")
        assert r["cerveau_choisi"] == "skill-pgcd"
        assert "PGCD(84, 120) = 12" in r["reponse"]

    def test_le_prenom_est_repondu_sans_cerveau(self, tmp_path):
        from aura.orchestrateur import RontoGPT
        ia = RontoGPT()
        ia._historique = MemoireConversation(chemin=tmp_path / "c.jsonl")
        ia._historique.ajouter("user", "Je m'appelle Pierre.")
        from aura import typographie
        assert ia.executer("Quel est mon prénom ?") == \
            typographie.nettoyer("Tu t'appelles Pierre.")
