"""Aiguilleur : petit modele local ou Gemini, selon la TACHE et la REPONSE.

Trois familles de tests :
- les regles de TACHE (la question demande une competence que le 1B n'a pas) ;
- les regles de REPONSE (le 1B a rendu une reponse faible) ;
- l'INTEGRATION (`executer_detaille`) : l'aiguilleur decide, gemini.repondre
  execute, et la reponse locale reste intacte quand rien ne declenche.
"""
import pytest

from aura import aiguilleur, gemini

_VARIABLES = ("AURA_CLOUD", "AURA_GEMINI", "AURA_GEMINI_HISTORIQUE",
              "AURA_AIGUILLEUR_REGLES",
              "AURA_AIGUILLEUR_DOMAINES", "AURA_GEMINI_REGLES",
              "AURA_GEMINI_DOMAINES")


@pytest.fixture(autouse=True)
def _isole(monkeypatch, tmp_path):
    """Pas de cle, pas de `.env` reel : le jugement part d'un etat connu.

    La fixture `_gemini_coupe` du conftest coupe la cible dans toute la suite ;
    ici on la re-active et on neutralise le `.env` de la racine en pointant
    `gemini._RACINE` sur un dossier temporaire vide.
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
    """Une cle ET l'envoi automatique (AURA_CLOUD=oui).

    Les tests d'integration ci-dessous verifient la MECANIQUE d'escalade. Le
    defaut du produit est `demander` (rien ne part sans clic) : il a ses
    propres tests plus bas (`fixture sans_envoi_auto`).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fausse-cle-test")
    monkeypatch.setenv("AURA_CLOUD", "oui")


@pytest.fixture
def sans_envoi_auto(monkeypatch):
    """Une cle, et le mode PAR DEFAUT du produit (AURA_CLOUD absent)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fausse-cle-test")
    monkeypatch.delenv("AURA_CLOUD", raising=False)


def _res(reponse, cerveau="gemma3-1b", **extra):
    return {"reponse": reponse, "cerveau_choisi": cerveau, **extra}


def _longue(n):
    """n phrases DISTINCTES (pas de boucle), ~45 caracteres chacune."""
    return " ".join(f"Phrase numero {i} sur la liberte et ses limites."
                    for i in range(n))


# -- disponibilite de la cible -------------------------------------------------

def test_sans_cible_disponible_jamais_d_escalade():
    # aucune cle -> disponible() est faux, et decider() garde toujours le local
    assert aiguilleur.disponible() is False
    d = aiguilleur.decider("Rédige une dissertation sur la liberté", _res(""))
    assert d.escalader is False and d.raison is None
    assert aiguilleur.decider("q", _res("Je ne sais pas.")).escalader is False


def test_interrupteur_general(avec_cle, monkeypatch):
    assert aiguilleur.disponible() is True
    monkeypatch.setenv("AURA_GEMINI", "0")
    assert aiguilleur.disponible() is False
    assert aiguilleur.decider("q", _res("")).escalader is False


# -- regles de TACHE -----------------------------------------------------------

def test_tache_dissertation_escalade_meme_sur_une_longue_reponse(avec_cle):
    """Une competence que le 1B n'a pas : on le sait AVANT sa reponse."""
    question = "Rédige une dissertation sur la liberté"
    d = aiguilleur.decider(question, _res(_longue(30)))
    assert d.escalader is True
    assert d.raison == "tache-redaction"
    assert d.moment == "tache"


@pytest.mark.parametrize("question", [
    "Rédige une dissertation sur la liberté",
    "Ecris un essai sur la democratie",
    "Fais un rapport de 3 pages sur le climat",
    "Produis un article de 1200 mots sur l'IA",
])
def test_les_demandes_longues_sont_des_taches_faibles(avec_cle, question):
    assert aiguilleur.raison(question, _res(_longue(30))) == "tache-redaction"


@pytest.mark.parametrize("question", [
    "Traduis ce texte en anglais",
    "traduit cette phrase en espagnol",
    "Peux-tu traduire ce paragraphe en allemand ?",
    "Fais la traduction de ce passage en italien",
])
def test_les_demandes_de_traduction_sont_des_taches_faibles(avec_cle, question):
    assert aiguilleur.raison(question, _res(_longue(30))) == "tache-traduction"


def test_une_question_theorique_sur_la_traduction_n_escalade_pas(avec_cle):
    # « traduction » comme sujet d'explication n'est pas une demande de traduction
    assert aiguilleur.raison(
        "Explique comment fonctionne la traduction automatique",
        _res(_longue(30))) is None


def test_les_regles_de_tache_se_coupent(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_AIGUILLEUR_REGLES", "vide")
    assert aiguilleur.raison("Rédige une dissertation sur la liberté",
                             _res(_longue(30))) is None
    # mais une reponse VIDE continue de declencher
    assert aiguilleur.raison("Rédige une dissertation sur la liberté",
                             _res("")) == "vide"


def test_une_tache_faible_ne_rejuge_pas_un_tier_exact(avec_cle):
    """Le RAG ou un solveur a repondu : la tache n'annule pas une preuve."""
    assert aiguilleur.raison(
        "Rédige une dissertation sur la liberté",
        _res("texte", "rag-binaire")) is None


# -- regles de REPONSE ---------------------------------------------------------

Q_NEUTRE = "Donne-moi des idees de repas pour la semaine"
REP_NEUTRE = ("Varie les legumes, les feculents et les proteines, et prepare "
              "quelques bases a l'avance pour gagner du temps.")
Q_PROBLEME = ("Lea a 12 billes et en perd 5 puis elle en gagne 8, combien en "
              "a-t-elle a la fin de la partie ?")
Q_FACTUEL = "Qui est le fondateur de Microsoft ?"


@pytest.mark.parametrize("question,reponse,attendu,moment", [
    (Q_NEUTRE, "", "vide", "reponse"),
    (Q_NEUTRE, "   ", "vide", "reponse"),
    (Q_NEUTRE, "[RontoGPT] Reflexion interrompue", "erreur", "reponse"),
    (Q_NEUTRE, "Je suis désolé, mais je ne peux pas répondre.", "refus", "reponse"),
    (Q_NEUTRE, "Cette information m'échappe : je ne sais pas.", "aveu", "reponse"),
    ("Décris la mer",
     " ".join(["La mer est grande et bleue et calme aujourd'hui."] * 5),
     "boucle", "reponse"),
    ("Quelle est la différence entre un virus et une bactérie ?",
     "Quelle est la différence entre un virus et une bactérie ?",
     "echo", "reponse"),
    ("Explique pourquoi le ciel est bleu", "À cause de la lumière.",
     "court", "reponse"),
    ("Rédige une dissertation sur la liberté", _longue(6),
     "redaction", "reponse"),
    (Q_PROBLEME, "Elle en a 15.", "probleme", "reponse"),
    (Q_FACTUEL, "Bill Gates.", "factuel", "reponse"),
])
def test_reponses_faibles_partent(avec_cle, question, reponse, attendu, moment):
    d = aiguilleur.decider(question, _res(reponse))
    assert d.escalader is True
    assert d.raison == attendu
    assert d.moment == moment
    assert d.fournisseur == "gemini"


@pytest.mark.parametrize("question,reponse", [
    (Q_NEUTRE, REP_NEUTRE),
    ("Explique pourquoi le ciel est bleu", _longue(6)),
    (Q_FACTUEL, "Selon la source, Bill Gates."),
])
def test_reponses_correctes_restent(avec_cle, question, reponse):
    r = _res(reponse, contexte_web="extrait verifie")
    assert aiguilleur.decider(question, r).escalader is False


def test_une_reponse_factuelle_sans_source_part(avec_cle):
    assert aiguilleur.raison(Q_FACTUEL, _res("Bill Gates.")) == "factuel"
    assert aiguilleur.raison(
        Q_FACTUEL, _res("Bill Gates.", contexte_web="extrait")) is None


def test_un_probleme_de_code_ne_part_pas_en_probleme(avec_cle):
    q = ("Ecris une fonction python qui renvoie la somme de 12 et 5 puis "
         "la multiplie par 3 et affiche le resultat final")
    assert aiguilleur.raison(q, _res("def f(): pass")) is None


def test_une_formule_exacte_ne_part_pas_en_probleme(avec_cle):
    assert aiguilleur.raison(
        Q_PROBLEME, _res("Elle en a 15.", formule="12 - 5 + 8")) is None


@pytest.mark.parametrize("cerveau", [
    "niveau0-instantane", "wikipedia", "pot-1b+ast", "constante",
    "constante-capacite", "horloge", "calcul-verbal", "calcul-taux",
    "rag-binaire", "micro-faits", "cache-semantique", "extraction-api",
    "clarification", "titulaires", "local", "",
])
def test_un_tier_exact_n_est_jamais_rejuge(avec_cle, cerveau):
    assert aiguilleur.raison(Q_NEUTRE, _res("Je ne sais pas.", cerveau)) is None


@pytest.mark.parametrize("cerveau", ["gemma3-1b", "llama-1b", "def-web+1b",
                                     "extraction-web", "qwen-1.5b"])
def test_un_tier_generatif_est_rejugeable(avec_cle, cerveau):
    assert aiguilleur.raison(
        Q_NEUTRE, _res("Je ne sais pas.", cerveau)) == "aveu"


@pytest.mark.parametrize("cerveau", ["niveau0-instantane", "wikipedia", "local"])
def test_une_reponse_vide_part_quel_que_soit_le_tier(avec_cle, cerveau):
    assert aiguilleur.raison(Q_NEUTRE, _res("", cerveau)) == "vide"


@pytest.mark.parametrize("prefixe", ["[Logique] Je ne sais pas.",
                                     "[Code] Je ne sais pas."])
def test_une_reponse_verifiee_n_est_pas_rejugee(avec_cle, prefixe):
    assert aiguilleur.raison(Q_NEUTRE, _res(prefixe)) is None


def test_une_relance_de_clarification_n_est_pas_une_reponse_faible(avec_cle):
    r = _res("Je ne sais pas.", "gemma3-1b", en_attente=True)
    assert aiguilleur.raison(Q_NEUTRE, r) is None


def test_mauvaise_langue(avec_cle):
    question = ("Explique pourquoi le ciel est bleu et donne les details de "
                "ce phenomene physique")
    anglais = (
        "the sky is blue because the light of the sun is scattered by the "
        "molecules of the air and the blue light of the spectrum is "
        "dispersed more than the red light which travels straight through "
        "the atmosphere without any deviation and this is what you see when "
        "you look at the sky")
    assert aiguilleur.raison(question, _res(anglais)) == "langue"


def test_domaine_designe_faible(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_AIGUILLEUR_DOMAINES", "code, math")
    assert aiguilleur.raison(Q_NEUTRE, _res(REP_NEUTRE), "math") == "domaine"
    assert aiguilleur.raison(Q_NEUTRE, _res(REP_NEUTRE), "web") is None
    assert aiguilleur.raison(
        Q_NEUTRE, _res(REP_NEUTRE, "calcul-verbal"), "math") is None


def test_l_ancien_nom_de_variable_reste_lu(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_GEMINI_DOMAINES", "code")
    assert aiguilleur.raison(Q_NEUTRE, _res(REP_NEUTRE), "code") == "domaine"


def test_regles_selectionnees(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_AIGUILLEUR_REGLES", "vide")
    assert aiguilleur.raison(Q_NEUTRE, _res("")) == "vide"
    assert aiguilleur.raison(Q_NEUTRE, _res("Je ne sais pas.")) is None
    monkeypatch.setenv("AURA_AIGUILLEUR_REGLES", "nimporte")
    assert aiguilleur.raison(Q_NEUTRE, _res("")) is None


# -- integration dans executer_detaille ---------------------------------------

def _pipeline_factice(monkeypatch, reponse_locale, cerveau="gemma3-1b",
                      contexte_web="extrait web"):
    """RontoGPT sans charger le modele : seul l'aiguillage est reel."""
    from aura import orchestrateur, cache_reponses
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


def test_pipeline_escalade_si_local_vide(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: vus.update(kw) or "reponse Gemini")
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert r["reponse"] == "reponse Gemini"
    assert r["cerveau_choisi"] == "gemini"
    assert r["repli"] == {"fournisseur": "gemini", "raison": "vide",
                          "moment": "reponse", "reponse_locale": ""}
    assert vus["contexte_web"] == "extrait web"


def test_pipeline_escalade_sur_la_tache(avec_cle, monkeypatch):
    """Le 1B a rendu une longue reponse : l'aiguilleur escalade quand meme."""
    appels = []
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: appels.append(q) or "dissertation Gemini")
    ia = _pipeline_factice(monkeypatch, _longue(30))
    r = ia.executer_detaille("Rédige une dissertation sur la liberté")
    assert appels, "la tache faible doit escalader meme sur une reponse correcte"
    assert r["cerveau_choisi"] == "gemini"
    assert r["repli"]["raison"] == "tache-redaction"
    assert r["repli"]["moment"] == "tache"


def test_pipeline_garde_la_reponse_locale_valide(avec_cle, monkeypatch):
    appels = []
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: appels.append(q) or "X")
    ia = _pipeline_factice(monkeypatch, "Paris est la capitale de la France.")
    r = ia.executer_detaille("question de test")
    assert r["reponse"] == "Paris est la capitale de la France."
    assert appels == []
    assert "repli" not in r


def test_pipeline_ne_rejuge_pas_un_tier_exact(avec_cle, monkeypatch):
    appels = []
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: appels.append(q) or "X")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.", "niveau0-instantane")
    r = ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert r["reponse"] == "Je ne sais pas."
    assert appels == []


def test_pipeline_echec_de_la_cible_laisse_la_reponse_locale(avec_cle, monkeypatch):
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: None)
    monkeypatch.setattr(gemini, "derniere_erreur", lambda: "quota depasse")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    r = ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert r["reponse"] == "Je ne sais pas."
    assert r["cerveau_choisi"] == "gemma3-1b"
    assert r["repli_echec"] == "quota depasse"


def test_pipeline_remplace_la_mauvaise_reponse_du_cache(avec_cle, monkeypatch):
    from aura import filtre_instantane
    vus = []
    monkeypatch.setattr(filtre_instantane, "mettre_a_jour",
                        lambda q, r: vus.append((q, r)) or True)
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: "Santiago.")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert vus == [("Quelle est la capitale du Chili ?", "Santiago.")]


def test_pipeline_remplace_la_reponse_dans_la_fenetre(
        avec_cle, monkeypatch, tmp_path):
    from aura.memoire_conversation import MemoireConversation
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: "Santiago.")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "Quelle est la capitale du Chili ?")
    ia._historique.ajouter("assistant", "Je ne sais pas.")
    ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert ia._historique.fenetre()[-1] == {"role": "assistant",
                                            "content": "Santiago."}


def test_pipeline_envoie_les_tours_precedents(avec_cle, monkeypatch, tmp_path):
    from aura.memoire_conversation import MemoireConversation
    vus = {}
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: vus.update(kw) or "Santiago.")
    ia = _pipeline_factice(monkeypatch, "")
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "je parle du Chili")
    ia._historique.ajouter("assistant", "Le Chili est un pays.")
    ia._historique.ajouter("user", "et sa capitale ?")      # reponse locale vide
    ia.executer_detaille("et sa capitale ?")
    assert vus["historique"] == [
        {"role": "user", "content": "je parle du Chili"},
        {"role": "assistant", "content": "Le Chili est un pays."}]


# -- fenetre de conversation ---------------------------------------------------

def test_tours_precedents_excluent_le_tour_en_cours(tmp_path):
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import RontoGPT
    ia = RontoGPT.__new__(RontoGPT)
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "u1")
    ia._historique.ajouter("assistant", "a1")
    ia._historique.ajouter("user", "u2")
    ia._historique.ajouter("assistant", "a2")
    assert ia._tours_precedents() == [{"role": "user", "content": "u1"},
                                      {"role": "assistant", "content": "a1"}]
