"""Provenance : le badge dit la verite sur le chemin de la reponse."""
import pytest

from aura import provenance as P
from aura import registre_experts


def _r(cerveau, reponse="42", **kw):
    return {"cerveau_choisi": cerveau, "reponse": reponse, **kw}


@pytest.mark.parametrize("cerveau,niveau", [
    ("calcul-verbal", "exact"), ("niveau0-instantane", "exact"),
    ("pot-1b+ast", "exact"), ("skill-algebre", "exact"),
    ("skill-identite", "base"), ("skill-memoire", "base"),
    ("skill-resume", "memoire"), ("documents", "document"),
    ("wikipedia", "source"), ("rag-binaire", "base"),
    ("cache-semantique", "memoire"), ("gemini", "cloud"),
    ("web-indisponible", "inconnu"),
])
def test_niveaux_connus(cerveau, niveau):
    assert P.provenance(_r(cerveau))["niveau"] == niveau


def test_chaque_expert_du_registre_est_classe():
    """Un expert ajoute au registre sans badge serait presente comme « local »."""
    # « experts-speciaux » est un etage du registre, pas une etiquette
    # publique : ses etiquettes sont tri-transitif / logique / potcode.
    for nom in registre_experts.noms():
        if nom == "experts-speciaux":
            continue
        assert nom in P._TABLE, "expert sans badge : " + nom


def test_les_skills_deterministes_sont_classes():
    """Un skill ajoute sans badge serait presente comme « local » a tort."""
    assert P.provenance(_r("skill-identite"))["niveau"] == "base"
    assert P.provenance(_r("skill-algebre"))["niveau"] == "exact"
    assert P.provenance(_r("skill-pgcd"))["verifie"] is True
    assert P.provenance(_r("skill-memoire"))["niveau"] == "base"
    assert P.provenance(_r("skill-resume"))["niveau"] == "memoire"


def test_modele_local_jamais_verifie():
    p = P.provenance(_r("gemma-3-1b-it-Q8_0-1b", "Une reponse redigee."))
    assert p["niveau"] == "local" and p["verifie"] is False


def test_modele_local_avec_web_le_dit():
    p = P.provenance(_r("gemma-1b", "ok", contexte_web="extrait"))
    assert "extraits web" in p["detail"]


def test_clarification_n_est_pas_une_reponse():
    assert P.provenance(_r("x", en_attente=True))["niveau"] == "precision"


def test_aveu_devient_je_ne_sais_pas():
    p = P.provenance(_r("gemma-1b", "Je ne peux pas verifier cette definition."))
    assert p["niveau"] == "inconnu" and p["badge"] == "Je ne sais pas"


def test_un_calcul_exact_n_est_pas_declasse_par_un_mot():
    assert P.provenance(_r("calcul-taux", "je ne sais pas"))["niveau"] == "exact"


def test_cloud_est_le_seul_a_quitter_la_machine():
    assert P.provenance(_r("gemini"))["quitte_la_machine"] is True
    assert P.provenance(_r("documents"))["quitte_la_machine"] is False


def test_ne_leve_jamais():
    assert P.provenance({})["niveau"] == "local"
    assert P.provenance({"reponse": None, "cerveau_choisi": None})["niveau"] == "local"
