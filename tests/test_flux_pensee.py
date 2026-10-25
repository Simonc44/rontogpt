"""Le flux a DEUX phases, et l'abonnement par defaut n'en prend qu'une.

Defaut reel : `publier_flux` envoyait tout, en vrac, a tous les abonnes. Le
modele ecrit sa reflexion dans le meme flux de tokens que sa reponse, donc :

  * l'utilisateur ne pouvait pas distinguer la reflexion de la reponse ;
  * un abonne qui attend le RESULTAT (cache, ancrage, graphe, verificateur)
    pouvait memoriser la reflexion comme si c'en etait un.

Le contrat est desormais explicite : `abonner_flux(fn)` = reponse SEULE
(choix sur par defaut), `abonner_flux(fn, LES_DEUX_PHASES)` = tout.

Les balises sont assemblees a partir de morceaux : ecrites d'un seul bloc,
certains outils de rendu les prennent pour du HTML et les suppriment.
"""
import pytest

from aura import llama_cerveau as lc

OUV = "<" "think>"
FER = "<" "/think>"


@pytest.fixture(autouse=True)
def _flux_propre():
    """Etat de MODULE remis a zero entre deux tests.

    Le registre d'abonnes ET la reflexion retenue sont globaux. Sans ca, un
    test depend de ce qu'un autre a laisse derriere lui — et la suite tourne
    en ordre ALEATOIRE : `test_reflexion_retiree_et_lisible` tombait ainsi
    dans un job sur quatre (mesure du 16 oct. 2026).
    """
    lc.effacer_reflexion()
    yield
    for phase in lc.LES_DEUX_PHASES:
        lc._abonnes_flux[phase].clear()
    lc.effacer_reflexion()


class TestAbonnementParPhase:
    def test_defaut_ne_recoit_que_la_reponse(self):
        vus: list[str] = []
        lc.abonner_flux(vus.append)
        lc.publier_flux("je pense", lc.PHASE_PENSEE)
        lc.publier_flux("la reponse", lc.PHASE_REPONSE)
        assert vus == ["la reponse"]

    def test_deux_phases_recoit_tout(self):
        vus: list[str] = []
        lc.abonner_flux(vus.append, lc.LES_DEUX_PHASES)
        lc.publier_flux("je pense", lc.PHASE_PENSEE)
        lc.publier_flux("la reponse", lc.PHASE_REPONSE)
        assert vus == ["je pense", "la reponse"]

    def test_desabonnement_retire_les_deux_phases(self):
        vus: list[str] = []
        off = lc.abonner_flux(vus.append, lc.LES_DEUX_PHASES)
        off()
        lc.publier_flux("x", lc.PHASE_PENSEE)
        lc.publier_flux("y", lc.PHASE_REPONSE)
        assert vus == []
        assert lc.flux_demande() is False

    def test_flux_actif_par_phase(self):
        lc.abonner_flux(lambda _t: None, (lc.PHASE_PENSEE,))
        assert lc.flux_actif(lc.PHASE_PENSEE) is True
        assert lc.flux_actif(lc.PHASE_REPONSE) is False
        # mais quelqu'un regarde : la generation doit streamer
        assert lc.flux_demande() is True

    def test_abonne_qui_plante_ne_bloque_pas(self):
        def explose(_texte):
            raise RuntimeError("abonne casse")

        recus: list[str] = []
        lc.abonner_flux(explose)
        lc.abonner_flux(recus.append)
        lc.publier_flux("morceau")
        assert recus == ["morceau"]


class TestRepartiteur:
    def test_texte_sans_balise_reste_en_reponse(self):
        r = lc._FluxTour()
        r.pousser("Bonjour, voici la reponse.")
        r.terminer()
        assert r.pensee == ""
        assert r.reponse == "Bonjour, voici la reponse."

    def test_reflexion_et_reponse_separees(self):
        vus: list[tuple[str, str]] = []
        lc.abonner_flux(lambda t: vus.append((lc.PHASE_PENSEE, t)),
                        (lc.PHASE_PENSEE,))
        lc.abonner_flux(lambda t: vus.append((lc.PHASE_REPONSE, t)),
                        (lc.PHASE_REPONSE,))
        r = lc._FluxTour()
        r.pousser(f"{OUV}j'hesite entre deux pistes.{FER}La reponse est 42.")
        r.terminer()
        assert r.pensee == "j'hesite entre deux pistes."
        assert r.reponse == "La reponse est 42."
        phases = [p for p, _ in vus]
        assert phases == [lc.PHASE_PENSEE, lc.PHASE_REPONSE]

    def test_balise_coupee_entre_deux_morceaux(self):
        """Le cas qui casse une detection naive : « <th » puis « ink> »."""
        r = lc._FluxTour()
        for morceau in ("<th", "ink>", "doute", "</th", "ink>", "Fini."):
            r.pousser(morceau)
        r.terminer()
        assert r.pensee == "doute"
        assert r.reponse == "Fini."

    def test_le_texte_nait_jamais_en_retard(self):
        """Un morceau publie ne doit pas etre retenu plus que necessaire."""
        morceaux: list[str] = []
        lc.abonner_flux(morceaux.append, lc.LES_DEUX_PHASES)
        r = lc._FluxTour()
        r.pousser("Une phrase entiere et longue qui n'annonce aucune balise.")
        # tout est publiable : rien ne doit rester coince dans le tampon
        assert "".join(morceaux) == "Une phrase entiere et longue qui n'annonce aucune balise."
        r.terminer()

    def test_balise_longue_prioritaire(self):
        """<thinking> ne doit pas etre lu comme  thinking suivi de « ing> »."""
        r = lc._FluxTour()
        r.pousser("<thinking>analyse</thinking>Reponse.")
        r.terminer()
        assert r.pensee == "analyse"
        assert r.reponse == "Reponse."

    def test_hors_phase_la_pensee_n_est_pas_jetee(self):
        """Sans abonne, la repartition reste calculee (memoire du tour)."""
        r = lc._FluxTour()
        r.pousser(f"{OUV}raisonnement{FER}Conclusion.")
        r.terminer()
        assert "raisonnement" in r.pensee


class TestReflexionDuTour:
    def test_reflexion_retiree_et_lisible(self):
        """La reflexion n'est pas jetee : elle est MISE DE COTE, nommement."""
        # `generer()` la remet a zero en production ; ici, on ne depend pas de
        # ce qu'un test precedent a laisse (voir la fixture _flux_propre).
        lc.effacer_reflexion()
        assert lc.reflexion_du_tour() == ""
        lc._retenir_reflexion("17 x 3 = 51, donc...")
        assert "17 x 3" in lc.reflexion_du_tour()
        lc.effacer_reflexion()
        assert lc.reflexion_du_tour() == ""

    def test_vide_n_ecrase_pas(self):
        lc.effacer_reflexion()
        lc._retenir_reflexion("premier")
        lc._retenir_reflexion("")
        assert lc.reflexion_du_tour() == "premier"
        lc.effacer_reflexion()


class TestAbonnementAvecPhase:
    """`abonner_flux_phase` : la variante qui DIT a quelle phase appartient
    chaque morceau.

    Defaut corrige le 9 oct. 2026 : l'API SSE et la GUI appelaient
    `abonner_flux` sans `phases`. Elles s'abonnaient donc a la reponse seule,
    et la reflexion — publiee phase par phase — n'avait AUCUN abonne : elle
    etait jetee dans les deux interfaces, alors que la demande produit est
    justement « l'utilisateur voit la pensee ET la reponse ».
    """

    def test_chaque_morceau_porte_sa_phase(self):
        vus: list[tuple[str, str]] = []
        lc.abonner_flux_phase(lambda m, p: vus.append((m, p)))
        lc.publier_flux("je pense", lc.PHASE_PENSEE)
        lc.publier_flux("la reponse", lc.PHASE_REPONSE)
        assert vus == [("je pense", lc.PHASE_PENSEE),
                       ("la reponse", lc.PHASE_REPONSE)]

    def test_defaut_prend_les_deux_phases(self):
        """Le defaut de CETTE variante est LES_DEUX_PHASES : c'est son objet.

        Le defaut sur (reponse seule) reste celui d'`abonner_flux`, inchange :
        un module qui n'a besoin que du resultat final n'en recoit toujours
        pas.
        """
        vus: list[str] = []
        lc.abonner_flux_phase(lambda m, _p: vus.append(m))
        lc.publier_flux("p", lc.PHASE_PENSEE)
        lc.publier_flux("r", lc.PHASE_REPONSE)
        assert vus == ["p", "r"]

    def test_desabonnement_retire_toutes_les_phases(self):
        vus: list[str] = []
        off = lc.abonner_flux_phase(lambda m, _p: vus.append(m))
        off()
        lc.publier_flux("p", lc.PHASE_PENSEE)
        lc.publier_flux("r", lc.PHASE_REPONSE)
        assert vus == []
        assert lc.flux_demande() is False

    def test_une_seule_phase_demandee(self):
        vus: list[str] = []
        lc.abonner_flux_phase(lambda m, _p: vus.append(m),
                              phases=(lc.PHASE_PENSEE,))
        lc.publier_flux("p", lc.PHASE_PENSEE)
        lc.publier_flux("r", lc.PHASE_REPONSE)
        assert vus == ["p"]


class TestLesClientsEcoutentLesDeuxPhases:
    """BRANCHE, pas seulement defini.

    Meme exigence que pour le garde-fou d'architecture du serveur : un contrat
    que personne n'appelle ne sert a rien. C'est exactement le defaut trouve
    ici — la mecanique des phases existait, testee, et aucun client ne s'y
    abonnait.
    """

    @pytest.mark.parametrize("module", ["aura/api.py"])
    def test_abonnement_avec_phase(self, module):
        from pathlib import Path
        racine = Path(__file__).resolve().parent.parent
        source = (racine / module).read_text(encoding="utf-8")
        assert "abonner_flux_phase" in source, (
            f"{module} s'abonne sans distinguer les phases : la reflexion du "
            "modele n'atteindrait jamais l'utilisateur")

    def test_la_gui_affiche_la_pensee_dans_sa_propre_bulle(self):
        """Le JS doit SEPARER les deux : concatener reviendrait a remettre la
        pensee et la reponse dans le meme fil, ce que la demande produit
        interdit."""
        from pathlib import Path
        racine = Path(__file__).resolve().parent.parent
        source = (racine / "aura" / "web" / "index.html").read_text(
            encoding="utf-8")
        assert "donnees.phase === 'pensee'" in source
        assert "bullePensee" in source
