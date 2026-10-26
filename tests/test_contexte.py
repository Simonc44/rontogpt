"""Compaction du contexte : estimation, saturation, tri des souvenirs.

Ces tests fixent les REGLES, pas les chiffres : `budget()` est monkeypatche,
donc ils ne dependent ni du modele charge ni de la machine. C'est le trou que
la couverture montrait (32 %) : le module decide ce qui est OUBLIE.
"""
from aura import contexte


def tour(role, texte):
    return {"role": role, "content": texte}


class FauxHistorique:
    """Le minimum que `contexte` demande a une memoire de conversation."""

    def __init__(self, tours):
        self._tours = list(tours)
        self.gardes = None

    def fenetre(self):
        return list(self._tours)

    def compacter_en_gardant(self, indices):
        gardes = set(indices)
        avant = len(self._tours)
        self._tours = [t for i, t in enumerate(self._tours) if i in gardes]
        self.gardes = sorted(gardes)
        return avant - len(self._tours)


class TestEstimation:
    def test_texte_vide(self):
        assert contexte.estimer("") == 0

    def test_ordre_de_grandeur(self):
        assert 1 <= contexte.estimer("a" * 300) <= 300

    def test_mode_par_defaut_et_recent(self, monkeypatch):
        monkeypatch.delenv("AURA_COMPACTION", raising=False)
        assert contexte.mode() == "importance"
        monkeypatch.setenv("AURA_COMPACTION", "recent")
        assert contexte.mode() == "recent"
        monkeypatch.setenv("AURA_COMPACTION", "n'importe quoi")
        assert contexte.mode() == "importance"


class TestGroupes:
    def test_un_groupe_par_reponse(self):
        tours = [tour("user", "q1"), tour("assistant", "r1"),
                 tour("user", "q2"), tour("assistant", "r2")]
        assert [len(g) for g in contexte._groupes(tours)] == [2, 2]

    def test_dernier_echange_incomplet_est_garde(self):
        tours = [tour("user", "q1"), tour("assistant", "r1"),
                 tour("user", "q2")]
        groupes = contexte._groupes(tours)
        assert len(groupes) == 2 and len(groupes[-1]) == 1

    def test_fenetre_vide(self):
        assert contexte._groupes([]) == []


class TestEtat:
    def test_sous_le_seuil(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 1000)
        mesure = contexte.etat(None, "bonjour")
        assert mesure["budget"] == 1000
        assert mesure["sature"] is False
        assert mesure["seuil"] == 90
        assert mesure["pourcentage"] < 90

    def test_saturation_atteinte(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 100)
        mesure = contexte.etat(FauxHistorique([tour("user", "a" * 300)]))
        assert mesure["sature"] is True
        assert mesure["pourcentage"] >= 90

    def test_les_champs_de_conversation_sont_hors_systeme(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 1000)
        mesure = contexte.etat(None, "q", systeme="regles " * 10)
        assert mesure["systeme_tokens"] > 0
        assert mesure["budget_conversation"] == 1000 - mesure["systeme_tokens"]
        assert mesure["conversation_tokens"] >= 0


class TestAutoCompaction:
    def test_sans_historique_ne_fait_rien(self):
        assert contexte.auto_compacter(None, "q") is None

    def test_sous_le_seuil_ne_touche_a_rien(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 100_000)
        historique = FauxHistorique([tour("user", "q"),
                                     tour("assistant", "r")])
        assert contexte.auto_compacter(historique, "q") is None
        assert historique.gardes is None

    def test_sature_retire_les_echanges_banals(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 200)
        monkeypatch.setenv("AURA_COMPACTION", "importance")
        echanges = []
        for i in range(6):
            question = ("merci" if i < 4
                        else "je m'appelle Simon et mon projet est Atlas")
            echanges += [tour("user", question),
                         tour("assistant", "blabla " * 20)]
        historique = FauxHistorique(echanges)

        # `systeme=""` : le prompt fixe du projet (~430 tokens) saturerait a
        # lui seul un budget de 200, et la fenetre serait videe d'office.
        mesure = contexte.auto_compacter(historique, "et ensuite ?",
                                        systeme="")

        assert mesure is not None and mesure["sature"] is True
        assert mesure["compacte"] > 0
        assert historique.gardes is not None
        # Le dernier echange n'est jamais celui qu'on jette.
        assert historique._tours[-1] is echanges[-1]
        assert len(historique._tours) >= 2

    def test_fenetre_minimale_le_dit_au_lieu_de_mentir(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 50)
        monkeypatch.setenv("AURA_COMPACTION", "recent")
        historique = FauxHistorique([tour("user", "a" * 300),
                                     tour("assistant", "z" * 300)])
        mesure = contexte.auto_compacter(historique, "q")
        assert mesure is not None
        assert mesure["compacte"] == 0
        assert mesure["raison"] == "fenetre minimale"
        assert historique.gardes is None


class TestSelection:
    def test_recent_garde_la_fin(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 1000)
        tours = [tour("user", f"q{i}") for i in range(6)]
        choix = contexte._selection_recent(tours, hors=0)
        # `GARDER` est un MINIMUM : tout ce qui tient est garde, et la fin de
        # la fenetre l'est toujours.
        assert len(tours) - 1 in choix
        assert choix == set(range(len(tours) - len(choix), len(tours)))

    def test_importance_garde_au_moins_le_dernier_echange(self, monkeypatch):
        monkeypatch.setattr(contexte, "budget", lambda: 1000)
        tours = [tour("user", "merci"), tour("assistant", "de rien"),
                 tour("user", "mon projet est Atlas"),
                 tour("assistant", "note")]
        choix = contexte._selection_importance(tours, hors=0)
        assert {len(tours) - 2, len(tours) - 1} <= choix
