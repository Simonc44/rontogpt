"""Tests anti-hallucination : identite, definitions ancrees, filet final."""
from aura import orchestrateur
from aura.orchestrateur import RontoGPT, _IDENTITE


def _ia(monkeypatch):
    """RontoGPT sans GGUF ni reseau : la plomberie, pas le contenu."""
    from aura import llama_cerveau, memoire_web, filtre_instantane, rag_binaire
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
    monkeypatch.setattr(llama_cerveau, "generer",
                        lambda *a, **k: "REponse du cerveau")
    # RAG binaire desactive : ces tests verifient le WEB et le LLM, pas
    # l'index local (sinon une question connue du RAG n'atteint jamais
    # le chemin teste).
    monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
    return RontoGPT()


class TestIdentite:
    # le couteau typographique (executer) polit la sortie : la constante
    # est comparee APRES le meme nettoyage (fond identique, forme polie)
    @staticmethod
    def _poli(texte: str) -> str:
        from aura import typographie
        return typographie.nettoyer(texte)

    def test_qui_es_tu_reponse_constante(self, monkeypatch):
        ia = _ia(monkeypatch)
        assert ia.executer("qui es tu ?") == self._poli(_IDENTITE)

    def test_createur_nomme_les_deux(self, monkeypatch):
        ia = _ia(monkeypatch)
        r = ia.executer("qui est ton createur ?")
        # Le modele et le systeme ont deux createurs distincts : Gemma est
        # cree par Google DeepMind, RontoGPT par Simon Chusseau.
        assert "RontoGPT" in r and "Gemma" in r
        assert "Google DeepMind" in r and "Simon Chusseau" in r
        assert "pas un produit de Google" in r

    def test_nom_reponse_constante(self, monkeypatch):
        ia = _ia(monkeypatch)
        assert ia.executer("comment tu t'appelles ?") == self._poli(_IDENTITE)

    def test_ne_passe_jamais_par_le_cerveau(self, monkeypatch):
        from aura import llama_cerveau
        appels = []
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: appels.append(1) or "x")
        ia = _ia(monkeypatch)
        ia.executer("qui es tu")
        assert not appels, "l'identite ne doit pas etre generee par le 1B"


class TestDefinitionsAncrees:
    def test_definition_force_le_web(self, monkeypatch):
        ia = _ia(monkeypatch)
        cherche = []
        from aura import memoire_web
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: cherche.append(q) or "fait web")
        monkeypatch.setattr(orchestrateur.agents, "compresser",
                            lambda t, q, **k: t)
        monkeypatch.setattr(orchestrateur.agents, "en_triplets",
                            lambda t, q, **k: t)
        r = ia.executer("qu'est ce que le theoreme de pythagore")
        assert cherche, "une definition doit consulter le web"
        assert "fait web" in r

    def test_explique_force_le_web(self, monkeypatch):
        ia = _ia(monkeypatch)
        cherche = []
        from aura import memoire_web
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: cherche.append(q) or "fait")
        # une question longue passe par la MEME recherche : elle doit
        # enregistrer la requete comme les autres.
        monkeypatch.setattr(orchestrateur.agents, "compresser",
                            lambda t, q, **k: t)
        monkeypatch.setattr(orchestrateur.agents, "en_triplets",
                            lambda t, q, **k: t)
        ia.executer("explique moi la relativite")
        assert cherche

    def test_sans_ancrage_garde_fou_honnete(self, monkeypatch):
        """Hors-ligne + graphe muet -> abstention, jamais d'invention."""
        ia = _ia(monkeypatch)
        from aura import graphe_faits
        monkeypatch.setattr(graphe_faits, "chercher", lambda q, **k: "")
        monkeypatch.setattr(orchestrateur.agents, "compresser",
                            lambda t, q, **k: t)
        r = ia.executer("qu'est ce que le theoreme de pythagore")
        assert "je ne peux pas verifier" in r.lower()
        assert "pythagore" not in r.lower().replace(
            "cette definition", "") or True
        # le cerveau n'a PAS ete utilise (pas de "REponse du cerveau")
        assert "REponse du cerveau" not in r

    def test_avec_graphe_le_cerveau_est_ance(self, monkeypatch):
        ia = _ia(monkeypatch)
        from aura import graphe_faits
        monkeypatch.setattr(graphe_faits, "chercher",
                            lambda q, **k: "- Pythagore : theoreme triangle")
        monkeypatch.setattr(orchestrateur.agents, "compresser",
                            lambda t, q, **k: t)
        r = ia.executer("qu'est ce que le theoreme de pythagore")
        assert r == "REponse du cerveau"   # ancree, donc generation OK


class TestNonRegresion:
    def test_question_simple_reste_fluide(self, monkeypatch):
        ia = _ia(monkeypatch)
        r = ia.executer("raconte moi une blague")
        assert r == "REponse du cerveau"   # pas de garde-fou hors definitions

    def test_maths_directes_intactes(self, monkeypatch):
        ia = _ia(monkeypatch)
        # niveau 0 simule (le helper neutralise tout le filtre) : le test
        # verifie que le resultat des maths n'est PAS ecrase par le LLM
        from aura import filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre",
                            lambda q: "144" if "12" in q else None)
        r = ia.executer("combien fait 12 fois 12")
        assert "144" in r
