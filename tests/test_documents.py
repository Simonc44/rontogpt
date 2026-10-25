"""RAG documents locaux : le besoin « parler a ses fichiers ».

Ce qui est verrouille ici :
- le decoupage borne et recouvrant (un passage coupe au milieu d'une phrase
  perd le contexte) ;
- l'index persistant et le remplacement d'un fichier re-ajoute ;
- la recherche avec repli lexical quand MiniLM est absent ;
- l'injection dans le pipeline et la ligne « Sources : », sans laquelle un
  extrait local n'est pas verifiable.
"""
import pytest

from aura import documents as doc


@pytest.fixture
def index_isole(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "DOSSIER", tmp_path / "index_docs")
    doc._vider_cache()
    yield tmp_path / "index_docs"
    doc._vider_cache()


class TestDecoupage:
    def test_borne_et_recouvrement(self):
        texte = " ".join(
            f"Phrase numero {i} sur le budget et les vacances." for i in range(120))
        morceaux = doc.decouper(texte)
        assert len(morceaux) >= 2
        assert all(len(m) <= doc.TAILLE_MORCEAU + 10 for m in morceaux)
        assert all(m.strip() for m in morceaux)
        # Le recouvrement duplique du texte : la somme depasse l'original.
        assert sum(map(len, morceaux)) > len(texte)

    def test_texte_vide(self):
        assert doc.decouper("") == []
        assert doc.decouper("   \n\n  ") == []

    def test_paragraphe_tres_long(self):
        texte = "A" * 2500
        morceaux = doc.decouper(texte)
        assert morceaux
        assert all(len(m) <= doc.TAILLE_MORCEAU + 10 for m in morceaux)


class TestIndex:
    def test_ajout_recherche_et_citation(self, index_isole):
        fichier = index_isole.parent / "notes.md"
        fichier.write_text(
            "# Budget\n\nLe budget 2026 du projet Atlas est de 42 000 euros.\n\n"
            "# Vacances\n\nLes vacances d'ete commencent en juillet.\n",
            encoding="utf-8")
        bilan = doc.ajouter(fichier)
        assert bilan["nom"] == "notes.md"
        assert bilan["morceaux"] >= 1

        hits = doc.chercher("Quel est le budget du projet Atlas ?", k=2)
        assert hits, "le passage budget doit etre retrouve"
        assert hits[0]["nom"] == "notes.md"
        assert "42 000" in hits[0]["texte"]

        resultat = doc.interroger("Quel est le budget du projet Atlas ?")
        assert resultat["sources"] == ["notes.md"]
        assert resultat["texte"].startswith("EXTRAITS DE DOCUMENTS LOCAUX")
        assert "42 000" in resultat["texte"]

    def test_persistance(self, index_isole):
        fichier = index_isole.parent / "notes.txt"
        fichier.write_text("Le budget Atlas est de 42 000 euros.", encoding="utf-8")
        doc.ajouter(fichier)
        doc._vider_cache()          # comme un nouveau processus
        assert len(doc.lister()) == 1
        assert doc.chercher("budget Atlas")

    def test_reajout_remplace_les_anciens_passages(self, index_isole):
        fichier = index_isole.parent / "notes.md"
        fichier.write_text("Le budget du projet Atlas est de 42 000 euros.",
                           encoding="utf-8")
        doc.ajouter(fichier)
        fichier.write_text("Les vacances d'ete commencent en juillet.",
                           encoding="utf-8")
        doc.ajouter(fichier)
        assert len(doc.lister()) == 1
        textes = " ".join(d["texte"] for d in doc._charger()[0])
        assert "Atlas" not in textes
        assert "vacances" in textes

    def test_suppression(self, index_isole):
        fichier = index_isole.parent / "notes.md"
        fichier.write_text("Le budget Atlas est de 42 000 euros.", encoding="utf-8")
        doc.ajouter(fichier)
        assert doc.supprimer("notes.md") >= 1
        assert doc.lister() == []
        assert doc.chercher("budget Atlas") == []

    def test_index_vide_ne_leve_pas(self, index_isole):
        assert doc.chercher("n'importe quoi") == []
        vide = doc.interroger("n'importe quoi")
        assert vide["texte"] == "" and vide["sources"] == []
        assert doc.statistiques()["morceaux"] == 0

    def test_repli_lexical_sans_minilm(self, index_isole, monkeypatch):
        monkeypatch.setattr(doc.embeddings, "encoder", lambda textes: None)
        fichier = index_isole.parent / "notes.md"
        fichier.write_text("Le budget Atlas est de 42 000 euros.", encoding="utf-8")
        doc.ajouter(fichier)
        hits = doc.chercher("budget Atlas")
        assert hits and hits[0]["nom"] == "notes.md"


class TestLecture:
    def test_extension_inconnue(self, tmp_path):
        fichier = tmp_path / "notes.docx"
        fichier.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            doc.lire(fichier)

    def test_fichier_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            doc.lire(tmp_path / "absent.txt")


class TestBoutonDocuments:
    """Le panneau de gauche doit fonctionner sans taper de chemin.

    Le CLI existait deja ; sans bouton, l'utilisateur devait ouvrir un
    terminal pour ajouter un PDF — c'est exactement le frottement qui fait
    abandonner une app locale.
    """

    class _FauxSelecteur:
        def __init__(self, choix):
            self.choix = choix

        def create_file_dialog(self, *a, **k):
            return self.choix

    def test_ajout_listage_et_retrait(self, index_isole, tmp_path):
        from aura.gui_web import Api

        fichier = tmp_path / "notes.md"
        fichier.write_text("# Projet Atlas\n\nLe budget 2026 est de 42 000 euros.",
                           encoding="utf-8")
        api = Api()
        api.attacher_fenetre(self._FauxSelecteur([str(fichier)]))

        r = api.documents_ajouter()
        assert r["ok"] is True and r["ajoutes"] == 1, r
        assert [f["nom"] for f in api.documents_liste()["fichiers"]] == ["notes.md"]

        assert api.documents_supprimer("notes.md")["supprimes"] == 1
        assert api.documents_liste()["fichiers"] == []

    def test_annulation_est_silencieuse(self, index_isole):
        from aura.gui_web import Api

        api = Api()
        api.attacher_fenetre(self._FauxSelecteur(None))
        r = api.documents_ajouter()
        assert r["ok"] is False and r["erreur"] == "annule"

    def test_un_fichier_illisible_ne_bloque_pas_les_autres(
            self, index_isole, tmp_path):
        from aura.gui_web import Api

        bon = tmp_path / "bon.txt"
        bon.write_text("Le budget est de 42 000 euros.", encoding="utf-8")
        mauvais = tmp_path / "mauvais.docx"
        mauvais.write_text("x", encoding="utf-8")
        api = Api()
        api.attacher_fenetre(self._FauxSelecteur([str(mauvais), str(bon)]))

        r = api.documents_ajouter()
        assert r["ok"] is True and r["ajoutes"] == 1, r
        assert len(r["erreurs"]) == 1
        assert [f["nom"] for f in r["fichiers"]] == ["bon.txt"]

    def test_index_indisponible_rend_une_erreur_propre(self, monkeypatch):
        from aura.gui_web import Api

        monkeypatch.setattr(doc, "lister",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        r = Api().documents_liste()
        assert r["ok"] is False and "boom" in r["erreur"]

    def test_le_bouton_de_la_gui_est_branche(self):
        """La methode peut exister sans qu'aucun bouton ne l'appelle."""
        from pathlib import Path
        racine = Path(__file__).resolve().parent.parent
        source = (racine / "aura" / "web" / "index.html").read_text(
            encoding="utf-8")
        assert "btn-doc-ajouter" in source
        assert "api.documents_ajouter()" in source
        assert "api.documents_liste()" in source


class TestPipeline:
    def test_les_documents_sont_injectes_et_cites(self, monkeypatch):
        from aura import agents, filtre_instantane, memoire_web
        from aura import documents as documents_reel
        from aura import orchestrateur as orch

        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
        monkeypatch.setattr(agents, "planifier", lambda q: "")
        monkeypatch.setenv("AURA_VERIF_WEB", "0")
        monkeypatch.setattr(
            documents_reel, "interroger",
            lambda q, **k: {"texte": "DOCUMENTS : le budget Atlas est de "
                                    "42 000 euros.",
                            "sources": ["notes.md"]})
        vus = {}

        def faux(question, *a, **k):
            vus["faits"] = k.get("contexte_faits") or ""
            return "Le budget du projet Atlas est de 42 000 euros."

        monkeypatch.setattr(orch.llama_cerveau, "generer", faux)
        ia = orch.RontoGPT()
        resultat = ia.executer_detaille("Quel est le budget du projet Atlas ?")
        assert "42 000" in vus["faits"], vus.get("faits")
        assert resultat["reponse"].rstrip().endswith("Sources : notes.md")

    def test_lookup_documentaire_direct_sans_generation(self, monkeypatch):
        """Similarite forte sur un lookup : on cite, on ne genere pas."""
        from aura import filtre_instantane, memoire_web
        from aura import documents as documents_reel
        from aura import orchestrateur as orch

        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
        monkeypatch.setenv("AURA_VERIF_WEB", "0")
        monkeypatch.setattr(
            documents_reel, "interroger",
            lambda q, **k: {
                "texte": "EXTRAITS DE DOCUMENTS LOCAUX :\n[1] notes.md : ...",
                "sources": ["notes.md"],
                "hits": [{"nom": "notes.md", "score": 0.9,
                          "texte": "Le budget Atlas est de 42 000 euros."}]})
        appels = []

        def faux(*a, **k):
            appels.append(1)
            return "reponse generee"

        monkeypatch.setattr(orch.llama_cerveau, "generer", faux)
        resultat = orch.RontoGPT().executer_detaille(
            "Qu'est-ce que le budget du projet Atlas ?")
        assert appels == [], "un lookup fort ne doit pas appeler le cerveau"
        assert resultat["cerveau_choisi"] == "documents"
        assert resultat["reponse"].startswith("D'apres notes.md")
        assert "42 000" in resultat["reponse"]
