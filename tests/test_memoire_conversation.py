"""Tests memoire de conversation : RLM (MIT), fenetre glissante, etat
evenementiel. L'historique complet vit dans un FICHIER consultable par
outil ; la fenetre du prompt ne contient que les derniers tours ; l'etat
civil est extrait par regex en JSON compact.
"""
import json

import pytest

from aura.memoire_conversation import MemoireConversation


@pytest.fixture()
def memoire(tmp_path):
    m = MemoireConversation(chemin=tmp_path / "conv.jsonl", fenetre=4)
    yield m
    # chaque fixture a son propre fichier : aucune fuite entre tests


class TestRlm:
    def test_historique_ecrit_en_fichier(self, memoire):
        memoire.ajouter("user", "je m'appelle Pierre")
        memoire.ajouter("assistant", "Enchante, Pierre.")
        lignes = memoire._chemin.read_text(encoding="utf-8").splitlines()
        assert len(lignes) == 2
        assert json.loads(lignes[0])["content"] == "je m'appelle Pierre"

    def test_recherche_trouve_le_vieux_tour(self, memoire):
        for i in range(30):
            memoire.ajouter("user", f"question numero {i} sur la physique")
            memoire.ajouter("assistant", f"reponse {i}")
        memoire.ajouter("user", "mon code secret est ZEBRE-42, oublie rien")
        for i in range(20):
            memoire.ajouter("user", f"autre fil {i}")
        hit = memoire.rechercher("ZEBRE")
        assert "ZEBRE-42" in hit

    def test_recherche_sans_resultat(self, memoire):
        memoire.ajouter("user", "bonjour")
        assert memoire.rechercher("xylophone-inexistant") == ""

    def test_recherche_insensible_accents_casse(self, memoire):
        memoire.ajouter("user", "J'habite a Caen depuis 2020")
        assert "Caen" in memoire.rechercher("caen")

    def test_outil_declenche_par_la_question(self, memoire):
        memoire.ajouter("user", "le mot de passe du wifi est CAROTTE-7")
        memoire.ajouter("assistant", "note.")
        for i in range(10):
            memoire.ajouter("user", f"blabla {i}")
            memoire.ajouter("assistant", f"ok {i}")
        r = memoire.tourner("cherche dans l'historique : CAROTTE")
        assert r is not None and "CAROTTE-7" in r

    def test_pas_doutil_sans_demande(self, memoire):
        assert memoire.tourner("quelle heure est-il ?") is None


class TestFenetreGlissante:
    def test_fenetre_tronquee(self, memoire):
        for i in range(10):
            memoire.ajouter("user", f"u{i}")
            memoire.ajouter("assistant", f"a{i}")
        f = memoire.fenetre()
        assert len(f) == 8                       # 4 tours = 8 messages
        assert f[-1]["content"] == "a9"          # le plus recent au bout
        assert all(t["content"] not in ("u0", "a0") for t in f[:2])

    def test_vider_keeps_fichier(self, memoire):
        memoire.ajouter("user", "secret : PIN-1234")
        memoire.vider()
        assert memoire.fenetre() == []
        assert memoire.rechercher("PIN-1234") != ""   # RLM preserve

    def test_compacter_garde_letat_durable(self, memoire):
        """Compacter jette la matiere brute mais garde le nom et les faits."""
        memoire.ajouter("user", "Je m'appelle Pierre")
        memoire.ajouter("assistant", "Enchante, Pierre.")
        assert memoire.compacter() == 2
        assert memoire.fenetre() == []                    # fenetre videe
        assert memoire.etat()["utilisateur"] == "Pierre"  # etat conserve
        assert memoire.rechercher("Pierre") != ""          # RLM intact


class TestEtatEvenementiel:
    def test_nom_extrait(self, memoire):
        memoire.ajouter("user", "Je m'appelle Pierre")
        assert memoire.etat()["utilisateur"] == "Pierre"

    def test_possession_extraite(self, memoire):
        memoire.ajouter("user", "j'ai acheté 4 serveurs")
        assert memoire.etat()["serveurs"] == "4"

    def test_etat_json_compact(self, memoire):
        memoire.ajouter("user", "moi c'est Alice")
        memoire.ajouter("user", "j'ai 2 chatons")
        j = memoire.etat_json()
        assert json.loads(j) == {"utilisateur": "Alice", "chatons": "2"}
        assert len(j) < 80                        # quelques octets

    def test_dernier_etat_gagne(self, memoire):
        memoire.ajouter("user", "j'ai 4 serveurs")
        memoire.ajouter("user", "j'ai vendu mes serveurs, j'ai 1 serveur")
        assert memoire.etat()["serveur"] == "1"

    def test_bloc_systeme_injecte_etat(self, memoire):
        memoire.ajouter("user", "Je m'appelle Pierre")
        bloc = memoire.bloc_systeme("Tu es RontoGPT.")
        assert "Tu es RontoGPT." in bloc               # ancre preservee
        assert "Pierre" in bloc                    # etat ajoute
        assert "historique" in bloc                # outil annonce


class TestRotation:
    """Rotation de l'historique : le fichier RLM grandit sans limite avec
    l'usage ; au-dela du seuil il est archive (date) et la recherche
    traverse les archives — la memoire longue reste accessible.
    """

    @pytest.fixture()
    def memoire_seuil(self, tmp_path, monkeypatch):
        import aura.memoire_conversation as mc
        monkeypatch.setattr(mc, "_TAILLE_ROTATION", 1024)   # seuil de test
        m = MemoireConversation(chemin=tmp_path / "conv.jsonl", fenetre=4)
        return m

    def _remplir(self, chemin, contenu="remplissage", n=30):
        with chemin.open("w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps({"role": "user", "content": contenu},
                                   ensure_ascii=False) + "\n")

    def test_rotation_archive_le_fichier(self, memoire_seuil):
        chemin = memoire_seuil._chemin
        self._remplir(chemin)                    # > seuil (1024 o)
        memoire_seuil.ajouter("user", "tour apres rotation")
        archives = memoire_seuil._archives()
        assert len(archives) == 1                # l'ancien est archive
        assert "tour apres rotation" in chemin.read_text(encoding="utf-8")
        assert "remplissage" in archives[0].read_text(encoding="utf-8")

    def test_recherche_traverse_les_archives(self, memoire_seuil):
        chemin = memoire_seuil._chemin
        self._remplir(chemin, "le code wifi est CAROTTE-9")
        memoire_seuil.ajouter("user", "tour neuf")   # declenche la rotation
        hit = memoire_seuil.rechercher("CAROTTE")
        assert "CAROTTE-9" in hit                # retrouve dans l'archive

    def test_purge_des_vieilles_archives(self, tmp_path, monkeypatch):
        import aura.memoire_conversation as mc
        monkeypatch.setattr(mc, "_TAILLE_ROTATION", 1024)
        chemin = tmp_path / "conv.jsonl"
        m = MemoireConversation(chemin=chemin, fenetre=4)
        for i in range(4):                       # 4 fausses archives anciennes
            (tmp_path / f"conv-2020-01-0{i}-000000.jsonl").write_text(
                "{}", encoding="utf-8")
        self._remplir(chemin)
        m.ajouter("user", "declenche")
        archives = m._archives()
        assert len(archives) == mc._MAX_ARCHIVES  # purge au-dela de 3
        # les plus vieilles ont ete supprimees, les plus recentes gardees
        assert not (tmp_path / "conv-2020-01-00-000000.jsonl").exists()
        assert not (tmp_path / "conv-2020-01-01-000000.jsonl").exists()
        assert (tmp_path / "conv-2020-01-03-000000.jsonl").exists()
        # archives[0] = l'archive du jour, avec l'historique bascule dedans
        assert archives[0].stat().st_size > 50

    def test_rotation_transparente_pour_la_fenetre(self, memoire_seuil):
        chemin = memoire_seuil._chemin
        self._remplir(chemin)
        memoire_seuil.ajouter("user", "je m'appelle Pierre")
        assert memoire_seuil.etat()["utilisateur"] == "Pierre"
        assert memoire_seuil.fenetre()[-1]["content"] == "je m'appelle Pierre"
