"""Tests des 2 derniers experts : logique exacte (mini-SAT) + PoT-code (sandbox).

Cibles : les 2 territoires ou le 8B gagnait encore — la logique SANS
nombres et le code.
"""
import pytest

from aura import logique, potcode
from aura.orchestrateur import RontoGPT


# ── 1. Solveur logique (backtracking pur Python) ────────────────────────

class TestLogique:
    def test_chevaliers_menteurs_symetrie_classique(self):
        """Une seule declaration = 2 solutions cohérentes (symétrie classique).

        Paul chevalier + Marie menteuse OU Paul menteur + Marie chevalière —
        les deux satisfont « Paul DIT Marie == menteur ». Le solveur doit
        les trouver TOUTES les deux (et refuser de trancher arbitrairement).
        """
        puzzle = logique.parser_puzzle(
            "ENTITES : Paul, Marie\n"
            "DOMAINE : chevalier, menteur\n"
            "Paul DIT Marie == menteur\n")
        r = logique.resoudre(puzzle)
        assert r["statut"] == "multiple"
        assert len(r["solutions"]) == 2
        types = {(s["Paul"], s["Marie"]) for s in r["solutions"]}
        assert types == {("chevalier", "menteur"), ("menteur", "chevalier")}

    def test_attributions_tous_differents(self):
        puzzle = logique.parser_puzzle(
            "ENTITES : Anna, Bob, Cal\n"
            "DOMAINE : rouge, vert, bleu\n"
            "TOUS DIFFERENTS\n"
            "CONDITION : Anna == rouge\n"
            "CONDITION : Bob != vert\n")
        r = logique.resoudre(puzzle)
        assert r["statut"] == "unique"
        s = r["solutions"][0]
        assert s["Anna"] == "rouge" and s["Bob"] == "bleu" and s["Cal"] == "vert"

    def test_aucune_solution_detectee(self):
        puzzle = logique.parser_puzzle(
            "ENTITES : A, B\nDOMAINE : x, y\nCONDITION : A == x\n"
            "CONDITION : A != x\n")
        assert logique.resoudre(puzzle)["statut"] == "aucune"

    def test_multiple_solutions_detecte(self):
        puzzle = logique.parser_puzzle(
            "ENTITES : A, B\nDOMAINE : x, y\nCONDITION : A == x\n")
        assert logique.resoudre(puzzle)["statut"] == "multiple"

    def test_parseur_refuse_le_bruit(self):
        with pytest.raises(ValueError):
            logique.parser_puzzle("Je pense que la reponse est peut-etre oui")
        with pytest.raises(ValueError):
            logique.parser_puzzle("ENTITES : A\nDOMAINE : x, y\nCONDITION : A == x")
        with pytest.raises(ValueError):
            logique.parser_puzzle("ENTITES : A, B\nDOMAINE : x, y\nCONDITION : Z == x")

    def test_expressions_composees(self):
        puzzle = logique.parser_puzzle(
            "ENTITES : A, B, C\nDOMAINE : v1, v2\n"
            "CONDITION : NON (A == v1) OU B == v2\n")
        r = logique.resoudre(puzzle)
        assert r["statut"] in ("unique", "multiple")

    def test_chevaliers_menteurs_unique_avec_declaration_fermee(self):
        """Declaration qui ferme : « A DIT ... ET CONDITION » brise la symétrie."""
        puzzle = logique.parser_puzzle(
            "ENTITES : Paul, Marie\n"
            "DOMAINE : chevalier, menteur\n"
            "Paul DIT Marie == menteur\n"
            "CONDITION : Marie == chevalier\n")
        r = logique.resoudre(puzzle)
        assert r["statut"] == "unique"
        assert r["solutions"][0]["Paul"] == "menteur"
        assert r["solutions"][0]["Marie"] == "chevalier"

    def test_pipeline_logique_complet(self, monkeypatch):
        """Le 1B ne fait QUE formaliser ; le solveur tranche ET redige.

        Contrat du 12 oct. 2026 : le 1B ne recoit jamais la consigne brute
        pour repondre (il begayait/reformulait) — un seul usage, l'extraction
        ENTITES/DOMAINE/CONDITION, et la reponse sort du solveur.
        """
        from aura import llama_cerveau
        formalisme = ("ENTITES : Anna, Bob, Cal\n"
                      "DOMAINE : rouge, vert, bleu\n"
                      "TOUS DIFFERENTS\n"
                      "CONDITION : Anna == rouge\n"
                      "CONDITION : Bob != vert")
        question = ("Anna, Bob et Cal portent chacun une couleur differente : "
                    "rouge, vert ou bleu. Anna porte du rouge. Bob ne porte "
                    "pas de vert. Qui porte quoi ?")
        appels = {"n": 0}

        def faux_generer(q, **kw):
            appels["n"] += 1
            assert kw.get("systeme") == logique._SYSTEME_LOGIQUE, (
                "le 1B ne doit servir qu'a extraire le formalisme")
            return formalisme

        monkeypatch.setattr(llama_cerveau, "generer", faux_generer)
        rep = RontoGPT()._resoudre_par_logique(question)
        assert "Anna : rouge" in rep
        assert "Bob : bleu" in rep
        assert "Cal : vert" in rep
        # majorite : 3 formalisations votees (identiques ici -> 3/3), et
        # AUCUN appel de synthese
        assert appels["n"] == 3


# ── 2. PoT-code (sandbox a builtins restreints) ─────────────────────────

class TestPotCode:
    BON_CODE = (
        "def resoudre(donnees):\n"
        "    return sorted(donnees)\n"
        "assert resoudre([3, 1, 2]) == [1, 2, 3]\n"
        "assert resoudre([]) == []\n"
        "assert resoudre([2]) == [2]\n"
    )

    def test_extraire_code(self):
        brut = "Voici :\n```python\n" + self.BON_CODE + "\n```\nCela devrait marcher."
        assert potcode.extraire_code(brut) == self.BON_CODE.strip()
        assert potcode.extraire_code("pas de bloc") is None

    def test_code_valide_passe(self):
        r = potcode.verifier_code(self.BON_CODE)
        assert r["statut"] == "valide" and r["nb_asserts"] == 3

    def test_assert_rate_refuse(self):
        code = ("def resoudre(d):\n    return d\n"
                "assert resoudre([1]) == [2]\n")
        with pytest.raises(ValueError, match="assert"):
            potcode.verifier_code(code)

    def test_import_interdit(self):
        code = ("import os\n"
                "def resoudre(d):\n    return d\n"
                "assert resoudre([]) == []\n")
        with pytest.raises(ValueError):
            potcode.verifier_code(code)

    def test_boucle_infinie_coupee(self):
        code = ("def resoudre(d):\n"
                "    i = 0\n"
                "    while True:\n"
                "        i = i + 1\n"
                "    return i\n"
                "assert resoudre([]) == 0\n")
        with pytest.raises(ValueError, match="budget"):
            potcode.verifier_code(code, budget=50_000)

    def test_fichier_interdit(self):
        code = ("def resoudre(d):\n"
                "    return open('/etc/passwd').read()\n"
                "assert resoudre(1)\n")
        with pytest.raises(ValueError):
            potcode.verifier_code(code)

    def test_aucun_assert_refuse(self):
        code = "def resoudre(d):\n    return d\n"
        with pytest.raises(ValueError, match="assert"):
            potcode.verifier_code(code)

    def test_fonction_absente_refusee(self):
        code = "x = 1\nassert x == 1\n"
        with pytest.raises(ValueError, match="resoudre"):
            potcode.verifier_code(code)

    def test_pipeline_code_complet(self, monkeypatch):
        from aura import llama_cerveau
        bloc = "```python\n" + self.BON_CODE + "\n```"

        def faux_generer(q, **kw):
            if kw.get("systeme") == potcode._SYSTEME_CODE:
                return bloc
            return "generique"

        monkeypatch.setattr(llama_cerveau, "generer", faux_generer)
        rep = RontoGPT()._resoudre_par_code("ecris une fonction de tri")
        assert "```python" in rep and "3 asserts" in rep

    def test_code_casse_replie_a_none(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, **kw: "```python\ndef resoudre(d):\n    return 1/0\nassert resoudre([]) == 0\n```")
        assert RontoGPT()._resoudre_par_code("du code") is None


# ── 3. Routage des experts speciaux ─────────────────────────────────────

class TestRoutageSpeciaux:
    def test_enigme_route_vers_logique(self, monkeypatch):
        monkeypatch.setattr(RontoGPT, "_resoudre_par_logique",
                            lambda self, q, t0=None: "solution logique")
        r = RontoGPT()._expert_special(
            "Deux chevaliers et un menteur : qui dit la verite ?", None, None)
        assert r["cerveau_choisi"] == "logique-1b+sat"

    def test_code_route_vers_sandbox(self, monkeypatch):
        monkeypatch.setattr(RontoGPT, "_resoudre_par_code",
                            lambda self, q, mode_auto=False:
                            "def resoudre(): pass")
        r = RontoGPT()._expert_special(
            "ecris une fonction python qui trie une liste", None, None)
        assert r["cerveau_choisi"] == "potcode-1b+sandbox"

    def test_donnees_numeriques_passent_avant(self, monkeypatch):
        appels = []
        monkeypatch.setattr(RontoGPT, "_resoudre_par_logique",
                            lambda self, q, t0=None: appels.append(1) or None)
        assert RontoGPT()._expert_special(
            "enigme de logique quelconque", [[1.0], [2.0]], [1.0, 4.0]) is None
        assert not appels

    def test_question_normale_ne_declenche_rien(self):
        assert RontoGPT()._expert_special(
            "capitale de l australie", None, None) is None
class TestSandboxEvasions:
    """`().__class__.__bases__[0]` ne demande aucun builtin : la barriere
    doit etre dans l'ARBRE, pas dans la liste des builtins autorises."""

    @pytest.mark.parametrize("code", [
        "\n".join(["def resoudre(d):",
                    "    return ().__class__.__bases__[0].__subclasses__()",
                    "assert resoudre([])"]),
        "\n".join(["def resoudre(d):",
                    "    return ''.__class__.__mro__",
                    "assert resoudre('')"]),
        "\n".join(["def resoudre(d):",
                    "    return __import__('os')",
                    "assert resoudre(1)"]),
        "\n".join(["def resoudre(d):",
                    "    return '{0.__class__}'.format(())",
                    "assert resoudre(1)"]),
        "\n".join(["import os",
                    "def resoudre(d):",
                    "    return d",
                    "assert resoudre(1)"]),
    ])
    def test_evasion_refusee(self, code):
        with pytest.raises(ValueError):
            potcode.verifier_code(code)

    def test_le_code_ordinaire_passe_toujours(self):
        code = "\n".join([
            "def resoudre(d):",
            "    return sorted(x * 2 for x in d if x > 0)",
            "assert resoudre([1, -2, 3]) == [2, 6]",
        ])
        assert potcode.verifier_code(code)["statut"] == "valide"
