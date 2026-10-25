"""Tests des 5 agents cognitifs d'RontoGPT.

1. Planificateur       : decoupe en etapes des taches actionables
2. Compresseur RAG     : 3 phrases les plus utiles au lieu du bruit web
3. Redacteur           : tics de langage retires, phrases dedoublees
4. Debugger recursif   : boucle sandbox -> erreur -> correction (max 3)
5. Masqueur            : prompt systeme par categorie routee

Plus : integration orchestrateur + failsafe (AURA_AGENTS=0).
"""

from aura import agents
from aura.orchestrateur import RontoGPT


def _ia_neutre(monkeypatch, analyse=None):
    from aura import filtre_instantane, llama_cerveau, rag_binaire
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
    monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
    if analyse is not None:
        monkeypatch.setattr(RontoGPT, "analyser", lambda self, q: analyse)
    return RontoGPT()


# ── 1. Planificateur ────────────────────────────────────────────────────

class TestPlanificateur:
    def test_signature_tache_complexe(self):
        assert agents.est_tache_complexe("organise un voyage a Rome")
        assert agents.est_tache_complexe("Prepares-moi un planning")
        assert agents.est_tache_complexe("liste les etapes pour monter un PC")
        assert not agents.est_tache_complexe("quelle heure est-il")
        assert not agents.est_tache_complexe("qui a decouvert le penicilline")

    def test_plan_valide(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "1. Reserver le vol\n2. Hotel\n3. Visites")
        plan = agents.planifier("organise un voyage a Rome")
        assert plan is not None and plan.startswith("1.") and "3. " in plan

    def test_plan_refuse_si_pas_numerote(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "D'abord le vol. Ensuite l'hotel.")
        assert agents.planifier("organise un voyage") is None

    def test_pas_de_plan_sur_question_simple(self):
        assert agents.planifier("quelle est la capitale du Japon") is None


# ── 2. Compresseur RAG ──────────────────────────────────────────────────

class TestCompresseur:
    LONG_WEB = ("Le prix de l or a clôturé à 2 650 dollars l once hier. "
                "La banque centrale a encore acheté 12 tonnes ce trimestre. "
                "Le rendement réel de l or reste faible en période de hausse des taux. "
                "Les bourses asiatiques ont rebondi de 1,2 pour cent. "
                "L inflation aux Etats-Unis ralentit à 2,4 pour cent sur un an. "
                "Les analysts s attendent à une consolidation du secteur minier. "
                "L or sert de valeur refuge en période d incertitude géopolitique.")

    def test_court_intact(self):
        assert agents.compresser("phrase courte", "question") == "phrase courte"

    def test_compresse_et_garde_le_pertinent(self):
        question = "quel est le prix de l or aujourd hui"
        sortie = agents.compresser(self.LONG_WEB, question)
        phrases = sortie.split(". ")
        assert 1 <= len(phrases) <= 3
        assert any("2 650" in p or "prix" in p.lower() for p in phrases)



# ── 3. Rédacteur ────────────────────────────────────────────────────────

class TestRedacteur:
    def test_tics_retires(self):
        brut = ("Il est important de noter que l or monte. "
                "Il faut souligner que le dollar baisse.")
        propre = agents.nettoyer_style(brut)
        assert "important de noter" not in propre.lower()
        assert "souligner" not in propre.lower()
        assert propre.startswith("L or monte")        # majuscule reprise

    def test_permets_de_conserve(self):
        brut = "Cet outil permet de calculer vite."
        assert agents.nettoyer_style(brut) == brut

    def test_phrase_dupliquee_une_fois(self):
        brut = "L or monte. L or monte. L or monte."
        assert agents.nettoyer_style(brut).count("L or monte") == 1

    def test_texte_normal_intact(self):
        brut = "La reponse est 42, calculee par le PGS en 12 millisecondes."
        assert agents.nettoyer_style(brut) == brut


# ── 4. Debugger récursif ────────────────────────────────────────────────

class TestDebuggerRecursif:
    BON = ("```python\ndef resoudre(n):\n"
           "    total = 0\n"
           "    for i in range(n):\n"
           "        total += i\n"
           "    return total\n"
           "assert resoudre(0) == 0\n"
           "assert resoudre(4) == 6\n"
           "assert resoudre(10) == 45\n```")
    CASSÉ = "```python\ndef resoudre(n):\n    return 1/0\nassert resoudre(1)\n```"

    def test_bon_code_premier_coup(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: self.BON)
        r = agents.boucle_code("somme des entiers")
        assert r is not None and "1 asserts" in r or "asserts" in r

    def test_correction_apres_echec(self, monkeypatch):
        """1er essai casse (ZeroDivision), 2e bon -> 1 correction auto."""
        from aura import llama_cerveau
        seq = iter([self.CASSÉ, self.BON])
        vus = []
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda p, **k: (vus.append(p), next(seq))[1])
        r = agents.boucle_code("somme des entiers")
        assert r is not None and "1 correction" in r
        # la relance contient le code casse ET l'erreur de la sandbox
        assert "1/0" in vus[1] and "ZeroDivisionError" in vus[1]

    def test_echec_total_renvoie_none(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: self.CASSÉ)
        assert agents.boucle_code("du code qui rate toujours") is None


# ── 5. Masqueur de personnalité ─────────────────────────────────────────

class TestMasqueur:
    def test_web(self):
        p = agents.personnalite_pour({"web"})
        assert p and "faits" in p.lower()

    def test_math(self):
        p = agents.personnalite_pour({"math"})
        assert p and "calcul" in p.lower()

    def test_general(self):
        p = agents.personnalite_pour({"general"})
        assert p is not None

    def test_rien_si_vide(self):
        assert agents.personnalite_pour(set()) is None
        assert agents.personnalite_pour(None) is None

    def test_composite_garde_la_base(self):
        p = agents.composer_personnalite({"web"}, "quel temps fait-il")
        assert p and p.startswith("You are a helpful assistant.")
        assert "faits" in p.lower()                  # masque ajoute

    def test_logique_garde_le_cot_natif(self):
        """Question de logique : None -> le suffixe CoT natif reste."""
        assert agents.composer_personnalite(
            {"web"}, "si Paul est plus grand que Leo, qui est le plus grand"
        ) is None


# ── Integration orchestrateur ───────────────────────────────────────────

class TestIntegrationOrchestrateur:
    def test_web_compresse_avant_llm(self, monkeypatch):
        """Le contexte web est filtre avant d'arriver au cerveau."""
        from aura import memoire_web
        analyse = {"experts": {"web"}, "scores": {"web": 0.9},
                   "methodes": ["t"]}
        ia = _ia_neutre(monkeypatch, analyse=analyse)
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: TestCompresseur.LONG_WEB)
        capturé = {}
        import aura.llama_cerveau as llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: capturé.update(ctx=a[1] if len(a) > 1
                                                           else k.get("contexte_web", ""))
                            or "ok")
        r = ia.executer("prix de l or aujourd hui")   # < 9 mots : mode simple
        assert r == "ok"
        assert len(capturé["ctx"].split(". ")) <= 4   # compresse

    def test_prompt_systeme_fixe_envoye_au_cerveau(self, monkeypatch):
        from aura import memoire_web
        analyse = {"experts": {"web"}, "scores": {"web": 0.9},
                   "methodes": ["t"]}
        ia = _ia_neutre(monkeypatch, analyse=analyse)
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: "Le fait verifie.")
        perso = {}
        import aura.llama_cerveau as llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, systeme=None, **k:
                            perso.update(s=systeme) or "ok")
        ia.executer("quel est le prix du petrole")
        assert perso["s"] == "You are a helpful assistant."

    def test_pipeline_general_intact(self, monkeypatch):
        """Chemin simple sans expert : comportement d'avant (pas de plan)."""
        analyse = {"experts": {"general"}, "scores": {"general": 0.9},
                   "methodes": ["t"]}
        ia = _ia_neutre(monkeypatch, analyse=analyse)
        assert ia.executer("bonjour, comment vas-tu") == "ok-llm"


# ── Failsafe ────────────────────────────────────────────────────────────

class TestFailsafe:
    def test_aura_agents_0_tout_desactive(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENTS", "0")
        assert not agents.actifs()
        assert agents.personnalite_pour({"web"}) is None
        # la compression renvoie le texte intact
        assert agents.compresser(TestCompresseur.LONG_WEB, "q") == \
            TestCompresseur.LONG_WEB

    def test_planificateur_inactif_sans_agents(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENTS", "0")
        from aura import llama_cerveau
        appels = []
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: appels.append(1) or "x")
        assert agents.planifier("organise un voyage") is None
        assert not appels                            # aucun appel cerveau
