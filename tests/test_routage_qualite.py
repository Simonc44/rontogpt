"""Routage qualite — garde-fous issus des defauts trouves par le banc.

Defauts mesures puis corriges (scripts/bench_qualite.py) :
    meta-long  : « Est-ce que tu es une IA ... » partait en dissertation (75 s)
    factuel    : « Quel est le prix du carburant ? » aussi (86 s)
    cache      : un refus en DOUBLE etait servi depuis le cache (0,1 s)
    tetes      : « Je suis desole, mais... » en tete d'une vraie reponse
    bribe      : « ok » -> le 1B ne repondait pas, il CONTINUAIT le format
                 « Q: ... R: ... » de la brique few-shot et inventait une
                 question attribuee a l'utilisateur
"""
from aura import llama_cerveau
from aura import orchestrateur as orch


class TestMetaEtReponseCourte:
    def test_meta_long_detectee(self):
        # etait 74,9 s de dissertation au banc
        assert orch._est_meta_ia(
            "Est-ce que tu es une intelligence artificielle vraiment capable "
            "de comprendre ce que tu ecris, ou tu simules seulement ?") is True

    def test_meta_courte_detectee(self):
        assert orch._est_meta_ia("es tu une ia aussi puissn qu'un 8B ou pluas")
        assert orch._est_meta_ia("Tu es plus fort que ChatGPT ?")
        assert orch._est_meta_ia("Tu es conscient de ce que tu dis ?")

    def test_questions_longues_non_meta(self):
        # pas de fausse positif : un essai reste un essai
        assert orch._est_meta_ia(
            "La puissance de la technologie est-elle une menace pour l'homme "
            "moderne ? Redige une dissertation structuree.") is False

    def test_reponse_courte_garde_un_budget_court(self):
        """« quel ... » / « combien ... » : budget court, pas une redaction.

        Le mode dissertation (multi-pass, 75-86 s mesures au banc) a ete
        supprime ; ce qui reste de cette garde, c'est le BUDGET : une
        question factuelle ne doit pas partir en 320 tokens de developpement.
        """
        for q in ("Quel est le prix du carburant en France aujourd'hui ?",
                  "Combien de temps dure le voyage ?"):
            assert orch.llama_cerveau._max_tokens_adaptatif(q) == 120, q

    def test_enonce_a_etapes_suit_le_cerveau(self):
        """Un PROBLEME a etapes recoit le budget du CERVEAU, pas un budget fixe.

        Depuis le 11 oct. 2026 le cerveau par defaut est un RAISONNEUR
        distille (R1) : un enonce a etapes recoit _TOKENS_RAISONNEMENT (768).
        Avec un cerveau NON raisonneur (AURA_GGUF=GEMMA), la meme question
        garde 120 : elargir le budget d'un non-raisonneur le fait DERIVER
        (mesure GSM8K : 0,12 -> 0,03). Le test suit donc le cerveau charge.
        """
        q = ("Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que "
             "Clara. Clara a 4 ans. Quel age a Alice ?")
        if orch.llama_cerveau._cerveau_raisonneur():
            attendu = orch.llama_cerveau._TOKENS_RAISONNEMENT
        else:
            attendu = 120
        assert orch.llama_cerveau._max_tokens_adaptatif(q) == attendu

    def test_redaction_demande_recoit_le_budget_developpe(self):
        """« Redige / Explique » : la reponse est longue, le budget doit l'etre."""
        assert orch.llama_cerveau._max_tokens_adaptatif(
            "Redige une analyse comparee du solaire et de l'eolien en "
            "France, avec les avantages et les limites de chaque filiere"
        ) == orch.llama_cerveau._TOKENS_DEVELOPPE
        assert orch.llama_cerveau._max_tokens_adaptatif(
            "Explique pourquoi la photosynthese est indispensable a la vie "
            "sur Terre en developpant son role dans la chaine alimentaire"
        ) == orch.llama_cerveau._TOKENS_DEVELOPPE


class TestRefus:
    REFUS_COURT = "Je suis desole, mais je ne peux pas repondre a cette demande."
    REFUS_PUIS_SUITE = (
        "Je suis desole, mais je ne peux pas repondre a cette demande. "
        + "Pourtant l'analyse de la question montre que la puissance se "
        "mesure en watts et que la notion recouvre des domaines tres "
        "varies : physique d'abord, puis metaphorique ensuite, si bien "
        "que la reponse depend du cadre retenu par l'interlocuteur. ")

    def test_refus_court_identifie(self):
        assert orch._est_refus(self.REFUS_COURT) is True

    def test_reponse_normale_nest_pas_un_refus(self):
        assert orch._est_refus("42") is False
        assert orch._est_refus("Le solaire progresse en Europe.") is False

    def test_refus_puis_suite_restaure(self):
        # la tete parasite est coupee, la vraie reponse reste
        propre = orch._retirer_refus_tete(self.REFUS_PUIS_SUITE)
        assert not propre.lower().startswith("je suis desole")
        assert "analyse de la question" in propre

    def test_refus_sans_suite_est_conserve(self):
        # un refus honnete (hors-ligne, garde-fou) ne doit pas etre vide
        assert orch._retirer_refus_tete(self.REFUS_COURT) == self.REFUS_COURT

    def test_reponse_sans_refus_intouchee(self):
        assert orch._retirer_refus_tete("42") == "42"
        assert orch._retirer_refus_tete("") == ""


class TestCachePropre:
    def test_refus_en_cache_est_bypass(self, monkeypatch):
        """Un refus sali dans le cache ne doit pas etre livre comme une
        reponse : le pipeline complet prend le relais."""
        from aura import llama_cerveau
        question = "Qui est le plus age ? Paul, Marie ou Leo ?"
        monkeypatch.setattr("aura.filtre_instantane.repondre",
                            lambda q: orch._RE_REFUS.pattern and
                            "Je suis desole, mais je ne peux pas repondre.")
        monkeypatch.setattr(orch.RontoGPT, "analyser",
                            lambda self, q: {"methodes": ["test"]})
        # ni RAG ni experts ni LLM : on s'arrete au resultat final
        monkeypatch.setattr("aura.rag_binaire.chercher", lambda q: None)
        monkeypatch.setattr(orch.RontoGPT, "_expert_special", lambda *a, **k: None)
        monkeypatch.setattr("aura.fast_lang.moteur.repondre", lambda q: None)
        monkeypatch.setattr("aura.calcul_verbal.resoudre", lambda q: None)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "Paul est le plus age, 12 ans.")
        r = orch.RontoGPT().executer_detaille(question)
        assert r["cerveau_choisi"] != "niveau0-instantane"
        assert "desole" not in r["reponse"].lower()


class TestLibellesDePlan:
    """Le 1B recopie parfois l'en-tete de son prompt dans sa redaction.

    Defaut mesure au banc : riche-puissance KO car « SECTION A REDIGER :
    PARTIE 2 - L'antithese » figurait litteralement dans la reponse (avec,
    en plus, la phrase d'intro recopiee avec un autre premier mot).
    """

    def test_libelle_colle_au_texte_puis_suite(self):
        brut = ("**SECTION A REDIGER : PARTIE 2 - L'antithese : la negation** "
                "Pourtant, la puissance se manifeste d'abord partout.")
        propre = llama_cerveau._sans_libelles(brut)
        assert "SECTION" not in propre
        assert "Pourtant, la puissance" in propre          # vrai texte garde

    def test_libelle_seul_sur_sa_ligne(self):
        brut = ("SECTION A REDIGER : PARTIE 1 : L'AFFIRMATION\n"
                "La these se pose des la premiere phrase.")
        propre = llama_cerveau._sans_libelles(brut)
        assert "SECTION A REDIGER" not in propre
        assert "La these se pose" in propre

    def test_partie_et_connecteurs_retires(self):
        assert "PARTIE 3" not in llama_cerveau._sans_libelles(
            "PARTIE 3 : La synthese\nTout se joue alors.")
        assert "CONNECTEURS" not in llama_cerveau._sans_libelles(
            "CONNECTEURS : neanmoins, donc, pourtant\nIl reste que...")

    def test_texte_sans_libelle_intouche(self):
        t = "La puissance se mesure. Elle se discute aussi."
        assert llama_cerveau._sans_libelles(t) == t

    def test_cache_sali_est_nettoye_au_service(self, monkeypatch):
        """Une vieille entree de cache contenant la balise est nettoyee
        AVANT d'etre servie (pas seulement a l'ecriture)."""
        sale = ("Bonjour. **SECTION A REDIGER : PARTIE 1 : L'AFFIRMATION** "
                "La these se pose des la premiere phrase, sans ambiguite.")
        monkeypatch.setattr("aura.filtre_instantane.repondre", lambda q: sale)
        monkeypatch.setattr(orch.RontoGPT, "analyser",
                            lambda self, q: {"methodes": ["test"]})
        monkeypatch.setattr("aura.rag_binaire.chercher", lambda q: None)
        monkeypatch.setattr(orch.RontoGPT, "_expert_special", lambda *a, **k: None)
        monkeypatch.setattr("aura.fast_lang.moteur.repondre", lambda q: None)
        monkeypatch.setattr("aura.calcul_verbal.resoudre", lambda q: None)
        r = orch.RontoGPT().executer_detaille(
            "La puissance de la technologie est-elle une menace ?")
        assert "SECTION A REDIGER" not in r["reponse"]
        assert "La these se pose" in r["reponse"]


class TestCanalisationDu1B:
    """Le 1B RECLAME son contexte au lieu de repondre.

    Mesure du 12 oct. 2026, question « Qui a gagne le tournoi de tennis de
    Roland-Garros en 2025 ? » avec un contexte web de 3535 car. POURTANT
    joint : le modele a recite son prompt systeme — « J'ai besoin des
    informations pour repondre a ta question. Dis-moi : « Qu'est-ce qui est
    fourni ci-dessous et est pertinent » (WEB / BASE DE FAITS) ... dis-moi ce
    que c'est. » — soit 82 s de calcul pour rendre la question a
    l'utilisateur.
    """

    FUITE = ("J'ai besoin des informations pour repondre a ta question. "
             "Dis-moi : « Qu'est-ce qui est fourni ci-dessous et est "
             "pertinent » (WEB / BASE DE FAITS) ... dis-moi ce que c'est.")

    def test_fuite_mesuree_retiree(self):
        propre = llama_cerveau._sans_parrotage(self.FUITE)
        assert propre == ""                  # rien de livrable dedans
        assert "BASE DE FAITS" not in propre
        assert "dis-moi" not in propre.lower()

    def test_variante_sources_retiree(self):
        # 2e mesure (12 oct. 2026), apres le 1er correctif : le 1B a change
        # de nom pour son besoin — « J'ai besoin des SOURCES pour repondre »
        # — et collait le titre de la source dans sa demande. La tournure
        # est donc reconnue sur sa FORME, pas sur le nom du besoin.
        variante = ("J'ai besoin des sources pour repondre : « Titre : "
                    "Roland-Garros 2025 : le recap des champions ! | Source : "
                    "https://www.rolandgarros.com/fr-fr/article/recap")
        propre = llama_cerveau._sans_parrotage(variante)
        assert "besoin des sources" not in propre
        assert "rolandgarros.com" not in propre

    def test_etiquette_au_milieu_dune_phrase(self):
        propre = llama_cerveau._sans_parrotage(
            "Le match a offert (WEB / BASE DE FAITS) un niveau de jeu rare.")
        assert "BASE DE FAITS" not in propre
        assert "Le match a offert" in propre
        assert "un niveau de jeu rare" in propre

    def test_phrase_de_demande_isolee_retiree(self):
        propre = llama_cerveau._sans_parrotage(
            "Canberra est la capitale. Dis-moi ce que tu veux savoir.")
        assert "Canberra est la capitale." in propre
        assert "Dis-moi" not in propre

    def test_reponse_saine_intacte(self):
        t = ("La diffusion de Rayleigh explique le bleu du ciel. La lumiere "
             "est diffusee par les molecules de l'atmosphere.")
        assert llama_cerveau._sans_parrotage(t) == t

    def test_absence_dautre_solution_gardee(self):
        # « je ne vois pas » n'est PAS du bavardage quand il porte sur une
        # SOLUTION : la phrase doit rester (sinon on ampute une vraie
        # reponse en croyant nettoyer une fuite).
        t = "Je ne vois pas d'autre solution que de comparer les couts totaux."
        assert llama_cerveau._sans_parrotage(t) == t

    def test_systeme_sans_la_phrase_recopiee(self):
        # la CAUSE est traitee : plus rien a reciter dans le prompt systeme
        assert "fournis ci-dessous" not in llama_cerveau._SYSTEME
        assert "FAITS WEB ou une FORMULE" not in llama_cerveau._SYSTEME
        assert llama_cerveau._SYSTEME == "You are a helpful assistant."


class TestAntiCopieProche:
    """Copie avec un premier mot different : la comparaison exacte rate
    (vu au banc), le fragment de 45+ signes la attrape.

    La detection inter-sections (`_sans_copies`) est partie avec le mode
    dissertation, qui etait seul a produire des sections. Ce qui reste —
    et qui sert a la livraison — c'est `_sans_doublons`, applique a la
    reponse finale : c'est lui que ces tests couvrent.
    """

    P1 = ("Pourtant, cette question souleve un debat complexe qui "
          "concerne notre societe actuelle et ses choix.")
    P2 = ("En definitive, cette question souleve un debat complexe qui "
          "concerne notre societe actuelle et ses choix.")

    def test_phrases_exactement_differentes(self):
        assert self.P1 != self.P2                    # le piege

    def test_reprise_a_debut_different_retiree(self):
        # la SECONDE occurrence doit partir, quel que soit son premier mot
        propre = llama_cerveau._sans_doublons(self.P2 + " " + self.P1)
        assert propre.count("souleve un debat complexe") == 1
        assert "En definitive" in propre           # c'est la 1re qui reste

    def test_sans_doublons_retire_la_copie_proche(self):
        propre = llama_cerveau._sans_doublons(
            self.P1 + " " + self.P2 + " Une chute utile et finale.")
        assert "En definitive" not in propre
        assert "Une chute utile et finale." in propre

    def test_texte_sain_intact(self):
        t = ("Le solaire progresse en Europe. L'eolien reste dependant "
             "du reseau. Une comparaison suppose de comparer les couts "
             "marginaux, pas les subventions declarees.")
        assert llama_cerveau._sans_doublons(t) == t


class TestFinitionDePhrase:
    """Jamais de phrase tronquee en fin de reponse (max_tokens ou coupure
    de disjoncteur au milieu d'un mot : « ... et d' » vu au banc)."""

    def test_queue_morte_coupee(self):
        t = ("Le solaire progresse en Europe. L'eolien reste dependant "
             "du reseau. La puissance de la technologie est")
        coupe = llama_cerveau._finir_phrase(t)
        assert coupe.endswith("reseau.")
        assert "technologie est" not in coupe

    def test_fin_compleete_intacte(self):
        t = "Le solaire progresse en Europe. L'eolien gagne du terrain"
        assert llama_cerveau._finir_phrase(t) == t

    def test_forcer_coupe_au_dernier_point(self):
        t = ("Intro solide et longue. Un debut de phrase coupee par le "
             "stream au moment ou le model")
        assert llama_cerveau._finir_phrase(t, forcer=True) == "Intro solide et longue."

    def test_texte_sans_aucune_ponctuation_intact(self):
        assert llama_cerveau._finir_phrase("un texte sans point", forcer=True) == \
            "un texte sans point"

    def test_queue_arretee_au_milieu_d_un_mot(self):
        # cas reels : max_tokens s'arrete en plein mot (« ... et d' »)
        attendu = "La these se pose sans ambiguite."
        t1 = (attendu + " Prendre conscience de ces risques et d")
        t2 = (attendu + " Prendre conscience de ces risques et d'")
        assert llama_cerveau._finir_phrase(t1) == attendu
        assert llama_cerveau._finir_phrase(t2) == attendu


class TestRepliContexte:
    """Fenetre d'historique trop longue pour la fenetre de contexte :
    erreur vue au banc (Requested tokens exceed context window).

    Deux lignes de defense :
    1. _limiter_messages rogne les VIEUX tours avant l'appel (la
       conversation continue avec les tours recents) ;
    2. si meme systeme + question debordent, le repli repart SANS
       historique au lieu de livrer « [RontoGPT] Erreur ... ».
    """

    def test_limite_les_vieux_tours_avant_l_appel(self, monkeypatch):
        """Prompt trop grand pour n_ctx -> seuls les anciens tours sautent,
        le systeme et la question courante restent (pas de repli brutal)."""
        class FauxLlm:
            _n_ctx = 400                      # fenetre minuscule (test)

            def __init__(self):
                self.appels: list[list] = []

            def tokenize(self, texte, add_bos=False):
                # ~1 token par caractere : le comptage est exact ici
                return list(texte)

            def create_chat_completion(self, messages, **kw):
                self.appels.append(list(messages))
                # se plante si plus de 2 tours d'historique passent
                if len(messages) > 4:
                    raise ValueError("Requested tokens (9999) exceed "
                                     "context window of 400")
                return {"choices": [{"message": {
                    "content": "Reponse bornee."}}]}

        faux = FauxLlm()
        monkeypatch.setattr(llama_cerveau, "_charger", lambda: faux)
        monkeypatch.setattr("aura.serveur_lora._actifs", lambda: False)
        hist = [{"role": "user", "content": "Tour un " + "x" * 150},
                {"role": "assistant", "content": "Reponse un " + "y" * 150},
                {"role": "user", "content": "Tour deux " + "z" * 150},
                {"role": "assistant", "content": "Reponse deux " + "w" * 150}]
        r = llama_cerveau.generer("Et maintenant ?", historique=hist)
        assert r == "Reponse bornee."
        appels = faux.appels[0]
        # systeme + question presents, tours trop vieux rognes
        assert appels[0]["role"] == "system"
        assert appels[-1]["content"].startswith("Et maintenant")
        assert len(appels) < 1 + len(hist) + 1

    def test_debordement_reessaie_sans_historique(self, monkeypatch):
        class FauxLlm:
            def __init__(self):
                self.appels: list[list] = []

            def create_chat_completion(self, messages, **kw):
                self.appels.append(list(messages))
                # systeme + question = 2 messages sans memoire
                if len(messages) > 2:
                    raise ValueError("Requested tokens (9999) exceed "
                                     "context window of 1536")
                return {"choices": [{"message": {
                    "content": "Reponse sans memoire."}}]}

        faux = FauxLlm()
        monkeypatch.setattr(llama_cerveau, "_charger", lambda: faux)
        monkeypatch.setattr("aura.serveur_lora._actifs", lambda: False)
        hist = [{"role": "user", "content": "Question precedente ?"},
                {"role": "assistant", "content": "Reponse precedente."}]
        r = llama_cerveau.generer("Et maintenant ?", historique=hist)
        assert r == "Reponse sans memoire."
        assert len(faux.appels[0]) > 2        # 1er essai : avec historique
        assert len(faux.appels[-1]) == 2      # repli : systeme + question


class TestFluxDirect:
    """Streaming GUI : abonnes aux morceaux + desabonnement garanti."""

    def test_abonnes_recoivent_puis_ne_recoivent_plus(self):
        recus: list[str] = []
        desabonner = llama_cerveau.abonner_flux(recus.append)
        try:
            llama_cerveau.publier_flux("bon")
            llama_cerveau.publier_flux("jour")
        finally:
            desabonner()
        assert recus == ["bon", "jour"]
        llama_cerveau.publier_flux("apres")      # desabonne : ignore
        assert recus == ["bon", "jour"]

    def test_abonne_en_erreur_ne_tue_pas_la_publication(self):
        recus: list[str] = []

        def _foireux(_m: str) -> None:
            raise RuntimeError("abonne ko")

        d1 = llama_cerveau.abonner_flux(_foireux)
        d2 = llama_cerveau.abonner_flux(recus.append)
        try:
            llama_cerveau.publier_flux("ok")
        finally:
            d1()
            d2()
        assert recus == ["ok"]


class TestMemoirePersistante:
    """Etat civil entre sessions + outil local « lis le fichier »."""

    @staticmethod
    def _memoire(tmp_path):
        from aura.memoire_conversation import MemoireConversation
        return MemoireConversation(chemin=tmp_path / "conv.jsonl")

    def test_etat_civil_survit_une_restauration(self, tmp_path):
        m = self._memoire(tmp_path)
        m.ajouter("user", "je m'appelle Simon")
        assert m.etat().get("utilisateur") == "Simon"
        # nouvelle session (nouvelle instance, meme disque) -> meme etat
        assert self._memoire(tmp_path).etat().get("utilisateur") == "Simon"

    def test_vider_efface_aussi_etat_disque(self, tmp_path):
        m = self._memoire(tmp_path)
        m.ajouter("user", "je m'appelle Simon")
        m.vider()
        assert m.etat() == {}
        assert self._memoire(tmp_path).etat() == {}

    def test_outil_fichier_inactif_sur_question_normale(self, tmp_path):
        m = self._memoire(tmp_path)
        assert m.lire_fichier("quel temps fait-il ?") is None

    def test_outil_fichier_lit_le_texte(self, tmp_path):
        cible = tmp_path / "notes.txt"
        cible.write_text("code du projet : 42", encoding="utf-8")
        m = self._memoire(tmp_path)
        contenu = m.lire_fichier(f"lis le fichier {cible}")
        assert contenu is not None and "42" in contenu

    def test_outil_fichier_signale_lintrouvable(self, tmp_path):
        m = self._memoire(tmp_path)
        r = m.lire_fichier(f"lis le fichier {tmp_path / 'absent.txt'}")
        assert r is not None and "introuvable" in r

    def test_outil_fichier_refuse_le_binaire(self, tmp_path):
        binaire = tmp_path / "data.bin"
        binaire.write_bytes(b"\x00\x01\x02taille quelconque")
        m = self._memoire(tmp_path)
        r = m.lire_fichier(f"lis le fichier {binaire}")
        assert r is not None and "binaire" in r


class TestBribesEtFewShot:
    """Defauts reels de conversation trouves en prod (2026-10).

    Sur « ok » le 1B ne repondait pas : prive de signal, il CONTINUAIT le
    format « Q: ... R: ... » de la brique few-shot et fabriquait une
    question attribuee a l'utilisateur (« Vous avez pose : Pourquoi le
    ciel est-il bleu ? »). Deux garde-fous : plus de marqueurs de
    tournure dans la brique, et zero LLM sur un accuse de reception nu.

    La portee du second garde-fou est MESUREE et volontairement etroite :
    les salutations restent au 1B (il les traite bien), et deux tests
    existants les utilisent comme question temoin.
    """

    FABRICATION = ("R: Quel est le taux de change du euro a la date "
                   "limite precedente ?")

    def test_bribe_ok_ne_touche_pas_le_llm(self, monkeypatch):
        # le LLM est piege : s'il est appele, il rend la fabrication
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: self.FABRICATION)
        r = orch.RontoGPT().executer_detaille("ok")
        assert r["cerveau_choisi"] == "constante"
        assert r["reponse"] == orch._BRIBE
        assert "R:" not in r["reponse"]

    def test_bribes_reconnues(self):
        for q in ("ok", "Ok !", "merci", "Merci beaucoup", "d'accord",
                  "super", "lol", "oui", "non", "exactement"):
            assert orch._est_bribe(q) is True, q

    def test_salutations_non_interceptees(self):
        # Mesure : avec la brique corrigee le 1B repond tres bien a une
        # salutation (« Bonjour ! Comment puis-je vous aider ? ») — les
        # intercepter degraderait la reponse. Et « bonjour » sert de
        # question temoin dans test_aura / test_jev.
        for q in ("bonjour", "salut", "coucou", "bonjour !", "ca va"):
            assert orch._est_bribe(q) is False, q

    def test_vraies_demandes_pas_des_bribes(self):
        for q in ("bonjour, comment vas-tu",
                  "Explique pourquoi le ciel est bleu",
                  "ok mais explique moi la photosynthese",
                  "combien font 17 x 23 ?",
                  "Quel est le plus grand mammifere du monde ?"):
            assert orch._est_bribe(q) is False, q

    def test_recopie_exacte_executee_sans_le_modele(self):
        # le 1B « corrigeait » la consigne : « je teste RontoGPT » -> « Je
        # tests RontoGPT. » ; une consigne de recopie se rend telle quelle
        assert orch._recopie("Ecris exactement le mot BANANE.") == "BANANE"
        assert orch._recopie("Ecris la phrase : je teste RontoGPT.") == \
            "je teste RontoGPT"

    def test_recopie_refuse_les_formes_libres(self):
        # pas de faux positif sur une vraie demande de redaction
        assert orch._recopie("Ecris la phrase correcte en anglais") is None
        assert orch._recopie("Ecris un poeme sur la mer") is None
        assert orch._recopie("Redige une analyse du solaire") is None

    def test_brique_fewshot_sans_marqueurs_de_tournure(self):
        from aura import agents
        for cle in ("general", "web", "math"):
            brique = agents.exemple_style("question", categorie=cle)
            assert brique
            assert "Q:" not in brique, cle
            assert "R:" not in brique, cle
            assert "recopier" in brique, cle


class TestRedactionSansContexteCopiable:
    """Le 1B RECOPIE le prompt au lieu de rediger -> prompt minimal.

    Defaut mesure au banc maison (11 oct. 2026, categorie « riche ») : 6
    questions sur 10 rendaient un ECHO du prompt systeme, identique d'une
    question a l'autre (911 car : bloc LEXIQUE, consignes d'outil,
    « [RontoGPT] Reflexion interrompue ») au lieu d'une redaction. Cause : sur
    une demande de developpement, le prompt complet — exemples de style +
    lexique de concepts + annonces d'outil — n'est que du texte A RECOPIER.
    Meme remede que l'AUTO-THINK des puzzles (cf. _SYSTEME_RAISONNEMENT).

    Contre-mesure mesuree : les 2 memes questions avec un prompt minimal
    rendent 1 423 et 1 412 car de redaction SUR LE SUJET.
    """

    #: Les 10 questions « riche » du banc, mot pour mot — si le detecteur
    #: en rate une, la question repart en echo et le banc reperd un point.
    RICHES = (
        "Redige une analyse comparee du solaire et de l'eolien en France, "
        "avec les avantages et les limites de chaque filiere",
        "La puissance de la technologie est-elle une menace pour l'homme "
        "moderne ? Redige une dissertation structuree.",
        "Explique pourquoi la lecture est importante pour la liberte de "
        "penser, en trois parties",
        "Analyse les avantages et les inconvenients du teletravail pour les "
        "entreprises et les salaries",
        "Compare la democratie athenienne et la democratie moderne",
        "La liberte peut-elle exister sans contrainte ? Redige une "
        "dissertation",
        "Analyse le role de l'art dans la societe contemporaine",
        "Compare l'intelligence artificielle et l'intelligence humaine sur "
        "le plan de la creativite",
        "Explique pourquoi les oceans sont essentiels a l'equilibre "
        "climatique",
        "Redige une dissertation structuree sur l'utilite du doute dans la "
        "demarche scientifique",
    )

    def test_les_10_questions_du_banc_sont_des_redactions(self):
        for q in self.RICHES:
            assert orch._est_redaction(q), q

    def test_formes_accentuees_reconnues(self):
        """La GUI ecrit accentue, le banc non : les deux doivent passer."""
        for q in ("Rédige une dissertation", "Analyse le rôle de l'art",
                  "Compare la démocratie et la monarchie",
                  "Explique pourquoi X", "Développe ce point",
                  "Écris une argumentation structurée"):
            assert orch._est_redaction(q), q

    def test_pas_de_faux_positif_sur_un_lookup(self):
        """« explique » SEUL reste un lookup : seuls les LOOKUPS sont cites.

        « Explique la photosynthese » demande un fait developpe mais
        verifiable : le passer en redaction libre ferait perdre l'ancrage.
        """
        for q in ("Qu'est-ce que la photosynthese ?",
                  "Explique la photosynthese",
                  "Quel est le carre de 13 ?",
                  "Combien font 17 x 23 ?",
                  "Ecris exactement le mot banane",
                  "Quelle est la capitale de la France ?"):
            assert not orch._est_redaction(q), q

    def test_redaction_recoit_un_prompt_sans_rien_a_recopier(self, monkeypatch):
        """Le systeme envoye ne porte NI brique de style, NI consigne d'outil."""
        from aura import agents, filtre_instantane, memoire_web
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
        monkeypatch.setattr(agents, "compresser", lambda t, q, **k: t)
        monkeypatch.setattr(agents, "en_triplets", lambda t, q: t)
        planifies = []
        monkeypatch.setattr(agents, "planifier",
                            lambda q, **k: planifies.append(q) or "")
        monkeypatch.setattr(agents, "exemple_style",
                            lambda *a, **k: "EXEMPLE DE STYLE A RECOPIER")
        from aura import agents_adk
        monkeypatch.setattr(agents_adk, "planifier_adk",
                            lambda q: planifies.append(q) or None)
        vus = []

        def faux_generer(question, contexte_web="", formule="", **k):
            vus.append(k.get("systeme") or "")
            # Assez long pour que les 3 blocs soient juges substantiels :
            # un texte court declencherait le repli sur le prompt general.
            return "Redaction de test sur le sujet. " * 20

        monkeypatch.setattr(orch.llama_cerveau, "generer", faux_generer)
        ia = orch.RontoGPT()
        ia.executer_detaille(self.RICHES[3])
        # Trois blocs (intro / developpement / conclusion), pas un seul tour :
        # c'est ce decoupage qui fait tenir la longueur sans deriver.
        assert len(vus) == 3, len(vus)
        # AUCUN plan : il serait retire du prompt par _rediger_developpe, et
        # le calculer coutait deux generations ADK + un repli (mesure : 24 s
        # de gagnees sur une dissertation).
        assert planifies == [], planifies
        # Le prompt est desormais DEDIE a la redaction (mesure du 24 oct. :
        # 2 051 et 2 674 car au lieu de 501-624 avec le prompt general). Le
        # contrat verifie reste le meme : rien de copiable, rien de l'outillage.
        from aura.detection_demande import _SYSTEME_REDACTION
        for systeme in vus:
            assert systeme.startswith(_SYSTEME_REDACTION), systeme[:200]
            assert "EXEMPLE DE STYLE" not in systeme, systeme[:200]
            assert "cherche dans l'historique" not in systeme, systeme[:200]
            assert "lis le fichier" not in systeme, systeme[:200]
            assert "LEXIQUE" not in systeme, systeme[:200]

    def test_question_ordinaire_prompt_fixe(self, monkeypatch):
        """Le prompt systeme est le meme pour TOUTES les questions."""
        vus = []
        monkeypatch.setattr(
            orch.llama_cerveau, "generer",
            lambda *a, **k: vus.append(k.get("systeme")) or "ok")
        ia = orch.RontoGPT()
        ia._generer("Quelle est la capitale de l'Australie ?", "", "")
        assert vus and (vus[-1] or "") == "You are a helpful assistant."
