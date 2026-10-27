"""Pipeline de reponse de RontoGPT : executer, cache, executer_detaille et chaine de generation. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Callable

from . import memoire_web, filtre_instantane
from . import registre_experts
from . import llama_cerveau, graphe_faits
from . import logique as solveur_logique
from . import potcode
from . import skills
from . import cache_reponses
from . import encyclopedie
from . import agents
from . import adaptateurs
from . import typographie as _typo
from . import provenance as _provenance
from .regles_reponse import _CAPACITE, _CAPACITES, _FONCTIONNEMENT, _IDENTITE, _MOTS_IDENTITE, _RE_CAPACITES, _RE_CAPACITE_MODELE, _RE_DEF, _RE_DEF_LOOKUP, _RE_FONCTIONNEMENT, _RE_IDENTITE, _RE_PHRASE_MIN, _est_meta_ia, _est_refus, _recopie, _repli_extractif, _reponse_bribe, _reponse_temps, _retirer_refus_tete, _sans_fragment_incomplet, _source_pertinente, _sujet_definition
from .detection_demande import (_DEMANDE_CODE, _RE_SORTIE_LONGUE,
                                _est_demande_ecriture, _est_redaction,
                                _sans_accents)

if TYPE_CHECKING:                 # import de typage seul, jamais execute
    from .clarification import Suivi
    from .memoire_conversation import MemoireConversation

LOG = logging.getLogger("aura.orchestrateur")


def _mode_cloud() -> str:
    """Que faire quand l'aiguilleur juge le modele local insuffisant ?

    `demander` (DEFAUT) : rien ne quitte l'ordinateur ; la reponse locale est
                          rendue avec `repli_propose`, et l'utilisateur
                          decide d'envoyer la question a Gemini (un clic).
    `oui`               : envoi automatique (ancien comportement).
    `non`               : jamais.
    """
    v = os.environ.get("AURA_CLOUD", "demander").strip().lower()
    return v if v in ("non", "demander", "oui") else "demander"


#: Similarite minimale pour servir un extrait de document TEL QUEL (lookup).
#: Mesure du 24 oct. 2026 sur un fichier de notes : question pertinente
#: 0,60-0,78, question hors sujet 0,39. 0,55 separe les deux sans generer.
_SEUIL_DOC_DIRECT = 0.55
#: Seuil plus haut pour une question factuelle COURTE (« Quel est le budget ? »)
#: elle n'est pas un lookup au sens de _RE_DEF_LOOKUP, mais elle attend une
#: valeur, pas une synthese. Sous ce seuil, on laisse le modele rediger.
_SEUIL_DOC_FACT = 0.65
_RE_QUESTION_FACTUELLE = re.compile(
    r"^\s*(quel|quelle|quels|quelles|combien|qui|quand|ou|où|comment)\b",
    re.IGNORECASE)
#: Verbes qui demandent une SYNTHESE, jamais un extrait brut.
_RE_SYNTHESE = re.compile(
    r"\b(resume|résume|synthese|synthèse|explique|analyse|compare|redige|"
    r"rédige|liste|developpe|développe)\b", re.IGNORECASE)


class PipelineMixin:
    """Pipeline de reponse de RontoGPT (methodes de RontoGPT)."""

    # CONTRAT STATIQUE (jamais execute) : les METHODES ci-dessous sont portees
    # par les AUTRES mixins et l'ETAT par RontoGPT.__init__ (montage fait dans
    # aura/orchestrateur.py). Les declarer ici, sous `TYPE_CHECKING`, rend le
    # mixin verifiable par mypy SANS import circulaire ni dependance runtime :
    # le bloc disparait a l'execution. L'implementation reste celle du mixin
    # qui la porte — on ne declare PAS de corps de methode, pour ne jamais
    # masquer l'originale dans l'ordre de resolution des methodes.
    if TYPE_CHECKING:
        # etat de l'instance (RontoGPT.__init__)
        _historique: "MemoireConversation"
        _clarif: "Suivi"
        _contexte_api: str
        _dernier_contexte: dict | None
        derniere_erreur: str | None
        # methodes portees par les autres mixins
        _consommer_relance: Callable[..., tuple[str, bool]]
        _decider: Callable[..., tuple[str | None, dict]]
        analyser: Callable[..., dict]
        _valider_experts: Callable[..., set]
        _est_puzzle: Callable[..., bool]
        _auto_think: Callable[..., bool]
        _expert_applicable: Callable[..., bool]
        _appel_expert: Callable[..., str | dict | None]
        _resultat_expert: Callable[..., dict | None]
        _porte_clarification: Callable[..., dict | None]
        _executer_experts: Callable[..., tuple[str, str]]
        _generer: Callable[..., str]
        _resoudre_par_pot: Callable[..., str | None]
        _extraire_chaine: Callable[..., list[tuple[str, str]]]
        _a_besoin_verification: Callable[..., bool]
        _verifier_au_web: Callable[..., str]


    # -- verification outillee (pattern CRITIC) -----------------------------


    # -- pipeline complet ---------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        # COUTEAU TYPOGRAPHIQUE (post-processeur regex, < 0,1 ms) : la
        # réponse finale sort propre (« l'arbre », NBSP devant ?!;:,
        # apostrophes typographiques) — l'intérieur des blocs ```code```
        # n'est JAMAIS touché (le code vérifié doit rester exécutable).
        return _typo.hors_code(
            self.executer_detaille(question, X, y)["reponse"])

    # -- cache de reponses en memoire (rapidite) ----------------------------
    # Une question IDENTIQUE reposee dans la foulee (reessai, deux fenetres,
    # relance du TUI) est servie en < 1 ms au lieu de repasser par le cerveau.
    # Prudent par construction : jamais de question contextuelle ou courte
    # (elle depend de la conversation), jamais de refus ni d'erreur, TTL court
    # pour le web (les faits bougent). AURA_CACHE_REP_TTL=0 le coupe.

    @staticmethod
    def _cle_reponse(question: str, X, y) -> str | None:
        return cache_reponses.cle_reponse(question, X, y)

    def _reponse_en_cache(self, cle: str | None) -> dict | None:
        if cle is None or self._clarif.en_attente():
            return None
        return cache_reponses.lire(cle)

    def _memoriser_reponse(self, cle: str | None, resultat: dict) -> None:
        cache_reponses.memoriser(cle, resultat)

    # -- aiguilleur : petit modele local, ou Gemini ----------------------------

    def _tours_precedents(self) -> list:
        """Les tours AVANT la question en cours (pour AURA_GEMINI_HISTORIQUE).

        `_executer_detaille` a deja ajoute la question a la fenetre, et la
        reponse locale SI elle n'etait pas vide (`ajouter` ignore le vide) : on
        retire donc, par ROLE, ce qui appartient au tour en cours. Liste vide
        si la memoire est absente.
        """
        try:
            tours = list(self._historique.fenetre())
        except Exception:
            return []
        if tours and tours[-1].get("role") == "assistant":
            tours.pop()
        if tours and tours[-1].get("role") == "user":
            tours.pop()
        return tours

    def _consolider_repli(self, question: str, texte: str) -> None:
        """La reponse du repli remplace la mauvaise reponse locale PARTOUT.

        Deja memorisee par `_executer_detaille`, la reponse locale faible
        serait resservie par le cache semantique (qui est consulte AVANT toute
        generation, donc avant tout repli) et mise dans la fenetre de
        conversation, ou la question suivante (« et lui ? ») s'appuierait
        dessus. `enregistrer` refuse un doublon : seule la reconsolidation
        remplace.
        """
        try:
            filtre_instantane.mettre_a_jour(question, texte)
        except Exception as e:
            LOG.info("[repli] cache non reconsolide (%s)", e)
        try:
            self._historique.remplacer_derniere_reponse(texte)
        except Exception as e:
            LOG.info("[repli] fenetre de conversation inchangee (%s)", e)

    def envoyer_a_gemini(self, question: str) -> dict:
        """Envoi EXPLICITE a Gemini : l'utilisateur vient de le demander.

        C'est le seul chemin (avec AURA_CLOUD=oui) par lequel une question
        quitte l'ordinateur. Rend un dict `ok` : en cas d'echec la raison est
        dite, et la reponse locale deja affichee reste valable.
        """
        from . import gemini
        if not gemini.disponible():
            return {"ok": False, "erreur": gemini.derniere_erreur()
                    or "Gemini n'est pas configure (cle absente ou desactivee)."}
        texte = gemini.repondre(question, historique=self._tours_precedents())
        if not texte:
            return {"ok": False, "erreur": gemini.derniere_erreur()
                    or "Gemini n'a pas repondu."}
        self._consolider_repli(question, texte)
        resultat = {
            "question": question, "reponse": texte,
            "cerveau_choisi": "gemini", "experts": [],
            "repli": {"fournisseur": "gemini", "raison": "demande-utilisateur",
                      "moment": "reponse", "reponse_locale": ""}}
        resultat["provenance"] = _provenance.provenance(resultat)
        return {"ok": True, **resultat}

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        # La jauge ne doit jamais refleter le tour PRECEDENT (cache, reponse
        # deterministe, erreur) : seul _generer() la renseigne.
        self._dernier_contexte = None
        # Documents cites par CE tour (nom + chemin) : l'API les rend au client
        # pour des citations cliquables. Jamais ceux du tour precedent.
        self._sources_docs: list[dict] = []
        cle = self._cle_reponse(question, X, y)
        en_cache = self._reponse_en_cache(cle)
        if en_cache is not None:
            LOG.info("[cache-rep] reponse servie depuis la memoire")
            self._historique.ajouter("user", question)
            self._historique.ajouter("assistant", str(en_cache.get("reponse", "")))
            en_cache["cache_instantane"] = True
            return en_cache
        from .orchestrateur import _BUDGET_S  # import local : evite le cycle
        t0 = time.time()                 # horloge du budget global (cf. _BUDGET_S)
        # ECHEANCE ABSOLUE (et non un simple drapeau) : elle est POSEE pour
        # ce fil et lue par la generation, qui la convertit en budget de
        # tokens. Sans ca, AURA_BUDGET_S ne bornait AUCUNE generation :
        # 136 s observees au banc pour un budget annonce de 45 s.
        # Passe par un ContextVar (llama_cerveau.poser_echeance) et non par
        # un kwarg : generer() est appele aussi par les agents, le PoT et
        # l'API, qui n'ont pas a connaitre cette valeur de contexte.
        budget_s = _BUDGET_S
        if _BUDGET_S > 0 and _RE_SORTIE_LONGUE.search(_sans_accents(question)):
            budget_long = float(os.environ.get("AURA_BUDGET_LONGUE_S", "120"))
            pages = re.search(r"\b([1-9]\d*)\s+pages?\b",
                              _sans_accents(question))
            facteur = min(int(pages.group(1)), 3) if pages else 1
            budget_s = max(_BUDGET_S, min(360, budget_long * facteur))
        deadline = (t0 + budget_s) if budget_s > 0 else None
        llama_cerveau.poser_echeance(deadline)
        try:
            resultat = self._executer_detaille(question, X, y, t0)
            # AIGUILLEUR (opt-in, GOOGLE_API_KEY / NVIDIA NIM) : il lit la TACHE
            # demandee (dissertation, traduction...) et la REPONSE rendue (vide,
            # refus, boucle, mauvaise langue, fait devine sans source...), puis
            # decide de garder le petit modele local ou de passer la main au
            # renfort distant (cf. aura/aiguilleur.py). Un tier exact ou verifie
            # n'est jamais rejuge ; sans cible disponible, rien ne part.
            try:
                from . import aiguilleur, gemini
                # Import optionnel de nemotron (nécessite openai)
                nemotron = None
                try:
                    from . import nemotron as _nemotron
                    nemotron = _nemotron
                except Exception:
                    pass
                categorie = getattr(self, "_derniere_categorie", None)
                decision = aiguilleur.decider(question, resultat, categorie)
                mode_cloud = _mode_cloud()
                # Backwards compat: "oui" -> "gemini"
                if mode_cloud == "oui":
                    mode_cloud = "gemini"
                if decision.escalader and mode_cloud == "gemini":
                    texte = gemini.repondre(
                        question,
                        contexte_web=str(resultat.get("contexte_web") or ""),
                        historique=self._tours_precedents())
                    if texte:
                        LOG.info("[aiguilleur] %s (%s) : %s -> gemini",
                                 decision.raison, decision.moment,
                                 resultat.get("cerveau_choisi"))
                        resultat["repli"] = {
                            "fournisseur": "gemini",
                            "raison": decision.raison,
                            "moment": decision.moment,
                            "reponse_locale": str(resultat.get("reponse") or "")}
                        resultat["reponse"] = texte
                        resultat["cerveau_choisi"] = "gemini"
                        self._consolider_repli(question, texte)
                    else:
                        resultat["repli_echec"] = gemini.derniere_erreur()
                elif decision.escalader and mode_cloud == "nemotron":
                    if nemotron is not None:
                        try:
                            client = nemotron.NemotronClient("default")
                            texte = client.repondre(
                                question,
                                contexte_web=str(resultat.get("contexte_web") or ""),
                                formule=str(resultat.get("formule") or ""),
                                max_tokens=4096)
                        except Exception as e:
                            LOG.warning("[aiguilleur] nemotron error: %s", e)
                            texte = None
                        if texte:
                            LOG.info("[aiguilleur] %s (%s) : %s -> nemotron",
                                     decision.raison, decision.moment,
                                     resultat.get("cerveau_choisi"))
                            resultat["repli"] = {
                                "fournisseur": "nemotron",
                                "raison": decision.raison,
                                "moment": decision.moment,
                                "reponse_locale": str(resultat.get("reponse") or "")}
                            resultat["reponse"] = texte
                            resultat["cerveau_choisi"] = "nemotron"
                            self._consolider_repli(question, texte)
                        else:
                            resultat["repli_echec"] = "Nemotron indisponible ou erreur"
                    else:
                        LOG.warning("[aiguilleur] nemotron requested but not available (openai not installed)")
                        resultat["repli_echec"] = "Nemotron non disponible (openai manquant)"
                elif decision.escalader and _mode_cloud() == "demander":
                    # RIEN NE PART : on PROPOSE. La reponse locale reste la
                    # reponse ; l'interface montre un bouton « Demander a
                    # Gemini » qui avertit que la question quittera l'ordinateur.
                    LOG.info("[aiguilleur] %s (%s) : envoi a gemini PROPOSE",
                             decision.raison, decision.moment)
                    resultat["repli_propose"] = {
                        "fournisseur": "gemini",
                        "raison": decision.raison,
                        "moment": decision.moment}
            except Exception as e:
                LOG.info("[aiguilleur] escalade impossible (%s)", e)
            resultat["provenance"] = _provenance.provenance(resultat)
            self._memoriser_reponse(cle, resultat)
            return resultat
        finally:
            # Nettoyage obligatoire : les workers de l'API reutilisent leur fil.
            llama_cerveau.poser_echeance(None)

    def _executer_detaille(self, question: str, X=None, y=None,
                           t0: float | None = None) -> dict:
        t0 = t0 if t0 is not None else time.time()
        # Remis a False a chaque tour ; la redaction Gemini le repasse a True
        # (cf. synthese_rontogpt._rediger_developpe).
        self._redaction_gemini = False
        # ── RELANCE DE CLARIFICATION EN ATTENTE (un seul tour) ─────────
        # Si la porte a demande une precision au tour precedent, ce message
        # EST la precision : on la recolle a la question d'origine et le
        # pipeline repart sur une question complete. C'est la seule chose
        # qui est faite avant les briques deterministes : sans ce recollage,
        # « Convertis 100 » + « en km » ne redeviendrait jamais la phrase que
        # `pal.py` sait calculer.
        question, relance = self._consommer_relance(question)
        # ── BRIBES CONVERSATIONNELLES (« ok », « merci », « salut ») ────
        # Ce n'est PAS une question : on repond par une constante, sans
        # jamais toucher au LLM (zero hallucination, zero cout).
        bribe = _reponse_bribe(question)
        if bribe is not None:
            return {"question": question, "experts": {"conversation"},
                    "analyse": {"methodes": ["bribe"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": bribe}
        # RECOPIE EXACTE : consigne deterministe, jamais confiee au modele.
        recopie = _recopie(question)
        if recopie is not None:
            return {"question": question, "experts": {"recopie"},
                    "analyse": {"methodes": ["recopie"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": recopie}
        # HEURE ET DATE : lues sur l'horloge, jamais generees par le modele.
        temps = _reponse_temps(question)
        if temps is not None:
            return {"question": question, "experts": {"temps"},
                    "analyse": {"methodes": ["temps"]},
                    "cerveau_choisi": "horloge",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": temps}
        # SKILLS (competences ciblees) : AVANT le routeur et le niveau 0, car
        # une identite, une equation lineaire ou un prenom retenu se repondent
        # exactement, sans 1B. Un skill qui ne sait pas (equation non
        # lineaire) rend None : le pipeline continue normalement.
        production_demandee = bool(
            _RE_SORTIE_LONGUE.search(_sans_accents(question))
            or _DEMANDE_CODE.search(_sans_accents(question))
            or _est_demande_ecriture(question))
        skill = skills.repondre(question, historique=self._historique,
                                production=production_demandee)
        if skill is not None:
            nom_skill, reponse_skill = skill
            # RECONSOLIDATION : une reponse exacte remplace la reponse
            # approchante du cache. Sans cela, l'ancienne hallucination
            # (« Je suis Gemma… ») reste servie a une question voisine,
            # meme apres correction du pipeline.
            if skills.doit_reconsolider(nom_skill):
                try:
                    filtre_instantane.mettre_a_jour(question, reponse_skill)
                except Exception as e:          # noqa: BLE001
                    LOG.info("[skills] cache non reconsolide (%s)", e)
            return {"question": question,
                    "experts": {f"skill:{nom_skill}"},
                    "analyse": {"methodes": [f"skill:{nom_skill}"]},
                    "cerveau_choisi": f"skill-{nom_skill}",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": reponse_skill}

        # ── CYCLE UNIQUE (idea JEV #3) ─────────────────────────────────
        # maths directes + routage en UN passage : la majorite des questions
        # quotidiennes n'ont PAS besoin du LLM. NB : depuis le 6 oct. 2026
        # cet etage est EXACT — le cache semantique a ete deplace apres les
        # solveurs (cf. plus bas), pour qu'une memoire approximative ne
        # puisse plus repondre avant un calcul juste.
        instant, analyse = self._decider(question)
        if instant is not None and not (X and y):
            # NETTOYAGE : le calcul direct peut rendre une chaine brute a
            # nettoyer (doublon, libelle de plan recopie).
            instant = llama_cerveau._sans_doublons(instant)
            instant = llama_cerveau._sans_libelles(instant) or instant
            # Idem cache : une recitation de prompt n'est pas un resultat —
            # vide, elle renvoie la question au pipeline (cf. _sans_parrotage).
            instant = llama_cerveau._sans_parrotage(instant)
            if not instant.strip() or _est_refus(instant):
                LOG.info("[niveau0] refus servi -> pipeline complet")
                instant, analyse = None, self.analyser(question)
            else:
                return {"question": question, "experts": {"instantane"},
                        "analyse": {"methodes": ["niveau0"]},
                        "cerveau_choisi": "niveau0-instantane",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": instant}

        # IDENTITE : reponse constante, hors des poids — l'apparence d'RontoGPT
        # ne se negocie pas avec un 1B (question « qui es tu » -> reponse
        # stable et veridique, 0 ms, zero hallucination possible).
        basse = question.lower()
        if (not production_demandee
                # « ton nom de domaine » parle d'un domaine, pas de moi.
                and "nom de domaine" not in basse
                and (any(_sans_accents(m) in _sans_accents(basse)
                         for m in _MOTS_IDENTITE)
                     or _RE_IDENTITE.search(_sans_accents(basse)))):
            return {"question": question, "experts": {"identite"},
                    "analyse": {"methodes": ["identite"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _IDENTITE}

        question_normalisee = _sans_accents(question)
        if (not production_demandee
                and _RE_FONCTIONNEMENT.search(question_normalisee)):
            return {"question": question, "experts": {"fonctionnement"},
                    "analyse": {"methodes": ["fonctionnement"]},
                    "cerveau_choisi": "constante-fonctionnement",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _FONCTIONNEMENT}
        if not production_demandee and _RE_CAPACITES.search(question_normalisee):
            return {"question": question, "experts": {"capacites"},
                    "analyse": {"methodes": ["capacites"]},
                    "cerveau_choisi": "constante-capacites",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _CAPACITES}

        # CAPACITE (meta-IA courte, « es tu aussi puissant qu'un 8B ») :
        # meme logique que l'identite — question fermee adressee au modele,
        # reponse constante verifiee, 0 s de CPU, zero risque de disser-
        # tation hors-sujet ou de boucle de repetition.
        if (not production_demandee
                and _RE_CAPACITE_MODELE.search(question_normalisee)
                and len(question.split()) <= 20):
            return {"question": question, "experts": {"identite"},
                    "analyse": {"methodes": ["capacite"]},
                    "cerveau_choisi": "constante-capacite",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _CAPACITE}
        # meta OUVERTE (longue) : elle repond, mais le budget de tokens est
        # borne (reponse courte, pas 1400 tokens de CPU).
        # ENCYCLOPEDIE : « qui est X » -> extrait source de Wikipedia, sans
        # LLM ; tolerant aux fautes. Absent ou douteux -> pipeline normal.
        if not production_demandee and (
                encyclopedie.sujet_de(question) is not None):
            llama_cerveau.publier_flux("Consultation de Wikipédia…",
                                       llama_cerveau.PHASE_ETAPE)
            wiki = encyclopedie.repondre(question)
            if wiki is not None:
                return {"question": question, "experts": {"encyclopedie"},
                        "analyse": {"methodes": ["encyclopedie"]},
                        "cerveau_choisi": "wikipedia",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": wiki}
        meta_ia = _est_meta_ia(question)

        # ── NIVEAUX 1-2 : experts + LLM ───────────────────────────────
        # VALIDATION AVANT EXECUTION (idea JEV #2 + confiance RouteLLM) :
        # chaque expert est verifie avant d'etre lance.
        experts = self._valider_experts(analyse, X, y)
        # Une intention temporelle ou une demande explicite de recherche
        # l'emporte sur le classement flou du routeur.
        demande_web = memoire_web.a_besoin_web(question)
        # VERROU D'AIGUILLAGE (fix bugs 3 et 4) : une demande de code (« ecris
        # une fonction », def, assert, Python...) ou une enigme de logique ne
        # doit JAMAIS declencher la recherche web — le code se genere et se
        # verifie en local (potcode), l'enigme se deduit (logique) : il n'y a
        # rien a chercher sur internet. Sans ce verrou, le mot-cle temporel
        # (« Python 3.12 », « qui est ») ou une definition forcee envoyait la
        # demande sur DuckDuckGo.
        _aiguillage = (analyse or {}).get("aiguillage")
        _verrou_code = bool(_DEMANDE_CODE.search(_sans_accents(question))
                            or potcode.est_demande_code(question)
                            or _aiguillage == "code")
        _verrou_logique = bool(not _verrou_code and (
            solveur_logique.est_enigme(question) or _aiguillage == "logique"))
        hors_web = _verrou_code or _verrou_logique
        if hors_web:
            demande_web = False
            experts.discard("web")
            if not experts:
                experts.add("general")
        if demande_web:
            experts.add("web")
        # ANCRAGE DES DEFINITIONS : « qu'est-ce que X » / « explique X » ->
        # le web expert est FORCE (si internet est la). Sans contexte, le
        # 1B invente des definitions fausses avec aplomb (« Pythagore » =
        # « plus petit carre commun ») : mieux vaut ancrer que deviner.
        # est_def est calcule sur la question ORIGINALE et reutilise partout
        # (le filet final ne doit JAMAIS tester la question enrichie : la
        # brique few-shot contient « formule », « fonctionne »... qui
        # declencheraient le filet a tort sur des questions innocentes).
        # UNE DEMANDE D'ECRITURE N'EST PAS UN LOOKUP : il n'y a rien a
        # chercher, il y a un texte a produire. Sans ce garde-fou,
        # « Ecris une courte histoire de trois phrases » partait en regime
        # extractif (cf. _RE_ECRITURE) et rendait une non-reponse.
        ecriture = _est_demande_ecriture(question)
        # REDACTION LONGUE : la question demande un TEXTE developpe (analyse,
        # dissertation, comparaison), pas un fait. Elle part avec le prompt
        # minimal _SYSTEME_REDACTION : mesure du 24 oct. 2026, il rend 2 051
        # et 2 674 caractères la ou le prompt general rend 501-624 (et parfois
        # une consigne recopiee). Une seule passe suffit ; le 1B n'a pas
        # besoin qu'on lui decoupe la tache.
        est_redaction = _est_redaction(question)
        est_def = bool(not ecriture and not hors_web and (
            _RE_DEF.search(question) or _RE_PHRASE_MIN.search(question)))
        # Filet final : uniquement les LOOKUPS (cf. _RE_DEF_LOOKUP).
        est_def_lookup = bool(not ecriture and not hors_web and (
            _RE_DEF_LOOKUP.search(question)
            or _RE_PHRASE_MIN.search(question)))
        if est_def:
            experts.discard("general")
            experts.add("web")
        # DETECTION UNIQUE (executes une seule fois) :
        # - est_puzzle pilote le gate RAG et la passe PoT ;
        # - auto_think pilote le brouillon <thinking> a la generation
        #   finale (les puzzles/enigmes qui ont RATE leurs experts
        #   deterministes repartent avec une passe de reflexion masquee).
        est_puzzle = self._est_puzzle(question)
        auto_think = self._auto_think(question)

        # ── EXPERTS, DANS L'ORDRE DECLARE (aura/registre_experts.py) ────
        # Ce bloc etait une SUITE DE `if` : l'ordre n'existait que dans
        # l'ordre des lignes, donc personne ne pouvait le relire — et c'est
        # ainsi que le cache semantique avait fini AVANT les solveurs
        # exacts. L'ordre est desormais declare (avec sa raison) et verifie
        # par tests/test_registre_experts.py.
        #
        # NE PAS re-inserer d'expert en `if` ici : ajouter une entree au
        # registre, son appel dans `_appel_expert` et sa garde dans
        # `_expert_applicable` — le test refuse un rang incoherent ou un
        # expert non branche.
        #
        # Les solveurs exacts n'ont AUCUNE condition : un probleme raconte
        # peut etre long, il reste un CALCUL — la longueur ne doit pas le
        # priver de l'exactitude.
        contexte = registre_experts.Contexte(
            question=question, est_puzzle=est_puzzle,
            a_donnees=bool(X and y))
        self._special_traite = False
        for nom_expert in registre_experts.noms():
            if not self._expert_applicable(nom_expert, contexte):
                continue
            reponse_expert = self._appel_expert(nom_expert, question, X, y,
                                                t0)
            if reponse_expert is None:
                continue
            resultat = self._resultat_expert(nom_expert, reponse_expert,
                                             question, analyse, experts)
            if resultat is None:        # hit refuse (ex : refus memorise)
                continue
            self._special_traite = True
            return resultat

        # ── PORTE DE CLARIFICATION (cf. registre_experts.PORTES) ───────
        # Question trop ouverte (« Explique » sans objet, « Ecris un poeme »
        # sans sujet, « Convertis 100 » sans unite cible) : plutot que de
        # laisser le 1B inventer un sujet, la porte pose 1 a 3 questions et
        # ne repond pas. APRES les experts (une question exacte doit rester
        # exacte), AVANT la generation (demander coute moins cher qu'une
        # reponse fausse). Court-circuitee quand une precision vient d'etre
        # recollee : si la precision reste insuffisante, la porte peut
        # demander un detail supplementaire (trois relances maximum).
        resultat_clarif = self._porte_clarification(question, X, y, relance)
        if resultat_clarif is not None:
            return resultat_clarif

        # FAN-OUT PARALLELE (idea JEV #1) : web ∥ PGS ∥ API publiques, un
        # aller-retour
        self._hors_web = hors_web
        try:
            contexte_web, formule = self._executer_experts(
                experts, question, X, y)
        finally:
            self._hors_web = False
        # Conserver la provenance avant compression du contexte web.
        sources_web = memoire_web.extraire_sources(contexte_web)
        # API PUBLIQUES (meteo, devises, definitions, livres, geoloc) :
        # recollee APRES le compresseur — une ligne de donnees structurees
        # ne doit pas etre jetee au top-3 — puis livree telle quelle si
        # elle est SEULE : zero generation (rapide), zero hallucination,
        # et le 1B ne peut plus refuser des donnees « en temps reel »
        # (refus observe en direct alors que la meteo etait au contexte).
        api = self._contexte_api
        self._contexte_api = ""
        sans_web = not contexte_web
        # AGENT 2 (compresseur RAG) : le contexte web brut est filtre —
        # seules les phrases utiles a la question alimentent le LLM
        if contexte_web:
            try:
                contexte_web = agents.compresser(contexte_web, question)
            except Exception as e:
                LOG.info("[agents] compression impossible (%s) -> texte brut", e)
        # BRIDAGE (fix bug 1) : 3 morceaux au plus (10 pour une recherche
        # complexe) — verrou materiel apres le compresseur.
        contexte_web = memoire_web.limiter_morceaux(contexte_web,
                                                    question=question)
        if api:
            contexte_web = (f"{contexte_web}\n{api}"
                            if contexte_web else api)
        if api and sans_web and not formule:
            LOG.info("[api] livraison extractive : %d car.", len(api))
            filtre_instantane.enregistrer(question, api)
            return {"question": question, "experts": experts | {"api"},
                    "analyse": analyse, "cerveau_choisi": "extraction-api",
                    "contexte_web": api, "formule": "",
                    "erreur_pgs": None, "reponse": api}
        if (memoire_web.demande_explicite_web(question)
                and not contexte_web and not api):
            message = ("Je n'ai pas réussi à obtenir de résultats Web pour "
                       "cette recherche. Vérifie la connexion ou réessaie "
                       "avec des termes plus précis ; je préfère ne pas "
                       "inventer une réponse en prétendant l'avoir vérifiée.")
            return {"question": question, "experts": experts | {"web"},
                    "analyse": analyse, "cerveau_choisi": "web-indisponible",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": message}
        if contexte_web:
            # DEFINITIONS : routage HYBRIDE selon la pertinence de la source.
            #
            # 1) Source FAIBLE (le sujet n'y figure pas en clair) -> ancrage
            #    extractif : on CITE le contexte verifie au lieu de le laisser
            #    reformuler par le 1B, qui invente avec aplomb (mesure du
            #    8 oct. 2026 : « fonction canonique » -> « si x = y alors
            #    f(x) = f(y) », faux). Extraction = zero generation = zero
            #    hallucination possible.
            # 2) Source FORTE (le sujet y figure) -> synthese ancree : le 1B
            #    redige a partir du VERIFIE. Lisible et, source claire aidant,
            #    fiable. Une definition reste un lookup, jamais une redaction
            #    libre.
            # GATE = `est_def_lookup`, PAS `est_def` : « explique X / montre-moi X »
            # demande un DEVELOPPEMENT (cf. _RE_DEF_LOOKUP) — le faire passer par
            # l'ancrage extractif le livrait comme un LOOKUP sans source. Mesure du
            # banc du 11 oct. 2026 : « Explique pourquoi les oceans sont essentiels a
            # l'equilibre climatique » rendait « Aucune source ne definit directement
            # « ... » » — une non-reponse a la place d'un developpement. `est_def`
            # reste vrai et continue de FORCER le web (l'ancrage aide dans les deux
            # cas) ; seul le regime d'EXTRACTION est reserve aux lookups.
            if not formule and est_def_lookup:
                if _source_pertinente(question, contexte_web):
                    LOG.info("[def] source pertinente -> synthese ancree")
                    rep = self._generer(question, contexte_web, "",
                                        categorie="web",
                                        est_def=est_def,
                                        est_def_lookup=est_def_lookup)
                    rep = llama_cerveau._sans_doublons(rep) or rep
                    # CANALISATION : ce chemin SORT ICI — il ne passe donc ni
                    # par le garde-fou du chemin general, ni par le nettoyage
                    # final. Le 1B y RECLAMAIT le contexte au lieu de le
                    # resumer (« Dis-moi : ... (WEB / BASE DE FAITS) ») : le
                    # bavardage part, les extraits verifies prennent sa place.
                    rep = (llama_cerveau._sans_parrotage(rep)
                           or _repli_extractif(contexte_web))
                    rep = memoire_web.ajouter_citations(rep, contexte_web)
                    filtre_instantane.enregistrer(question, rep)
                    return {"question": question, "experts": experts | {"web"},
                            "analyse": analyse, "cerveau_choisi": "def-web+1b",
                            "contexte_web": contexte_web, "formule": "",
                            "erreur_pgs": None, "reponse": rep}
                # dedup : deux extraits de recherche identiques ne doivent
                # pas etre livres cote a cote (phrase repetee a l'ecran).
                # Honnetete : quand le sujet ne figure nulle part, on le DIT
                # avant de citer — sinon l'utilisateur croit a une reponse.
                _sujet = _sujet_definition(question)
                amorce = (f"Aucune source ne definit directement « {_sujet} ». "
                          if len(_sujet) >= 3 else "")
                reponse_extraite = (
                    f"{amorce}D'apres les sources verifiees en ligne :\n"
                    + llama_cerveau._sans_doublons(
                        _sans_fragment_incomplet(contexte_web)))
                reponse_extraite = memoire_web.ajouter_citations(
                    reponse_extraite, contexte_web)
                filtre_instantane.enregistrer(question, reponse_extraite)
                return {"question": question, "experts": experts | {"web"},
                        "analyse": analyse, "cerveau_choisi": "extraction-web",
                        "contexte_web": contexte_web, "formule": formule,
                        "erreur_pgs": None, "reponse": reponse_extraite}
            # AGENT 2b (hyper-compression) : phrases -> triplets semantiques
            # (sujet | relation | objet) — protege la fenetre du 1B et
            # nourrit le graphe de faits au passage
            try:
                contexte_web = agents.en_triplets(contexte_web, question)
            except Exception as e:
                LOG.info("[agents] triplets impossible (%s) -> texte compresse", e)
            liens = [f"Titre : {titre} | Source : {url}"
                     for titre, url in sources_web]
            if liens:
                contexte_web += "\nSOURCES WEB :\n" + "\n".join(liens)

        # AGENT 1 (planificateur) : les taches actionables complexes sont
        # decoupees en 2-3 etapes AVANT la redaction.
        # JAMAIS POUR UNE REDACTION : le plan y est explicitement retire
        # (cf. _rediger_developpe, ou il a deja produit du code hors sujet).
        # Le calculer coutait DEUX generations ADK (planificateur +
        # relecteur) puis une troisieme en repli — pour un texte jete.
        plan = ""
        if not formule and not est_redaction:
            try:
                from . import agents_adk
                plan = (agents_adk.planifier_adk(question)
                        or agents.planifier(question) or "")
            except Exception as e:
                LOG.info("[agents] planification impossible (%s) -> sans plan", e)
            if plan:
                LOG.info("[agents] plan insere : %d lignes", len(plan.splitlines()))

        # RAISONNEMENT PoT (Program-of-Thoughts) : les puzzles d'ages/
        # relations sans donnees numeriques vont au raisonneur — le 1B
        # ecrit les etapes, l'AST les verifie exactement. Repli auto.
        if not (X and y) and not contexte_web and est_puzzle:
            pot = self._resoudre_par_pot(question, t0=t0)
            if pot is not None:
                filtre_instantane.enregistrer(question, pot)
                return {"question": question, "experts": experts | {"pot"},
                        "analyse": analyse, "cerveau_choisi": "pot-1b+ast",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": pot}

        # GRAPHE DE FAITS (MiniRAG-lite) : retrouver au lieu de deviner —
        # uniquement des faits VERIFIES web (jamais d'hallucination dedans).
        # La chaine utilisee est memorisee : si la reponse finale est
        # confirmee par un expert symbolique, elle sera RENFORCEE (MLT).
        contexte_faits = "" if contexte_web else \
            graphe_faits.chercher(question)
        self._chaine_faits = self._extraire_chaine(contexte_faits)
        # DOCUMENTS LOCAUX (RAG) : importer SES fichiers et repondre a partir
        # d'eux, avec le nom de la source. Index vide = ~0 ms (lecture d'un
        # fichier absent) ; MiniLM absent = recouvrement lexical. Ce contexte
        # est verifie par construction (extraits de l'utilisateur) : il sert
        # aussi d'ancrage au filet anti-hallucination, comme le graphe.
        contexte_docs = ""
        sources_docs: list[str] = []
        trouve: dict = {"texte": "", "sources": [], "hits": []}
        try:
            from . import documents as _documents
            trouve = _documents.interroger(question)
            contexte_docs = trouve["texte"]
            sources_docs = trouve["sources"]
            vus_chemins: set[str] = set()
            for hit in trouve.get("hits") or []:
                chemin_doc = str(hit.get("source") or "")
                if chemin_doc and chemin_doc not in vus_chemins:
                    vus_chemins.add(chemin_doc)
                    self._sources_docs.append(
                        {"nom": str(hit.get("nom") or chemin_doc),
                         "chemin": chemin_doc})
        except Exception as e:      # noqa: BLE001 - jamais bloquant
            LOG.info("[documents] recherche impossible (%s)", e)
        if contexte_docs:
            contexte_faits = (f"{contexte_faits}\n\n{contexte_docs}"
                               if contexte_faits else contexte_docs)
            # LOOKUP DOCUMENTAIRE DIRECT : quand un extrait repond avec une
            # similarite forte, on le CITE au lieu de le faire reecrire par
            # le 1B. Zero generation = zero invention, et ~0 s au lieu de
            # ~15 s. Reserve aux questions de lookup : « resume mes notes »
            # n'est pas un lookup et garde la synthese.
            meilleur = (trouve.get("hits") or [None])[0]
            factuelle = bool(
                _RE_QUESTION_FACTUELLE.match(question)
                and len(question.split()) <= 14
                and not _RE_SYNTHESE.search(question))
            seuil_doc = (_SEUIL_DOC_DIRECT if est_def_lookup else
                         _SEUIL_DOC_FACT if factuelle else None)
            if (seuil_doc is not None and meilleur
                    and meilleur.get("score", 0) >= seuil_doc):
                extrait = meilleur["texte"].strip()
                if len(extrait) > 500:
                    extrait = extrait[:500].rsplit(" ", 1)[0] + "…"
                reponse_doc = (f"D'apres {meilleur['nom']} : « {extrait} »")
                filtre_instantane.enregistrer(question, reponse_doc)
                self._historique.ajouter("user", question)
                self._historique.ajouter("assistant", reponse_doc)
                return {"question": question, "experts": experts | {"doc"},
                        "analyse": analyse, "cerveau_choisi": "documents",
                        "contexte_web": contexte_web,
                        "contexte_docs": contexte_docs, "formule": formule,
                        "erreur_pgs": None, "reponse": reponse_doc}
        # PROMPT SYSTEME FIXE : plus de masque de personnalite par categorie
        # (aura/llama_cerveau.py). Le plan (agent 1) complete toujours la
        # question ; le web compresse (agent 2) part en contexte.
        question_envoyee = f"{question}\n\nPLAN A SUIVRE :\n{plan}" if plan else question
        # HOT-SWAP ADAPTATEUR (LoRA) : si un adaptateur existe pour la
        # categorie routee, il est monte sur le contexte charge SANS
        # recharger le modele (AURA_ADAPTATEURS=1 pour activer ; defaut
        # OFF = zero surcout, chemin inchange). Echec -> cerveau brut.
        categorie = ("web" if "web" in experts else
                     "math" if "math" in experts else
                     "code" if _DEMANDE_CODE.search(question) else "general")
        self._derniere_categorie = categorie
        # ROUTAGE MoE TOP-2 : le routeur classe la question, les DEUX
        # meilleurs experts sont actives et renormalises (le grain per-token
        # n'existe pas dans un LoRA : llama-server tient un vecteur
        # d'echelles par generation). Repli : la categorie routee seule
        # quand le routeur n'a rendu aucun score.
        try:
            from . import serveur_lora as _srv
            if _srv._actifs():
                _srv.activer_top2(analyse.get("scores") or {},
                                  defaut=categorie)
        except Exception as e:
            LOG.info("[serveur_lora] routage top-2 impossible (%s) -> "
                     "categorie par defaut", e)
        try:
            from .llama_cerveau import llm_charge as _llm_charge
            _llm = _llm_charge()
            if _llm is not None:
                adaptateurs.appliquer(_llm, categorie)
        except Exception as e:
            LOG.info("[adaptateurs] hot-swap impossible (%s) -> cerveau brut", e)
        # meta-ia ouverte : budget borne (reponse courte, jamais 1400 tokens)
        # AUTO-THINK : puzzles, enigmes et problemes en mots seulement — et
        # seulement s'il reste du budget.
        from .orchestrateur import _BUDGET_S  # import local : evite le cycle
        _budget_ok = _BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S
        _use_think = auto_think and _budget_ok
        reponse = self._generer(question_envoyee, contexte_web, formule,
                                 contexte_faits=contexte_faits,
                                 categorie=categorie, est_def=est_def,
                                 est_def_lookup=est_def_lookup,
                                 redaction=est_redaction,
                                 max_tokens=(110 if meta_ia and not
                                             _RE_SORTIE_LONGUE.search(
                                                 question_normalisee) else None),
                                 think=_use_think)
        # REPLI SANS-REFLEXION : la reflexion coute 2x de budget et le 1B
        # peut ne jamais la fermer -> le cerveau renvoie un message
        # d'interruption. Sur un probleme de taux, le chemin DIRECT repond
        # souvent (mal, mais utilisable) : rendre ce message serait une
        # regression. On retente UNE fois sans reflexion, et seulement si
        # l'echeance laisse le temps d'une generation.
        if (_use_think
                and reponse.startswith("[RontoGPT] Reflexion interrompue")
                and (_BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S)):
            LOG.info("[auto-think] reflexion non terminee -> repli sans "
                     "reflexion")
            reponse = self._generer(question_envoyee, contexte_web, formule,
                                     contexte_faits=contexte_faits,
                                         categorie=categorie, est_def=est_def,
                                     est_def_lookup=est_def_lookup,
                                     redaction=est_redaction,
                                     max_tokens=(110 if meta_ia and not
                                                 _RE_SORTIE_LONGUE.search(
                                                     question_normalisee) else None),
                                     think=False)
        # AGENT 3 (redacteur) : les tics de langage du 1B sont retires
        try:
            reponse = agents.nettoyer_style(reponse)
        except Exception as e:
            LOG.info("[agents] nettoyage impossible (%s) -> texte brut", e)
        # CANALISATION DU 1B (mesure du 12 oct. 2026) : au lieu de rediger, le
        # 1B RECLAMAIT le contexte — « J'ai besoin des informations pour
        # repondre a ta question. Dis-moi : "... fourni ci-dessous et est
        # pertinent" (WEB / BASE DE FAITS) ... dis-moi ce que c'est. » — 82 s
        # de calcul pour rendre la question, alors que 3535 car. de contexte
        # verifie etaient joints. Le bavardage part ici ; s'il ne reste RIEN,
        # on sert les extraits verifies au lieu du vide.
        try:
            propre = llama_cerveau._sans_parrotage(reponse)
            if propre != reponse:
                LOG.info("[canalisation] recitation de prompt retiree "
                         "(%d -> %d car.)", len(reponse), len(propre))
                reponse = propre
            if not reponse.strip() and contexte_web:
                LOG.info("[canalisation] reponse vide -> ancrage extractif")
                reponse = _repli_extractif(contexte_web)
        except Exception as e:
            LOG.info("[canalisation] impossible (%s)", e)
        # GARDE-FOU FINAL (disjoncteur) : JAMAIS une reponse en boucle ne
        # part vers l'utilisateur — on deduplique les phrases repetees ;
        # si c'est une boucle serree (n-grammes), on coupe avant repetition.
        # PUIS tete de refus parasite : « Je suis desole, mais... » suivi
        # d'une vraie suite est coupe (un refus SANS suite reste conserve).
        try:
            # LIBELLES DE CONSIGNE : « SECTION A REDIGER : PARTIE n »
            # recopie par le 1B ne doit JAMAIS arriver a l'utilisateur
            # (observe au banc sur cette balise ecolee).
            reponse = llama_cerveau._sans_libelles(reponse) or reponse
            # PHRASES COPIEES : mot pour mot OU par fragment de 45 signes
            # (« Pourtant, cette question souleve... » puis « En definitive,
            # cette question souleve... » reste une copie malgre son debut).
            sans_dbl = llama_cerveau._sans_doublons(reponse)
            if sans_dbl != reponse:
                LOG.warning("[disjoncteur] phrases copiees retirees en fin "
                            "de chaine")
                reponse = sans_dbl
            if llama_cerveau._en_boucle(reponse):
                reponse = llama_cerveau._couper_boucle(reponse)
                LOG.warning("[disjoncteur] reponse nettoyee en fin de chaine")
            # phrase tronquee en fin de texte (max_tokens) : finition propre
            reponse = llama_cerveau._finir_phrase(reponse)
            reponse = _retirer_refus_tete(reponse)
        except Exception as e:
            LOG.info("[disjoncteur] garde-fou impossible (%s)", e)
        # VERIFICATION OUTILLEE (pattern CRITIC) : preuve web avant livraison
        verifiee = False
        if ((_BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S)
                and not hors_web
                # Une redaction Gemini est un texte long : la verification
                # web court la remplacerait par un extrait local.
                and not getattr(self, "_redaction_gemini", False)
                and self._a_besoin_verification(question, contexte_web,
                                                reponse)):
            avant_verification = reponse
            reponse = self._verifier_au_web(question, reponse)
            # N'apprendre que si la reponse revisee a change et porte au
            # moins deux liens distincts. Un simple extrait ou un seul site
            # ne suffit pas a consolider un fait dans la memoire longue.
            verifiee = (
                reponse.strip() != avant_verification.strip()
                and len(memoire_web.extraire_sources(reponse)) >= 2)
            # RECONSOLIDATION : la reponse verifiee remplace l'ancienne au
            # cache (sinon une reponse fausse d'avant reste collée a vie)
            if verifiee:
                filtre_instantane.mettre_a_jour(question, reponse)
        # APPRENTISSAGE DU GRAPHE : les reponses verifiees web nourrissent
        # le graphe de faits (extraction conservatrice) — le systeme devient
        # plus savant a chaque question factuelle confirmee.
        if verifiee:
            try:
                graphe_faits.apprendre_de_reponse(reponse, verifiee_web=True)
            except Exception as e:
                LOG.info("[graphe] apprentissage impossible : %s", e)
        # MEMOIRE A LONG TERME AUTONOME : une reponse confirmee par un
        # expert symbolique (validation exacte : calcul, logique, code,
        # preuve web) renforce les liens de la chaine de faits utilisee —
        # les schemas de pensee eprouves remontent, les liens inutiles
        # retombent (oubli doux). Aucun re-entrainement requis.
        if self._chaine_faits:
            confirmee = (verifiee or "math" in experts or "pot" in experts
                         or reponse.startswith(("[Logique]", "[Code]")))
            if confirmee:
                for s, o in self._chaine_faits:
                    try:
                        graphe_faits.renforcer(s, o)
                    except Exception as e:
                        LOG.debug("[graphe] renforcement de %s oublie : %s",
                                  s, e)
        # NETTOYAGE FINAL AVANT LIVRAISON ET MEMORISATION : quoi qu'aient
        # produit la double passe ou la verification web, aucun libelle de
        # plan, aucune phrase copiee ni fin tronquee ne doit etre servi
        # (ni servi ulterieurement depuis le cache).
        try:
            reponse = llama_cerveau._sans_libelles(reponse) or reponse
            reponse = llama_cerveau._sans_doublons(reponse)
            reponse = llama_cerveau._finir_phrase(reponse)
            # CANALISATION : aucune recitation de prompt ne part a l'ecran,
            # meme si elle vient de la memoire ou de la verification web.
            # Tout bavardage => les extraits verifies, jamais le vide.
            propre = llama_cerveau._sans_parrotage(reponse)
            if propre != reponse:
                LOG.info("[canalisation] recitation retiree en fin de "
                         "chaine (%d -> %d car.)", len(reponse), len(propre))
                reponse = propre
            if not reponse.strip() and contexte_web:
                reponse = _repli_extractif(contexte_web)
        except Exception as e:
            LOG.info("[nettoyage final] impossible (%s)", e)
        reponse = memoire_web.ajouter_citations(reponse, contexte_web)
        # CITATION DES DOCUMENTS : un extrait local sans provenance n'est pas
        # verifiable. La ligne est ajoutee ICI (chemin de generation) et pas
        # par le modele, qui l'oublierait ou l'inventerait.
        if sources_docs and "Sources :" not in reponse:
            reponse = f"{reponse}\n\nSources : {', '.join(sources_docs)}"
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.ajouter("user", question)
        self._historique.ajouter("assistant", reponse)
        # PENSEE ET AUTO-CORRECTION DU TOUR : elles sont jointes au resultat
        # EXPLICITEMENT, sous leurs propres cles. C'est le pendant cote
        # donnees de la separation des flux : `reponse` ne contient QUE la
        # reponse, et qui veut la reflexion doit demander `pensee`. Un
        # consommateur qui ignore ces cles (cache, ancrage, graphe) ne voit
        # donc jamais la reflexion du modele passer pour une reponse.
        return {"question": question, "experts": experts,                "analyse": analyse, "cerveau_choisi": self._etiquette_cerveau(), "contexte_web": contexte_web,
                "contexte_docs": contexte_docs,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "pensee": llama_cerveau.reflexion_du_tour(),
                "auto_correction": llama_cerveau.alerte_du_tour(),
                "reponse": reponse}

    def _etiquette_cerveau(self) -> str:
        """Nom public du cerveau pour ce tour.

        Une redaction confiee a Gemini ne doit pas etre presentee comme une
        reponse du modele local : le badge de provenance lit ce nom.
        """
        return ("gemini" if getattr(self, "_redaction_gemini", False)
                else self._nom_cerveau())

    @staticmethod
    def _nom_cerveau() -> str:
        """Identifiant du cerveau REELLEMENT charge, pour le diagnostic.

        La valeur etait ecrite en dur (« llama-3.2-1b ») : au moment du
        passage a Gemma, toute la chaine de diagnostic aurait continue
        d'annoncer Llama — un journal qui ment sur le modele charge coute
        plus cher que pas de journal. La famille est desormais LUE dans
        l'en-tete du GGUF (aura/gguf_meta.py), pas deduite d'un nom de
        fichier ni d'une constante.
        """
        return f"{llama_cerveau.famille() or 'inconnu'}-1b"
