"""Raisonnement de RontoGPT : pilier logique, Program-of-Thoughts, puzzles et routage de la reflexion. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
import re
from typing import TYPE_CHECKING

from . import clarification
from . import llama_cerveau, raisonneur
from . import logique as solveur_logique
from . import potcode
from . import arbre_reflexion
from . import agents
from .regles_reponse import _est_bribe
from .detection_demande import _ENIGME_LOGIQUE, _PHRASE_TAUX, _SUJET_MOUVEMENT, _SYSTEME_RECOPIE, _cle_reponse_pot, _contient_valeur

if TYPE_CHECKING:       # types du contrat des mixins : jamais importes au runtime
    from .arbre_reflexion import ArbreReflexion
    from .clarification import Suivi
    from .expert_symbolique import ExpertSymbolique
    from .memoire_conversation import MemoireConversation

LOG = logging.getLogger("aura.orchestrateur")


class RaisonnementMixin:
    """Raisonnement de RontoGPT (methodes de RontoGPT)."""

    # CONTRAT STATIQUE (jamais execute) : ces noms sont portes par RontoGPT
    # (aura/orchestrateur.py) au montage. Les declarer ici rend le mixin
    # verifiable par mypy sans import circulaire NI dependance au runtime :
    # le bloc disparait a l'execution.
    if TYPE_CHECKING:
        expert: "ExpertSymbolique"
        _clarif: "Suivi"
        _historique: "MemoireConversation"
        # Cree et REMIS A None par ce mixin lui-meme ; declare ICI parce que
        # RontoGPT.__init__ l'initialise et doit pouvoir le remettre a None
        # (sans la declaration, mypy infere `ArbreReflexion` du premier
        # `self._dernier_arbre = arbre` et refuse l'initialisation).
        _dernier_arbre: "ArbreReflexion | None"

    # -- pilier logique ----------------------------------------------------

    def resoudre_numerique(self, X, y) -> str:
        formule = self.expert.resoudre(X, y)
        self.derniere_formule = formule
        self.derniere_erreur = self.expert.erreur
        return formule

    # -- raisonnement Program-of-Thoughts -----------------------------------

    @staticmethod
    def _est_puzzle(question: str) -> bool:
        """Puzzle d'ages/relations : le terrain ou PoT rapporte le plus."""
        try:
            return raisonneur.est_puzzle(question)
        except Exception:
            return False

    @staticmethod
    def _auto_think(question: str) -> bool:
        """Puzzle d'ages ou enigme de logique -> brouillon <thinking> utile.

        Le 1B derape justement sur ces terrains : une passe de reflexion
        masquee (separee par separer_reflexion, jamais montree a
        l'utilisateur) coute ~2x de tokens mais est le levier de
        profondeur restant sur un cerveau 1B sans reflexion native.
        """
        q = question.lower()
        return (RaisonnementMixin._est_puzzle(question)
                or bool(_ENIGME_LOGIQUE.search(q))
                # probleme de taux : les DEUX signatures, pas l'une ou l'autre
                or bool(_SUJET_MOUVEMENT.search(q)
                        and _PHRASE_TAUX.search(q)))

    # -- enigme logique (mini-SAT pur Python) -------------------------------

    def _resoudre_par_logique(self, question: str,
                              t0: float | None = None) -> str | None:
        """Le 1B formalise (entites/domaine/dits), le solveur deduit exactement.

        MAJORITE D'EXPERTS : N formalisations independantes sont echantillon-
        nees et CHAQUE vote avec sa SOLUTION — une formalisation fausse est
        sortie du lot par les autres ; ex aequo -> la premiere solution
        unique vue (ancien comportement, jamais plus strict qu'avant).
        UNE relance de correction (self-debug) si le 1er essai est rejete.
        Aucun essai exploitable -> None (chemin normal).
        """
        from .orchestrateur import _nb_majorite  # import local : evite le cycle
        n = _nb_majorite(t0)
        voix: dict[frozenset, int] = {}
        details: dict[frozenset, tuple] = {}
        ordre: list[frozenset] = []
        relance = False
        try:
            for i in range(n):
                # cadrage « Enigme : … / Formalisme : » : mesuré en réel, c'est
                # celui qui fait respecter le format au 1B (noms exacts)
                brut = llama_cerveau.generer(
                    f"Enigme :\n{question}\n\nFormalisme :",
                    systeme=solveur_logique._SYSTEME_LOGIQUE,
                    max_tokens=300)
                try:
                    puzzle = solveur_logique.parser_puzzle(brut)
                except ValueError as e:
                    if relance or i > 0:
                        # la relance self-debug est unique (cout borne) :
                        # les essais suivants sont simplement eclates
                        LOG.info("[logique] formalisation %d/%d rejetee (%s)",
                                 i + 1, n, e)
                        continue
                    relance = True
                    LOG.info("[logique] formalisation rejetee (%s) -> 1 relance", e)
                    second = llama_cerveau.generer(
                        question + "\n\nTa reponse precedente etait :\n" + brut
                        + "\n\nElle a ete REJETEE car : " + str(e)
                        + "\nReponds a nouveau, STRICTEMENT selon le format, "
                          "avec les vrais noms de l'enigme et SANS inventer de "
                          "contrainte.",
                        systeme=solveur_logique._SYSTEME_LOGIQUE,
                        max_tokens=300)
                    try:
                        puzzle = solveur_logique.parser_puzzle(second)
                    except ValueError as e2:
                        LOG.info("[logique] relance rejetee (%s)", e2)
                        continue
                # Les entites DOIVENT venir de l'enonce : un 1B qui recopie un
                # exemple de format (ou invente des noms) est rejete ici.
                if not solveur_logique.entites_dans_question(puzzle, question):
                    LOG.info("[logique] entites absentes de l'enonce -> "
                             "formalisation rejetee (essai %d/%d)", i + 1, n)
                    continue
                r = solveur_logique.resoudre(puzzle)
                if r["statut"] != "unique":
                    LOG.info("[logique] statut %s (essai %d/%d)",
                             r["statut"], i + 1, n)
                    continue
                cle = frozenset(tuple(sorted(s.items()))
                                for s in r["solutions"])
                if cle not in voix:
                    voix[cle] = 0
                    details[cle] = (puzzle, r["solutions"])
                    ordre.append(cle)
                voix[cle] += 1
            if not voix:
                LOG.info("[logique] aucun essai unique -> chemin normal")
                return None
            # majorite si elle existe, sinon la 1re solution unique vue
            cle_gagnante = max(ordre, key=lambda c: voix[c])
            if len(voix) > 1:
                LOG.info("[logique] vote : %d solutions divergentes, "
                         "gagnante %d/%d voix", len(voix),
                         voix[cle_gagnante], n)
            puzzle_gagnant, solutions_gagnantes = details[cle_gagnante]
            # Le 1B n'a servi QU'A formaliser (ENTITES/DOMAINE/CONDITION) :
            # la reponse est rendue par le solveur, jamais redigee par le
            # modele (qui, devant la consigne brute, begayait/reformulait).
            finale = solveur_logique.phrase_solution(
                puzzle_gagnant, solutions_gagnantes)
            LOG.info("[logique] enigme resolue exactement (%d essai(s))", n)
            return finale
        except (ValueError, Exception) as e:      # noqa: B014 — jamais bloquant
            LOG.info("[logique] formalisation/refus (%s) -> chemin normal", e)
            return None

    # -- demande de code (PoT-code) -----------------------------------------

    def _resoudre_par_code(self, question: str, mode_auto: bool = False) -> str | None:
        """Le 1B ecrit fonction + asserts, la sandbox les execute.

        Un assert rate = code refuse = None (jamais de code casse livre).
        mode_auto : l'agent debugger recursif prend le relais (jusqu'a 3
        essais, l'erreur exacte renvoyee au modele a chaque relance) ;
        echec de la boucle = None (chemin normal, jamais pire qu'avant).
        """
        try:
            brut = llama_cerveau.generer(question,
                                         systeme=potcode._SYSTEME_CODE,
                                         max_tokens=400)
            code = potcode.extraire_code(brut)
            if not code:
                return None
            rapport = potcode.verifier_code(code)
            LOG.info("[potcode] valide : %d asserts", rapport["nb_asserts"])
            return "```python\n" + code + "\n```\n\n(Code verifie : " \
                   f"{rapport['nb_asserts']} asserts passes en sandbox.)"
        except (ValueError, Exception) as e:      # noqa: B014 — jamais bloquant
            LOG.info("[potcode] code refuse (%s) -> chemin normal", e)
            if not mode_auto:
                return None
            # AGENT 4 (debugger recursif) : boucle de self-debug max 3
            try:
                return agents.boucle_code(question, max_tentatives=3)
            except Exception as e2:
                LOG.info("[agents] boucle code impossible (%s) -> chemin normal", e2)
                return None

    def _consommer_relance(self, question: str) -> tuple[str, bool]:
        """Recolle la precision du tour precedent, s'il y en avait une.

        Renvoie (question, relance). `relance` vrai = une porte a deja ete
        posee pour ce tour : la porte est alors COURT-CIRCUITEE, ce qui
        garantit « un seul tour » (l'utilisateur ne peut pas se voir poser
        une seconde serie de questions, meme s'il repond vaguement).
        """
        if not self._clarif.en_attente():
            return question, False
        if _est_bribe(question):
            # « ok » n'apporte AUCUNE precision : recoller produirait
            # « Explique ok ». On referme la relance et on traite le tour
            # normalement (jamais de conversation bloquee en attente).
            LOG.info("[clarification] acquittement sans precision -> "
                     "relance abandonnee")
            self._clarif.annuler()
            return question, False
        fusion = self._clarif.consommer(question)
        if not fusion:
            return question, False
        LOG.info("[clarification] precision recollee : %s", fusion[:120])
        return fusion, True

    def _porte_clarification(self, question: str, X, y,
                             relance: bool = False) -> dict | None:
        """Porte : question trop ouverte -> questions de relance, pas de reponse."""
        if not relance:
            self._clarif.annuler()
        if not clarification.actif():
            return None
        if X is not None and y is not None:
            return None                 # calcul numerique demande : rien a preciser
        besoin = clarification.detecter(
            question,
            a_contexte=clarification.a_un_sujet(self._historique.fenetre()))
        if besoin is None:
            self._clarif.annuler()
            return None
        if relance and not self._clarif.peut_relancer():
            self._clarif.annuler()
            return None
        self._clarif.poser(question, besoin)
        LOG.info("[clarification] %s (%s) : %s", besoin.famille,
                 besoin.raison, " | ".join(besoin.questions))
        reponse = besoin.texte()
        # La relance est un TOUR de conversation a part entiere : le cerveau
        # doit la voir pour ne pas repondre deux fois a la meme question.
        # En revanche elle n'est JAMAIS mise au cache des reponses : ce
        # n'est pas une reponse a la question posee.
        self._historique.ajouter("user", question)
        self._historique.ajouter("assistant", reponse)
        return {"question": question, "experts": {"clarification"},
                "analyse": {"methodes": ["clarification"]},
                "cerveau_choisi": "clarification",
                "contexte_web": "", "formule": "", "erreur_pgs": None,
                "clarification": {"famille": besoin.famille,
                                  "manque": list(besoin.manque),
                                  "questions": list(besoin.questions),
                                  "raison": besoin.raison},
                "en_attente": True,
                "reponse": reponse}

    def _resoudre_par_pot(self, question: str,
                          t0: float | None = None) -> str | None:
        """Passe PoT : le 1B ecrit les etapes, l'AST verifie chaque calcul.

        MAJORITE D'EXPERTS + ARBRE DE REFLEXION : chaque essai echantillonne
        devient une BRANCHE de l'arbre (structures MCTS), notee par l'AST
        etape par etape (validee 1.0 / cassee 0.0) ; les branches valides
        votent sur la reponse finale — la majorite gagne, ex aequo -> la
        meilleure branche. La branche gagnante est reinjectee a la synthese
        comme brouillon deja prouve (branche_vers_prompt).

        Renvoie None si AUCUN essai ne produit >= 2 etapes calculables —
        dans ce cas on retombe sur le chemin normal (jamais pire qu'avant).
        """
        from .orchestrateur import _nb_majorite  # import local : evite le cycle
        n = _nb_majorite(t0)
        arbre = arbre_reflexion.ArbreReflexion(question=question)
        self._dernier_arbre = arbre
        essais: list = []      # (etapes, resultats, cle de vote, branche)
        for i in range(n):
            try:
                brut = llama_cerveau.generer(question,
                                             systeme=raisonneur._SYSTEME_POT,
                                             max_tokens=280)
            except Exception as e:
                LOG.info("[pot] generation %d/%d impossible : %s", i + 1, n, e)
                break
            etapes = raisonneur.extraire_etapes(brut)
            if len(etapes) < 2:
                LOG.info("[pot] pas assez d'etapes calculables (%d, essai %d/%d)",
                         len(etapes), i + 1, n)
                continue
            branche = arbre.nouvelle_branche()
            resultats: list = []
            valide = True
            for phrase, calcul in etapes:
                try:
                    res = raisonneur.verifier_calculs([(phrase, calcul)])[0]
                    ok = True
                except ValueError:
                    res, ok, valide = None, False, False
                branche.ajouter(f"{phrase} = {calcul}", expert="pot",
                                score=1.0 if ok else 0.0, valide=ok)
                resultats.append(res)
            if not valide:
                continue       # branche KO conservee dans l'arbre (trace)
            essais.append((etapes, resultats,
                           _cle_reponse_pot(brut, resultats[-1]), branche))
        if not essais:
            LOG.info("[pot] aucun essai valide -> chemin normal")
            return None
        # VOTE : la cle = la reponse annoncee (chiffre) ; majorite si elle
        # existe, ex aequo -> l'arbre tranche (branche seriee puis notee).
        voix: dict[str, int] = {}
        for essai in essais:
            voix[essai[2]] = voix.get(essai[2], 0) + 1
        cle_gagnante = max(voix, key=lambda c: voix[c])   # ex aequo -> 1re vue
        candidats = [e for e in essais if e[2] == cle_gagnante]
        # La branche dont le CALCUL AST confirme la reponse votee passe
        # devant (mesure : une etape mal ordonnee faisait dire « 8 » a la
        # synthese alors que le vote portait « 9 »).
        confirmes = [e for e in candidats
                     if e[1] and raisonneur._fmt(e[1][-1]) == cle_gagnante]
        etapes, resultats, cle, branche = max(
            confirmes or candidats,
            key=lambda e: (e[3].serie,
                           e[3].score if e[3].score is not None else -1.0))
        if len(essais) > 1:
            LOG.info("[pot] majorite %d/%d voix vers %r", voix[cle],
                     len(essais), cle)
        LOG.info("[pot] arbre de reflexion :\n%s", arbre.trace())
        verification = raisonneur.construire_verification(etapes, resultats)
        brouillon = arbre_reflexion.branche_vers_prompt(branche)
        if brouillon:
            verification += ("\n\nREFLEXION PROUVEE A METTRE EN FRANCAIS :\n"
                             + brouillon)
        try:
            finale = llama_cerveau.generer(question,
                                           contexte_web=verification,
                                           max_tokens=160)
        except Exception as e:
            LOG.info("[pot] synthese impossible : %s", e)
            return None
        # GARANTIE DE LA VALEUR VERIFIEE : la synthese LLM re-derive parfois
        # en ignorant le bloc verifie (mesure : « 6 » / « 7 » repondus alors
        # que la derniere etape AST valait 9). Une relance explicite, puis —
        # si le modele s'obstine — une phrase construite PAR LE PROGRAMME :
        # la valeur prouvee ne disparait jamais de la reponse.
        cibles = [raisonneur._fmt(resultats[-1])]     # dernier resultat AST
        if cle and cle not in cibles:
            cibles.append(cle)                        # majorite des essais
        if not _contient_valeur(finale, cibles):
            try:
                relance = llama_cerveau.generer(
                    question,
                    systeme=_SYSTEME_RECOPIE,
                    contexte_web=(verification
                                  + f"\nLa valeur exacte est {cibles[0]}. "
                                  "Reponds en UNE seule phrase qui contient "
                                  "cette valeur."),
                    max_tokens=80)
                if _contient_valeur(relance, cibles):
                    finale = relance
            except Exception as e:
                LOG.info("[pot] relance impossible : %s", e)
        if not _contient_valeur(finale, cibles):
            # deux echecs : on livre la reponse PAR LE CODE, en une phrase
            # seule plutot qu'en accumulant deux ages contradictoires
            LOG.info("[pot] synthese sans la valeur verifiee -> reponse du code")
            if (cle and cle != cibles[0]
                    and re.fullmatch(r"\d+(?:[.,]\d+)?", cle)):
                # vote et calcul AST divergent : on annonce la reponse VOTEE
                # (contrat du vote), jamais un chiffre que le vote rejette
                finale = f"Reponse retenue par la majorite des essais : {cle}."
            else:
                finale = raisonneur.phrase_verifiee(etapes, resultats)
        LOG.info("[pot] puzzle resolu : %d etapes verifiees (%d essai(s))",
                 len(etapes), len(essais))
        return finale
