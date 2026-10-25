"""Orchestrateur autonome : AUCUNE dependance externe (pas Ollama, pas API).

Cerveau local configurable (`aura.toml` / `AURA_GGUF`) :
- Gemma-3-1B-it par defaut dans la configuration courante (llama.cpp),
  instruction-tuned, bon FR + EN,
  flash attention + KV cache q8_0 + memoire de conversation multi-tours

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le routeur (TF-IDF + LogReg) choisit les experts a activer ; Llama synthetise.

Cycle de decision inspire de JEV ultrafast (browser-use) :
- niveau 0 exact d'abord ; le routeur ne tourne que si le niveau 0 ne repond pas
- validation de chaque expert AVANT execution (pas de PGS sans donnees)
- fan-out parallele : les experts partent ensemble, echecs isoles
- verification outillee (pattern CRITIC) : le factuel LLM est prouve au web
- confiance calibree (style RouteLLM) : classifieur peu sur -> chemin sur
"""
import logging
import os
import time


# REEXPORTS (convention du module : `# noqa: F401 (reexport)`). Ces noms ne
# servent PAS dans ce fichier, mais ils SONT son API : les tests et les
# extensions les lisent en `orch.<nom>` ou `orchestrateur.<nom>`. Les
# supprimer casse 71 tests a l'import — c'est arrive en nettoyant les imports
# "inutilises", et rien d'autre ne l'aurait signale.
from . import memoire_web  # noqa: F401 (reexport)
from . import llama_cerveau  # noqa: F401 (reexport)
from . import agents  # noqa: F401 (reexport)
from . import expert_symbolique, filtre_instantane
from . import clarification
from . import arbre_reflexion
from .memoire_conversation import MemoireConversation
from .routeur import RouteurIntelligent
from .experts_rontogpt import ExpertsMixin, _SEUIL_CONFIANCE  # noqa: F401 (reexport)
from .synthese_rontogpt import SyntheseMixin
from .verification_rontogpt import VerificationMixin
from .raisonnement_rontogpt import RaisonnementMixin
from .pipeline_rontogpt import PipelineMixin
from .regles_reponse import (  # noqa: F401 -- reexportes : lus par les tests via orch.*
    _BRIBE, _RE_REFUS, _CAPACITE, _IDENTITE, _est_bribe, _est_meta_ia, _est_refus, _recopie,
    _repli_extractif, _reponse_temps, _retirer_refus_tete)
from .detection_demande import (  # noqa: F401 -- reexportes : lus via orch.*
    _SYSTEME_REDACTION, _cle_reponse_pot, _est_redaction)

LOG = logging.getLogger("aura.orchestrateur")

# BUDGET TEMPS GLOBAL : les etapes FACULTATIVES (double passe de style,
# verification web) ne demarrent plus si le pipeline a deja depasse ce
# plafond — un passage long ne doit pas se voir empiler 2 passes de +.
# Ce garde-fou ne coupe pas une generation en cours (impossible sans tuer
# le process llama), il empeche surtout l'empilement qui a produit les
# 153 s observes. AURA_BUDGET_S=0 pour toujours les executer.
_BUDGET_S = float(os.environ.get("AURA_BUDGET_S", "45"))

# MAJORITE D'EXPERTS (self-consistency) : sur les terrains EXACTS (puzzles
# d'ages, enigmes de logique), le meme cerveau est echantillonne N fois et
# CHAQUE essai passe un solveur deterministe — la reponse retenue est celle
# qui recueille le plus de voix VERIFIEES (un essai casse ne vote pas).
# AURA_MAJORITE_N=1 coupe (ancien comportement : un seul essai).
# Borne a 3 : au-dela le gain mesurable est negligeable face au cout CPU.
_MAJORITE_MAX = 3


def _nb_majorite(t0: float | None = None) -> int:
    """Nombre d'essais a lancer (1 si coupe ou budget epuise)."""
    try:
        n = int(os.environ.get("AURA_MAJORITE_N", "3"))
    except ValueError:
        n = 3
    n = max(1, min(n, _MAJORITE_MAX))
    if n > 1 and t0 is not None and _BUDGET_S > 0 \
            and time.time() - t0 >= _BUDGET_S:
        return 1                     # budget epuise : plus d'essai supp.
    return n


class RontoGPT(PipelineMixin, VerificationMixin, RaisonnementMixin, ExpertsMixin, SyntheseMixin):
    """Systeme MoE autonome : cerveau Llama + experts PGS / web."""

    def __init__(self):
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._routeur = RouteurIntelligent()
        self._llama = None
        # MEMOIRE DE CONVERSATION (RLM MIT + fenetre glissante + etat
        # evenementiel) : l'historique complet vit dans un FICHIER externe
        # consultable par outil (recherche arriere), seuls les 4 derniers
        # tours bruts alimentent la fenetre, et l'etat civil (nom,
        # possessions...) est extrait par regex en JSON compact. Fin de
        # l'historique brut injecte en masse dans le 1B.
        # Fenetre proportionnelle a n_ctx (profil_materiel.tours_fenetre) :
        # 4 echanges pour une fenetre de 4096, jusqu'a 12 pour 12288 et plus.
        try:
            from . import profil_materiel
            _tours = profil_materiel.tours_fenetre()
        except Exception:  # noqa: BLE001
            _tours = 4
        self._historique = MemoireConversation(fenetre=_tours)
        self._chaine_faits: list = []        # chaine graphe de la question
        self._derniere_categorie = "general"
        # Preference d'interface, appliquee au VRAI prompt systeme par
        # SyntheseMixin. "auto" laisse le modele suivre la question.
        self._langue_reponse = "auto"
        # Vrai si un expert exact (logique, code, calcul, fast-lang) a deja
        # tranche : lu par _generer() pour le filet anti-hallucination.
        # Initialise ici : sans ca, appeler _generer() hors du pipeline
        # complet (tests, scripts) levait AttributeError.
        self._special_traite = False
        # Jauge de contexte du dernier tour : mesure de saturation et
        # compaction eventuelle (aura/contexte.py). None = sous le seuil.
        self._dernier_contexte: dict | None = None
        # ETAT DE LA PORTE DE CLARIFICATION (aura/clarification.py) : la
        # relance en attente vit ICI, cote instance — un seul cerveau par
        # process (API, GUI), donc un seul suivi de conversation possible.
        # Un seul tour : des que la precision est recollee, l'etat est vide.
        self._clarif = clarification.Suivi()
        # Contexte API publique recolte par _executer_experts (consomme par
        # le pipeline APRES le compresseur : la ligne de donnees ne doit
        # pas partir au top-3 des phrases « utiles »).
        self._contexte_api = ""
        # Dernier arbre de reflexion construit (PoT) : trace exploitable
        # pour le debug/les tests (meme esprit que derniere_formule).
        self._dernier_arbre: arbre_reflexion.ArbreReflexion | None = None

    # -- memoire de conversation -------------------------------------------

    def reinitialiser_conversation(self):
        """Oublie la conversation en cours (nouveau sujet).

        Fenetre + etat civil remis a zero ; le fichier RLM est PRESERVE :
        la memoire longue reste accessible via l'outil de recherche.
        """
        self._historique.vider()
        # une relance de clarification appartient a la conversation qu'on
        # vient de quitter : la garder fusionnerait une precision avec une
        # question d'un autre sujet.
        self._clarif.annuler()

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] fallback (%s)", e)
            return self._routeur.routeur_classique(question)

    # -- decision en un cycle (idea JEV #3) ---------------------------------

    def _decider(self, question: str):
        """Un cycle de decision : niveau 0 exact, puis routeur si necessaire.

        Le niveau 0 repond en moins d'une milliseconde. Le lancer en parallele
        du routeur faisait perdre ce travail sur chaque hit exact : le thread
        de classification continuait en arriere-plan meme si son resultat
        etait ignore. On ne paie le routage que pour les questions qui en ont
        besoin ; son surcout est negligeable face a une generation LLM.
        Renvoie (instant, analyse) — analyse vaut None si le niveau 0 a
        tranche tout seul.

        `filtre_instantane.repondre` est ici l'etage EXACT seul. Le cache
        semantique n'est PLUS consulte ici : il l'est plus loin, apres tous
        les solveurs exacts (cf. le bloc « CACHE SEMANTIQUE » dans
        executer_detaille). C'est une contrainte d'ordre, pas une
        optimisation : un approximatif ne doit pas primer sur un exact.
        """
        instant = filtre_instantane.repondre(question)
        if instant is not None:
            return instant, None
        return None, self.analyser(question)

    @staticmethod
    def _extraire_chaine(contexte_faits: str) -> list[tuple[str, str]]:
        """Extrait [(sujet, objet)] du contexte graphe (lignes '- s : r o')."""
        chaine = []
        for ligne in (contexte_faits or "").splitlines():
            ligne = ligne.strip().lstrip("- ")
            if " : " not in ligne:
                continue
            gauche, _, droite = ligne.partition(" : ")
            gauche = gauche.strip()
            droite = droite.split(" ", 1)[-1].strip()   # saute la relation
            if gauche and droite:
                chaine.append((gauche, droite))
        return chaine




# Ancien nom public conservé pour les extensions et scripts existants.
Aura1B = RontoGPT
