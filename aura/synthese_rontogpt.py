"""Synthese de la reponse de RontoGPT : prompt systeme, garde-fous de
contamination et generation. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
import os
from typing import TYPE_CHECKING, Callable

from . import autoamelioration
from .regles_reponse import _rejoue_le_tour_precedent, _reste_substantiel, _sans_amorce_meta

if TYPE_CHECKING:
    from .memoire_conversation import MemoireConversation

LOG = logging.getLogger("aura.orchestrateur")

#: Prompt de redaction pour Gemini. Le 1B local est decoupe en trois blocs ;
#: Gemini peut rendre une dissertation complete en UN appel, avec sa structure.
_SYSTEME_REDACTION_GEMINI = (
    "Tu rédiges en français une dissertation, un essai, un article ou un "
    "rapport structuré. Introduction : présente le sujet, pose une "
    "problématique et annonce deux ou trois axes. Développement : deux ou "
    "trois parties argumentées, avec des exemples précis. Conclusion : "
    "réponds à la problématique et ouvre une perspective. Reste strictement "
    "sur le sujet ; n'invente ni citation, ni chiffre, ni source. N'écris "
    "aucun préambule, aucun méta-commentaire et aucun titre de section si on "
    "ne te le demande pas. Vise 900 à 1 500 caractères, sauf si la question "
    "impose une longueur."
)


def _gemini_redaction_active() -> bool:
    """L'utilisateur a-t-il explicitement confie les redactions a Gemini ?"""
    if os.environ.get("AURA_CLOUD", "demander").strip().lower() == "non":
        return False
    option = os.environ.get("AURA_GEMINI_REDACTION", "0").strip().lower()
    if option in ("", "0", "non", "false"):
        return False
    from . import gemini
    return gemini.disponible()


class SyntheseMixin:
    """Generation de la reponse finale (methodes de RontoGPT)."""

    # CONTRAT STATIQUE (jamais execute) : ces noms sont portes par RontoGPT
    # (aura/orchestrateur.py) au montage des mixins. Les declarer ici, sous
    # `TYPE_CHECKING`, rend le mixin verifiable par mypy sans import
    # circulaire ni dependance au runtime : le bloc disparait a l'execution.
    if TYPE_CHECKING:
        _llama: Callable[..., str] | None
        _historique: "MemoireConversation"
        _special_traite: bool
        _dernier_contexte: dict | None

    #: Seuil du banc qualite pour une reponse developpee (cf.
    #: scripts/bench_qualite.py, _defauts_riche).
    _LONGUEUR_REDACTION = 900

    def _systeme_avec_langue(self, systeme: str) -> str:
        """Ajoute la preference utilisateur au prompt systeme reel."""
        langue = getattr(self, "_langue_reponse", "auto")
        if langue == "fr":
            return systeme + "\n\nLANGUE DE REPONSE : reponds uniquement en francais."
        if langue == "en":
            return systeme + "\n\nRESPONSE LANGUAGE: answer only in English."
        return systeme

    def _generer(self, question: str, contexte_web: str, formule: str,
                 contexte_faits: str = "",
                 categorie: str = "",
                 est_def: bool = False,
                 est_def_lookup: bool = False,
                 max_tokens: int | None = None,
                 think: bool | None = None,
                 redaction: bool = False) -> str:
        # Remis a False a chaque tour : seul le chemin Gemini de redaction
        # le repasse a True, et le pipeline s'en sert pour la provenance.
        self._redaction_gemini = False
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        # JAUGE DE CONTEXTE : prompt + fenetre atteignent 90 % de la fenetre
        # du cerveau -> les tours les plus anciens sont retires AVANT la
        # generation. Le dernier echange et l'etat durable restent ; le reste
        # vit toujours dans le fichier RLM. Mesure sur le texte reellement
        # envoye (systeme, web, faits, formule), pas sur la seule question.
        from .llama_cerveau import _SYSTEME as systeme_base
        systeme_final = self._systeme_avec_langue(systeme_base)
        from . import contexte as _contexte
        mesure_contexte = _contexte.auto_compacter(
            self._historique, question, contexte_web=contexte_web,
            contexte_faits=contexte_faits, formule=formule, systeme=systeme_final)
        # Un second _generer() dans le MEME tour (repli sans reflexion) ne
        # doit pas effacer la saturation deja signalee : on ne remplace que
        # par une mesure non nulle.
        if mesure_contexte is not None:
            self._dernier_contexte = mesure_contexte
        # REDACTION LONGUE : prompt minimal dedie (mesure : 2 051 et 2 674
        # caracteres sur les deux questions « riche » du banc, contre 501-624
        # avec le prompt general) et decoupage en trois blocs. Le chemin
        # normal reste inchange.
        if redaction:
            return self._rediger_developpe(
                question, contexte_web, formule, contexte_faits,
                categorie, max_tokens)
        # PROMPT SYSTEME FIXE : plus de personnalite par categorie, de brique
        # de style, de lexique de concepts, ni d'etat de conversation injecte.
        # La seule valeur envoyee est celle du cerveau (aura/llama_cerveau.py).
        # OUTIL RLM (MIT) : la question demande-t-elle l'historique ?
        # Resolution symbolique (recherche fichier arriere) — zero LLM,
        # zero token d'historique injecte.
        resultat_rlm = self._historique.tourner(question)
        if resultat_rlm is not None:
            return self._llama(
                f"{question}\n\nRESULTAT DE LA RECHERCHE DANS L'HISTORIQUE :\n"
                f"{resultat_rlm}", contexte_web, formule,
                historique=self._historique,
                contexte_faits=contexte_faits,
                systeme=systeme_final,
                categorie=categorie,
                max_tokens=max_tokens)
        # OUTIL LOCAL : « lis le fichier <chemin> » -> contenu injecte
        # (resolution symbolique, meme esprit que le RLM — zero LLM).
        contenu_fichier = self._historique.lire_fichier(question)
        if contenu_fichier is not None:
            return self._llama(
                f"{question}\n\nCONTENU DU FICHIER :\n{contenu_fichier}",
                contexte_web, formule,
                historique=self._historique,
                contexte_faits=contexte_faits,
                systeme=systeme_final,
                categorie=categorie,
                max_tokens=max_tokens)
        # FILET FINAL (anti-hallucination) : question de LOOKUP (definition,
        # fait chiffre) SANS aucun ancrage (web injoignable, graphe muet, pas
        # d'expert special) -> on n'invente pas : garde-fou honnete + contexte
        # du graphe s'il existe. Un « je ne peux pas verifier » vaut mieux
        # qu'une definition fausse dite avec assurance.
        # `est_def_lookup` et non `est_def` : une demande de DEVELOPPEMENT
        # (« explique la photosynthese ») n'est pas une definition a verifier,
        # cf. _RE_DEF_LOOKUP. Les deux sont calcules sur la question
        # ORIGINALE (jamais sur la question + brique few-shot, cf.
        # executer_detaille).
        if (est_def_lookup and not contexte_web and not contexte_faits
                and not self._special_traite):
            return ("Je ne peux pas verifier cette definition en ce moment "
                    "(hors-ligne et absente de ma base de faits). Pour ne "
                    "pas t'inventer une reponse fausse, je prefere m'abstenir. "
                    "Reessaie avec internet, ou apprends-moi ce fait : "
                    "graphe_faits.ajouter(...).")
        # Le brouillon de raisonnement n'herite pas de l'historique distant :
        # garder au plus le dernier echange utile pour resoudre les references
        # conversationnelles, sans injecter les vieux exemples/corrections.
        historique_generation = (
            self._historique.fenetre()[-2:] if think else self._historique)
        reponse = self._llama(question, contexte_web, formule,
                              historique=historique_generation,
                              contexte_faits=contexte_faits,
                              systeme=systeme_final,
                              categorie=categorie,
                              max_tokens=max_tokens,
                              **({"think": think} if think else {}))
        # -- GARDE-FOU CONTAMINATION (cf. _RE_AMORCE_META) ---------------
        # On retire l'amorce meta ; si la reponse n'etait que cette amorce,
        # ou si elle rejoue le tour precedent, on regenere SANS HISTORIQUE.
        reponse = _sans_amorce_meta(reponse)
        if (not _reste_substantiel(reponse)
                or _rejoue_le_tour_precedent(reponse, self._historique)):
            LOG.info("[anti-contamination] tour rejoue ou amorce meta vide "
                     "-> regeneration sans historique")
            propre = _sans_amorce_meta(
                self._llama(question, contexte_web, formule,
                            historique=[],
                            contexte_faits=contexte_faits,
                            systeme=systeme_final,
                            categorie=categorie,
                            max_tokens=max_tokens,
                            **({"think": think} if think else {})))
            if propre.strip():
                reponse = propre
        return reponse

    def _rediger_developpe(self, question: str, contexte_web: str, formule: str,
                           contexte_faits: str, categorie: str,
                           max_tokens: int | None) -> str:
        """Redige une reponse longue en TROIS blocs structurels.

        Le postulat « un 1B ne peut pas ecrire 900 caracteres en une passe »
        etait faux avec `_SYSTEME_REDACTION` : une passe rend 2 051 et 2 674
        car sur les deux questions « riche » du banc. Mais une passe unique
        derive en fin de texte (boucles, meta-commentaires). Le decoupage
        intro / developpement / conclusion garde la longueur et la structure
        sans jamais redonner au modele du texte a commenter.

        Coût : trois requetes a chaud sur `llama-server` (contexte deja
        charge) ; un serveur muet retombe sur le chemin normal.
        """
        # GEMINI POUR LES REDACTIONS LONGUES (opt-in) : une seule requete,
        # pas de brouillon local jete. Le sujet et les extraits partent chez
        # Google ; l'historique n'est PAS joint (une redaction repart du
        # sujet nu). Sans cle, en echec, ou avec AURA_CLOUD=non : les trois
        # blocs locaux restent le chemin par defaut.
        if _gemini_redaction_active():
            from . import flux_cerveau, gemini
            flux_cerveau.publier_flux("Rédaction Gemini en cours…",
                                      flux_cerveau.PHASE_ETAPE)
            texte = gemini.repondre(
                question, contexte_web=contexte_web, historique=[],
                contexte_faits=contexte_faits,
                systeme=_SYSTEME_REDACTION_GEMINI, max_tokens=max_tokens)
            if texte:
                self._redaction_gemini = True
                # Aucune reflexion locale n'a eu lieu : ne pas laisser celle
                # du tour precedent dans le canal de pensee.
                from .flux_cerveau import effacer_reflexion
                effacer_reflexion()
                LOG.info("[redaction] Gemini a rendu le texte (%d car.)",
                         len(texte))
                return texte.strip()
            LOG.info("[redaction] Gemini indisponible (%s) -> blocs locaux",
                     gemini.derniere_erreur())

        from . import llama_cerveau as lc
        from .detection_demande import _SYSTEME_REDACTION

        llama = self._llama
        if llama is None:
            llama = lc.generer
            self._llama = llama
        # PAS D'HISTORIQUE et PAS DE PLAN : mesures du 24 oct. 2026. Le plan
        # genere pour une demande longue peut etre du CODE (le 1B confond
        # « analyse » et « fonction ») et l'historique persistant contient la
        # question precedente — deux sources de contamination constatees :
        # la question « puissance » a rendu du code sur le solaire/eolien.
        # Une redaction repart de la question NUE ; le contexte verifie
        # (web/faits) reste fourni.
        consigne = question.split("\n\nPLAN A SUIVRE :")[0].strip()

        def passe(q: str, budget: int) -> str:
            brut = llama(q, contexte_web, formule,
                               historique=[],
                               contexte_faits=contexte_faits,
                               systeme=_SYSTEME_REDACTION,
                               categorie=categorie,
                               max_tokens=budget)
            return _sans_amorce_meta(lc.separer_reflexion(brut)[0]).strip()

        # TROIS BLOCS, PAS UNE CONTINUATION. Le 1B ne sait pas « poursuivre »
        # un texte : quand on lui redonne le texte deja ecrit, il le COMMENTE
        # (« Le texte fourni : que faut-il faire ? »), mesure du 24 oct. 2026.
        # Chaque bloc repart de la question nue avec une consigne de section :
        # pas de texte a recopier, donc pas de meta-commentaire possible.
        # BUDGETS SERRES (mesure du 24 oct. 2026) : 940 tokens produisaient
        # ~3 000 caracteres en 56 s, soit TROIS FOIS le contrat du banc
        # (>900 car). Sur ce CPU (~11 tok/s), chaque token en trop se paie
        # ~90 ms pour du remplissage. 350 tokens rendent ~1 500 car en ~30 s,
        # au-dessus du contrat, sans derive du 1B en fin de texte.
        sections = (
            ("Rédige l'introduction du texte : présente le sujet et annonce "
             "le plan en deux ou trois axes. 3 à 4 phrases.", 90),
            ("Rédige le développement en deux paragraphes : d'abord les "
             "apports ou avantages, ensuite les limites ou les risques. "
             "5 à 7 phrases.", 180),
            ("Rédige la conclusion : réponds clairement à la question, puis "
             "ouvre une perspective. 2 à 3 phrases.", 80),
        )
        morceaux = []
        for instruction, budget in sections:
            texte = passe(f"{consigne}\n\n{instruction}", budget)
            if texte and len(texte) > 60:
                morceaux.append(texte)
        if not morceaux:
            # Serveur muet : on retombe sur le chemin normal, qui sait gerer
            # le repli in-process et les garde-fous.
            return self._generer(question, contexte_web, formule,
                                 contexte_faits=contexte_faits,
                                 categorie=categorie, max_tokens=max_tokens,
                                 redaction=False)
        texte = "\n\n".join(morceaux)
        if lc._en_boucle(texte):
            texte = lc._couper_boucle(texte) or texte
        return texte.strip()

    # -- prompt structure ---------------------------------------------------

    @staticmethod
    def _construire_prompt(question, contexte_web, formule) -> str:
        sections = [f"QUESTION : {question}"]
        ctx_corr = autoamelioration.construire_contexte_corrections(question)
        if ctx_corr:
            sections.append(f"\nCORRECTIONS :\n{ctx_corr}")
        if contexte_web:
            sections.append(f"\nWEB :\n{contexte_web}")
        if formule:
            sections.append(f"\nFORMULE : Y = {formule}")
        return "\n".join(sections)

