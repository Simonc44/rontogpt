"""Regles de reponse (identite, temps, refus, definitions) : constantes
 et detecteurs DETERMINISTES utilises par RontoGPT. Extrait de
 aura/orchestrateur.py ; aucun changement de comportement."""
import re
from datetime import datetime
from . import llama_cerveau
from . import skills as _skills
from .detection_demande import (
    _sans_accents
)



# Questions d'identite (« qui es tu », « qui t'a cree ») : reponse non
# hallucinable — tiree du skill `identite` (skills.toml), jamais des poids
# du 1B. Le modele de base est nomme honnetement, sans revendiquer Google.
_IDENTITE = _skills.IDENTITE
_MOTS_IDENTITE = ("qui es tu", "qui es-tu", "qui t'a", "qui t'a cree",
                  "ton createur", "ton nom", "tu t'appelles", "tu es qui")

# identite tolerante aux fautes (« qui est tu », « t'es qui », « presente toi »)
_RE_IDENTITE = re.compile(
    r"\b(qui\s+(?:es|est|etes)[- ]?(?:tu|vous)\b|t'?es\s+qui\b|"
    r"presente[- ]?(?:toi|vous)\b|"
    r"(?:peux|pourrais)[- ]?tu\s+te\s+presenter\b|"
    r"comment\s+tu\s+t'?appelles?\b)")
# une suite de conversation (« et lui ? », « ca marche comment ») depend du
# tour precedent : jamais mise en cache
_RE_SUITE_CONVERSATION = re.compile(
    r"^(et|ca|cela|ce|celui|celle|ceux|il|elle|ils|elles|donc|alors|mais|oui|non|"
    r"repete|redis|recommence|refais|encore)\b")

_RE_FONCTIONNEMENT = re.compile(
    r"\b(comment|explique|decris)\b.{0,70}\b(fonctionne|marche|"
    r"travaille|architecture|pipeline|systeme)\b|"
    r"\b(fonctionnement|architecture)\b.{0,50}\b(rontogpt|toi|assistant)\b",
    re.IGNORECASE)
_RE_CAPACITES = re.compile(
    r"\b(que (?:sais|peux|peut)[- ]tu faire|tu sais faire quoi|"
    r"tes capacites|quelles sont tes capacites|tes fonctionnalites|"
    r"a quoi (?:sers|peux tu servir)|"
    r"quelles taches|quels types de taches)\b", re.IGNORECASE)
_RE_CAPACITE_MODELE = re.compile(
    # « puissance » et « performances » doivent viser LE MODELE : nus, ils
    # attrapaient toute question qui parle de puissance ou de rendement.
    # Defaut reel mesure par scripts/bench_qualite.py : « La puissance de la
    # technologie est-elle une menace… Redige une dissertation » (17 mots)
    # recevait la reponse-constante d'identite au lieu d'une dissertation,
    # alors que `_est_meta_ia` dit correctement que ce n'est PAS une question
    # meta (cf. tests/test_disjoncteur.py).
    r"\b(plus puissant|"
    # Comparaison DIRECTE avec un autre assistant : question fermee sur le
    # modele, elle merite la reponse-constante (0 s). Sans ce motif,
    # « Tu es plus fort que ChatGPT ? » partait au 1B et rendait 7-34 s de
    # digression (echec meta-capacite du banc : cible < 5 s).
    r"(?:plus |aussi )?(?:fort|bon|meilleur) (?:que|qu')\s*"
    r"(?:chatgpt|gpt[- ]?\d*|claude|gemini|copilot|mistral|deepseek)|"
    r"(?:ta|tes|ton|votre|vos)\s+(?:puissance|performances?)|"
    r"taille du modele|quel modele|combien de parametres|\d{1,3}[\s-]?b|"
    r"aussi puissant|es[- ]tu une ia|es[- ]tu une intelligence artificielle)\b",
    re.IGNORECASE)

# QUESTIONS META SUR L'IA (« es-tu une ia », « aussi puissant qu'un 8B ») :
# ni une tache de redaction ni un fait a verifier. Sans ce garde-fou, la
# longueur seule (>= 9 mots) declenchait le mode dissertation (plan +
# 3 sections + double passe = ~1400 tokens de CPU, soit ~150 s) sur une
# question fermee — d'ou la sortie hors-sujet repetee observee en prod.
# Le mode dissertation a ete supprime, mais ce garde-fou reste : il borne
# le BUDGET de tokens d'une question adressee au modele (110 tokens), ce qui
# n'a rien a voir avec le mode.
_RE_META_IA = re.compile(
    r"(\bes[-\s]?tu\b|\bt'es[-\s]?tu\b|\bqui es[-\s]?tu\b|"
    r"\best[-\s]?ce que tu (es|sois|puisses|saches|comprennes)\b|"
    r"\btu es\s+(?:une?\s+)?(ia|intelligence artificielle|robot|modele|"
    r"assistant|conscient|intelligent|capable)\b|"
    r"\bton modele\b|\bton algorithme\b|\bton architecture\b|"
    r"\bta puissance\b|\btes capacites\b|\btes performances\b|"
    r"\bce que tu (peux|sais|connais|comprends)\b|"
    r"\ben tant qu'ia\b|\ben tant que ia\b|"
    r"\bcomment tu (fonctionnes|marches|es construit|travailles)\b|"
    # « intelligence artificielle » SEUL a ete retire : il faisait passer
    # pour une question SUR RontoGPT toute question dont l'IA est le SUJET
    # (« Compare l'intelligence artificielle et l'intelligence humaine »
    # recevait la reponse constante « je ne suis pas un 8B » au lieu d'une
    # analyse — bug reel trouve par scripts/banc_100.py). Les formulations
    # dirigees vers le modele restent couvertes plus haut : « es tu »,
    # « es-ce que tu es », « tu es une IA ».
    r"\b\d{1,3}[\s-]?b\b|\bgpt[-\s]?\d|\bgpt\b|\bchatgpt\b|\bopenai\b|"
    r"\bllama[-\s]?\d|\bmistral\b|\bdeepseek\b|\bclaude[-\s]?\d|\bgemini\b)",
    re.IGNORECASE)

# Reponse-constante pour les questions de capacite (0 s, zero hallucina-
# tion possible, meme esprit que _IDENTITE) : honnete et technique.
_FONCTIONNEMENT = (
    "RontoGPT traite une demande en plusieurs etapes : il cherche d'abord si "
    "un outil exact peut la resoudre (par exemple un calcul), puis choisit les "
    "experts utiles. Selon la question, il peut consulter une base locale, une "
    "recherche Web ou une API publique. Le modele de langage redige ensuite la "
    "reponse à partir de la demande et des éléments trouvés ; des filtres "
    "retirent certaines répétitions et des liens de sources peuvent être "
    "ajoutés. Le modèle local peut encore se tromper : les outils réduisent "
    "certaines erreurs, sans garantir toutes les réponses. La recherche Web "
    "nécessite une connexion Internet."
)
_CAPACITES = (
    "Je peux expliquer et rédiger, aider à programmer, résoudre certains "
    "calculs avec des outils dédiés, retrouver des éléments de la conversation "
    "et chercher des informations sur le Web si la connexion est disponible. "
    "La précision dépend de la tâche et des sources accessibles ; je peux "
    "encore produire des erreurs."
)
_CAPACITE = (
    "RontoGPT est un système qui combine un modèle de langage local avec des "
    "outils de calcul, de routage, de mémoire et de recherche. Le profil fourni "
    "par défaut utilise Gemma 3 1B quantifié, mais la configuration peut choisir "
    "un autre modèle compatible. Un modèle plus grand peut mieux réussir "
    "certaines tâches, au prix de ressources supplémentaires. Il n'existe pas "
    "de score unique qui permette de dire que RontoGPT équivaut partout à un "
    "modèle 3B, 8B ou autre : cela dépend du modèle, des outils et de la tâche."
)


# ACCUSE DE RECEPTION (« ok », « merci », « d'accord ») : ni une question
# ni une tache, mais ce n'est PAS neutre pour autant. Sans garde-fou le
# 1B, privé de signal, se mettait à CONTINUER le format de tournure de la
# brique few-shot d'agents.py et rendait « R: Quel est le taux de change
# du euro a la date limite precedente ? » — en prod l'exemple du ciel
# était recopié et attribué a l'utilisateur (« Vous avez pose : Pourquoi
# le ciel est-il bleu ? »). Bug reel reproduit, 2026-10.
#
# PORTÉE MESURÉE : les SALUTATIONS sont volontairement EXCLUES. Avec la
# brique de style corrigée (cf. agents.py), le 1B répond parfaitement à
# « bonjour » -> « Bonjour ! Comment puis-je vous aider aujourd'hui ? ».
# Les intercepter dégraderait la réponse pour rien — et deux tests
# (test_aura, test_jev) utilisent « bonjour » comme question témoin.
_RE_BRIBE = re.compile(
    r"^(?:ok|okay|okey|d'?accord|entendu|compris|parfait|super|cool|"
    r"nickel|merci(?: beaucoup| bien)?|thanks|thx|"
    r"oui|non|ouais|yep|nope|exact|exactement|mdr|lol|ptdr|"
    r"ça marche|ca marche)[\s!.,;:?…]*$", re.IGNORECASE)

_BRIBE = ("Bien reçu. Dis-moi ton besoin — un calcul exact, un fait "
          "vérifié ou une explication — et je m'en occupe.")


def _est_bribe(question: str) -> bool:
    """Vrai si le message n'est QUE de la conversation, sans contenu.

    Ancre des deux bouts, 4 mots et 40 caracteres au maximum : une
    vraie demande courte (« combien font 2+2 ») ne tombe pas dedans.
    """
    q = (question or "").strip()
    if not q or len(q) > 40 or len(q.split()) > 4:
        return False
    return bool(_RE_BRIBE.match(q))


_SYSTEME_REQUETE = (
    "Tu transformes une question en requête de moteur de recherche. "
    "Réponds par UNE SEULE ligne de 4 à 8 mots-clés : pas de phrase, "
    "pas de guillemets, pas de ponctuation, pas d'explication.")
_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
          "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")
# Questions entieres seulement (ancrees des deux bouts) : « quelle heure
# est-il » oui, « quelle heure de train pour Lyon ? » non.
_RE_HEURE = re.compile(
    r"^(?:quelle\s+heure(?:\s+(?:est[\s-]il|il\s+est))?|"
    r"l'heure(?:\s+(?:actuelle|qu'il\s+est))?)[\s?!.]*$")
_RE_DATE = re.compile(
    r"^(?:quel\s+jour(?:\s+(?:on\s+est|sommes[\s-]nous|est[\s-]on))?|"
    r"quelle\s+(?:est\s+la\s+)?date(?:\s+(?:d'aujourd'hui|du\s+jour))?|"
    r"date\s+(?:d'aujourd'hui|du\s+jour)|on\s+est\s+quel\s+jour)"
    r"[\s?!.]*$")


def _reponse_temps(question: str, maintenant: datetime | None = None
                   ) -> str | None:
    """Heure ou date lues sur l'horloge locale, sinon None.

    Reponse VOLATILE : jamais memorisee (cf. _memoriser_reponse).
    """
    q = _sans_accents(question).lower().replace("\u2019", "'").strip()
    now = maintenant or datetime.now()
    if _RE_HEURE.match(q):
        return f"Il est {now.hour} h {now.minute:02d}."
    if _RE_DATE.match(q):
        return (f"Nous sommes le {_JOURS[now.weekday()]} {now.day} "
                f"{_MOIS[now.month - 1]} {now.year}.")
    return None


def _reponse_bribe(question: str) -> str | None:
    """Reponse constante pour un accuse de reception, sinon None."""
    return _BRIBE if _est_bribe(question) else None


# RECOPIE EXACTE : « ecris exactement le mot BANANE », « ecris la
# phrase : je teste RontoGPT ». Ce n'est PAS une tache de redaction — le 1B
# « corrigeait » la consigne au lieu de l'executer : au banc 100 il
# rendait « Je tests RontoGPT. » pour « je teste RontoGPT » et autre chose que
# BANANE. Une consigne de recopie se rend TELLE QUELLE, par le programme.
# Deux formes seulement (jamais de forme libre) pour eviter les faux
# positifs du genre « ecris la phrase correcte en anglais ».
_RE_RECOPIE = re.compile(
    r"\becris?\s+(?:moi\s+)?(?:"
    r"exactement\s+(?:le\s+mot|la\s+phrase|la\s+formule|le\s+texte)\s+(.+?)"
    r"|"
    r"(?:le\s+mot|la\s+phrase|la\s+formule|le\s+texte)\s*:\s*(.+?)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL)


def _recopie(question: str) -> str | None:
    """Le texte a recopier tel quel, ou None si ce n'en est pas une."""
    m = _RE_RECOPIE.search((question or "").strip())
    if not m:
        return None
    texte = (m.group(1) or m.group(2) or "").strip()
    texte = texte.rstrip(" .,;:")      # ponctuation de la CONSIGNE
    if not texte or len(texte) > 300:
        return None
    return texte


# Question qui attend une VALEUR/UN FAIT court (« quel age a Alice »,
# « quel est le prix du carburant ») : la dissertation multi-pass (75-86 s
# mesures au banc) y est du gachis — on reste en reponse courte.
_RE_REPONSE_COURTE = re.compile(
    r"(\bquel\s+(age|est|sont|prix|nombre|est-ce)|\bquelle?\s+(est|sont|la "
    r"capitale|la reponse)|\bcombien\b|\bqui (est|a|trouve)|"
    r"\bo[ùu] se trouve\b|\bquand (est-ce|a-t-il|est ne)\b|\bprix de\b)",
    re.IGNORECASE)

# Refus INTEGRAL (le modele declina puis se tait) : ne doit jamais etre
# mis en cache ni livre comme si c'etait une reponse. S'il y a une vraie
# suite derriere, c'est un simple tic de tete -> retire.
_RE_REFUS = re.compile(
    r"^\s*(je suis desole|je suis desolee|je ne peux pas|"
    r"je ne suis pas en mesure|desole, mais|pardon, mais)", re.IGNORECASE)


def _est_refus(texte: str) -> bool:
    """Refus court = pas une reponse (refus long avec suite = garde-fou
    legitime, cf. filet hors-ligne)."""
    return bool(texte and _RE_REFUS.search(texte) and len(texte) < 400)


def _retirer_refus_tete(texte: str) -> str:
    """Si la reponse REFUSE puis repond quand meme, coupe le refus de tete
    (« Je suis desole, mais... » suivi d'une vraie analyse = tic parasite)."""
    if not texte or not _RE_REFUS.search(texte):
        return texte
    phrases = re.split(r"(?<=[.!?])\s+", texte.strip())
    if len(phrases) < 2:
        return texte
    reste = " ".join(phrases[1:]).strip()
    # on ne coupe que si la suite est substantielle (sinon on garde le
    # garde-fou : un refus honnete reste un refus honnete)
    return reste if len(reste) >= 200 else texte


# --- CONTAMINATION PAR L'HISTORIQUE ------------------------------------
# En conversation, le 1B (1 milliard de parametres) confond « repondre » et
# « continuer » : il recopie la consigne de source, ou rejoue le tour
# precedent, sources comprises. Deux mesures du 11 oct. 2026, meme question
# seule dans un processus neuf : la reponse est JUSTE (nom, sources) ; la
# meme question apres un tour precedent rend « Je vais repondre avec les
# faits fournis (WEB / BASE DE FAITS). Sources : [...] » au lieu du nom.
# Le remede : retirer l'amorce meta, et, s'il ne reste rien d'utile ou si le
# tour precedent est rejoue, REGENERER SANS HISTORIQUE — plus rien a copier.
_RE_AMORCE_META = re.compile(
    r"^\W*\**(?:je vais (?:repondre|utiliser|me baser|reformuler)|"
    r"compris\b|d'accord\b|tres bien\b|"
    r"d'apres les (?:faits|extraits|sources) (?:fournis|fournies)|"
    r"voici (?:une|ma) reponse|en me basant sur)", re.IGNORECASE)


def _sans_amorce_meta(texte: str) -> str:
    # Coupe une amorce meta en tete. Renvoie "" si la reponse n'etait QUE ca :
    # l'appelant la traite comme vide et regenere sans historique.
    if not texte or not _RE_AMORCE_META.search(texte):
        return texte
    phrases = re.split(r"(?<=[.!?])\s+", texte.strip())
    return " ".join(phrases[1:]).strip()


_RE_LIGNE_SOURCE = re.compile(r"^\W*sources?\s*:", re.IGNORECASE)


def _reste_substantiel(texte: str) -> bool:
    # Vrai s'il reste une VRAIE reponse une fois les lignes de sources
    # retirees. « Sources : [...] » seul n'est pas une reponse : c'est
    # exactement ce que laissait « Compris, j'utiliserai ces extraits...
    # Sources : [...] » apres retrait de l'amorce meta (cas reel du
    # 11 oct. 2026, ou l'utilisateur a vu cette ligne a la place d'une
    # reponse).
    if not texte:
        return False
    reste = "\n".join(ligne for ligne in texte.splitlines()
                       if not _RE_LIGNE_SOURCE.match(ligne.strip()))
    return bool(reste.strip())


def _rejoue_le_tour_precedent(reponse: str, historique) -> bool:
    # Vrai si la reponse recopie au moins 60 signes d'affilee du dernier tour.
    if not reponse or historique is None:
        return False
    try:
        tours = [t for t in historique.fenetre()
                 if t.get("role") == "assistant"]
    except Exception:                       # objet sans fenetre() exploitable
        return False
    if not tours:
        return False
    precedent = tours[-1].get("content") or ""
    if not precedent:
        return False
    from difflib import SequenceMatcher
    commun = SequenceMatcher(None, reponse, precedent).find_longest_match(
        0, len(reponse), 0, len(precedent)).size
    return commun >= 60


def _est_meta_ia(question: str) -> bool:
    """Question adressee AU modele sur lui-meme (capacites, taille, identite)."""
    return bool(_RE_META_IA.search(question or ""))

# Definition/savoir (« qu'est-ce que X », « explique X ») : reponses que le
# 1B invente avec aplomb — le juge d'un petit modele est mauvais, donc on
# ANCRE : contexte web obligatoire (ou graphe de faits) avant de repondre.
_RE_DEF = re.compile(
    r"\b(qu'?est.?ce.?que|qu'?est.?ce.?qu'|c'?est.?quoi|qu'elle? est la "
    r"definition|definis|explique(.?moi)?|montre.?moi)\b", re.IGNORECASE)
_RE_PHRASE_MIN = re.compile(r"\b(theoreme|theorie|loi|principe|concept|"
    r"formule|definition|signifie|histoire de| fonctionne)", re.IGNORECASE)

# SOUS-ENSEMBLE STRICT de _RE_DEF, pour le FILET FINAL seulement.
#
# Deux demandes differentes etaient confondues dans `est_def` :
#   * « qu'est-ce que X / c'est quoi X / definis X » demande un FAIT
#     VERIFIABLE : sans ancrage, s'abstenir est la bonne reponse — inventer
#     une definition fausse avec aplomb est le defaut mesure sur ce projet ;
#   * « explique X / montre-moi X » demande un DEVELOPPEMENT : le modele
#     peut le produire de memoire, et s'abstenir y serait une non-reponse.
# `est_def` reste tel quel pour FORCER l'ancrage web (un contexte aide dans
# les deux cas). Seul le filet final distingue les deux.
#
# Consequence visible, et c'est ce qui a fait remonter le probleme : tant que
# le mode dissertation existait, toute question contenant « explique »
# partait en redaction AVANT d'atteindre le filet — la confusion etait sans
# effet. Sa suppression l'a rendue visible : hors-ligne, « Explique la
# photosynthese » repondait « je ne peux pas verifier cette definition ».
_RE_DEF_LOOKUP = re.compile(
    r"\b(qu'?est.?ce.?que|qu'?est.?ce.?qu'|c'?est.?quoi|definis|"
    r"qu'elle? est la definition|la definition de|signifie)\b", re.IGNORECASE)


_AMORCE_DEF = re.compile(
    r"^\W*(?:qu'?est[\s-]?ce[\s-]?qu(?:e|')|c'?est[\s-]?quoi|"
    r"explique(?:[\s-]?moi)?|definis|montre[\s-]?moi|que\s+signifie|"
    r"definition\s+de)\s*", re.IGNORECASE)
_ARTICLE_DEF = re.compile(r"^(?:une?|le|la|les|des|du|de\s+la|d'|l')\s+",
                          re.IGNORECASE)


def _sujet_definition(question: str) -> str:
    """Le sujet nu d'une question de definition (« fonction canonique »)."""
    sujet = _AMORCE_DEF.sub("", _sans_accents(question or "")).strip()
    sujet = _ARTICLE_DEF.sub("", sujet).strip()
    sujet = re.sub(r"[?!.;,]+$", "", sujet).strip()
    return sujet


def _repli_extractif(contexte_web: str) -> str:
    """Reponse de secours : citer le VERIFIE au lieu de servir du bavardage.

    Utilise quand la canalisation du 1B (`llama_cerveau._sans_parrotage`) a
    vide la reponse : le modele avait RECLAME le contexte au lieu de
    rediger. Les extraits partent tels quels, sans generation — donc sans
    nouvelle occasion d'inventer.
    """
    return ("D'apres les sources verifiees en ligne :\n"
            + llama_cerveau._sans_doublons(
                _sans_fragment_incomplet(contexte_web)))


def _sans_fragment_incomplet(texte: str) -> str:
    """Retire les fragments finaux en pleine phrase (snippets tronques).

    Un moteur de recherche renvoie des extraits coupes net, souvent juste
    avant une formule (« ... une factorisation par a : »). Les citer tels
    quels DONNE L'IMPRESSION que la reponse s'arrete au milieu. On recule
    donc au dernier point de fin de phrase : le texte rendu est complet.
    """
    lignes: list[str] = []
    for ligne in (texte or "").splitlines():
        t = ligne.rstrip()
        if not t:
            continue
        if t[-1] in ".!?…»\"":
            lignes.append(t)
            continue
        fins = list(re.finditer(r"[.!?…](?=\s)", t))
        lignes.append(t[:fins[-1].end()].rstrip() if fins else t)
    return "\n".join(lignes)


def _source_pertinente(question: str, contexte_web: str) -> bool:
    """La source parle-t-elle VRAIMENT du sujet demande ?

    Strict a dessein : le sujet doit apparaitre comme phrase contigue. Un
    simple recouvrement de mots (« fonction » et « canonique » presents
    separement, dans un autre sens) ne suffit pas — c'est exactement le cas
    qui doit rester en citation.
    """
    sujet = _sujet_definition(question)
    if len(sujet) < 3 or not contexte_web:
        return False
    return sujet in _sans_accents(contexte_web)
