"""Clarification interactive : demander une precision AVANT de repondre.

Le probleme mesure
------------------
Le 1B ne dit jamais « ta question est trop vague ». Face a « Explique » ou
« Ecris un poeme », il INVENTE un sujet et redige 300 mots a cote — la
reponse est fausse non parce que le cerveau se trompe, mais parce que la
demande ne portait aucun contenu. Un grand modele s'en sort par la
conversation ; un 1B, non : il n'a pas les tokens de contexte necessaires
pour deviner, et il n'a pas la prudence de demander.

Ici, la porte pose 1 a 3 questions de relance et NE REPOND PAS. Le tour
suivant apporte la precision, la question d'origine et la precision sont
RECOLLEES (concatenation), et le pipeline repart — cette fois avec un
sujet.

Trois regles de conception
--------------------------
1. **Deterministe.** Seules des regles de forme decident. Le juge d'un
   petit modele est mauvais (`orchestrateur._CAPACITE` est ne de ce
   constat) : lui demander « cette question est-elle assez precise ? »
   reviendrait a ajouter une source de bruit et ~0,5 s de latence a
   CHAQUE question non resolue. Les regles sont relisibles et testables.

2. **Apres les experts, jamais avant.** La porte vit dans la cascade a la
   position declaree dans `registre_experts.PORTES` : apres la derniere
   couche qui REPOND, avant la generation. Une question que `calcul_taux`
   ou le cache savent trancher ne doit jamais se voir opposer « precise
   ta demande » — sinon on aurait refait, a l'envers, le bug du cache
   semantique (corrige le 6 oct. 2026).

3. **Un seul tour.** `Suivi` n'accepte qu'une relance : si la reponse de
   l'utilisateur reste vague, on repond avec ce qu'on a. Une porte qui
   peut se reposer indefiniment est un questionnaire, pas un assistant.

Ce qui declenche, et ce qui ne declenche JAMAIS
-----------------------------------------------
  * `verbe-nu`         « Explique », « Aide-moi », « Continue »
  * `objet-generique`  « Ecris un poeme », « Fais un plan » (sans sujet)
  * `renvoi-sans-sujet` « Explique ca », « Et ensuite ? » — SANS antecedent
  * `conversion-incomplete` « Convertis 100 » (unite cible absente)
  * `mot-de-passe`     « Genere un mot de passe » (longueur/criteres absents)

`objet-generique` et `renvoi-sans-sujet` ne se declenchent QUE s'il n'y a
aucun sujet precedent dans la conversation : apres « parlons des volcans »,
« Ecris un poeme » designe evidemment les volcans. Prendre la suite d'un
echange pour une question vide serait le pire faux positif possible — celui
qui fait perdre du temps a un utilisateur qui vient de parler.

Ce qui ne declenche pas, volontairement : « raconte moi une blague » (le
sujet n'ajoute rien — n'importe quelle blague repond), « Ecris un poeme sur
la mer », « Resume la Revolution francaise », et les 100 questions de
`scripts/banc_100.py` (cf. tests/test_clarification.py).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

LOG = logging.getLogger("aura.clarification")

# Coupe-circuit : AURA_CLARIFICATION=0 rend la porte transparente (la
# cascade repart exactement comme avant le module).
_VAR = "AURA_CLARIFICATION"


def actif() -> bool:
    """La porte est-elle branchee ? (AURA_CLARIFICATION=0 pour couper)"""
    import os
    return os.environ.get(_VAR, "1") not in ("0", "off", "false", "")


# ── normalisation -----------------------------------------------------------

def _norm(texte: str) -> str:
    """Minuscules, sans accents, apostrophes droites, espaces normalises."""
    t = unicodedata.normalize("NFKD", (texte or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(t.split())


# ── 1. le lexique des demandes ---------------------------------------------
# Formes verbales qui DEMANDENT quelque chose (par opposition aux questions
# factuelles : « quelle est la capitale... » n'est pas dans cette liste).
_VERBES = frozenset("""
explique expliques expliquer explication
detaille detailles detailler
developpe developpes developper
resume resumes resumer
redige rediges rediger
ecris ecrit ecrire
traduis traduit traduire
corrige corriges corriger
ameliore ameliores ameliorer
reformule reformules reformuler
analyse analyses analyser
compare compares comparer
decris decrit decrire
liste listes lister
enumere enumeres enumerer
calcule calcules calculer
raconte racontes raconter
fais fait faire
donne donnes donner
genere generes generer
propose proposes proposer
conseille conseilles conseiller
aide aides aider
montre montres montrer
trouve trouves trouver
cherche cherches chercher
continue continues continuer
poursuis poursuit poursuivre
simplifie simplifies simplifier
planifie planifies planifier
construis construire
complete completes completer
precise precises preciser
illustre illustres illustrer
parle parles parler
raconte racontes raconter
dis dit dire
""".split())

# Objets GENERIQUES : le verbe a bien un objet, mais cet objet ne dit pas
# DE QUOI il parle. « Ecris un poeme » ne permet pas de choisir un sujet —
# il ne permet que d'ecrire un poeme quelconque.
#
# « blague » est volontairement ABSENT : n'importe quelle blague repond a
# « raconte moi une blague » (sujet sans enjeu), et ce cas est teste
# ailleurs (tests/test_anti_hallucination.py).
_OBJETS_GENERIQUES = frozenset((
    "plan", "resume", "texte", "histoire", "poeme", "exemple", "liste",
    "calcul", "exercice", "code", "script", "fonction", "programme",
    "algorithme", "recette", "itineraire", "cv", "lettre", "email", "mail",
    "rapport", "fiche", "dissertation", "redaction", "argumentaire",
    "tableau", "graphique", "formule", "solution", "reponse", "question",
    "conte", "chanson", "discours", "resume de cours", "plan de cours",
))

_ARTICLES = frozenset((
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "mon",
    "ma", "mes", "ton", "ta", "tes", "ce", "cet", "cette", "ces", "quelque",
    "quelques",
))

# Pronoms de renvoi : leur sens est AILLEURS (dans un tour precedent).
_DEICTIQUES = frozenset((
    "ca", "cela", "ceci", "ce", "cette", "cet", "le", "la", "les", "l",
    "lui", "elle", "eux", "celui", "celle", "chose", "truc", "machin",
    "meme", "sujet", "probleme", "question",
))

_TRAITEMENTS_VIDES = frozenset(("un", "peu", "donc", "vite", "maintenant"))

# Datifs (« explique-moi », « donne nous ») : ils ne portent aucun contenu.
_DATIFS = frozenset(("moi", "nous", "me", "m"))

# Pronoms elides ecrits SEULS (sans apostrophe) en tete de message.
_ELISIONS_SEULES = frozenset(("m", "t", "s", "j", "l", "d", "n", "c",
                              "qu"))

# Verbes SEQUENTIELS : leur objet est le TOUR PRECEDENT. « Continue » apres
# un echange n'a pas besoin qu'on lui demande « sur quoi ? » — la porte
# s'efface alors qu'elle se declenche sur « Explique » (dont l'objet n'est
# jamais donne par le seul contexte).
_VERBES_SEQUENTIELS = frozenset((
    "continue", "continues", "continuer", "poursuis", "poursuit",
    "poursuivre", "precise", "precises", "preciser", "developpe",
    "developpes", "developper", "reprends", "recommence", "refais",
    "simplifie", "reformule", "ameliore", "corrige", "traduis",
))

# Elisions : « m'expliques », « c'est », « l'atome » — l'apostrophe n'est pas
# un mot. Sans ce decoupage, « m'expliques » n'etait pas reconnu comme le
# verbe « expliques » et la porte laissait passer une demande sans objet.
_RE_ELISION = re.compile(r"\b([a-z])'")

# Prepositions suspendues : « aide moi a » s'arrete au milieu d'une phrase.
_PREPOSITIONS = frozenset(("a", "de", "d", "en", "sur", "pour", "avec",
                           "vers", "dans", "par", "que"))

# Mots qui ne portent AUCUN sujet (salutations, politesse, formules).
# Sert a decider si la conversation a deja un sujet : les messages de
# pre-chauffage de la GUI (« Bonjour ») ne doivent pas compter comme un
# debut de conversation, sinon la porte ne se poserait jamais.
_MOTS_SOCIAUX = frozenset((
    # salutations et acquittements
    "bonjour", "salut", "bonsoir", "coucou", "hello", "hey", "yo", "merci",
    "ok", "okay", "oui", "non", "ouais", "stp", "svp", "please", "ca", "va",
    "vas", "tu", "vous", "je", "comment", "au", "revoir", "bientot",
    "bonne", "journee", "soiree", "bien", "tres", "et", "toi", "moi",
    "est", "ce", "que", "qui", "quoi", "de", "du", "la", "le", "les", "un",
    "une", "des", "il", "elle", "on", "s'il", "te", "plait", "allez",
    # politesse, formules, verbes de volonte — un tour qui n'en contient pas
    # d'autres ne porte AUCUN sujet (« merci beaucoup » n'est pas un sujet)
    "beaucoup", "trop", "peu", "peux", "peut", "pourrais", "pourriez",
    "voudrais", "veux", "veut", "aimerais", "besoin", "aide", "aider",
    "sympa", "gentil", "chose", "choses", "truc", "trucs", "vraiment",
    "toujours", "jamais", "encore", "voila", "parfait", "super", "cool",
    "nickel", "d'accord", "daccord", "accord", "compris", "entendu",
    "desole", "mieux", "plus", "moins", "entre", "avec", "sans", "pour",
    "dans", "sur", "sous", "quand", "alors", "donc", "aussi", "autre",
))

# ── 2. politesse -------------------------------------------------------------
# Le texte oral arrive sans apostrophe (« s il te plait », « c est quoi ») :
# les motifs tolerent l'espace a la place de l'apostrophe, sinon une demande
# dictee passerait pour une demande a objet.
_SVP = r"s\s*'?\s*il\s+te\s+plait|s\s*'?\s*il\s+vous\s+plait"
_RE_AVANT = re.compile(
    r"^(?:(?:" + _SVP + r"|svp|stp|please|peux[- ]tu|"
    r"pourrais[- ]tu|pourriez[- ]vous|peux[- ]vous|"
    r"est[- ]ce que tu peux|est[- ]ce que vous pouvez|tu peux|vous pouvez|"
    r"je (?:veux|voudrais) que tu|j\s*'?\s*aimerais(?: que tu)?)\s+)+")
_RE_APRES = re.compile(
    r"\s+(?:" + _SVP + r"|svp|stp|please|merci|"
    r"un peu|maintenant|pour moi|rapidement|vite)+$")


def _depouiller(t: str) -> str:
    """Retire l'emballage poli (« peux-tu ... s'il te plait »)."""
    avant = ""
    while avant != t:
        avant = t
        t = _RE_AVANT.sub("", t)
        t = _RE_APRES.sub("", t)
    return t


def _mots(t: str) -> list[str]:
    """Jetons d'une demande depouillee (tirets, elisions, ponctuation)."""
    t = _RE_ELISION.sub(" ", t)
    t = t.replace("-", " ").strip(" \t.,;:!?…\"'")
    return re.findall(r"[a-z0-9']+", t)


def _nettoyer_bords(mots: list[str]) -> list[str]:
    """Retire les bords sans contenu d'une demande.

    En FIN : mots vides et prepositions suspendues — « aide moi a » s'arrete
    au milieu d'une phrase ; sans ce nettoyage une demande incomplete
    passerait pour une demande a objet.
    En TETE : pronoms elides ecrits sans apostrophe (« m aider » pour
    « m'aider ») — sinon le verbe n'est plus reconnu comme premier mot.
    """
    while mots and mots[-1] in (_TRAITEMENTS_VIDES | _PREPOSITIONS):
        mots.pop()
    while len(mots) > 1 and mots[0] in _ELISIONS_SEULES:
        mots.pop(0)
    return mots


# ── 3. les regles -----------------------------------------------------------

# « et ensuite ? », « pourquoi ? », « c'est quoi ? », « continue »
# Ces formes n'ont de sens QUE par rapport a un tour precedent : elles ne
# sont donc signalees que sans sujet en conversation (cf. `a_contexte`).
_RE_SUITE_NUE = re.compile(
    r"^(?:et\s+)?(?:ensuite|apres|puis|donc|alors)\s*\??$"
    r"|^(?:pourquoi|comment|quand|ou|combien|quoi|ensuite)\s*\??$"
    r"|^(?:continue|continues|poursuis|poursuit|precise|vas y|vas-y|"
    r"encore|reprends)\s*$"
    r"|^(?:c\s*'?\s*est quoi|et c\s*'?\s*est quoi|"
    r"qu\s*'?\s*est[- ]ce que c\s*'?\s*est)\s*\??$")

_RE_CONVERSION = re.compile(r"\b(converti\w*|change\w*|transforme\w*)\b")
_RE_NOMBRE = re.compile(r"\d")
_RE_UNITE_CIBLE = re.compile(r"\ben\s+\S")
_RE_PASSE = re.compile(r"\bmot de passe\b|\bmots de passe\b|\bpassword\b")
_RE_VERBE_PASSE = re.compile(
    r"\b(genere\w*|cree\w*|creer|fais|fait|donne\w*|propose\w*|trouve\w*|"
    r"invente\w*|ecris|ecrire|suggere\w*)\b")
# NB : « mot » en toutes lettres est volontairement ABSENT — il est deja
# dans « mot de passe » et ferait donc matcher la longueur sur la seule
# presence du sujet (bug attrape au premier essai : « genere un mot de
# passe » n'etait jamais signale).
_RE_LONGUEUR = re.compile(
    r"\b\d|\b(long|longueur|court|caractere\w*|lettre\w*|symbole\w*|"
    r"chiffre\w*|aleatoire|solide|robuste|simple|memorisable|"
    r"phrase)\b")

# Intentions formulees sans objet concret : elles ne commencent pas toujours
# par un verbe d'action ("je voudrais faire un truc", "j'ai besoin d'aide").
# On ne declenche que sur ces formes explicites pour eviter les faux positifs.
_RE_INTENTION_VAGUE = re.compile(
    r"^(?:je\s+(?:veux|voudrais|souhaite|souhaiterais|dois)\s+"
    r"(?:faire|creer|construire|preparer|organiser|ameliorer|corriger|"
    r"comprendre|resoudre|ecrire|rediger|traduire|choisir|apprendre)"
    r"|j\s*'\s*aimerais\s+(?:faire|creer|construire|preparer|organiser|"
    r"ameliorer|corriger|comprendre|resoudre|ecrire|rediger|traduire|"
    r"choisir|apprendre))"
    r"(?:\s+(?:quelque\s+chose|un\s+truc|un\s+peu|des\s+choses))?"
    r"\s*[.!?]*$")
_RE_BESOIN_AIDE_VAGUE = re.compile(
    r"^(?:j\s*'\s*ai\s+besoin\s+d\s*'\s*aide|"
    r"je\s+(?:suis\s+perdu|ne\s+sais\s+pas\s+quoi\s+faire))\s*[.!?]*$")


@dataclass(frozen=True)
class Besoin:
    """Ce qui manque a une question pour qu'une reponse soit possible."""
    question: str                 # la question d'origine, telle quelle
    famille: str                  # la regle qui a conclu
    manque: tuple[str, ...]       # cles de ce qui manque
    questions: tuple[str, ...]    # 1 a 3 relances a poser
    raison: str                   # pourquoi on refuse de repondre

    def texte(self) -> str:
        """Le message livre a l'utilisateur (les questions, rien d'autre)."""
        if len(self.questions) == 1:
            corps = f"1. {self.questions[0]}"
        else:
            corps = "\n".join(f"{i}. {q}"
                              for i, q in enumerate(self.questions, 1))
        # Texte DESTINE A L'UTILISATEUR (meme convention que _BRIBE) :
        # phrase accentuee, pas cle technique.
        return (f"Il me manque une précision avant de répondre :\n\n{corps}\n\n"
                "Réponds à cela et je te donne la réponse complète.")


_RELANCE: dict[str, str] = {
    "objet": "Quel résultat veux-tu obtenir exactement ? Donne-moi le sujet ou la tâche précise.",
    "sujet": "Quel sujet veux-tu que je traite ?",
    "contraintes": "Quelles contraintes veux-tu (longueur, ton, public) ?",
    "reference": "De quoi parles-tu exactement ? Je n'ai pas d'échange "
                 "précédent auquel me raccrocher.",
    "unite": "Vers quelle unité veux-tu convertir cette valeur ?",
    "format": "Quelle longueur et quels critères veux-tu "
              "(taille, majuscules, chiffres, symboles) ?",
}


def _besoin(question: str, famille: str, manque: tuple[str, ...],
            raison: str) -> Besoin:
    questions = tuple(_RELANCE[c] for c in manque[:3])
    return Besoin(question=question.strip(), famille=famille, manque=manque,
                  questions=questions, raison=raison)


def detecter(question: str, a_contexte: bool = False) -> Besoin | None:
    """Analyse une question : None si elle est assez precise, sinon le besoin.

    `a_contexte` : vrai si un tour UTILISATEUR precedent portait deja un
    sujet. Sans sujet precedent, un renvoi (« explique ca ») ou un objet
    generique (« ecris un poeme ») ne designe rien ; avec un sujet, ils
    designent ce dont on vient de parler — et la porte s'efface.
    """
    t = _depouiller(_norm(question))
    if not t:
        return None

    if _RE_INTENTION_VAGUE.match(t) or _RE_BESOIN_AIDE_VAGUE.match(t):
        return _besoin(question, "objectif-absent", ("objet",),
                       "intention sans objet ni resultat attendu")

    # 1. CONVERSION INCOMPLETE : « convertis 100 » — sans unite cible, la
    #    reponse est impossible (100 quoi, vers quoi ?). Teste EN PREMIER :
    #    ces verbes ne sont pas dans _VERBES, mais l'ordre rend la regle
    #    lisible dans les traces.
    if _RE_CONVERSION.search(t) and not _RE_UNITE_CIBLE.search(t) \
            and (_RE_NOMBRE.search(t) or t.rstrip().endswith(" en")):
        return _besoin(question, "conversion-incomplete", ("unite",),
                       "verbe de conversion sans unite cible")

    # 2. MOT DE PASSE : « genere un mot de passe » — la longueur et les
    #    criteres changent entierement la reponse demandee.
    if _RE_PASSE.search(t) and _RE_VERBE_PASSE.search(t) \
            and not _RE_LONGUEUR.search(t):
        return _besoin(question, "mot-de-passe", ("format",),
                       "generation de mot de passe sans longueur ni criteres")

    # 3. RENVOI SANS SUJET : « explique ca », « et ensuite ? » — le sens est
    #    dans un tour precedent qui n'existe pas.
    if not a_contexte and _RE_SUITE_NUE.match(t):
        return _besoin(question, "renvoi-sans-sujet", ("reference",),
                       "renvoi a un sujet precedent inexistant")

    mots = _nettoyer_bords(_mots(t))
    if not mots:
        return None
    if mots[0] not in _VERBES:
        return None
    reste = mots[1:]
    while reste and reste[0] in _DATIFS:      # « explique MOI ca »
        reste = reste[1:]
    if len(reste) == 1 and reste[0] in _DEICTIQUES:
        if not a_contexte:
            return _besoin(question, "renvoi-sans-sujet", ("reference",),
                           "renvoi a un sujet precedent inexistant")
        return None

    # 4. VERBE NU : « Explique », « Resume », « Aide-moi ».
    #    Aucun objet du tout — la reponse serait inventee de bout en bout.
    #    Exception mesuree : un verbe SEQUENTIEL apres un tour utilisateur
    #    a objet (« Continue », « Developpe ») designe ce tour.
    if not reste:
        if a_contexte and mots[0] in _VERBES_SEQUENTIELS:
            return None
        return _besoin(question, "verbe-nu", ("objet",),
                       "demande sans objet")

    # 5. OBJET GENERIQUE : « ecris un poeme », « fais un plan ». L'objet est
    #    la, mais il ne dit pas DE QUOI il parle.
    if not a_contexte:
        noyau = [m for m in reste if m not in _ARTICLES]
        if len(noyau) == 1 and noyau[0] in _OBJETS_GENERIQUES:
            return _besoin(question, "objet-generique",
                           ("sujet", "contraintes"),
                           "objet generique sans sujet")
        if " ".join(noyau) in _OBJETS_GENERIQUES:
            return _besoin(question, "objet-generique",
                           ("sujet", "contraintes"),
                           "objet generique sans sujet")
    return None


# ── 4. le tour suivant -------------------------------------------------------

def a_un_sujet(tours) -> bool:
    """Un tour UTILISATEUR precedent portait-il un sujet exploitable ?

    « bonjour », « merci », « comment vas-tu » ne sont pas des sujets : le
    pre-chauffage `executer_detaille("Bonjour")` de la GUI ne doit pas
    suffire a faire croire que la conversation a commence. On cherche donc
    un mot PLEIN (>= 4 lettres) hors du vocabulaire social.
    """
    for tour in tours or ():
        if str(tour.get("role", "")) != "user":
            continue
        for mot in _mots(_norm(str(tour.get("content", "")))):
            if len(mot) >= 4 and mot not in _MOTS_SOCIAUX \
                    and mot not in _OBJETS_GENERIQUES:
                return True
    return False


def fusionner(origine: str, precision: str) -> str:
    """Recolle la precision a la question d'origine.

    La CONCATENATION est le bon recollage pour les trois familles :
      « Explique »        + « la photosynthese » -> « Explique la ... »
      « Ecris un poeme »  + « sur la mer »       -> « Ecris un poeme sur la mer »
      « Convertis 100 »   + « en km »            -> « Convertis 100 en km »

    Ce n'est pas cosmetique : sur la 3e, la phrase recollee est reprisable
    par les solveurs EXACTS (pal.py) alors qu'une question enrichie a la
    main (« Precision : en km ») ne l'est pas — le calcul redeviendrait une
    generation.
    """
    o = (origine or "").strip()
    p = (precision or "").strip()
    if not p:
        return o
    if not o:
        return p
    return f"{o} {p}"


class Suivi:
    """Suit une clarification avec une limite de trois relances successives."""

    MAX_RELANCES = 3

    def __init__(self) -> None:
        self._origine = ""
        self._famille = ""
        self._questions: tuple[str, ...] = ()
        self._nombre_relances = 0

    def en_attente(self) -> bool:
        return bool(self._origine)

    def poser(self, question: str, besoin: Besoin) -> None:
        self._origine = question
        self._famille = besoin.famille
        self._questions = besoin.questions
        self._nombre_relances += 1

    def peut_relancer(self) -> bool:
        return self._nombre_relances < self.MAX_RELANCES

    def consommer(self, precision: str) -> str | None:
        """Fusionne la precision et remet l'etat a zero.

        Renvoie la question recollee, ou None si rien n'etait en attente.
        """
        if not self._origine:
            return None
        fusion = fusionner(self._origine, precision)
        self._origine = ""
        self._famille = ""
        self._questions = ()
        return fusion

    def annuler(self) -> None:
        """Oublie la relance en cours (nouvelle conversation, echec)."""
        self._origine = ""
        self._famille = ""
        self._questions = ()
        self._nombre_relances = 0

    def etat(self) -> dict:
        """Trace exploitable (journal, tests, API)."""
        return {"origine": self._origine, "famille": self._famille,
                "questions": list(self._questions)}
