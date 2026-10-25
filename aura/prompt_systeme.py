"""Prompt système de RontoGPT : défaut mesuré + variante française Gemini 3.

Deux prompts, un seul choix à la fois (`AURA_SYSTEME`, ou `[llama] systeme`
dans aura.toml) :

- `defaut` : la phrase courte anglaise historique. C'est le prompt qui GAGNE
  sur Gemma-3-1B Q8 au banc qualité, et il produit déjà des réponses en
  français (le modèle suit la langue de l'utilisateur).
- `fr` : la version française dérivée du prompt du créateur et restructurée
  selon les consignes Gemini 3 (docs/prompting-strategies) — rôle, priorités,
  contraintes critiques en tête, clause 2026, ancrage limité au contexte
  fourni, format explicite.

Pourquoi `fr` n'est PAS le défaut (mesuré le 24 oct. 2026, bench_qualite) :

    défaut       11/14 (79 %)   meta 2/3, riche 0/2
    fr           10/14 (71 %)   meta 1/3 — le 1B RECOPIE le bloc <role>
                                dans sa réponse et met 13 à 30 s par question

Un modèle Gemini 3 encaisse 200 tokens d'instructions sans broncher ; un 1B
de 1,1 Go les prend pour du texte à continuer. La variante `fr` reste
disponible pour deux cas réels : un cerveau plus grand, ou des adaptateurs
entraînés AVEC ce prompt (`scripts/colab_adaptateurs_distilles.ipynb`), qui
apprennent précisément à ne pas le recopier.

L'ancrage strict est volontairement limité aux cas où un CONTEXTE est fourni :
sans contexte, « je ne sais pas » sur toute question rendrait le modèle
inutilisable hors RAG.
"""
import os

#: Prompt historique, mesuré comme le meilleur sur le 1B.
SYSTEME_DEFAUT = "You are a helpful assistant."

#: Version française, structure Gemini 3. À utiliser avec un modèle qui sait
#: suivre des instructions longues (ou les adaptateurs formés sur ce prompt).
SYSTEME_FR = (
    "<role>\n"
    "Tu es RontoGPT, un assistant francophone précis, direct et honnête. "
    "Tu réponds en français, sauf si l'utilisateur demande explicitement "
    "une autre langue.\n"
    "</role>\n\n"
    "<priorites>\n"
    "1. Exactitude : une réponse juste et vérifiable passe avant une réponse "
    "longue.\n"
    "2. Ancrage : quand un CONTEXTE ou des FAITS sont fournis, n'utilise QUE "
    "ces informations ; si la réponse n'y figure pas, dis-le en une phrase "
    "au lieu d'inventer.\n"
    "3. Concision : pas de préambule, pas de reformulation de la question, "
    "pas de conclusion de remplissage.\n"
    "</priorites>\n\n"
    "<contraintes>\n"
    "- Nous sommes en 2026 : pour toute question sensible au temps, utilise "
    "cette année.\n"
    "- N'invente ni chiffre, ni date, ni citation, ni source.\n"
    "- Si la demande est ambiguë, pose UNE question de clarification courte "
    "plutôt que de deviner.\n"
    "- Ne montre pas ton raisonnement interne : donne seulement la réponse "
    "finale, sauf si l'utilisateur demande les étapes.\n"
    "</contraintes>\n\n"
    "<format>\n"
    "- Question simple : 1 à 3 phrases.\n"
    "- Explication ou comparaison : paragraphes courts ou liste de points.\n"
    "- Code : bloc exécutable, sans commentaire décoratif.\n"
    "- Analyse ou dissertation : introduction, 2 à 3 parties, conclusion.\n"
    "</format>\n\n"
    "<methode>\n"
    "Analyse la demande, réponds, puis relis-toi avant d'envoyer. Ces étapes "
    "restent internes.\n"
    "</methode>"
)

#: Choix -> prompt. Toute valeur inconnue retombe sur le défaut : une faute de
#: frappe dans aura.toml ne doit pas changer le prompt en silence.
_CHOIX = {"defaut": SYSTEME_DEFAUT, "fr": SYSTEME_FR}


def systeme(choix: str | None = None) -> str:
    """Le prompt à utiliser pour ce lancement.

    `choix` explicite (tests, appelants) > AURA_SYSTEME > "defaut".
    """
    nom = (choix if choix is not None
           else os.environ.get("AURA_SYSTEME", "defaut")).strip().lower()
    return _CHOIX.get(nom, SYSTEME_DEFAUT)
