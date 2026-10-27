"""Tests des syllogismes : inclusion, exclusion, instances, conclusions."""
from aura import syllogismes as s


# ── _sing : pluriel grossier -> singulier ──────────────────────────────────


def test_sing_pluriel_en_singulier():
    assert s._sing("felins") == "felin"
    assert s._sing("chats") == "chat"
    assert s._sing("un") == "un"            # mot court : on garde


# ── _jetons : normalisation (casse, accents, ponctuation) ──────────────────


def test_jetons_normalise_majuscules_accents_punctuation():
    assert s._jetons("Tous les CHATS, sont des FELINS !") == \
        ["tous", "les", "chats", "sont", "des", "felins"]
    assert s._jetons("Les Écureuils sont des rongeurs") == \
        ["les", "ecureuils", "sont", "des", "rongeurs"]


# ── _couper : separation des premisses et de la question ───────────────────


def test_couper_avec_peut_on_dire():
    j = ["tous", "les", "chats", "sont", "des", "felins",
         "peut", "on", "dire", "que", "felix", "est", "un", "felin"]
    pre, q = s._couper(j)
    assert pre == ["tous", "les", "chats", "sont", "des", "felins"]
    assert q[0] == "peut" and q[-1] == "felin"


def test_couper_sans_trigger_retourne_question_nulles():
    assert s._couper(["tous", "les", "chats", "sont", "des", "felins"]) == \
        (["tous", "les", "chats", "sont", "des", "felins"], None)


# ── _regles : inclusions / exclusions / instances ──────────────────────────


def test_regles_inclusion_tous():
    incl, excl, inst = s._regles(["tous", "les", "chats", "sont", "des", "felins"])
    assert incl == [("chat", "felin")]
    assert excl == []
    assert inst == []


def test_regles_inclusion_tout():
    incl, excl, inst = s._regles(["tout", "chat", "est", "felin"])
    assert incl == [("chat", "felin")]


def test_regles_exclusion():
    incl, excl, inst = s._regles(["aucun", "felin", "n", "est", "chat"])
    assert excl == [("felin", "chat")]
    assert incl == []


def test_regles_instance():
    incl, excl, inst = s._regles(["felix", "est", "un", "chat"])
    assert inst == [("felix", "chat")]
    assert incl == [] and excl == []


# ── _conclusion : membre / inclus / exclus / absente ───────────────────────


def test_conclusion_membres():
    assert s._conclusion(["peut", "on", "dire", "que", "felix", "est", "un", "felin"]) \
        == ("membre", "felix", "felin")


def test_conclusion_absente():
    # Sans pattern « X est un Y » exploable : pas de conclusion.
    assert s._conclusion(["peut", "on", "dire", "que", "felix", "a", "des", "chats"]) \
        is None


# ── _supersets : fermeture transitive ──────────────────────────────────────


def test_supersets_fermeture_transitive():
    sup = s._supersets([("chat", "felin"), ("felin", "animal")])
    assert sup["chat"] >= {"animal"}


# ── repondre : oui / non / abstention ───────────────────────────────────────


def test_repondre_oui_par_instance_derivee():
    # Felicem est un chat ; tous les chats sont des felins -> felicem est un felin
    r = s.repondre(
        "Felicem est un chat. Tous les chats sont des felins. "
        "Peut on dire que felicem est un felin ?")
    assert r == "Oui."


def test_repondre_absente_ne_invente_pas():
    # Question sans conclusion exploitable : on s'abstient.
    assert s.repondre("Tous les chats sont des felins. Peut on dire que le ciel est bleu ?") \
        is None
