"""Chemins INSTANTANES : salutations, heure, date — sans modele, < 1 ms.

Mesure (13 oct. 2026) : « bonjour » passait par le pipeline complet et
coutait ~6,5 s sur CPU. Une salutation, l'heure ou la date n'ont rien a
chercher dans un modele : une reponse constante (ou une lecture d'horloge)
est plus rapide ET plus juste (un 1B ne connait pas l'heure qu'il est).

Principe
--------
- l'ANCRAGE aux deux bouts : le message doit etre ENTIEREMENT une
  salutation / une question d'heure ou de date. « Bonjour, combien font
  2+2 ? » n'est PAS capte — il continue dans la cascade ;
- la salutation est miroir (« Bonsoir » -> « Bonsoir ! »), tutoiement par
  defaut, vouvoiement si le message vouvoie ;
- l'heure et la date viennent de l'horloge de la machine (jamais du modele) ;
- `AURA_INSTANT=0` coupe tout (les tests de repli LLM s'en servent).

Ces reponses varient avec le temps : l'orchestrateur ne les met JAMAIS dans
le cache de reponses (cf. `_memoriser_reponse`).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import unicodedata

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
          "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


def _actif() -> bool:
    return os.environ.get("AURA_INSTANT", "1") != "0"


def _norm(texte: str) -> str:
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.replace("’", "'")
    n = re.sub(r"[^a-z0-9' ]+", " ", n)
    return " ".join(n.split())


_POLI = re.compile(
    r"^(?:(?:dis|dites) moi |peux tu me dire |tu peux me dire |"
    r"pouvez vous me dire |je voudrais savoir |j'aimerais savoir |"
    r"stp |svp |s'il te plait |s'il vous plait )+")
_QUEUE = re.compile(
    r"(?: (?:aujourd'hui|aujourdhui|maintenant|actuellement|en ce moment|"
    r"s'il te plait|s'il vous plait|stp|svp|exactement))+$")

# ── salutations ────────────────────────────────────────────────────────────
_SALUT = r"(?P<s>bonjour|bonsoir|salut|coucou|hello|hey|hi|yo|slt|bjr|bsr|hola|cc)"
_ADRESSE = (r"(?: (?:a )?(?:toi|vous|tous|rontogpt|aura|tout le monde|"
            r"l'ia|ia|mon ami|mec))?")
_NOUVELLES = (r"(?P<n>(?: et)?(?: comment)?(?: ca va(?: bien)?| tu vas bien|"
              r" vas tu(?: bien)?| allez vous(?: bien)?| quoi de neuf|"
              r" la forme))")
_RE_SALUT = re.compile(rf"^{_SALUT}{_ADRESSE}{_NOUVELLES}?{_ADRESSE}$")
_RE_NOUVELLES = re.compile(
    r"^(?:comment ca va|comment vas tu|comment allez vous|tu vas bien|"
    r"quoi de neuf)$")

_MIROIR = {"bonjour": "Bonjour", "bonsoir": "Bonsoir", "bjr": "Bonjour",
           "bsr": "Bonsoir", "hello": "Hello", "hi": "Hello", "hey": "Hey",
           "hola": "Hola"}

# ── heure / date ───────────────────────────────────────────────────────────
_RE_HEURE = re.compile(
    r"^(?:quelle heure est il|quelle heure il est|il est quelle heure|"
    r"quelle heure est ce|quelle heure|l'heure qu'il est|l'heure|"
    r"donne moi l'heure|quelle est l'heure)$")
_RE_DATE = re.compile(
    r"^(?:quel jour (?:sommes nous|est on|on est|est il|est ce)|"
    r"on est quel jour|nous sommes quel jour|quelle est la date"
    r"(?: d'aujourd'hui| du jour| actuelle)?|la date(?: d'aujourd'hui| du jour)?|"
    r"quelle date (?:sommes nous|est on|on est)|quel est le jour|"
    r"le jour d'aujourd'hui|on est le combien|nous sommes le combien|"
    r"quel jour de la semaine (?:sommes nous|est on|on est)|"
    r"quel jour on est)$")
_RE_ANNEE = re.compile(
    r"^(?:quelle annee (?:sommes nous|est on|on est)|"
    r"on est en quelle annee|nous sommes en quelle annee|"
    r"en quelle annee (?:sommes nous|est on|on est))$")


def _salutation(q: str) -> str | None:
    m = _RE_SALUT.match(q)
    if m is None and not _RE_NOUVELLES.match(q):
        return None
    vous = "vous" in q.split()
    aide = "Comment puis-je vous aider ?" if vous else "Comment puis-je t'aider ?"
    if m is None:                       # « comment ca va ? » seul
        return ("Ça va bien, merci ! " + aide)
    debut = _MIROIR.get(m.group("s"), "Salut")
    if m.group("n"):
        return f"{debut} ! Ça va bien, merci. {aide}"
    return f"{debut} ! {aide}"


def _date_fr(d: _dt.date) -> str:
    jour = "1er" if d.day == 1 else str(d.day)
    return f"{_JOURS[d.weekday()]} {jour} {_MOIS[d.month - 1]} {d.year}"


def repondre(question: str, maintenant: _dt.datetime | None = None) -> dict | None:
    """{"reponse", "genre"} pour une salutation / l'heure / la date, sinon None."""
    if not _actif():
        return None
    brut = (question or "").strip()
    if not brut or len(brut) > 60:
        return None
    q = _QUEUE.sub("", _POLI.sub("", _norm(brut)))
    if not q:
        return None
    salut = _salutation(q)
    if salut is not None:
        return {"reponse": salut, "genre": "salutation"}
    if _RE_HEURE.match(q) or _RE_DATE.match(q) or _RE_ANNEE.match(q):
        maintenant = maintenant or _dt.datetime.now().astimezone()
        if _RE_HEURE.match(q):
            return {"reponse": (f"Il est {maintenant.hour} h "
                                f"{maintenant.minute:02d} (heure de ton "
                                f"ordinateur)."), "genre": "heure"}
        if _RE_ANNEE.match(q):
            return {"reponse": f"Nous sommes en {maintenant.year}.",
                    "genre": "date"}
        return {"reponse": f"Nous sommes le {_date_fr(maintenant.date())}.",
                "genre": "date"}
    return None
