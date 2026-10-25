"""Profils d'echantillonnage : UNE source pour les DEUX chemins d'inference.

Defaut reel
-----------
RontoGPT genere par deux chemins : le SERVEUR llama.cpp (HTTP, actif par defaut)
et le chemin IN-PROCESS (llama-cpp-python, repli). Les deux ecrivaient leurs
parametres SEPAREMENT, et ils avaient deja diverge :

  * le serveur n'envoyait AUCUNE penalite alors que l'in-process envoyait
    repeat_penalty 1,1 -> le chemin ACTIF etait le moins protege ;
  * le disjoncteur anti-boucle (relance en penalite 1,25)
    n'existait QUE cote in-process.

Un parametre ecrit deux fois finit toujours par diverger : il n'est donc
ecrit qu'ICI. `pour_serveur` et `pour_inprocess` ne different que par ce que
chaque backend accepte — jamais par les valeurs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Profil:
    temperature: float
    top_p: float
    min_p: float
    repeat_penalty: float
    #: le sampler DRY de llama.cpp (penalise les SEQUENCES repetitives longues)
    #: n'existe que cote serveur : le marquer ici rend la difference EXPLICITE
    #: au lieu de la laisser dans un `if` d'un seul des deux chemins.
    dry: bool = False


#: Temperature FIXE a 1.0 sur les DEUX profils : elle ne varie plus par usage
#: ni apres une boucle. Seule la penalite de repetition change (la relance
#: reste plus punitive que la generation normale).
NORMAL = Profil(1.0, 0.9, 0.05, 1.1, dry=True)
#: Relance apres detection d'une boucle : meme temperature, penalite forte.
RELANCE = Profil(1.0, 0.9, 0.05, 1.25)

#: Balises d'arret GENERIQUES (toujours valables) : la fin de tour de la
#: famille chargee est ajoutee par `stops()` — cf. llama_cerveau
#: ._STOPS_FAMILLE. Avant, `<|eot_id|>` (Llama) etait ecrit ici EN DUR : le
#: jour ou le cerveau change de famille, on envoyait au modele une balise
#: qui n'existe pas dans son vocabulaire, sans que rien ne le signale.
STOPS: tuple[str, ...] = ("\nUtilisateur:", "\nUser:")


def stops() -> list[str]:
    """Balises d'arret pour le cerveau REELLEMENT charge.

    Import paresseux : llama_cerveau importe ce module, donc cet appel ne
    peut pas etre au niveau module (cycle).
    """
    balises = list(STOPS)
    try:
        from . import llama_cerveau
        balises = list(llama_cerveau.stops_famille()) + balises
    except Exception:            # noqa: BLE001 — jamais bloquant
        pass
    return balises

_DRY = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2}


def pour_serveur(p: Profil, sans_dry: bool | None = None) -> dict:
    """Corps de requete /v1/chat/completions pour le profil `p`.

    `sans_dry` par defaut suit AURA_SANS_DRY=1 (retour au comportement
    d'origine du serveur, sans le sampler DRY).
    """
    if sans_dry is None:
        sans_dry = os.environ.get("AURA_SANS_DRY", "") == "1"
    corps: dict = {"temperature": p.temperature, "top_p": p.top_p,
                   "min_p": p.min_p, "repeat_penalty": p.repeat_penalty,
                   "stop": stops()}
    if p.dry and not sans_dry:
        corps.update(_DRY)
    return corps


def pour_inprocess(p: Profil) -> dict:
    """Kwargs de llama_cpp.create_chat_completion pour le profil `p`."""
    return {"temperature": p.temperature, "top_p": p.top_p,
            "min_p": p.min_p, "repeat_penalty": p.repeat_penalty,
            "stop": stops()}
