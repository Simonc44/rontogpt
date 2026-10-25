"""Metadonnees GGUF : lire l'ARCHITECTURE REELLE d'un cerveau, sans le charger.

Le probleme resolu
------------------
`adaptateurs.py` portait deja l'avertissement : un adaptateur LoRA est un
DELTA DE POIDS calcule pour une architecture precise ; monte sur une autre
base il produit du bruit SANS erreur visible. Le controle de ce risque
comparait… des CHAINES DE CARACTERES dans le nom du fichier
(`"llama-3.2-1b" in nom`). Un fichier renomme passait donc le controle, et
un cerveau du bon type range sous un autre nom le ratait.

Les nombres qui comptent (nombre de couches, dimension, tetes, vocabulaire)
sont DANS le fichier : l'en-tete GGUF est un tableau cle/valeur que ce
module lit directement (quelques dizaines de Ko lus, aucun modele charge,
aucune dependance).

Ce que ca change, concretement, pour la migration Llama -> Gemma
---------------------------------------------------------------
    Llama-3.2-1B : 16 couches, dim 2048, 32 tetes / 8 KV, vocab 128256
    Gemma-3-1B   : 26 couches, dim 1152,  4 tetes / 1 KV, ctx   32768

Les deux nombres ne se ressemblent pas : le verdict « incompatible » devient
un FAIT mesure sur le fichier au lieu d'une convention de nommage.
"""
from __future__ import annotations

import struct
from pathlib import Path

# Types de valeurs GGUF (spec ggml, gguf.h).
_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

#: Champs d'architecture, communs a toutes les familles (le prefixe de la
#: famille est ajoute automatiquement : `gemma3.block_count`, etc.).
_CLES = (
    "block_count", "embedding_length", "feed_forward_length",
    "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "context_length", "vocab_size",
)

#: Cache : {chemin: meta}. L'en-tete d'un GGUF ne change pas sous nos pieds.
_CACHE: dict[str, dict] = {}


def lire(chemin: str | Path, tout: bool = False) -> dict:
    """Toutes les metadonnees du GGUF, dans l'ordre du fichier.

    `tout=False` s'arrete a l'architecture : c'est ce dont on a besoin, et
    ca evite de decoder un `chat_template` de plusieurs Ko a chaque appel.
    """
    cle = str(chemin)
    if not tout and cle in _CACHE:
        return _CACHE[cle]
    meta: dict = {}
    with open(chemin, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{chemin} n'est pas un GGUF (magic absent)")
        meta["_version"] = struct.unpack("<I", f.read(4))[0]
        meta["_nb_tenseurs"], n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]
            nom = f.read(n).decode("utf-8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            meta[nom] = _valeur(f, t, tout)
    if not tout:
        _CACHE[cle] = meta
    return meta


def lire_selection(chemin: str | Path, prefixes: tuple[str, ...]) -> dict:
    """Metadonnees dont la cle commence par l'un des `prefixes`.

    Les autres valeurs sont SAUTEES sans etre decodees (tableaux compris) :
    detecter une table Engram (`<famille>.ple.*`) ne doit pas charger le
    vocabulaire de 200 000 tokens ni le chat_template. Contrairement a
    `lire()`, les tableaux RETENUS sont rendus element par element, pas sous
    forme de `<N elements>`.
    """
    meta: dict = {}
    with open(chemin, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{chemin} n'est pas un GGUF (magic absent)")
        f.read(4)                                    # version
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]
            nom = f.read(n).decode("utf-8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            if nom.startswith(prefixes):
                meta[nom] = _valeur(f, t, True)
            else:
                _sauter(f, t)                        # consomme sans decoder
    return meta


def _sauter(f, t: int) -> None:
    """Avance au-dela d'une valeur GGUF SANS la decoder.

    `_valeur(..., profond=False)` saute les tableaux de types fixes, mais
    pas les tableaux de CHAINES (`tokenizer.ggml.tokens`, ~200 000 entrees)
    qu'il decode une par une. Les ignorer vraiment demande de lire chaque
    longueur et de se deplacer : c'est ce que fait cette fonction.
    """
    if t == 8:                                       # chaine
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if t == 9:                                       # tableau
        type_elem = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if type_elem == 8:                           # tableau de chaines
            for _ in range(n):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            return
        taille = _TAILLE.get(type_elem)
        if taille is None:
            raise ValueError(f"type GGUF de tableau inconnu : {type_elem}")
        f.seek(taille * n, 1)
        return
    fmt = _FMT.get(t)
    if fmt is None:
        raise ValueError(f"type GGUF inconnu : {t}")
    f.seek(struct.calcsize(fmt), 1)


def _valeur(f, t: int, profond: bool):
    """Une valeur GGUF. `profond=False` saute le CONTENU des tableaux longs
    (vocabulaire, template) : on garde leur NOMBRE, pas leurs elements."""
    if t == 8:                                   # chaine
        n = struct.unpack("<Q", f.read(8))[0]
        if not profond and n > 4096:             # chat_template geant
            f.seek(n, 1)
            return f"<{n} octets>"
        return f.read(n).decode("utf-8", "replace")
    if t == 9:                                   # tableau
        type_elem = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if not profond:
            taille = _TAILLE.get(type_elem)
            if taille is not None:
                f.seek(taille * n, 1)
                return f"<{n} elements>"
        return [_valeur(f, type_elem, profond) for _ in range(n)]
    fmt = _FMT.get(t)
    if fmt is None:
        raise ValueError(f"type GGUF inconnu : {t}")
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


_TAILLE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def architecture(chemin: str | Path) -> dict:
    """Les seules cles qui decident si un adaptateur LoRA peut se monter.

    Renvoie un dict vide (jamais une exception) si le fichier est absent ou
    illisible : l'appelant teste `if not arc:` au lieu d'attraper large.
    """
    try:
        meta = lire(chemin)
    except Exception:
        return {}
    famille = str(meta.get("general.architecture") or "")
    if not famille:
        return {}
    arc: dict = {"famille": famille}
    for suffixe in _CLES:
        v = meta.get(f"{famille}.{suffixe}", meta.get(suffixe))
        if v is not None and not isinstance(v, str):
            arc[suffixe.split(".")[-1]] = int(v)
    arc["_nom"] = str(meta.get("general.name") or "")
    return arc


def famille(chemin: str | Path) -> str:
    """'gemma3', 'llama', ... — ou '' si le fichier n'est pas lisible."""
    return str(architecture(chemin).get("famille") or "")


def compatible(chemin: str | Path, cible: dict) -> tuple[bool, str]:
    """L'adaptateur calcule pour `cible` peut-il se monter sur `chemin` ?

    `cible` est un dict du meme format que `architecture()` (cf.
    `adaptateurs.ARCHITECTURE_CIBLE`). On compare les nombres, pas les noms.
    """
    arc = architecture(chemin)
    if not arc:
        return False, "architecture illisible (fichier absent ou tronque)"
    ecarts = []
    for cle in ("famille", "block_count", "embedding_length",
                "head_count", "head_count_kv"):
        attendu = cible.get(cle)
        if attendu is None:
            continue
        vu = arc.get(cle)
        if vu != attendu:
            ecarts.append(f"{cle} : attendu {attendu}, lu {vu}")
    if ecarts:
        return False, "; ".join(ecarts)
    return True, ""


def resume(chemin: str | Path) -> str:
    """Une ligne lisible, pour les journaux et les diagnostics."""
    arc = architecture(chemin)
    if not arc:
        return f"{Path(str(chemin)).name} : architecture illisible"
    return (f"{Path(str(chemin)).name} : {arc['famille']} "
            f"{arc.get('block_count', '?')} couches, "
            f"dim {arc.get('embedding_length', '?')}, "
            f"{arc.get('head_count', '?')}/{arc.get('head_count_kv', '?')} tetes, "
            f"ctx {arc.get('context_length', '?')}")
