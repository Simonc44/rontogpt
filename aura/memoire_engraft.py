"""ENGRAFT — couche overlay `.pleo` (n-grammes), SANS injection dans le modele.

Ce que fait ENGRAFT
-------------------
ENGRAFT (fulvian/engraft-ngram, 2026) ecrit des faits dans la **table
n-grammes (Engram / PLE)** d'un modele qui en possede une : Qwen4Exp
(Qwen3.8-Flash-Next), DeepSeek V4.1 Flash a terme. A chaque position, les
deux et trois derniers tokens sont haches en 16 lignes de table, lues puis
ajoutees au flux residuel au debut du reseau. Un fichier `.pleo` ne contient
que des **vecteurs de remplacement pour des lignes globales** ; le fork
llama.cpp les substitue au moment du `gather`. Aucun poids n'est modifie,
retirer l'overlay restaure le modele a l'identique.

Ce que fait CE module
---------------------
- lit et ecrit le format `.pleo` (magic `PLEO`, `uint32 n`, `uint32 dim=160`,
  `n` index `int32`, puis `n * 160` valeurs `float32` little-endian) ;
- reproduit l'**adressage** de `llm_graph_input_ple::set_input` : hash mixte
  sur `[x_t, x_{t-1}, x_{t-2}]`, multiplicateurs du modele, un modulo par
  tete, EOS qui coupe le contexte vers l'arriere ;
- indexe les lignes d'un overlay et dit, pour une suite de tokens, quelles
  positions lisent une ligne remplacee (`positions_touchees`,
  `lignes_lues`) — utile pour verifier qu'un overlay « tire » sans charger
  le modele ;
- detecte la **capacite** d'un GGUF (`capacite_engram`) et **refuse** de
  pretendre appliquer un overlay sur le moteur de ce projet
  (`verifier_moteur`).

Ce que ce module ne fait PAS, et pourquoi
-----------------------------------------
Il n'injecte aucun fait dans Gemma-3-1B. Ce modele n'a pas de table Engram :
il n'y a aucune ligne a remplacer, et `llama-cpp-python` (0.3.36) n'expose
aucun hook de gather PLE. Tout « hook d'activation » dans `llama_cerveau.py`
serait une simulation (contexte ou biais de logits), donc un cout en tokens
et une revendication « 0 token » fausse. Le jour ou un modele Engram et le
fork `engine/` d'ENGRAFT tournent ici, `capacite_engram` le detectera et
`verifier_moteur` devra etre etendu ; en attendant il rend False avec la
raison exacte.

Sources du format et de l'adressage : `docs/lens.md`, `docs/mechanism.md` et
`engraft/table.py` du depot fulvian/engraft-ngram.
"""
from __future__ import annotations

import dataclasses
import os
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from . import gguf_meta

#: Une ligne de table est un vecteur de 160 valeurs float32.
ROW_LEN = 160
#: Magic du format d'overlay (4 octets, ASCII).
PLEO_MAGIC = b"PLEO"
#: magic + uint32 n + uint32 dim.
_ENTETE = 4 + 4 + 4
#: Masque du hash 64 bits du fork.
_MASQUE64 = (1 << 64) - 1
#: Prefixe des metadonnees de la table dans le GGUF (sous la famille).
_SUFFIXE_PLE = ".ple."


# ── CONFIGURATION DE LA TABLE (lue dans le GGUF du modele) ──────────────

@dataclasses.dataclass(frozen=True)
class ConfigEngram:
    """Parametres d'adressage que le fork lit dans le GGUF du modele.

    `heads_per_ngram` tetes pour le bigramme, autant pour le trigramme (et
    au-dela) : `n_heads == (ngram_size - 1) * heads_per_ngram`. Les
    `head_offsets` donnent la position de chaque tete dans l'espace GLOBAL
    des lignes (indice local + offset de la tete).
    """

    eos_token_id: int
    ngram_size: int
    heads_per_ngram: int
    layer_multipliers: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]
    image_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.ngram_size < 2:
            raise ValueError("ngram_size doit valoir au moins 2")
        if self.heads_per_ngram < 1:
            raise ValueError("heads_per_ngram doit etre positif")
        if self.eos_token_id < 0:
            raise ValueError("eos_token_id negatif")
        if len(self.layer_multipliers) < self.ngram_size:
            raise ValueError(
                "layer_multipliers plus court que ngram_size "
                f"({len(self.layer_multipliers)} < {self.ngram_size})")
        if any(m < 0 for m in self.layer_multipliers):
            raise ValueError("layer_multipliers negatif")
        if len(self.head_vocab_sizes) != len(self.head_offsets):
            raise ValueError(
                "head_vocab_sizes et head_offsets de longueurs differentes "
                f"({len(self.head_vocab_sizes)} != {len(self.head_offsets)})")
        attendu = (self.ngram_size - 1) * self.heads_per_ngram
        if len(self.head_vocab_sizes) != attendu:
            raise ValueError(
                f"{len(self.head_vocab_sizes)} tetes pour ngram_size="
                f"{self.ngram_size} et heads_per_ngram={self.heads_per_ngram} "
                f"(attendu {attendu})")
        if any(v <= 0 for v in self.head_vocab_sizes):
            raise ValueError("head_vocab_sizes doit etre strictement positif")
        if any(o < 0 for o in self.head_offsets):
            raise ValueError("head_offsets negatif")
        for i in range(len(self.head_offsets) - 1):
            fin = self.head_offsets[i] + self.head_vocab_sizes[i]
            if fin > self.head_offsets[i + 1]:
                raise ValueError(
                    f"les lignes des tetes {i} et {i + 1} se chevauchent")

    @property
    def n_heads(self) -> int:
        return len(self.head_vocab_sizes)

    @property
    def n_lignes_par_position(self) -> int:
        """Nombre de lignes lues a CHAQUE position (16 sur les modeles cibles)."""
        return (self.ngram_size - 1) * self.heads_per_ngram


# ── OVERLAY `.pleo` : FORMAT, LECTURE, ECRITURE, FUSION ─────────────────

@dataclasses.dataclass(frozen=True, eq=False)
class Overlay:
    """Lignes globales remplacees et leurs vecteurs (float32 [n, 160])."""

    lignes: np.ndarray
    valeurs: np.ndarray
    chemin: str = ""

    def __post_init__(self) -> None:
        if self.lignes.ndim != 1:
            raise ValueError(f"lignes doit etre 1-D, ndim={self.lignes.ndim}")
        attendu = (self.lignes.shape[0], ROW_LEN)
        if self.valeurs.shape != attendu:
            raise ValueError(
                f"valeurs.shape={self.valeurs.shape}, attendu {attendu}")

    @property
    def n(self) -> int:
        return int(self.lignes.shape[0])

    def __len__(self) -> int:
        return self.n


def _valider(lignes: object, valeurs: object) -> tuple[np.ndarray, np.ndarray]:
    """Valide et normalise des lignes globales + vecteurs (format `.pleo`)."""
    lignes = np.asarray(lignes)
    if lignes.ndim != 1:
        raise ValueError(f"lignes doit etre 1-D, ndim={lignes.ndim}")
    if not np.issubdtype(lignes.dtype, np.integer):
        raise TypeError(f"lignes doit etre entiere, recu {lignes.dtype}")
    lignes = lignes.astype(np.int64, copy=False)
    if lignes.size and int(lignes.min()) < 0:
        raise ValueError("indice de ligne global negatif")
    if np.unique(lignes).size != lignes.size:
        raise ValueError("indice de ligne global present deux fois")
    valeurs = np.asarray(valeurs, dtype=np.float32)
    attendu = (lignes.size, ROW_LEN)
    if valeurs.shape != attendu:
        raise ValueError(
            f"valeurs.shape={valeurs.shape}, attendu {attendu}")
    if not np.isfinite(valeurs).all():
        raise ValueError("vecteur non fini (NaN ou infini) dans l'overlay")
    return lignes, valeurs


def lire_overlay(chemin: str | Path) -> Overlay:
    """Lit un `.pleo` et valide TOUT ce que le format permet de verifier.

    La taille du fichier est comparee a celle annoncee par l'en-tete AVANT
    toute allocation : un `n` enorme dans un petit fichier est refuse, pas
    alloue. Les doublons de ligne sont refuses (le fork doit substituer une
    seule valeur par ligne), comme les vecteurs non finis.
    """
    p = Path(chemin)
    taille = p.stat().st_size
    if taille < _ENTETE:
        raise ValueError(
            f"{p} : fichier trop court pour un .pleo ({taille} octets)")
    with p.open("rb") as f:
        magic = f.read(4)
        if magic != PLEO_MAGIC:
            raise ValueError(
                f"{p} : magic {magic!r}, attendu {PLEO_MAGIC!r}")
        n, dim = struct.unpack("<II", f.read(8))
        if dim != ROW_LEN:
            raise ValueError(
                f"{p} : dim={dim}, attendu {ROW_LEN}")
        attendu = _ENTETE + 4 * n + 4 * n * dim
        if taille != attendu:
            raise ValueError(
                f"{p} : taille {taille} octets, attendu {attendu} pour "
                f"n={n}, dim={dim} (fichier tronque ou octets en trop)")
        lignes = np.fromfile(f, dtype="<i4", count=n)
        valeurs = np.fromfile(f, dtype="<f4", count=n * dim).reshape(n, dim)
    lignes, valeurs = _valider(lignes, valeurs)
    return Overlay(lignes=lignes, valeurs=valeurs, chemin=str(p))


def ecrire_overlay(chemin: str | Path, lignes: object, valeurs: object,
                   *, trier: bool = False) -> Overlay:
    """Ecrit un `.pleo` (atomiquement) et renvoie l'overlay relu en memoire.

    `trier=False` conserve l'ordre de l'appelant, comme `write_pleo` du
    depot d'origine. Le fichier n'est remplace qu'une fois complet : un
    overlay interrompu ne laisse pas un fichier a moitie ecrit.
    """
    lignes, valeurs = _valider(lignes, valeurs)
    if trier and lignes.size:
        ordre = np.argsort(lignes, kind="stable")
        lignes = lignes[ordre]
        valeurs = valeurs[ordre]
    p = Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporaire = p.with_name(p.name + ".tmp")
    n = int(lignes.shape[0])
    try:
        with temporaire.open("wb") as f:
            f.write(PLEO_MAGIC)
            f.write(struct.pack("<II", n, ROW_LEN))
            f.write(lignes.astype("<i4", copy=False).tobytes())
            f.write(valeurs.astype("<f4", copy=False).tobytes())
        os.replace(temporaire, p)
    finally:
        if temporaire.exists():
            temporaire.unlink()
    return Overlay(lignes=lignes, valeurs=valeurs, chemin=str(p))


def fusionner(overlays: Iterable[Overlay]) -> Overlay:
    """Fusionne des overlays par ligne globale, en dedupliquant.

    Deux overlays qui donnent des vecteurs IDENTIQUES pour la meme ligne
    sont acceptes (c'est le cas d'un meme fait ecrit deux fois). Deux
    vecteurs DIFFERENTS pour la meme ligne sont une erreur : le fork ne
    saurait pas lequel appliquer, et la resoudre en silence masquerait un
    bug de construction.
    """
    vus: dict[int, np.ndarray] = {}
    for overlay in overlays:
        for ligne, vecteur in zip(overlay.lignes.tolist(), overlay.valeurs):
            if ligne in vus:
                if not np.array_equal(vus[ligne], vecteur):
                    raise ValueError(
                        f"ligne globale {ligne} : deux vecteurs differents "
                        "dans les overlays fusionnes")
            else:
                vus[ligne] = vecteur
    if not vus:
        return Overlay(lignes=np.empty(0, dtype=np.int64),
                       valeurs=np.empty((0, ROW_LEN), dtype=np.float32))
    lignes = np.array(sorted(vus), dtype=np.int64)
    valeurs = np.stack([vus[int(i)] for i in lignes]).astype(np.float32)
    return Overlay(lignes=lignes, valeurs=valeurs)


# ── ADRESSAGE DES N-GRAMMES (miroir de llm_graph_input_ple::set_input) ──

def _tokens_valides(tokens: Iterable[int]) -> list[int]:
    out = []
    for token in tokens:
        t = int(token)
        if t < 0:
            raise ValueError(f"token id negatif : {t}")
        out.append(t)
    return out


def adresses_locales(tokens: Sequence[int], config: ConfigEngram) -> np.ndarray:
    """Lignes LOCALES lues a chaque position : int64 [T, n_lignes].

    Miroir de `engraft.table.PleTable.ngram_addresses` : contexte
    `[x_t, x_{t-1}, x_{t-2}]` ; un predecesseur manquant ou un EOS renconte
    en remontant coupe le contexte (positions plus anciennes remplacees par
    l'EOS du modele), mais l'EOS du token COURANT ne coupe pas son propre
    contexte.
    """
    suite = _tokens_valides(tokens)
    n_lignes = config.n_lignes_par_position
    out = np.zeros((len(suite), n_lignes), dtype=np.int64)
    for t, token in enumerate(suite):
        contexte = [token]
        coupe = False
        for s in range(1, config.ngram_size):
            pos = t - s
            precedent = None if (pos < 0 or coupe) else suite[pos]
            if precedent is None or precedent == config.eos_token_id:
                coupe = True
                precedent = config.eos_token_id
            contexte.append(precedent)
        colonne = 0
        for n in range(2, config.ngram_size + 1):
            melange = ((contexte[0] * config.layer_multipliers[0])
                       & _MASQUE64)
            for j in range(1, n):
                melange ^= ((contexte[j] * config.layer_multipliers[j])
                            & _MASQUE64)
            melange &= _MASQUE64
            base = (n - 2) * config.heads_per_ngram
            for g in range(config.heads_per_ngram):
                tete = base + g
                out[t, colonne] = melange % config.head_vocab_sizes[tete]
                colonne += 1
    return out


def adresses_globales(tokens: Sequence[int],
                      config: ConfigEngram) -> np.ndarray:
    """Lignes GLOBALES lues a chaque position : int64 [T, n_lignes]."""
    locales = adresses_locales(tokens, config)
    offsets = np.asarray(config.head_offsets, dtype=np.int64)
    return locales + offsets[None, :]


# ── INDEX D'UN OVERLAY : quelles positions lisent une ligne remplacee ───

@dataclasses.dataclass(frozen=True, eq=False)
class IndexLignes:
    """Lignes d'un overlay triees, avec leur position d'origine."""

    lignes_triees: np.ndarray
    positions: np.ndarray


def indexer(overlay: Overlay) -> IndexLignes:
    """Trie les lignes de l'overlay une fois (reutilisable sur N requetes)."""
    ordre = np.argsort(overlay.lignes, kind="stable")
    return IndexLignes(lignes_triees=overlay.lignes[ordre],
                       positions=ordre.astype(np.int64))


def positions_dans_overlay(overlay: Overlay, lignes_globales: object,
                           index: IndexLignes | None = None) -> np.ndarray:
    """Position dans l'overlay de chaque ligne globale, -1 si absente."""
    demandees = np.asarray(lignes_globales, dtype=np.int64)
    if overlay.n == 0:
        return np.full(demandees.shape, -1, dtype=np.int64)
    idx = index if index is not None else indexer(overlay)
    if idx.lignes_triees.size == 0:
        return np.full(demandees.shape, -1, dtype=np.int64)
    place = np.searchsorted(idx.lignes_triees, demandees)
    borne = np.clip(place, 0, idx.lignes_triees.size - 1)
    trouve = idx.lignes_triees[borne] == demandees
    return np.where(trouve, idx.positions[borne], -1).astype(np.int64)


def lignes_lues(tokens: Sequence[int], config: ConfigEngram, overlay: Overlay,
                index: IndexLignes | None = None) -> np.ndarray:
    """Nombre de lignes remplacees lues a chaque position : int64 [T]."""
    globales = adresses_globales(tokens, config)
    if globales.size == 0:
        return np.zeros(0, dtype=np.int64)
    idx = index if index is not None else indexer(overlay)
    coups = positions_dans_overlay(overlay, globales.reshape(-1), idx) >= 0
    return coups.reshape(globales.shape).sum(axis=1).astype(np.int64)


def positions_touchees(tokens: Sequence[int], config: ConfigEngram,
                       overlay: Overlay,
                       index: IndexLignes | None = None) -> np.ndarray:
    """Bool [T] : la position lit-elle au moins une ligne de l'overlay ?"""
    return lignes_lues(tokens, config, overlay, index) > 0


# ── CAPACITE ENGRAM DU MODELE, ET GARDE-FOU DU MOTEUR ───────────────────

@dataclasses.dataclass(frozen=True)
class Capacite:
    """Le GGUF porte-t-il une table Engram exploitable par un overlay ?"""

    disponible: bool
    raison: str = ""
    famille: str = ""
    config: ConfigEngram | None = None


def _entier(meta: dict, cle: str) -> int:
    valeur = meta[cle]
    if isinstance(valeur, bool) or not isinstance(valeur, (int, np.integer)):
        raise ValueError(f"{cle} : entier attendu, recu {type(valeur).__name__}")
    return int(valeur)


def _entiers(meta: dict, cle: str) -> tuple[int, ...]:
    valeur = meta[cle]
    if isinstance(valeur, (str, bytes)) or not isinstance(
            valeur, (list, tuple, np.ndarray)):
        raise ValueError(f"{cle} : tableau d'entiers attendu")
    return tuple(int(v) for v in valeur)


def capacite_depuis_meta(meta: dict) -> Capacite:
    """Verdict a partir des metadonnees GGUF deja lues (testable sans GGUF).

    La table Engram n'existe que sur les architectures qui la declarent
    (`<famille>.ple.*`). Gemma-3 n'en a pas : c'est un FAIT lu dans les
    metadonnees, pas une liste de noms de modeles.
    """
    famille = str(meta.get("general.architecture") or "")
    if not famille:
        return Capacite(False, "architecture absente des metadonnees GGUF")
    prefixe = f"{famille}{_SUFFIXE_PLE}"
    requises = ("eos_token_id", "ngram_size", "heads_per_ngram",
                "layer_multipliers", "head_vocab_sizes", "head_offsets")
    presentes = {cle: f"{prefixe}{cle}" in meta for cle in requises}
    if not any(presentes.values()):
        return Capacite(
            False,
            f"architecture {famille} : aucune table Engram "
            f"(cles {prefixe}* absentes)", famille)
    manquantes = [f"{prefixe}{cle}" for cle, ok in presentes.items() if not ok]
    if manquantes:
        return Capacite(
            False, f"table Engram incomplete : {', '.join(manquantes)}",
            famille)
    try:
        config = ConfigEngram(
            eos_token_id=_entier(meta, f"{prefixe}eos_token_id"),
            ngram_size=_entier(meta, f"{prefixe}ngram_size"),
            heads_per_ngram=_entier(meta, f"{prefixe}heads_per_ngram"),
            layer_multipliers=_entiers(meta, f"{prefixe}layer_multipliers"),
            head_vocab_sizes=_entiers(meta, f"{prefixe}head_vocab_sizes"),
            head_offsets=_entiers(meta, f"{prefixe}head_offsets"),
            image_token_id=(_entier(meta, f"{prefixe}image_token_id")
                            if f"{prefixe}image_token_id" in meta else None),
        )
    except (KeyError, TypeError, ValueError) as e:
        return Capacite(
            False, f"metadonnees Engram incoherentes : {e}", famille)
    return Capacite(True, "", famille, config)


def capacite_engram(chemin: str | Path | None = None) -> Capacite:
    """Lit les metadonnees du cerveau configure et rend le verdict Engram.

    Aucun modele n'est charge : seuls les quelques Ko d'en-tete GGUF sont
    lus, et seulement les cles `<famille>.ple.*` (le vocabulaire de 200 000
    tokens n'est jamais decode).
    """
    if chemin is None:
        from . import llama_cerveau
        chemin = llama_cerveau.chemin_gguf()
    p = str(chemin)
    if not os.path.isfile(p):
        return Capacite(False, f"GGUF introuvable : {p}")
    try:
        meta = gguf_meta.lire(p)
    except Exception as e:                       # noqa: BLE001 — verdict, pas panne
        return Capacite(False, f"metadonnees GGUF illisibles ({p}) : {e}")
    famille = str(meta.get("general.architecture") or "")
    if not famille:
        return Capacite(False, f"architecture absente de {p}")
    try:
        precis = gguf_meta.lire_selection(
            p, ("general.architecture", f"{famille}{_SUFFIXE_PLE}"))
    except Exception as e:                       # noqa: BLE001
        return Capacite(False, f"metadonnees GGUF illisibles ({p}) : {e}")
    return capacite_depuis_meta(precis)


def verifier_moteur(chemin: str | Path | None = None) -> tuple[bool, str]:
    """Le moteur d'inference de CE projet peut-il appliquer un `.pleo` ?

    Non, et le module le dit au lieu de simuler : `llama-cpp-python` ne
    expose pas le gather PLE. Meme sur un GGUF Engram, il faudrait le fork
    `engine/` d'ENGRAFT. Le jour ou ce fork est branche, c'est ICI que la
    detection doit etre etendue (et pas dans un hook Python de
    `llama_cerveau.py`, qui ne verrait jamais les activations).
    """
    capacite = capacite_engram(chemin)
    if not capacite.disponible:
        return False, capacite.raison
    return False, (
        "table Engram detectee, mais le moteur de ce projet "
        "(llama-cpp-python) n'applique pas les overlays .pleo : le gather du "
        "fork ENGRAFT (engine/) est requis")
