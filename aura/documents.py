"""RAG de documents locaux : importer, chercher, citer.

Le besoin n°1 apres le chat est de pouvoir parler a SES documents sans les
envoyer dans le cloud. Ce module indexe des fichiers PDF / Markdown / texte,
les decoupe en passages, les encode avec MiniLM (aura/embeddings.py) et
retrouve les passages pertinents pour une question.

Garanties :
- **local** : tout reste dans `.cache_docs/` (ou `AURA_DOCS_DIR`) ;
- **sans dependance dure** : si MiniLM est absent, la recherche retombe sur
  un recouvrement lexical (aura/rag_binaire.tokeniser), jamais sur une erreur ;
- **incremental** : re-ajouter un fichier remplace ses anciens passages ;
- **citable** : chaque passage porte le nom du fichier d'origine, et
  `interroger()` rend la liste des sources a afficher.

Usage CLI :
    python -m aura.documents add notes.md rapport.pdf
    python -m aura.documents add-dossier ~/Documents/mes-notes
    python -m aura.documents list
    python -m aura.documents search "que dit le rapport sur le budget ?"
    python -m aura.documents rm rapport.pdf
    python -m aura.documents stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

from . import embeddings

LOG = logging.getLogger("aura.documents")

RACINE = Path(__file__).resolve().parent.parent
#: Dossier d'index (surchargeable pour les tests et les profils multiples).
DOSSIER = Path(os.environ.get("AURA_DOCS_DIR") or (RACINE / ".cache_docs"))

EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
TAILLE_MORCEAU = 700          # caracteres par passage (~180 tokens)
CHEVAUCHEMENT = 120
SEUIL_COSINUS = 0.28
SEUIL_LEXICAL = 0.10
MAX_OCTETS = 20 * 1024 * 1024
MAX_CAR_CONTEXTE = 1800

_verrou = threading.Lock()
_cache_docs: list[dict] | None = None
_cache_vecs: np.ndarray | None = None
_cache_sig: tuple | None = None


# -- chemins ---------------------------------------------------------------

def _index_path() -> Path:
    return DOSSIER / "index.jsonl"


def _vecteurs_path() -> Path:
    return DOSSIER / "vecteurs.npy"


def _vider_cache() -> None:
    """Les tests changent DOSSIER : l'ancien index ne doit pas resservir.

    SANS verrou : il est appele depuis `_ecrire`, qui tourne deja sous
    `_verrou` (ajouter/supprimer). Un Lock non reentrant y provoquait un
    interblocage silencieux.
    """
    global _cache_docs, _cache_vecs, _cache_sig
    _cache_docs, _cache_vecs, _cache_sig = None, None, None


# -- lecture des fichiers --------------------------------------------------

def _lire_pdf(chemin: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:            # pragma: no cover - dependance declaree
        raise RuntimeError(
            "Lecture PDF indisponible : installer pypdf "
            "(`uv pip install pypdf`).") from e
    try:
        lecteur = PdfReader(str(chemin))
        if lecteur.is_encrypted:
            raise RuntimeError(f"{chemin.name} : PDF chiffre, illisible")
        pages = [(page.extract_text() or "") for page in lecteur.pages]
    except RuntimeError:
        raise
    except Exception as e:              # noqa: BLE001 - message utilisateur
        raise RuntimeError(f"{chemin.name} : PDF illisible ({e})") from e
    return "\n\n".join(p for p in pages if p.strip())


def lire(chemin: str | Path) -> str:
    """Texte brut d'un fichier supporte (txt / md / pdf)."""
    chemin = Path(chemin).expanduser()
    if not chemin.is_file():
        raise FileNotFoundError(f"fichier introuvable : {chemin}")
    if chemin.stat().st_size > MAX_OCTETS:
        raise ValueError(
            f"{chemin.name} : trop volumineux "
            f"({chemin.stat().st_size / 1e6:.1f} Mo > {MAX_OCTETS / 1e6:.0f} Mo)")
    suffixe = chemin.suffix.lower()
    if suffixe not in EXTENSIONS:
        raise ValueError(
            f"{chemin.name} : format non supporte "
            f"(attendu : {', '.join(sorted(EXTENSIONS))})")
    if suffixe == ".pdf":
        return _lire_pdf(chemin)
    return chemin.read_text(encoding="utf-8", errors="replace")


# -- decoupage -------------------------------------------------------------

def _unites(texte: str) -> list[str]:
    """Paragraphes, ou phrases pour les paragraphes trop longs."""
    unites: list[str] = []
    for paragraphe in re.split(r"\n\s*\n", texte):
        # Les titres Markdown (`#`, `##`) ne sont pas du contenu : les garder
        # les fait recopier tels quels dans la reponse et pollue l'embedding.
        # On garde le TEXTE du titre, termine par un point, pour que « Projet
        # Atlas » ne colle pas a la phrase suivante dans un extrait cite.
        paragraphe = re.sub(r"(?m)^#{1,6}\s*(.+?)\s*$", r"\1.", paragraphe)
        paragraphe = re.sub(r"[ \t]+", " ", paragraphe).strip()
        if not paragraphe:
            continue
        if len(paragraphe) <= TAILLE_MORCEAU:
            unites.append(paragraphe)
            continue
        for phrase in re.split(r"(?<=[.!?])\s+", paragraphe):
            phrase = phrase.strip()
            if phrase:
                unites.append(phrase)
    return unites


def _queue(texte: str, longueur: int) -> str:
    """Fin de passage propre (coupee sur un mot) pour le chevauchement."""
    if longueur <= 0 or not texte:
        return ""
    fin = texte[-longueur:]
    if " " in fin:
        fin = fin.split(" ", 1)[1]
    return fin.strip()


def decouper(texte: str, taille: int = TAILLE_MORCEAU,
             chevauchement: int = CHEVAUCHEMENT) -> list[str]:
    """Decoupe en passages d'environ `taille` caracteres, avec recouvrement."""
    if not texte or not texte.strip():
        return []
    morceaux: list[str] = []
    courant = ""
    for unite in _unites(texte):
        # unite plus longue que la cible : tranchage brut, borné
        while len(unite) > taille:
            if courant:
                morceaux.append(courant)
                courant = ""
            morceaux.append(unite[:taille])
            unite = _queue(unite, taille - chevauchement)
        if not unite:
            continue
        if not courant:
            courant = unite
        elif len(courant) + 1 + len(unite) <= taille:
            courant = f"{courant} {unite}"
        else:
            morceaux.append(courant)
            courant = f"{_queue(courant, chevauchement)} {unite}".strip()
    if courant.strip():
        morceaux.append(courant.strip())
    # dedup en gardant l'ordre (un chevauchement peut produire un doublon)
    vus: set[str] = set()
    propres: list[str] = []
    for morceau in morceaux:
        if morceau and morceau not in vus:
            vus.add(morceau)
            propres.append(morceau)
    return propres


# -- index -----------------------------------------------------------------

def _empreinte(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8", "replace")).hexdigest()[:16]


def _charger() -> tuple[list[dict], np.ndarray | None]:
    """Index en cache, invalide par mtime (index OU vecteurs)."""
    global _cache_docs, _cache_vecs, _cache_sig
    index = _index_path()
    vecteurs = _vecteurs_path()
    if not index.is_file():
        return [], None
    sig = (str(index), index.stat().st_mtime_ns,
           vecteurs.stat().st_mtime_ns if vecteurs.is_file() else 0)
    if _cache_sig == sig and _cache_docs is not None:
        return _cache_docs, _cache_vecs
    docs: list[dict] = []
    try:
        with index.open(encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    docs.append(json.loads(ligne))
                except json.JSONDecodeError:
                    LOG.warning("[documents] ligne d'index illisible ignoree")
    except OSError as e:
        LOG.warning("[documents] index illisible : %s", e)
        return [], None
    vecs = None
    if vecteurs.is_file():
        try:
            vecs = np.load(vecteurs)
            if len(vecs) != len(docs):
                LOG.warning("[documents] vecteurs desynchronises -> lexical")
                vecs = None
        except Exception as e:              # noqa: BLE001
            LOG.warning("[documents] vecteurs illisibles (%s) -> lexical", e)
    _cache_docs, _cache_vecs, _cache_sig = docs, vecs, sig
    return docs, vecs


def _ecrire(docs: list[dict], vecs: np.ndarray | None) -> None:
    DOSSIER.mkdir(parents=True, exist_ok=True)
    index = _index_path()
    tmp = index.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp, index)
    vecteurs = _vecteurs_path()
    if vecs is not None and len(vecs) == len(docs):
        tmpv = vecteurs.with_suffix(".npy.tmp")
        with tmpv.open("wb") as f:
            np.save(f, vecs.astype(np.float32))
        os.replace(tmpv, vecteurs)
    elif vecteurs.is_file():
        vecteurs.unlink()
    _vider_cache()


def ajouter(chemin: str | Path) -> dict:
    """Indexe un fichier (ou le remplace s'il etait deja present).

    Rend un bilan : ``{"nom", "morceaux", "caracteres"}``.
    """
    chemin = Path(chemin).expanduser()
    texte = lire(chemin)
    morceaux = decouper(texte)
    if not morceaux:
        raise ValueError(f"{chemin.name} : aucun texte exploitable")
    source = str(chemin.resolve())
    with _verrou:
        anciens, anciens_vecs = _charger()
        # On retire l'ancienne version du fichier, vecteurs compris :
        # re-ajouter un document ne doit jamais melanger deux versions.
        gardes, gardes_vecs = [], []
        for i, doc in enumerate(anciens):
            if doc.get("source") == source:
                continue
            gardes.append(doc)
            if anciens_vecs is not None and i < len(anciens_vecs):
                gardes_vecs.append(anciens_vecs[i])
        mtime = chemin.stat().st_mtime
        nouveaux_docs = [
            {"source": source, "nom": chemin.name, "morceau": i,
             "texte": m, "sha": _empreinte(m), "mtime": mtime}
            for i, m in enumerate(morceaux)]
        docs = gardes + nouveaux_docs
        vecs = None
        nouveaux_vecs = embeddings.encoder(morceaux)
        if nouveaux_vecs is not None:
            if anciens_vecs is not None and len(gardes_vecs) == len(gardes):
                vecs = (np.vstack([np.asarray(gardes_vecs, dtype=np.float32),
                                   nouveaux_vecs]).astype(np.float32)
                        if gardes_vecs else nouveaux_vecs.astype(np.float32))
            elif not anciens:
                vecs = nouveaux_vecs.astype(np.float32)
            else:
                # Index existant sans vecteurs (MiniLM etait absent) : on
                # encode TOUT, sinon la recherche serait incoherente.
                vecs = embeddings.encoder([d["texte"] for d in docs])
        _ecrire(docs, vecs)
    LOG.info("[documents] %s : %d passage(s) indexes", chemin.name, len(morceaux))
    return {"nom": chemin.name, "morceaux": len(morceaux),
            "caracteres": len(texte)}


def ajouter_dossier(dossier: str | Path, recursif: bool = True) -> list[dict]:
    dossier = Path(dossier).expanduser()
    if not dossier.is_dir():
        raise NotADirectoryError(f"dossier introuvable : {dossier}")
    motif = "**/*" if recursif else "*"
    bilans = []
    for fichier in sorted(dossier.glob(motif)):
        if fichier.is_file() and fichier.suffix.lower() in EXTENSIONS:
            try:
                bilans.append(ajouter(fichier))
            except (ValueError, RuntimeError, OSError) as e:
                LOG.warning("[documents] %s ignore : %s", fichier.name, e)
    return bilans


def lister() -> list[dict]:
    """Une entree par fichier source : nombre de passages et taille."""
    docs, _ = _charger()
    par_source: dict[str, dict] = {}
    for doc in docs:
        entree = par_source.setdefault(
            doc.get("source", "?"),
            {"source": doc.get("source", "?"), "nom": doc.get("nom", "?"),
             "morceaux": 0, "caracteres": 0})
        entree["morceaux"] += 1
        entree["caracteres"] += len(doc.get("texte", ""))
    return sorted(par_source.values(), key=lambda d: d["nom"].lower())


def supprimer(cible: str) -> int:
    """Supprime un fichier de l'index (nom, chemin ou radical). Rend le nb."""
    cible = cible.strip()
    with _verrou:
        docs, vecs = _charger()
        gardes, gardes_vecs = [], []
        supprimes = 0
        for i, doc in enumerate(docs):
            nom = str(doc.get("nom", ""))
            source = str(doc.get("source", ""))
            if (cible in (nom, source) or Path(source).stem == cible
                    or source.endswith(cible)):
                supprimes += 1
                continue
            gardes.append(doc)
            if vecs is not None and i < len(vecs):
                gardes_vecs.append(vecs[i])
        if supprimes:
            nouveaux_vecs = (np.asarray(gardes_vecs, dtype=np.float32)
                             if gardes_vecs and vecs is not None else None)
            _ecrire(gardes, nouveaux_vecs)
    return supprimes


# -- recherche -------------------------------------------------------------

def _scores_lexicaux(question: str, docs: list[dict]) -> np.ndarray:
    from . import rag_binaire

    mots_q = rag_binaire.tokeniser(question)
    if not mots_q:
        return np.zeros(len(docs), dtype=np.float32)
    return np.asarray(
        [len(mots_q & rag_binaire.tokeniser(d.get("texte", ""))) / len(mots_q)
         for d in docs], dtype=np.float32)


def chercher(question: str, k: int = 3,
             seuil: float | None = None) -> list[dict]:
    """Passages les plus proches de la question (liste vide si rien)."""
    question = (question or "").strip()
    if not question:
        return []
    docs, vecs = _charger()
    if not docs:
        return []
    scores = None
    if vecs is not None:
        vecteur = embeddings.encoder1(question)
        if vecteur is not None and len(vecteur) == vecs.shape[1]:
            scores = vecs @ vecteur.astype(np.float32)
            seuil = SEUIL_COSINUS if seuil is None else seuil
    if scores is None:
        scores = _scores_lexicaux(question, docs)
    # Le seuil peut rester None si aucun vecteur n'est disponible
    # (`seuil` explicite prioritaire, sinon le seuil du chemin emprunte).
    plancher = SEUIL_LEXICAL if seuil is None else seuil
    ordre = np.argsort(-scores)
    resultats = []
    for i in ordre:
        score = float(scores[i])
        if score < plancher:
            continue
        doc = docs[int(i)]
        resultats.append({"source": doc.get("source", ""),
                          "nom": doc.get("nom", "?"),
                          "texte": doc.get("texte", ""),
                          "score": round(score, 3)})
        if len(resultats) >= k:
            break
    return resultats


def interroger(question: str, k: int = 3,
               max_car: int = MAX_CAR_CONTEXTE) -> dict:
    """Contexte pret a injecter + sources a citer.

    Rend ``{"texte": "", "sources": []}`` quand l'index est vide ou que rien
    ne depasse le seuil : l'appelant n'a rien a nettoyer.
    """
    hits = chercher(question, k=k)
    if not hits:
        return {"texte": "", "sources": [], "hits": []}
    lignes: list[str] = []
    sources: list[str] = []
    total = 0
    for i, hit in enumerate(hits, 1):
        bloc = f"[{i}] {hit['nom']} : {hit['texte'].strip()}"
        if lignes and total + len(bloc) > max_car:
            break
        lignes.append(bloc)
        total += len(bloc)
        if hit["nom"] not in sources:
            sources.append(hit["nom"])
    # PAS DE CONSIGNE dans l'entete : un 1B recopie les instructions qu'on
    # lui donne (« appuie-toi dessus et cite... »). La provenance est portee
    # par le format [nom] et la ligne « Sources : » ajoutee par le pipeline.
    entete = "EXTRAITS DE DOCUMENTS LOCAUX :"
    return {"texte": entete + "\n" + "\n".join(lignes),
            "sources": sources, "hits": hits}


def contexte(question: str, k: int = 3) -> str:
    """Raccourci : le texte seul (compatibilite avec les autres RAG)."""
    return interroger(question, k=k)["texte"]


def statistiques() -> dict:
    docs, vecs = _charger()
    sources = {d.get("source") for d in docs}
    return {"dossier": str(DOSSIER), "fichiers": len(sources),
            "morceaux": len(docs),
            "vecteurs": "minilm" if vecs is not None else "lexical"}


# -- CLI -------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aura.documents",
                                description="Documents locaux d'RontoGPT")
    sous = p.add_subparsers(dest="commande", required=True)

    a = sous.add_parser("add", help="indexer un ou plusieurs fichiers")
    a.add_argument("fichiers", nargs="+")
    ad = sous.add_parser("add-dossier", help="indexer un dossier")
    ad.add_argument("dossier")
    sous.add_parser("list", help="lister les fichiers indexes")
    s = sous.add_parser("search", help="chercher dans les documents")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=3)
    r = sous.add_parser("rm", help="retirer un fichier de l'index")
    r.add_argument("cible")
    sous.add_parser("stats", help="etat de l'index")

    args = p.parse_args(argv)
    try:
        if args.commande == "add":
            for fichier in args.fichiers:
                b = ajouter(fichier)
                print(f"+ {b['nom']} : {b['morceaux']} passage(s), "
                      f"{b['caracteres']} caracteres")
        elif args.commande == "add-dossier":
            bilans = ajouter_dossier(args.dossier)
            for b in bilans:
                print(f"+ {b['nom']} : {b['morceaux']} passage(s)")
            print(f"{len(bilans)} fichier(s) indexe(s)")
        elif args.commande == "list":
            fichiers = lister()
            if not fichiers:
                print("(aucun document indexe)")
            for f in fichiers:
                print(f"{f['nom']:40s} {f['morceaux']:3d} passages "
                      f"{f['caracteres']:7d} car")
        elif args.commande == "search":
            resultat = interroger(args.question, k=args.k)
            if not resultat["texte"]:
                print("(aucun passage pertinent)")
                return 1
            print(resultat["texte"])
            print("\nSOURCES : " + ", ".join(resultat["sources"]))
        elif args.commande == "rm":
            n = supprimer(args.cible)
            print(f"{n} passage(s) supprime(s)" if n
                  else "aucun fichier correspondant")
        elif args.commande == "stats":
            print(json.dumps(statistiques(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as e:
        print(f"[documents] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
