"""Amplifie le dataset `general` par reformulations SURES de la question.

Ce que ce script ne fait PAS
----------------------------
Inverser l'ordre des mots, passer a la voix passive ou substituer des synonymes
du CONTENU. Sur une paire question/reponse, ces transformations changent le
sens ou rendent la reponse fausse : le 1B apprend alors une association qui
n'existe pas. On ne reformule que l'ENVELOPPE de la question (formule
d'introduction, ponctuation, casse) ; la reponse reste identique.

Le gain est donc reel mais borne : robustesse a la formulation, pas nouvelles
connaissances. 31 paires francaises propres x 8 enveloppes = ~248 exemples,
sous la cible de 300 du notebook.

Usage :
    python scripts/amplifier_general.py
    python scripts/amplifier_general.py --max 300 --par-question 10
    python scripts/amplifier_general.py --source datasets/general.jsonl \
        --sortie datasets/general_amplifie.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

#: Enveloppes neutres : elles n'ajoutent aucune contrainte de contenu.
_ENVELOPPES = (
    "Peux-tu me dire : {q}",
    "Réponds brièvement : {q}",
    "J'aimerais savoir : {q}",
    "Explique-moi : {q}",
    "Question : {q}",
    "J'ai besoin d'une réponse claire : {q}",
    "Donne la réponse à cette question : {q}",
    "Réponds en français : {q}",
)

#: Prefixes deja poses par une generation precedente : sans ce nettoyage,
#: chaque passage ajoutait une couche (« Dis-moi : Dis-moi : … »).
_PREFIXE = re.compile(
    r"^\s*(?:dis[- ]moi|peux[- ]tu m'expliquer|peux[- ]tu me dire|"
    r"j'aimerais savoir|j'ai besoin d'une réponse claire|"
    r"réponds brièvement|réponds en français|explique[- ]moi|"
    r"question|donne la réponse à cette question)\s*:\s*",
    re.IGNORECASE)


def nettoyer(question: str) -> str:
    """Retire les enveloppes empilees et normalise la ponctuation finale."""
    q = (question or "").strip()
    while True:
        nouveau = _PREFIXE.sub("", q, count=1).strip()
        if nouveau == q:
            break
        q = nouveau
    q = re.sub(r"\s+", " ", q).strip()
    if q and q[-1] not in "?!.":
        q += " ?"
    return q


def variantes(question: str, par_question: int = 8) -> list[str]:
    """La question nue + ses enveloppes, sans doublon."""
    q = nettoyer(question)
    if not q:
        return []
    vues = {q.casefold()}
    sortie = [q]
    for modele in _ENVELOPPES:
        if len(sortie) >= max(1, par_question):
            break
        candidate = modele.format(q=q)
        if candidate.casefold() in vues:
            continue
        vues.add(candidate.casefold())
        sortie.append(candidate)
    return sortie


def lisible(question: str, reponse: str) -> bool:
    """Le filtre de qualite, identique a celui du notebook."""
    q, r = (question or "").strip(), (reponse or "").strip()
    if not (8 <= len(q) <= 300) or not (20 <= len(r) <= 2000):
        return False
    if any(mauvais in q + r for mauvais in ("http://", "https://",
                                             "<|", "As an AI", "I cannot")):
        return False
    return True


def amplifier(source: Path, sortie: Path, maximum: int = 300,
              par_question: int = 8, graine: int = 42) -> dict:
    """Ecrit le JSONL amplifie et renvoie un rapport (compte par source)."""
    if not source.is_file():
        raise FileNotFoundError(f"dataset introuvable : {source}")
    sources = [json.loads(ligne)
               for ligne in source.read_text(encoding="utf-8").splitlines()
               if ligne.strip()]
    vues: set[str] = set()
    exemples: list[dict] = []
    ecartes = 0
    for e in sources:
        q, r = e.get("question", ""), e.get("reponse", "")
        if not lisible(q, r):
            ecartes += 1
            continue
        for variante in variantes(q, par_question):
            cle = variante.casefold()
            if cle in vues:
                continue
            vues.add(cle)
            exemples.append({"question": variante, "reponse": r,
                             **({"contexte": e["contexte"]}
                                if e.get("contexte") else {})})
            if len(exemples) >= maximum:
                break
        if len(exemples) >= maximum:
            break

    random.Random(graine).shuffle(exemples)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8") as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return {"source": str(source), "sortie": str(sortie),
            "sources_lues": len(sources), "ecartes": ecartes,
            "exemples": len(exemples)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="amplifier_general")
    ap.add_argument("--source", default=str(RACINE / "datasets" / "general.jsonl"))
    ap.add_argument("--sortie", default=str(RACINE / "datasets"
                                           / "general_amplifie.jsonl"))
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--par-question", type=int, default=8)
    ap.add_argument("--graine", type=int, default=42)
    args = ap.parse_args(argv)
    rapport = amplifier(Path(args.source), Path(args.sortie),
                        maximum=args.max, par_question=args.par_question,
                        graine=args.graine)
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
