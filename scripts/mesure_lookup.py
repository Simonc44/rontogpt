"""Mesure du brouillon STATISTIQUE (n-gram) cote serveur llama.cpp.

Question posee : « le modele symbolique peut-il calculer par les maths les
mots les plus probables (comptage de n-grammes) et depasser un LLM en
vitesse ? »

Ce qui est mesurable ici, c'est le TAUX D'ACCEPTATION du brouillon
(draft_n_accepted / draft_n) : un brouillon n'est jamais cru sur parole, le
modele cible verifie chaque token propose. Un brouillon faux ne coute donc
que le temps de le proposer et de le rejeter.

Deux regimes sont mesures separement :

  * 4 questions `riche` du banc -> texte NOUVEAU, c'est le cas nominal ;
  * 2 controles repetitifs        -> seul regime ou un compteur de n-grammes
    peut devancer le reseau.

Note de version (build 11111, commit 4ceb17191) : la liste des
implementations speculatives compilees est
draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache.
Il n'y a AUCUNE implementation « lookup » : les options -lcs/-lcd sont
parsees mais ne branchent rien cote serveur.

Usage : .venv/Scripts/python.exe scripts/mesure_lookup.py [--tokens N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from aura.llama_cerveau import chemin_gguf

_RACINE = Path(__file__).resolve().parent.parent
_BIN = _RACINE / "serveur" / "vulkan" / "llama-server.exe"
_GGUF = Path(chemin_gguf())   # cerveau par defaut (le GGUF Llama a ete supprime)
_LORAS = ",".join(str(_RACINE / "serveur" / f"aura-{n}-T.gguf")
                  for n in ("code", "general", "math", "web"))
_PORT = 8763

# Filtre de prompts (rempli par main) : evite de mesurer tous les regimes
# quand un seul est en question.
_PREFIXES: tuple[str, ...] = ()

# Taille de contexte (remplie par main) : la PRODUCTION tourne en 8192.
_CTX: int = 4096

# Flags du serveur de PRODUCTION (cf. ligne de commande du processus 8612) :
# on mesure le brouillon dans les conditions reelles, LoRA compris.
_BASE = ["-c", str(_CTX), "-t", "6", "--no-warmup", "-fa", "on",
         "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-ngl", "99",
         "--lora", _LORAS]

# (etiquette, arguments ajoutes)
_VARIANTES: list[tuple[str, list[str]]] = [
    ("off", []),
    ("ngram-cache", ["--spec-type", "ngram-cache"]),
    ("ngram-simple-n2", ["--spec-type", "ngram-simple",
                         "--spec-ngram-simple-size-n", "2"]),
    ("ngram-map-k-n2", ["--spec-type", "ngram-map-k",
                        "--spec-ngram-map-k-size-n", "2"]),
    ("ngram-map-k-n2-m12", ["--spec-type", "ngram-map-k",
                            "--spec-ngram-map-k-size-n", "2",
                            "--spec-ngram-map-k-size-m", "12"]),
    ("ngram-mod-n8", ["--spec-type", "ngram-mod",
                      "--spec-ngram-mod-n-min", "8"]),
]

# 4 questions `riche` reelles du banc (texte nouveau) + 2 controles.
_PROMPTS: list[tuple[str, str]] = [
    ("riche-02", "La puissance de la technologie est-elle une menace pour "
                 "l'homme moderne ? Redige une dissertation structuree."),
    ("riche-06", "La liberte peut-elle exister sans contrainte ? Redige une "
                 "dissertation"),
    ("riche-08", "Compare l'intelligence artificielle et l'intelligence "
                 "humaine sur le plan de la creativite"),
    ("riche-10", "Redige une dissertation structuree sur l'utilite du doute "
                 "dans la demarche scientifique"),
    ("ctrl-repet", "Repete exactement 20 fois, une par ligne, sans rien "
                   "changer :\nle chat dort sous la table\n"),
    ("ctrl-rejoue", "Recopie exactement ce qui suit, sans rien ajouter :\n"
                    "Au futur, le verbe « etre » se conjugue : je serai, tu "
                    "seras, il/elle sera, nous serons, vous serez, "
                    "ils/elles seront."),
    # Regime STRUCTURE : c'est la ou RontoGPT passe l'essentiel de son temps
    # de generation (plan de la passe 1 contraint par GBNF, listes,
    # code de potcode). Si le brouillon n-gram doit payer quelque part,
    # c'est ici.
    ("struct-plan",
     "Genere UNIQUEMENT le plan de ta reponse, au format exact :\n"
     "PARTIE 1 : <LA these : l'affirmation, avec sa preuve>\n"
     "PARTIE 2 : <L'ANTITHESE : la negation nuancee, avec son exemple>\n"
     "PARTIE 3 : <LA SYNTHESE : ce qui depasse l'opposition>\n"
     "CONNECTEURS : <5 connecteurs logiques avances separes par des "
     "virgules>\n\nQuestion : la liberte peut-elle exister sans "
     "contrainte ?"),
    ("struct-puce",
     "Donne 10 avantages du teletravail, un par ligne, chaque ligne "
     "commencant exactement par « Avantage 1 : », « Avantage 2 : », etc."),
    ("struct-code",
     "Ecris une fonction Python qui calcule la moyenne d'une liste de "
     "nombres, en gerant la liste vide."),
]


def _sante(port: int, attente: float = 90.0) -> bool:
    fin = time.time() + attente
    while time.time() < fin:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _completion(port: int, prompt: str, n_predict: int) -> dict:
    corps = json.dumps({"prompt": prompt, "n_predict": n_predict,
                        "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=corps,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def _mesurer(etiq: str, extra: list[str], n_predict: int) -> list[dict]:
    base = ["-c", str(_CTX), *_BASE[2:]]
    cmd = [str(_BIN), "-m", str(_GGUF), "--port", str(_PORT),
           "--host", "127.0.0.1", *base, *extra]
    log = open(_RACINE / "serveur" / f"mesure_lookup_{etiq}.log", "wb",
               buffering=0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(_RACINE))
    lignes: list[dict] = []
    try:
        if not _sante(_PORT):
            print(f"[lookup] {etiq:16s} serveur muet -> abandon")
            return []
        _completion(_PORT, "Bonjour !", 8)          # chauffe (mmap, pages)
        for nom, prompt in _PROMPTS:
            if _PREFIXES and not nom.startswith(_PREFIXES):
                continue
            t0 = time.time()
            rep = _completion(_PORT, prompt, n_predict)
            dt = time.time() - t0
            t = rep.get("timings", {}) or {}
            n_draft = int(t.get("draft_n") or 0)
            acc = int(t.get("draft_n_accepted") or 0)
            lignes.append({
                "variante": etiq, "prompt": nom,
                "tok_s": round(float(t.get("predicted_per_second") or 0), 2),
                "n_pred": int(t.get("predicted_n") or 0),
                "draft_n": n_draft, "acceptes": acc,
                "taux": round(acc / n_draft, 4) if n_draft else 0.0,
                "mur_s": round(dt, 2),
            })
            print(f"[lookup] {etiq:16s} {nom:11s} "
                  f"{lignes[-1]['tok_s']:5.2f} tok/s  "
                  f"accept={lignes[-1]['taux']:.3f} "
                  f"({acc}/{n_draft})  {dt:.1f}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()
    return lignes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mesure_lookup")
    ap.add_argument("--tokens", type=int, default=120,
                    help="tokens generes par prompt (defaut 120)")
    ap.add_argument("--variantes", default="",
                    help="filtre par prefixe d'etiquette (ex: ngram)")
    ap.add_argument("--prompts", default="",
                    help="filtre par prefixe de prompt (ex: struct,ctrl)")
    ap.add_argument("--ctx", type=int, default=4096,
                    help="taille de contexte (production : 8192)")
    args = ap.parse_args(argv)

    if not _BIN.is_file() or not _GGUF.is_file():
        print("[lookup] binaire ou GGUF introuvable")
        return 1

    # filtre global (variable de module) lu par _mesurer -> evite de
    # trainer tous les prompts quand on ne mesure qu'un regime.
    global _PREFIXES, _CTX
    _PREFIXES = tuple(p for p in args.prompts.split(",") if p)
    _CTX = args.ctx

    prefixes_var = tuple(p for p in args.variantes.split(",") if p)
    variantes = [v for v in _VARIANTES
                 if not prefixes_var or v[0].startswith(prefixes_var)]
    toutes: list[dict] = []
    for etiq, extra in variantes:
        toutes += _mesurer(etiq, extra, args.tokens)

    # le nom porte le ctx ET le filtre : sans ca, deux runs de regimes
    # differents s'ecrasent et on perd la trace de la mesure.
    nom = ("mesure_lookup_c%d" % _CTX) + (
        "_" + "_".join(_PREFIXES) if _PREFIXES else "") + ".json"
    sortie = _RACINE / "serveur" / nom
    sortie.write_text(json.dumps(toutes, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n[lookup] resultats -> {sortie}")
    _tableau(toutes)
    return 0


def _tableau(lignes: list[dict]) -> None:
    par_var: dict[str, list[dict]] = {}
    for ligne in lignes:
        par_var.setdefault(ligne["variante"], []).append(ligne)
    print(f"\n{'variante':16s} {'moy tok/s':>9s} {'moy taux':>9s} "
          f"{'taux riche':>11s} {'taux ctrl':>10s}")
    for v, ls in par_var.items():
        riches = [ligne for ligne in ls if ligne["prompt"].startswith("riche")]
        ctrls = [ligne for ligne in ls if ligne["prompt"].startswith("ctrl")]
        moy = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        print(f"{v:16s} {moy([ligne['tok_s'] for ligne in ls]):9.2f} "
              f"{moy([ligne['taux'] for ligne in ls]):9.3f} "
              f"{moy([ligne['taux'] for ligne in riches]):11.3f} "
              f"{moy([ligne['taux'] for ligne in ctrls]):10.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
