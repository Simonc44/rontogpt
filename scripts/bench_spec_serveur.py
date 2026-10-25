"""Micro-bench du speculative decoding cote SERVEUR (llama-server).

Compare des boots identiques du binaire embarque (serveur/llama-server.exe)
avec et sans speculative decoding :

    sans spec :  ... --no-warmup
    avec spec :  ... --no-warmup --spec-type ngram-simple   (ou ngram-mod...)

Contrairement au chemin in-process (scripts/bench_spec.py), le serveur ne
force PAS logits_all : le brouillon n-gram ne coute donc rien a verifier
hors le calcul target deja paye. On mesure predicted_per_second renvoye par
POST /completion + le taux d'acceptation du draft (draft_n_accepted/draft_n).

Usage : .venv/Scripts/python.exe scripts/bench_spec_serveur.py
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from aura.llama_cerveau import chemin_gguf

_RACINE = Path(__file__).resolve().parent.parent
_GGUF = Path(chemin_gguf())   # cerveau par defaut (plus de nom en dur)
_BIN = _RACINE / "serveur" / "llama-server.exe"

# "repétitif" = sortie LITTERALEMENT identique a chaque ligne : c'est le seul
# cas ou un brouillon n-gram peut devancer le modele (un chiffre qui change
# a chaque ligne casse deja le match n-gram).
_PROMPTS = {
    "repétitif": ("Repète exactement 40 fois, une par ligne, sans rien changer :\n"
                  "le chat dort sous la table\n"),
    "neuf": "Explique en 3 phrases pourquoi le ciel est bleu pendant la journée.",
}


def _sante(port: int, attente: float = 60.0) -> bool:
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


def _completion(port: int, prompt: str, n_predict: int = 160) -> dict:
    corps = json.dumps({"prompt": prompt, "n_predict": n_predict,
                        "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=corps,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def _mesurer(port: int, spec: str | None, etiq: str | None = None) -> None:
    etiq = etiq or spec or "off"
    cmd = [str(_BIN), "-m", str(_GGUF), "--port", str(port),
           "--host", "127.0.0.1", "-c", "1536", "-t", "4", "--no-warmup"]
    if spec:
        cmd += ["--spec-type", spec]
    if etiq == "mod-nmin64":
        # match minimal de 64 tokens : ne draft que sur du texte TRES repetitif
        cmd += ["--spec-ngram-mod-n-min", "64"]
    log = open(_RACINE / "serveur" / f"bench_spec_{etiq}.log",
               "wb", buffering=0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(_RACINE))
    try:
        if not _sante(port):
            print(f"[bench] serveur (spec={etiq}) ne repond pas -> abandon")
            return
        # 1 tour de chauffe (mmap, pages chaudes), puis mesure
        _completion(port, "Bonjour !")
        for nom, prompt in _PROMPTS.items():
            t0 = time.time()
            rep = _completion(port, prompt)
            dt = time.time() - t0
            t = rep.get("timings", {}) or {}
            tps = float(t.get("predicted_per_second") or 0)
            draf = ""
            if t.get("draft_n"):
                draf = (f" draft={t.get('draft_n_accepted', 0)}/{t['draft_n']}"
                        f" acceptés")
            print(f"[bench] spec={etiq:12s} {nom:10s} {tps:5.1f} tok/s "
                  f"({t.get('predicted_n', '?')} tok en {dt:.1f}s){draf}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log.close()


def main() -> int:
    if not _BIN.is_file() or not _GGUF.is_file():
        print("[bench] binaire ou GGUF introuvable")
        return 1
    # 1re serie froide ignoree (cache fichier), 2e serie = mesure retenue
    variantes = [None, "ngram-map-k", "ngram-map-k4v", "ngram-cache"]
    for passe in (1, 2):
        print(f"--- passe {passe} ---")
        for variante in variantes:
            if variante is None:
                _mesurer(8761, None)
            else:
                type_, _, etiq = variante.partition(":")
                _mesurer(8761, type_, etiq or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
