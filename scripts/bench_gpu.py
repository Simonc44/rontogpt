"""Bench CPU vs Vulkan d'RontoGPT (meme question, memes budgets).

Compare les deux binaires embarques :
    CPU     : serveur/llama-server.exe   (build cpu-x64, ce qu'RontoGPT utilise)
    VULKAN  : serveur/vulkan/...         (build win-vulkan-x64, -ngl 99)

Mesure sur /completion : prefill (prompt tok/s) + decode (predicted tok/s),
2 passes (froid puis chaud) pour ne pas conclure sur un cache de fichier.

Usage : .venv/Scripts/python.exe scripts/bench_gpu.py
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from aura.llama_cerveau import chemin_gguf

_RACINE = Path(__file__).resolve().parent.parent
# Cerveau par defaut du projet (le GGUF Llama 3.2 a ete supprime : ne pas
# coder un nom de fichier ici, `chemin_gguf()` est la source unique).
_GGUF = Path(chemin_gguf())

# sortie longue NON bornable : le modele ne peut pas finir avant le
# budget, donc le decode se mesure sur les 160 tokens demandes
_PROMPT = ("Redige un long developpement sur la foret, ses saisons, ses "
           "animaux, ses utilites pour l'homme et l'avenir de la filiere "
           "bois. Detaille chaque idee en phrases completes.\n")

CONFIGS = [
    ("cpu", _RACINE / "serveur" / "llama-server.exe", []),
    ("vulkan", _RACINE / "serveur" / "vulkan" / "llama-server.exe", ["-ngl", "99"]),
]


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


def _completion(port: int, n_predict: int = 256) -> dict:
    corps = json.dumps({"prompt": _PROMPT, "n_predict": n_predict,
                        "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=corps,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def _mesurer(label: str, binaire: Path, drapeaux: list[str], port: int) -> None:
    if not binaire.is_file():
        print(f"[gpu] {label:8s} binaire absent : {binaire}")
        return
    cmd = [str(binaire), "-m", str(_GGUF), "--port", str(port),
           "--host", "127.0.0.1", "-c", "1536", "-t", "4", "--no-warmup",
           *drapeaux]
    log = open(_RACINE / "serveur" / f"bench_gpu_{label}.log", "wb",
               buffering=0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(_RACINE))
    try:
        if not _sante(port):
            print(f"[gpu] {label:8s} ne repond pas (voir bench_gpu_{label}.log)")
            return
        for passe in (1, 2):                 # 1 = froid, 2 = chaud
            rep = _completion(port)
            t = rep.get("timings", {}) or {}
            print(f"[gpu] {label:8s} passe{passe} "
                  f"decode {float(t.get('predicted_per_second') or 0):5.1f} tok/s "
                  f"| prefill {float(t.get('prompt_per_second') or 0):5.1f} tok/s "
                  f"| {t.get('predicted_n')} tokens")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log.close()
        time.sleep(1.0)


def main() -> int:
    if not _GGUF.is_file():
        print("[gpu] GGUF introuvable")
        return 1
    for passe_label, port in (("serie A", 8781), ("serie B", 8782)):
        print(f"--- {passe_label} ---")
        for label, binaire, drapeaux in CONFIGS:
            _mesurer(label, binaire, drapeaux, port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
