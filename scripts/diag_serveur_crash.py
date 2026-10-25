"""Diagnostic : sous quelle requete llama-server (4 LoRA) plante-t-il ?

Le 22/09 la meme configuration repondait (voir serveur/autogere.log ~l.566) ;
depuis, une requete courte avec message systeme declenche :
    GGML_ASSERT(obj_new) failed / not enough space in the context's memory pool

On rejoue une matrice de requetes, un boot par requete (le plantage tue le
serveur), pour isoler le declencheur : longueur du prompt, systeme, max_tokens.

Usage : .venv/Scripts/python.exe scripts/diag_serveur_crash.py
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from aura.llama_cerveau import chemin_gguf

_RACINE = Path(__file__).resolve().parent.parent
_GGUF = Path(chemin_gguf())   # cerveau par defaut
_BIN = _RACINE / "serveur" / "llama-server.exe"
_LORA = sorted((_RACINE / "serveur").glob("aura-*-T.gguf"))

CAS = {
    "systeme_court": ([{"role": "system", "content": "Tu es RontoGPT."},
                       {"role": "user", "content": "Repete 3 fois : bonjour monde"}], 80, False),
    "court+stop": ([{"role": "system", "content": "Tu es RontoGPT."},
                    {"role": "user", "content": "Repete 3 fois : bonjour monde"}], 80, True),
    "user_court": ([{"role": "user", "content": "Repete 3 fois : bonjour monde"}], 80, False),
    "systeme_long": ([{"role": "system", "content": "Tu es RontoGPT."},
                      {"role": "user", "content": "Explique le cycle de l'eau "
                       "en 5 phrases detaillees avec des exemples concrets."}], 80, False),
}

_STOP = ["<|eot_id|>", "\nUtilisateur:", "\nUser:"]


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


def _essai(nom: str, messages: list[dict], max_tokens: int, stop: bool) -> None:
    cmd = [str(_BIN), "-m", str(_GGUF), "--port", "8765", "--host", "127.0.0.1",
           "-c", "1536", "-t", "4", "--no-warmup"]
    for p in _LORA:
        cmd += ["--lora", str(p)]
    log = open(_RACINE / "serveur" / "diag_crash.log", "ab", buffering=0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(_RACINE))
    try:
        if not _sante(8765):
            print(f"[diag] {nom:16s} -> serveur ne repond pas")
            return
        corps = json.dumps({"messages": messages, "max_tokens": max_tokens,
                            "temperature": 0.4, "top_p": 0.9,
                            **({"stop": _STOP} if stop else {})}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8765/v1/chat/completions", data=corps,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                rep = json.loads(r.read().decode())
            txt = (rep.get("choices") or [{}])[0].get("message", {}).get("content", "")
            print(f"[diag] {nom:16s} -> OK  {len(txt)} car.")
        except Exception as e:  # noqa: BLE001
            print(f"[diag] {nom:16s} -> CRASH ({e.__class__.__name__})")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        else:
            print(f"[diag] {nom:16s} -> serveur mort (code {proc.returncode})")
        log.close()
        time.sleep(1.0)


def main() -> int:
    if not _BIN.is_file() or not _GGUF.is_file() or not _LORA:
        print("[diag] binaire/GGUF/LoRA introuvable")
        return 1
    print(f"[diag] {len(_LORA)} adaptateurs")
    for nom, (msgs, budget, stop) in CAS.items():
        _essai(nom, msgs, budget, stop)
    # chemin reel : aura.serveur_lora._demarrer() + chat() (meme port 8757)
    print("[diag] --- via serveur_lora.chat() ---")
    import os
    os.environ["AURA_SERVEUR"] = "1"
    from aura import serveur_lora as s
    for nom in ("chat_api",):
        ok = s._demarrer()
        if not ok:
            print(f"[diag] {nom:16s} -> boot KO ({s._indispo})")
            continue
        try:
            r = s.chat([{"role": "system", "content": "Tu es RontoGPT."},
                        {"role": "user", "content": "Repete 3 fois : bonjour monde"}],
                       max_tokens=80)
            print(f"[diag] {nom:16s} -> {'OK  ' + str(len(r)) + ' car.' if r else 'CRASH/VIDE'}")
        finally:
            if s._process and s._process.poll() is None:
                s._process.terminate()
                try:
                    s._process.wait(timeout=10)
                except Exception:
                    s._process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
