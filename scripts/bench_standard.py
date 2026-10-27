#!/usr/bin/env python3
"""Banc standard RontoGPT — suite de benchmarks reproductible.

Usage :
    python scripts/bench_standard.py [--quick]

Ce script execute les benchmarks de reference :
1. Banc interne 100 questions (pipeline complet)
2. GSM8K cerveau nu (lm-eval, 100 exemples, 5-shot)
3. Optionnel : banc interne complet (si --quick non passe)

Prerequis :
- Serveur llama.cpp sur port 8757 (AURA_SERVEUR=1 python -m aura &)
- lm-eval installe (uv pip install --python .venv/Scripts/python.exe lm-eval)

Resultats attendus (27 oct. 2026) :
- Banc interne : 94/100 (85/85 exact, 9/15 formulaire)
- GSM8K cerveau nu : 0.16 flexible, 0.00 strict
- GSM8K pipeline : 0.12 flexible, 0.00 strict
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def run(cmd: list[str], description: str, timeout: int = 600) -> tuple[int, str]:
    """Execute une commande et retourne (code, sortie)."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8", errors="replace")
        if res.stdout:
            print(res.stdout[-2000:])  # dernieres lignes
        if res.stderr:
            print(res.stderr[-2000:], file=sys.stderr)
        return res.returncode, res.stdout
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (> {timeout}s)", file=sys.stderr)
        return -1, ""


def main():
    ap = argparse.ArgumentParser(description="Banc standard RontoGPT")
    ap.add_argument("--quick", action="store_true",
                    help="saute le banc interne complet (plus lent)")
    args = ap.parse_args()

    t0 = time.time()
    results = []

    # 1. Verifier que le serveur tourne
    print("\n>> Verification du serveur llama.cpp sur :8757...")
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8757/health", timeout=2)
        print("  OK : serveur accessible")
    except Exception as e:
        print(f"  ERREUR : serveur non accessible ({e})")
        print("  Lancez : AURA_SERVEUR=1 python -m aura &")
        return 1

    # 2. Banc interne (pipeline complet)
    if not args.quick:
        code, out = run(
            [sys.executable, "-m", "scripts.harnais_evaluation", "--banc", "100"],
            "Banc interne 100 questions (pipeline complet)",
            timeout=1200
        )
        results.append(("Banc interne", code, out))

    # 3. GSM8K cerveau nu (lm-eval)
    code, out = run(
        [sys.executable, "-m", "scripts.lm_eval_aura",
         "--taches", "gsm8k", "--limite", "100", "--fewshot", "5"],
        "GSM8K cerveau nu (100 exemples, 5-shot)",
        timeout=1200
    )
    results.append(("GSM8K cerveau nu", code, out))

    # 4. GSM8K pipeline complet
    if not args.quick:
        code, out = run(
            [sys.executable, "-m", "scripts.harnais_evaluation",
             "--taches", "gsm8k", "--limite", "100"],
            "GSM8K pipeline complet (pipeline RontoGPT)",
            timeout=1200
        )
        results.append(("GSM8K pipeline", code, out))

    # Resume
    print("\n" + "="*60)
    print("  RESUME")
    print("="*60)
    for name, code, _ in results:
        status = "OK" if code == 0 else "ECHEC"
        print(f"  {name:25s} : {status}")
    print(f"\nTemps total : {time.time() - t0:.0f}s")

    return 0 if all(c == 0 for _, c, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())