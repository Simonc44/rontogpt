"""Telecharge le cerveau d'RontoGPT dans modeles/.

Par defaut : Gemma-3-1B-it (unsloth/gemma-3-1b-it-GGUF) en Q8_0 — le cerveau
configure dans aura/llama_cerveau.py.
MiniCPM5-1B (OpenBMB, cerveau « thinking ») reste disponible :
AURA_CERVEAU=minicpm — il a besoin d'AURA_THINK=1 pour donner son mieux.

Usage :
  python scripts/telecharger_llama.py                  # Gemma Q8_0
  AURA_CERVEAU=llama python scripts/telecharger_llama.py            # Llama Q4_K_M
  AURA_CERVEAU=minicpm python scripts/telecharger_llama.py          # MiniCPM Q4
  AURA_CERVEAU=minicpm python scripts/telecharger_llama.py Q8_0    # MiniCPM Q8
  AURA_CERVEAU=llama python scripts/telecharger_llama.py Q6_K      # Llama Q6
"""
import os
import socket
import sys
from pathlib import Path

REPOS = {
    "gemma": "unsloth/gemma-3-1b-it-GGUF",
    "llama": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "minicpm": "openbmb/MiniCPM5-1B-GGUF",
}
FICHIERS = {
    "gemma": {
        "Q8_0": "gemma-3-1b-it-Q8_0.gguf",
    },
    "llama": {
        "Q4_K_M": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Q6_K": "Llama-3.2-1B-Instruct-Q6_K.gguf",
    },
    "minicpm": {
        "Q4_K_M": "MiniCPM5-1B-Q4_K_M.gguf",
        "Q8_0": "MiniCPM5-1B-Q8_0.gguf",
    },
}
DEST = Path(__file__).resolve().parent.parent / "modeles"


def _est_connecte(
    hote="huggingface.co", port=443, delai=5
):
    """Vrai si le DNS et le port TCP 443 semblent ouverts."""
    try:
        with socket.create_connection((hote, port), timeout=delai):
            return True
    except OSError:
        return False


def main() -> int:
    choix = os.environ.get("AURA_CERVEAU", "gemma").lower()
    quant = (sys.argv[1] if len(sys.argv) > 1 else "Q8_0").upper()
    if choix not in REPOS:
        print(f"[!] cerveau inconnu : {choix} (choix : {', '.join(REPOS)})")
        return 1
    fichier = FICHIERS[choix].get(quant)
    if not fichier:
        print(f"[!] quant inconnue pour {choix} : {quant} "
              f"(choix : {', '.join(FICHIERS[choix])})")
        return 1
    # Petit test de connectivite avant de lancer le telechargement, pour
    # que l'echec soit imminent et lisible (timeout court, pas de silence).
    if not _est_connecte():
        print("[!] hors ligne detectee : impossible de joindre huggingface.co.")
        print("    Verifiez votre connexion, un proxy ou un blocage DNS.")
        return 2

    from huggingface_hub import hf_hub_download
    DEST.mkdir(exist_ok=True)
    print(f"Telechargement de {REPOS[choix]}/{fichier}...")
    chemin = hf_hub_download(REPOS[choix], fichier, local_dir=str(DEST))
    print(f"OK : {chemin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
