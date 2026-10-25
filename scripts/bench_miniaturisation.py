"""Benchmark de miniaturisation : Q4_K_M (807 Mo) vs IQ3_M (657 Mo).

Mesure sur machine reelle : taille, chargement, vitesse (tok/s) et
qualite (3 faits deterministes, temperature 0 pour la reproductibilite).
"""
import os
import time

from llama_cpp import Llama

# Exactement la config du cerveau (aura/llama_cerveau.py)
REGLAGES = dict(
    n_ctx=1536,
    n_batch=768,
    n_threads=6,
    n_threads_batch=6,
    flash_attn=True,
    type_k=1,   # q8_0 : KV cache /2 en RAM
    type_v=1,
    verbose=False,
)

QUESTIONS = [
    ("Quelle est la capitale de l'Australie ?", "Canberra"),
    ("Quelle est la capitale du Japon ?", "Tokyo"),
    ("En quelle annee la tour Eiffel a-t-elle ete construite ?", "1889"),
]

# ~380 tokens de prompt (identiques pour les 2 modeles)
REMPLISSAGE = "Reponds en une phrase concise. " * 60


def bench(chemin: str) -> None:
    nom = os.path.basename(chemin)
    taille = os.path.getsize(chemin) / 1e6
    print("=" * 62)
    print(f"{nom}  —  {taille:.1f} MB")
    print("=" * 62)

    t0 = time.time()
    llm = Llama(model_path=chemin, **REGLAGES)
    print(f"chargement : {time.time() - t0:.1f} s")

    # --- vitesse : gros prompt + generation greedy ---
    prompt = REMPLISSAGE + "Explique ce qu'est la photosynthese."
    t0 = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
        temperature=0.0,
    )
    dt = time.time() - t0
    n_gen = out["usage"]["completion_tokens"]
    print(f"vitesse    : {n_gen / dt:.1f} tok/s  ({n_gen} tokens en {dt:.1f} s)")

    # --- qualite : faits deterministes ---
    bons = 0
    for question, attendu in QUESTIONS:
        r = llm.create_chat_completion(
            messages=[{"role": "user", "content": question}],
            max_tokens=30,
            temperature=0.0,
        )
        texte = r["choices"][0]["message"]["content"]
        ok = attendu.lower() in texte.lower()
        bons += ok
        print(f"  {'OK' if ok else 'KO'}  {question[:48]}  ->  {texte.strip()[:50]}")
    print(f"qualite    : {bons}/3")

    del llm
    print()


if __name__ == "__main__":
    from aura.llama_cerveau import chemin_gguf
    bench(chemin_gguf())
    bench("modeles/IQ3_M.download.gguf")
