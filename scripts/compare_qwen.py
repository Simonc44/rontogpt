"""Benchmark head-to-head : Gemma 3 1B (RontoGPT) vs Qwen 2.5 1.5B (Ollama).

Memes questions, memes regles (temp 0.1, 150 tokens max), reponses verifiables.
Usage : python scripts/compare_qwen.py   (necessite qwen2.5:1.5b dans Ollama)
"""
import time

import requests
from llama_cpp import Llama

from aura.llama_cerveau import chemin_gguf

QUESTIONS = [
    # (question, sous-chaine attendue dans la reponse)
    ("Sophie a plus de livres que Marc. Marc en a plus que Lea. "
     "Qui en a le moins ? Reponds juste le prenom.", "lea"),
    ("Calcule : 47 + 38 - 15. Reponds juste le nombre.", "70"),
    ("Quelle est la capitale de l'Australie ? Reponds juste la ville.", "canberra"),
    ("Qui a ecrit 'L'Etranger' ? Reponds juste le nom de famille.", "camus"),
    ("Paul fait 3 ans de plus que Marie. Marie a 14 ans. "
     "Quel age a Paul ? Reponds juste le nombre.", "17"),
]
PROMPT_VITESSE = "Explique la photosynthese en deux phrases."
OPTS_LLAMA = dict(n_ctx=1024, n_threads=6, n_batch=512, flash_attn=True,
                  verbose=False, type_k=8, type_v=8)
GGUF = chemin_gguf()   # cerveau par defaut du projet


def demander_ollama(question: str, modele: str, max_tokens: int = 150):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": modele, "stream": False,
        "messages": [{"role": "user", "content": question}],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }, timeout=300)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"].strip(), d.get("eval_count", 0), d.get("eval_duration", 0)


def demander_llama(llm: Llama, question: str, max_tokens: int = 150):
    t0 = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens, temperature=0.1)
    n = out["usage"]["completion_tokens"]
    return (out["choices"][0]["message"]["content"].strip(),
            n, (time.time() - t0))


def main() -> None:
    print("=" * 74)
    print(" HEAD-TO-HEAD : Llama 3.2 1B (RontoGPT) vs Qwen 2.5 1.5B (Ollama)")
    print("=" * 74)

    scores = {"llama": 0, "qwen": 0}
    llm = Llama(model_path=GGUF, **OPTS_LLAMA)

    for q, attendu in QUESTIONS:
        # Qwen d'abord (API), puis Llama (meme question)
        try:
            rq, nq, dq = demander_ollama(q, "qwen2.5:1.5b-instruct")
            vq = rq.lower()
            tok_q = nq / (dq / 1e9) if dq else 0
        except Exception as e:
            rq, vq, tok_q = f"(erreur : {e})", "", 0
        rl, nl, dl = demander_llama(llm, q)
        vl = rl.lower()
        ok_q, ok_l = attendu in vq, attendu in vl
        scores["qwen"] += ok_q
        scores["llama"] += ok_l
        print(f"\nQ : {q[:70]}")
        print(f"   QWEN  [{'OK' if ok_q else 'KO'}] {rq[:60]!r}")
        print(f"   LLAMA [{'OK' if ok_l else 'KO'}] {rl[:60]!r}")

    # vitesse sur le meme prompt
    _, nq, dq = demander_ollama(PROMPT_VITESSE, "qwen2.5:1.5b-instruct", 80)
    tok_q = nq / (dq / 1e9) if dq else 0
    _, nl, dl = demander_llama(llm, PROMPT_VITESSE, 80)
    tok_l = nl / dl
    print(f"\nVITESSE : Qwen {tok_q:.1f} tok/s | Llama {tok_l:.1f} tok/s")
    print(f"\nSCORE RAISONNEMENT/CONNAISSANCES : Qwen {scores['qwen']}/5"
          f" | Llama {scores['llama']}/5")


if __name__ == "__main__":
    main()
