"""Point d'entree : `python -m rontogpt`.

Aiguillage vers `aura.__main__:main` — une seule implementation de la CLI,
pas de copie a maintenir. `main()` fixe deja `prog="rontogpt"`, donc l'aide
affiche bien « usage: rontogpt ».

Particularite de ce module : il s'importe AVANT `aura` (dont le paquet
`__init__` tire gplearn / llama_cpp). C'est donc la seule porte ou l'on peut
encore choisir son interpreteur : lance avec le Python global (« python »
tout court), le projet recharge ses dependances depuis `.venv` au lieu
d'echouer sur « No module named 'gplearn' ».
"""
from __future__ import annotations

import sys
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent      # racine du projet

#: Dependances du projet qu'un Python global n'a pas. Leur absence suffit a
#: diagnostiquer « mauvais interpreteur », sans charger quoi que ce soit.
_DEPS = ("numpy", "sklearn", "gplearn", "llama_cpp")


def _python_venv() -> Path | None:
    """L'interpreteur du venv du projet, s'il existe."""
    for cand in (_RACINE / ".venv" / "Scripts" / "python.exe",
                 _RACINE / ".venv" / "bin" / "python"):
        if cand.is_file():
            return cand
    return None


def _manquantes() -> list[str]:
    import importlib.util
    return [m for m in _DEPS if importlib.util.find_spec(m) is None]


def _relancer_si_besoin(argv: list[str]) -> None:
    """Relance dans `.venv` si l'interpreteur courant est incomplet.

    Aucune boucle possible : dans la venv, `deja_venv` est vrai, donc la
    deuxieme branche l'emporte et affiche le message au lieu de relancer.
    """
    manquantes = _manquantes()
    if not manquantes:
        return
    deja_venv = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    py = _python_venv()
    if py is not None and not deja_venv:
        import subprocess
        print(f"[rontogpt] {', '.join(manquantes)} absent(s) de "
              f"{sys.executable} -> relance avec {py}")
        raise SystemExit(
            subprocess.call([str(py), "-m", "rontogpt", *argv]))
    raise SystemExit(
        "[rontogpt] modules manquants : " + ", ".join(manquantes) + "\n"
        "Installe-les avec :  uv sync\n"
        "Puis lance avec    :  "
        + str(_RACINE / ".venv" / "Scripts" / "python.exe") + " -m rontogpt")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _relancer_si_besoin(args)
    from aura.__main__ import main as _cli          # import paresseux
    return _cli(args)


if __name__ == "__main__":
    sys.exit(main())
