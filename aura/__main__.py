"""CLI RontoGPT : cerveau Llama 3.2 1B (Q4_K_M) + experts symboliques.

Usage :
  python -m aura                 # interface terminal OpenTUI (defaut)
                                 # TUI absent -> conversation en console
  python -m aura "question"      # une question, reponse sur la sortie standard
  python -m aura --console       # ancienne console texte
  python -m aura --demo
  python -m aura "question" --x "[[1],[2],[3]]" --y "[1,8,27]"
  python -m aura --superviser      # API surveillee (relancee si elle plante) + TUI
  python -m aura --superviser --api  # API seule, surveillee
  python -m aura --superviser --api --demon  # idem, detache en arriere-plan

POINT D'ENTREE UNIQUE : le superviseur vivait dans un `main.py` a la racine ;
il est desormais ici (cf. aura/superviseur.py), donc un seul lanceur a
connaitre — `rontogpt` (console script) ou `python -m aura`.
"""
import argparse
import json
import sys
import time

# CONFIGURATION (aura.toml + profils) : AVANT tout import du cerveau,
# les modules lisent leurs AURA_* au moment de l'import.
from . import config as _config
_config.appliquer()

# noqa: E402 — l'ordre est le PROPOS de ces trois lignes : `appliquer()` doit
# avoir pose les AURA_* AVANT que l'orchestrateur (et tout ce qu'il importe)
# ne les lise au moment de son import. Remonter cet import le casserait.
from .orchestrateur import RontoGPT  # noqa: E402


def _demo(ia: RontoGPT) -> int:
    X = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    y = [0.0, 0.015625, 0.125, 0.421875, 1.0]
    r = ia.executer_detaille(
        "Dernieres decouvertes en physique et resous X cube", X, y)
    print("=" * 60)
    print(" RontoGPT — Llama 3.2 1B + PGS + Web (autonome)")
    print("=" * 60)
    print(f"Experts actives : {r['experts']}")
    print(f"Cerveau         : {r['cerveau_choisi']}")
    if r["contexte_web"]:
        print(f"\nWeb :\n{r['contexte_web'][:300]}")
    print(f"\nFormule PGS    : Y = {r['formule']}  (erreur {r['erreur_pgs']:.3g})")
    print(f"\nReponse :\n{r['reponse'][:300]}")
    return 0


def _chat(ia: RontoGPT) -> int:
    """Mode conversationnel multi-tours (le modele se souvient)."""
    print('RontoGPT chat — "exit" pour quitter, "reset" pour oublier la conversation.')
    while True:
        try:
            q = input("toi> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            return 0
        if q.lower() == "reset":
            ia.reinitialiser_conversation()
            print("[conversation oubliée]")
            continue
        t0 = time.time()
        print(ia.executer(q))
        print(f"[{time.time()-t0:.1f}s]")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rontogpt")
    p.add_argument("question", nargs="?", default="")
    p.add_argument("--x", help="liste JSON de vecteurs, ex. '[[1],[2],[3]]'")
    p.add_argument("--y", help="liste JSON de valeurs, ex. '[1,8,27]'")
    p.add_argument("--chat", action="store_true", help="mode conversationnel multi-tours")
    p.add_argument("--tui", action="store_true",
                   help="interface terminal OpenTUI (defaut quand aucune "
                        "question n'est donnee)")
    p.add_argument("--console", action="store_true",
                   help="console texte brute, sans interface")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--notify", action="store_true",
                   help="pousse le resultat vers l'ile Coucou (Mochi), "
                        "si elle tourne (pont local, Windows)")
    p.add_argument("--selfplay", type=int, metavar="N", default=0,
                   help="self-play : N defis auto-generes, valides exactement, "
                        "succes reintegres (MLT + dataset)")
    p.add_argument("--niveau", type=int, default=1, choices=(1, 2),
                   help="niveau des defis self-play (1 = calcul, 2 = + enigmes)")
    p.add_argument("--superviser", action="store_true",
                   help="l'API tourne SOUS SURVEILLANCE (relancee apres un "
                        "plantage, avec delai croissant) pendant que "
                        "l'interface s'y connecte")
    p.add_argument("--api", action="store_true",
                   help="avec --superviser : API seule, sans interface "
                        "(l'ancien « python main.py --api »)")
    p.add_argument("--demon", action="store_true",
                   help="avec --superviser : detache le processus et rend la "
                        "main (API seule, journal logs/aura-demon.log)")
    args = p.parse_args(argv)

    if args.selfplay:
        from . import selfplay
        import logging as _l
        _l.basicConfig(level=_l.INFO)
        print(json.dumps(selfplay.session(args.selfplay, niveau=args.niveau),
                         ensure_ascii=False, indent=2))
        return 0

    # SUPERVISION (ancien `main.py` racine) : le lanceur s'occupe de l'API et
    # de l'interface ensemble, donc il passe AVANT le choix d'interface.
    if args.superviser:
        from . import superviseur
        if args.demon:
            # Un TUI detache n'aurait aucun terminal : --demon vaut --api.
            return superviseur.lancer_demon()
        return superviseur.lancer(interface=not args.api)

    # INTERFACE PAR DEFAUT : le TUI. Tout ce qui demande un chemin TEXTE
    # (question, --x/--y, --demo, --chat, --console) garde la console :
    # une question ponctuelle ne doit pas ouvrir un ecran.
    chemin_texte = bool(args.question or args.x or args.y or args.demo
                        or args.chat or args.console)
    if args.tui or not chemin_texte:
        from . import tui as _tui
        code = _tui.lancer()
        if code != _tui.INDISPONIBLE:
            return code
        # Interface non installable : on ne bloque pas, on continue en texte.
        print("[tui] repli sur la console texte.", file=sys.stderr)

    ia = RontoGPT()
    # --console (ou --chat) SANS question ouvre la conversation : c'est le
    # sens de l'option. Avec une question, elle reste une question ponctuelle.
    if args.chat or (args.console and not (args.question or args.x or args.y)):
        return _chat(ia)
    if args.demo:
        return _demo(ia)
    if not args.question and not (args.x and args.y):
        # Aucune question et pas d'interface installee : on ouvre la
        # CONVERSATION. La demo X3 est un test de plomberie, pas un accueil —
        # un utilisateur qui tape `rontogpt` sans argument veut parler au
        # modele, pas voir une formule mathematique.
        return _chat(ia)

    X = json.loads(args.x) if args.x else None
    y = json.loads(args.y) if args.y else None
    reponse = ia.executer(args.question, X, y)
    print(reponse)
    if args.notify:
        from . import coucou
        resultat = coucou.termine(reponse)
        print(f"[ile Coucou] {resultat.get('raison', '?')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
