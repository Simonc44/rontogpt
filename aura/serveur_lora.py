"""Cerveau via llama-server officiel : le chemin LoRA qui FONCTIONNE.

Le hot-swap in-process (adaptateurs.py via ctypes) est casse par un bug
llama.cpp (LoRA x CPU_REPACK, access violation au decode). Le serveur
officiel llama-server (ggml-org) charge plusieurs adaptateurs au boot
et les active/desactive a la volee via POST /lora-adapters (echelles) :
c'est le mecanisme utilise par les pipelines de production.

Protocol (verifie sur b11111) :
- GET  /health          -> {"status":"ok"}
- GET  /lora-adapters   -> [{"id":0,"path":...,"scale":1.0}, ...]
- POST /lora-adapters   -> corps JSON [{"id":N,"scale":0.0|1.0}] (partiel OK)
- POST /v1/chat/completions (OpenAI-compatible)

Cycle de vie (AURA_SERVEUR=1 pour activer) :
- le serveur est DEMARRE par RontoGPT au premier besoin (port AURA_SERVEUR_PORT,
  defaut 8757), binaire serveur/llama-server.exe, base modeles/<cerveau>,
  les 4 adaptateurs serveur/aura-*.gguf montes au boot ;
- si un serveur tourne deja sur le port, il est REUTILISE (pas de double
  boot) ;
- echec de boot ou d'appel -> None (l'appelant retombe sur le chemin
  in-process), jamais bloquant.
"""
import json
import logging
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import urllib.request

from . import adaptateurs as adaptateurs_mod
from . import echantillonnage

LOG = logging.getLogger("aura.serveur_lora")

_RACINE = Path(__file__).resolve().parent.parent

# categorie routee -> id de l'adaptateur au boot du serveur
_IDS = {"math": 0, "web": 1, "code": 2, "general": 3}

_ADRESSE = os.environ.get("AURA_SERVEUR_ADRESSE", "127.0.0.1:8757")

_process: subprocess.Popen | None = None
_actif: tuple | None = None        # mix actuellement applique ((cat, scale)...)
_indispo: str | None = None        # raison d'une indisponibilite (log 1 fois)
# DERNIERS TIMINGS llama-server (mesure REELLE, pas d'estimation) :
#   prompt_n / prompt_ms   -> taille et duree du prefill (le contexte paye)
#   predicted_n / predicted_ms -> taille et duree du decode (le goulot)
# llama-server les calcule deja et RontoGPT les JETAIT : impossible de savoir
# combien de tokens on envoie vraiment, ni ce que coute chaque etape.
_TIMINGS: dict[str, float] = {}
_TIMINGS_CLES = ("prompt_n", "prompt_ms", "prompt_per_second",
                 "predicted_n", "predicted_ms", "predicted_per_second")


def derniers_timings() -> dict[str, float]:
    """Timings du DERNIER appel serveur (vide si aucun / non fournis).

    Permet a l'appelant (orchestrateur, banc, diagnostic) de lire la taille
    reelle du prompt et le debit de decode au lieu de l'estimer.
    """
    return dict(_TIMINGS)


def _oublier_timings() -> None:
    """Vide les timings au DEBUT de chaque appel : jamais de valeur perimee.

    Sans ca, un appel suivant qui ne renvoie pas de timings laisserait
    `derniers_timings()` afficher ceux de l'appel PRECEDENT, et le diagnostic
    attribuerait a une question la taille du prompt d'une autre.
    """
    _TIMINGS.clear()


def _memoriser_timings(reponse: dict) -> None:
    """Retient les timings d'une reponse /v1/chat/completions, si presents."""
    brut = reponse.get("timings")
    if not isinstance(brut, dict):
        return
    _TIMINGS.clear()
    for cle in _TIMINGS_CLES:
        valeur = brut.get(cle)
        if isinstance(valeur, (int, float)):
            _TIMINGS[cle] = float(valeur)


def _actifs() -> bool:
    """Interrupteur : AURA_SERVEUR (aura.toml [serveur] actif = true par
    defaut). AURA_SERVEUR=0 force le chemin in-process."""
    return os.environ.get("AURA_SERVEUR", "0") != "0"


def _port() -> int:
    return int(_ADRESSE.rsplit(":", 1)[1])


def _url(chemin: str) -> str:
    return f"http://{_ADRESSE}{chemin}"


def _existe_serveur(timeout: float = 1.0) -> bool:
    """Un serveur repond deja sur le port ? (reuse si oui)"""
    try:
        with urllib.request.urlopen(_url("/health"), timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def _modele_servi(timeout: float = 1.0) -> str | None:
    """Le fichier GGUF que le serveur DEJA en place declare servir.

    llama-server l'expose sur /v1/models ; selon la version la liste s'appelle
    `models` ou `data`. Illisible -> None, jamais une exception : l'appelant
    doit pouvoir ne rien conclure.
    """
    try:
        with urllib.request.urlopen(_url("/v1/models"), timeout=timeout) as r:
            charge = json.loads(r.read().decode())
    except Exception:
        return None
    liste = charge.get("models") or charge.get("data") or []
    if not isinstance(liste, list) or not liste:
        return None
    premier = liste[0]
    if not isinstance(premier, dict):
        return None
    valeur = premier.get("model") or premier.get("name") or premier.get("id")
    return str(valeur) if valeur else None


def _meme_cerveau(chemin_servi: str | None) -> bool:
    """Le modele servi est-il le cerveau configure ? (MESURE, pas nom)

    On compare les NOMBRES de l'en-tete GGUF (aura/gguf_meta.py), jamais les
    noms de fichiers : un autre QUANT DU MEME cerveau doit passer (Q6_K au
    lieu de Q4_K_M est le meme modele), et un Llama-3.2 range sous le bon nom
    doit etre refuse. Des que la comparaison est impossible (fichier absent,
    en-tete illisible), on repond True : on ne refuse pas ce qu'on n'a pas su
    mesurer.
    """
    from . import gguf_meta, llama_cerveau

    cerveau = llama_cerveau.chemin_gguf()
    if not chemin_servi or not cerveau:
        return True
    # Le serveur peut annoncer un chemin RELATIF (cwd du lancement) : on le
    # resout d'abord depuis la racine du projet.
    candidats = [Path(chemin_servi)]
    if not Path(chemin_servi).is_absolute():
        candidats.append(_RACINE / chemin_servi)
    servi = next((p for p in candidats if p.is_file()), None)
    if servi is None:
        return True
    cible = gguf_meta.architecture(cerveau)
    if not cible:
        return True
    ok, _ = gguf_meta.compatible(servi, cible)
    return ok


_conflit_signale = False


def _conflit_cerveau() -> str | None:
    """Le message a journaliser si un serveur ETRANGER occupe le port.

    Defaut reel, constate le 9 oct. 2026 : `_existe_serveur()` ne dit que
    « quelque chose repond sur le port ». Or un llama-server sert le modele
    avec lequel il a ETE LANCE — un serveur reste d'un lancement precedent
    (autre cerveau, et ici les 4 adaptateurs Llama) repond exactement comme
    le bon. RontoGPT l'adoptait donc en silence et repondait avec l'ANCIEN
    cerveau pendant que tout le reste parlait du nouveau : ecart de qualite
    invisible et inexplicable, exactement ce que le docstring de
    `_gguf_cerveau()` promettait d'eviter. Le verdict vient desormais de la
    meme mesure d'architecture que celle des adaptateurs.
    """
    if not _existe_serveur():
        return None
    if _meme_cerveau(_modele_servi()):
        return None
    from . import llama_cerveau

    return ("un serveur llama-server DEJA en place sur %s sert un autre "
            "modele que le cerveau configure (AURA_GGUF=%s, %s). Il est "
            "laisse tel quel : RontoGPT passe par le chemin in-process plutot "
            "que de repondre avec le mauvais cerveau. Arreter ce processus "
            "libere le chemin serveur (bien plus rapide)."
            % (_ADRESSE, llama_cerveau.cle_cerveau(),
               Path(llama_cerveau.chemin_gguf()).name))


_aides: dict[str, str] = {}


def _aide_binaire(bin_: Path) -> str:
    """Texte de --help du binaire (mis en cache, 1 probe au maximum)."""
    cle = str(bin_)
    if cle not in _aides:
        try:
            h = subprocess.run([str(bin_), "--help"], capture_output=True,
                               text=True, timeout=15)
            _aides[cle] = h.stdout + h.stderr
        except Exception:
            _aides[cle] = ""
    return _aides[cle]


def _vulkan() -> Path | None:
    """Build officiel win-vulkan de llama.cpp, si present et non desactive.

    Installable par scripts/installer_vulkan.ps1 (zip bXXXX officiel).
    Mesure sur ce PC (scripts/bench_gpu.py, Llama-3.2-1B Q4_K_M, 256 tok) :
        CPU     : decode 18,6-19,0 tok/s | prefill  66-73 tok/s
        VULKAN  : decode 20,4-20,6 tok/s | prefill 216-227 tok/s (x3,1)
    Le prefill est ce qui compte : chaque passe du multi-pass re-traite son
    prompt. AURA_GPU=0 force le chemin CPU.
    """
    if os.environ.get("AURA_GPU") == "0":
        return None
    exe = _RACINE / "serveur" / "vulkan" / "llama-server.exe"
    if exe.is_file() and (exe.parent / "ggml-vulkan.dll").is_file():
        return exe
    return None


def _binaire() -> Path | None:
    """Le binaire llama-server : AURA_SERVEUR_BIN, sinon serveur/llama-server.exe
    (recherche la version recente telechargee et les extractions datees)."""
    env = os.environ.get("AURA_SERVEUR_BIN")
    if env and Path(env).is_file():
        return Path(env)
    vulkan = _vulkan()
    if vulkan is not None:
        return vulkan
    dossier = _RACINE / "serveur"
    if not dossier.is_dir():
        return None
    # Preferer le plus recent (b11111 > b7400) : llama-server.exe direct ou
    # dans un sous-dossier. Le tri mtime seul se trompe ici (serveur/b7400/
    # extraite APRES la racine donc mtime plus jeune) — or ce b7400 est un
    # ancien build qui plante avec les 4 LoRA (GGML_ASSERT au 1er token, cf.
    # scripts/diag_serveur_crash.py) et ignore --spec-type. Signal de
    # version fiable : la presence de --spec-type dans le --help.
    candidats = sorted(dossier.rglob("llama-server.exe"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidats:
        return None
    for c in candidats:
        if "--spec-type" in _aide_binaire(c):
            return c
    return candidats[0]


def _gguf_cerveau() -> Path | None:
    """Le GGUF du cerveau — SOURCE UNIQUE : llama_cerveau.chemin_gguf().

    Ce module avait sa propre liste de noms de fichiers (Llama-3.2-1B...).
    Deux listes = deux cerveaux possibles pour un meme lancement, et un
    oubli ne produit pas d'erreur : il produit un serveur qui tourne sur
    l'ANCIEN modele pendant que le reste d'RontoGPT parle au nouveau. Le
    resultat serait un ecart de qualite invisible et inexplicable.
    """
    from . import llama_cerveau

    chemin = llama_cerveau.chemin_gguf()
    return Path(chemin) if chemin and Path(chemin).is_file() else None


def _adaptateurs_boot() -> list[Path]:
    """Adaptateurs du dossier serveur, dans l'ORDRE de _IDS.

    L'ORDRE compte : llama-server numerote les adaptateurs dans l'ordre
    des arguments --lora, donc activer("math") n'agit sur le bon
    fichier que si l'ordre de boot est celui de _IDS. Un `sorted()` ici
    (ordre alphabetique : code, general, math, web) decalait tous les
    identifiants en silence — les echelles s'appliquaient au mauvais
    adaptateur. On enumere donc _IDS, la meme source que activer_mixte.

    Depuis le 10 oct. 2026 le convertisseur llama.cpp ecrit lora_a/lora_b
    dans la convention attendue par le serveur : la transposition
    (anciens `aura-*-T.gguf`, cf. scripts/transpose_lora.py) n'est plus
    necessaire. Les deux noms restent acceptes pour ne rien casser.
    """
    d = _RACINE / "serveur"
    trouves: list[Path] = []
    for cat in _IDS:                    # math, web, code, general
        for nom in (f"aura-{cat}.gguf", f"aura-{cat}-T.gguf"):
            p = d / nom
            if p.is_file():
                trouves.append(p)
                break
    return trouves


_spec_supporte: dict[str, bool] = {}


def _spec_type_ok(bin_: Path) -> bool:
    """Le binaire accepte-t-il --spec-type (speculative decoding n-gram) ?

    Le dossier sert/ peut contenir plusieurs versions (b7400 sans spec,
    b11111 avec) : on sonde --help une fois par binaire pour ne pas faire
    crasher un ancien. AURA_SERVEUR_SPEC=0 coupe tout (AURA_SPEC ne
    concerne QUE le chemin in-process, qui lui ralentit — cf.
    llama_cerveau._draft_model).

    Mesure (scripts/mesure_lookup.py, build 11111, LoRA x4, ctx 4096 ET
    8192, derive controlee A-B-C-A : `off` reproduit a +/-1-2 %) :

        variante        texte repete      plan / liste / code (structure)
        off             10,1-11,8 tok/s   10,4-11,7 tok/s (reference)
        ngram-cache     14,5-15,3 (+35%)  6,3-10,1  (** -16 a -42 % **)
        ngram-mod n8    27,9-30,4 (+170%) identique a off (0 brouillon)

    ngram-cache BROUILLONNE sur le texte structure et n'accepte RIEN
    (1,7 % / 0 % / 0 % pour 60 / 24 / 88 tokens brouillons) : c'est du cout
    pur, et le plan de la passe 1 comme le code de potcode tombent dedans.
    ngram-mod avec n-min 8 ne brouillonne que sur du texte vraiment repete
    (acceptation 100 %) ; ailleurs il ne propose AUCUN token, donc son
    calcul est identique a `off` et la perte est nulle par construction.

    Ancien banc (scripts/bench_spec_serveur.py : CPU seul, ctx 1536, UN
    seul prompt "neuf" de ~60 tokens) concluait "ngram-cache neutre sur
    texte neuf" : c'etait un faux neutre — le banc ne contenait aucun
    texte structure. Le prompt-lookup (-lcs/-lcd) n'existe PAS dans ce
    build (aucune implementation `lookup`, options parsees puis ignorees).
    """
    if os.environ.get("AURA_SERVEUR_SPEC", "1") == "0":
        return False
    cle = str(bin_)
    if cle not in _spec_supporte:
        aide = _aide_binaire(bin_)
        _spec_supporte[cle] = ("--spec-type" in aide and "ngram-mod" in aide)
        LOG.info("[serveur_lora] speculative decoding (ngram-mod) : %s",
                 "actif" if _spec_supporte[cle] else "non supporte par ce binaire")
    return _spec_supporte[cle]


def _n_ctx() -> int:
    """Fenetre de contexte du cerveau, EXACTEMENT la meme que le chemin
    in-process (miroir de llama_cerveau._reglages_materiel) :
      1. AURA_TUNING -> auto-tuning LOCAL de la machine ;
      2. la config compilee du .aef (AURA_CONFIG) ;
      3. sinon l'auto-tuning de la machine (profil_materiel).
    Une valeur figee plus petite ferait deborder en mode serveur tout prompt
    dimensionne pour la fenetre du cerveau (« exceed context window »)."""
    from . import profil_materiel as pm

    if os.environ.get("AURA_TUNING"):
        try:
            return int(pm.reglages_optimaux()["n_ctx"])
        except Exception as e:
            LOG.debug("[serveur] auto-tuning local indisponible : %s", e)
    try:
        import json as _json
        cfg = _json.loads(os.environ.get("AURA_CONFIG", "{}"))
        if cfg.get("n_ctx"):
            return int(cfg["n_ctx"])
    except Exception as e:
        LOG.debug("[serveur] n_ctx du .aef illisible : %s", e)
    try:
        return int(pm.reglages_optimaux()["n_ctx"])
    except Exception:
        return 4096


def _mlock() -> bool:
    """Les poids restent-ils VERROUILLES en RAM ? (AURA_MLOCK=0 coupe)

    mlock empeche Windows de paginer/compresser les 1,1 Go de poids : le
    serveur ne repaie pas de lecture disque apres une veille ou une pression
    memoire, et le premier token d'une instance deja vivante reste immediat.
    En revanche, mlock NE RACCOURCIT PAS le chargement initial — un boot a
    froid lit toujours le GGUF. Le « boot instantane » vient de garder le
    serveur vivant (mode demon), pas du verrou lui-meme.
    """
    return os.environ.get("AURA_MLOCK", "1") != "0"


def _args_mlock(aide: str) -> list[str]:
    """Options de verrouillage RAM acceptees par CE binaire.

    b11111 remplace `--mlock` par `-lm/--load-mode` (`mlock`, `mmap+mlock`).
    On ne pose que ce que le binaire declare dans --help : une option
    inconnue ferait echouer le boot et on perdrait le chemin serveur.
    `mmap+mlock` garde le mapping (demarrage rapide, pages partagees) tout
    en verrouillant les pages touchees.
    """
    if not _mlock():
        return []
    if "--load-mode" in aide:
        return ["--load-mode", "mmap+mlock"]
    if "--mlock" in aide:
        return ["--mlock"]
    return []


def _demarrer() -> bool:
    """Demarre le serveur avec le cerveau + les adaptateurs au boot."""
    global _process, _indispo
    bin_ = _binaire()
    gguf = _gguf_cerveau()
    if bin_ is None or gguf is None:
        _indispo = "binaire llama-server ou GGUF introuvable"
        LOG.info("[serveur_lora] %s -> chemin in-process", _indispo)
        return False
    # -c <n_ctx du cerveau> : la MEME fenetre que le chemin in-process
    # (histrique 4 tours + few-shot + budget de generation sans
    # « exceed context window »).
    # --no-warmup : le warmup avec 4 LoRA montes consomme ~10x la memoire
    # de contexte d'un boot simple (GGML_ASSERT sinon) — pas double.
    # THREADS : profil_materiel calcule le nombre optimal pour CETTE machine
    # (coeurs logiques, +26% vs physiques mesure). Le « 4 » fige laissait le
    # chemin SERVEUR — le chemin ACTIF (serveur.actif = true) — sur 4 threads,
    # alors que le chemin in-process utilisait tous les coeurs : deux
    # empreintes CPU differentes selon le chemin pris, sans le dire.
    # AURA_SERVEUR_THREADS pour forcer une valeur : sur un hybride P+E, plus
    # de threads n'est pas toujours mieux, la mesure tranche.
    try:
        from . import profil_materiel as _pm
        _threads_auto = int(_pm.reglages_optimaux().get("n_threads") or 4)
    except Exception:
        _threads_auto = max(os.cpu_count() or 4, 1)
    _threads = int(os.environ.get("AURA_SERVEUR_THREADS", _threads_auto))
    cmd = [str(bin_), "-m", str(gguf), "--port", str(_port()),
           "--host", _ADRESSE.rsplit(":", 1)[0], "-c", str(_n_ctx()),
           "-t", str(_threads), "--no-warmup"]
    LOG.info("[serveur_lora] threads : %d", _threads)
    if _spec_type_ok(bin_):
        # brouillon n-gram (cache) extrait de l'historique : +170 % sur du
        # texte REPETE, et cout NUL ailleurs (0 brouillon propose, donc
        # calcul identique a `off`). Ne pas revenir a ngram-cache : mesure
        # 16-42 % plus LENT sur le plan de la passe 1, les listes et le
        # code — cf. le tableau dans _spec_type_ok et la mesure
        # scripts/mesure_lookup.py (serveur/mesure_lookup_*.json).
        cmd += ["--spec-type", "ngram-mod"]
        # n-min bas = chercher des correspondances plus courtes, donc plus
        # souvent. Mesure : a 8, 0 brouillon sur plan/liste/code (inertie
        # parfaite) et 100 % d'acceptation sur le texte repete.
        if "--spec-ngram-mod-n-min" in _aide_binaire(bin_):
            cmd += ["--spec-ngram-mod-n-min", "8"]
    # PARITE AVEC LE CHEMIN IN-PROCESS : profil_materiel demande flash
    # attention et un cache KV q8_0 (RAM du cache divisee par 2). Le serveur
    # ignorait les deux, donc la MEME config donnait deux empreintes memoire
    # selon le chemin pris. Chaque option n'est posee que si le binaire la
    # connait (--help) : sinon il ne demarrerait pas et on perdrait le chemin
    # serveur entier (repli in-process silencieux).
    try:
        from . import profil_materiel as _pm2
        _reg = _pm2.reglages_optimaux()
    except Exception:
        _reg = {}
    _aide = _aide_binaire(bin_)
    if _reg.get("flash_attn") and ("--flash-attn" in _aide or "-fa," in _aide):
        # « on » EXPLICITE : -fa prend une valeur optionnelle et avalerait
        # sinon l'argument suivant (constate : -fa --cache-type-k q8_0 ->
        # « unknown value for --flash-attn: '--cache-type-k' », serveur mort,
        # repli in-process silencieux).
        cmd += ["-fa", "on"]
    if _reg.get("type_kv") == 8 and "--cache-type-k" in _aide:
        cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    _opts_mlock = _args_mlock(_aide)
    if _opts_mlock:
        cmd += _opts_mlock
        LOG.info("[serveur_lora] poids verrouilles en RAM (%s)",
                 " ".join(_opts_mlock))
    adaptateurs = _adaptateurs_boot()
    # GARDE-FOU D'ARCHITECTURE (aura/adaptateurs.py) : un adaptateur LoRA
    # est un DELTA DE POIDS pour une architecture precise. Monte sur une
    # autre base, il ne specialise rien et produit du bruit — sans erreur
    # visible. On refuse les adaptateurs plutot que de servir du bruit.
    if adaptateurs:
        ok_cerveau, pourquoi = adaptateurs_mod.verifier_cerveau(gguf)
        if not ok_cerveau:
            LOG.warning("[serveur_lora] %s", pourquoi)
            adaptateurs = []
    # Build GPU (Vulkan) : tous les layers sur l'iGPU (repli CPU automatique
    # par llama.cpp si aucun device — jamais de plantage).
    if _vulkan() == bin_:
        cmd += ["-ngl", "99"]
    if "--spec-type" in _aide_binaire(bin_):
        # build recent : --lora accepte une liste separee par des virgules
        # (les occurrences repetees declenchent un warning deprecation).
        if adaptateurs:
            cmd += ["--lora", ",".join(str(p) for p in adaptateurs)]
    else:
        for p in adaptateurs:
            cmd += ["--lora", str(p)]
    try:
        journal = open(_RACINE / "serveur" / "autogere.log",
                       "ab", buffering=0)
        _process = subprocess.Popen(
            cmd, stdout=journal, stderr=journal, cwd=str(_RACINE),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        _indispo = f"boot impossible : {e}"
        LOG.info("[serveur_lora] %s", _indispo)
        return False
    # attente de la sante (boot du 1B : ~5-15 s)
    for _ in range(60):
        if _process.poll() is not None:
            _indispo = f"serveur mort (code {_process.returncode})"
            LOG.info("[serveur_lora] %s", _indispo)
            return False
        if _existe_serveur():
            LOG.info("[serveur_lora] pret sur %s (%d adaptateurs)",
                     _ADRESSE, len(_adaptateurs_boot()))
            return True
        time.sleep(1.0)
    _indispo = "timeout de boot (20 s + 40 s d'attente)"
    LOG.info("[serveur_lora] %s", _indispo)
    return False


# Verrou de BOOT : le rechauffage en tache de fond (api) et le
# premier appel chat() peuvent arriver en meme temps — sans ce verrou,
# deux llama-server partiraient sur le meme port.
_verrou_boot = threading.Lock()
_rechauffe = False


def _assurer() -> bool:
    """Serveur operationnel (demarre si besoin)."""
    global _conflit_signale
    if not _actifs():
        return False
    if _existe_serveur():
        conflit = _conflit_cerveau()
        if conflit is None:
            return True
        # Signale UNE fois : le message est long, et il se repeterait a
        # chaque question. On ne pose PAS `_indispo` : si l'utilisateur
        # arrete le serveur etranger, le chemin serveur se retablit de
        # lui-meme au coup suivant.
        if not _conflit_signale:
            _conflit_signale = True
            LOG.warning("[serveur_lora] %s", conflit)
        return False
    if _indispo is not None:
        return False
    with _verrou_boot:
        if _existe_serveur():          # un autre thread a deja boot
            return True
        if _indispo is not None:
            return False
        return _demarrer()


def rechauffer() -> None:
    """Rechauffage en tache de fond (thread daemon unique).

    Demarre llama-server AU LANCEMENT (GUI / api) : la premiere question
    n'attend plus les 5-15 s de boot. No-op si le serveur est coupe
    (AURA_SERVEUR=0), deja pret, deja rechauffe, ou deja indisponible
    (binaire absent -> repli in-process automatique partout).
    """
    global _rechauffe
    if not _actifs() or _rechauffe or _indispo is not None:
        return
    if _existe_serveur():              # serveur externe deja la
        _rechauffe = True
        return
    _rechauffe = True
    LOG.info("[serveur_lora] rechauffage en tache de fond...")
    threading.Thread(target=_assurer, daemon=True,
                     name="aura-rechauffage").start()


# ── hot-swap ────────────────────────────────────────────────────────────

# Verrou global : serialise hot-swap (POST /lora-adapters) et generation
# (/v1/chat/completions). Un swap pendant une generation modifie les
# pointeurs de tenseurs en memoire C de llama.cpp sous ses pieds —
# resultat : sortie corrompue (melange de specialites) ou crash.
_verrou = threading.Lock()


# ── ROUTAGE MoE : TOP-2 EXPERTS ─────────────────────────────────────────
#
# CE QUE LE RUNTIME PERMET, ET CE QU'IL NE PERMET PAS
# ---------------------------------------------------
# Un adaptateur LoRA est un DELTA DE POIDS LINEAIRE : llama-server applique
# UN vecteur d'echelles pour toute la generation (`POST /lora-adapters`).
# Il n'y a pas de porte par token dans ce format — un routage Top-2 PAR
# TOKEN suppose des poids de routeur DANS le modele (architecture MoE),
# pas un overlay LoRA. Le grain reellement disponible est donc la REQUETE.
#
# Ancien comportement : tout expert dont la probabilite depassait 0,15
# etait active avec son score brut. Trois consequences :
#   - jusqu'a 4 deltas superposes a faible echelle : la specialite
#     dominante etait diluee ;
#   - la somme des echelles tombait sous 1 des qu'un expert etait ecarte,
#     donc la reponse perdait de la specialisation sans que rien ne le dise ;
#   - « deux vrais sujets » (math + web) n'etait pas distingue de « bruit »
#     (trois scores a 0,2).
# Ici : les DEUX meilleurs seulement, renormalises pour que le budget de
# specialisation reste constant.
TOP_K = 2
SEUIL_EXPERT = 0.15


def routeur_top2(scores: dict[str, float] | None,
                 seuil: float = SEUIL_EXPERT) -> dict[str, float]:
    """Top-2 des experts, poids renormalises (somme = 1).

    - seules les categories SERVIES par le serveur (`_IDS`) sont retenues :
      un score « pot » ou « rag » n'a pas d'adaptateur monte ;
    - un score sous `seuil`, non numerique ou non fini est ecarte — un
      expert a 2 % n'est pas un expert, c'est du bruit ;
    - egalite : l'ordre de boot de `_IDS` tranche (deterministe, donc le
      meme mix pour la meme question d'un run a l'autre) ;
    - dict vide / aucun candidat -> {} : le repli appartient a l'appelant
      (en general la categorie routee, cf. `activer_top2`).
    """
    if not scores:
        return {}
    candidats: list[tuple[int, str, float]] = []
    for cat, i in _IDS.items():
        try:
            score = float(scores.get(cat, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score >= seuil:
            candidats.append((i, cat, score))
    if not candidats:
        return {}
    candidats.sort(key=lambda t: (-t[2], t[0]))
    retenus = candidats[:TOP_K]
    total = sum(score for _, _, score in retenus)
    return {cat: round(score / total, 3) for _, cat, score in retenus}


def activer_top2(scores: dict[str, float] | None,
                 defaut: str | None = None) -> bool:
    """Applique le mix Top-2 du routeur ; repli `defaut` si aucun score.

    `defaut` est la categorie deduite par l'orchestrateur (experts ou
    regex code) : elle sert UNIQUEMENT quand le routeur n'a rien donne
    (methode « mots-cles », score vide), jamais a forcer un troisieme
    expert dans un Top-2 deja rempli.
    """
    poids = routeur_top2(scores)
    if not poids:
        if not defaut:
            return False
        poids = {defaut: 1.0}
    return activer_mixte(poids)

def activer(categorie: str) -> bool:
    """Monte l'adaptateur de la categorie seul (API simple = mix 100/0)."""
    return activer_mixte({categorie: 1.0} if categorie else {})


def activer_mixte(poids: dict[str, float]) -> bool:
    """Monte PLUSIEURS adaptateurs simultanement (multi-LoRA a taille
    constante) : poids = {categorie: echelle 0..1}.

    Ex : {"math": 0.7, "web": 0.3} — une question qui touche deux sujets
    profite des deux specialites, sans recharger quoi que ce soit (le
    serveur ne change que les echelles, cout ~0).
    Renvoie True si le serveur a accepte.

    SECURITE THREADS : le verrou serialise POST /lora-adapters vs
    /v1/chat/completions — sans lui, une generation en cours pendant le
    swap des pointeurs de tenseurs en memoire C peut produire une sortie
    corrompue (melange de deux specialites) ou un crash du serveur.
    """
    global _actif
    cle = tuple(sorted((k, round(v, 3)) for k, v in poids.items() if v > 0))
    if cle == _actif:
        return True        # deja applique : zero appel reseau
    with _verrou:          # aucun chat() pendant la modification des echelles
        if cle == _actif:  # double-check : un autre thread a deja applique
            return True
        if not _assurer():
            return False
        corps = [{"id": i, "scale": round(float(poids.get(cat, 0.0)), 3)}
                 for cat, i in _IDS.items()]
        try:
            req = urllib.request.Request(
                _url("/lora-adapters"), method="POST",
                data=json.dumps(corps).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                ok = json.loads(r.read().decode()).get("success", False)
            if not ok:
                return False
            _actif = cle
            LOG.info("[serveur_lora] mix actif : %s",
                     dict(cle) if cle else "(cerveau brut)")
            return True
        except Exception as e:
            LOG.info("[serveur_lora] mix impossible : %s", e)
            return False


def chat(messages: list[dict], max_tokens: int = 256,
         publier: Callable[[str], None] | None = None,
         interrompre: Callable[[], bool] | None = None) -> str | None:
    """Generation via /v1/chat/completions (OpenAI-compatible).

    None = echec -> l'appelant retombe sur le chemin in-process.
    Le verrou garantit qu'aucun hot-swap ne modifie les echelles des
    adaptateurs PENDANT la generation (sinon : sortie corrompue).

    publier (optionnel) : si fourni, on passe en flux SSE (stream: true)
    et chaque delta est pousse a ce callback au fur et a mesure — la GUI
    ecrit la reponse pendant qu'elle arrive du serveur.

    interrompre (optionnel, flux seulement) : appele a chaque delta ; s'il
    renvoie True, on CESSE de lire le flux et on renvoie ce qui a ete
    recu. C'est le point d'accroche du Self-Correction Stream
    (aura/auto_correction.py) : une erreur exacte reperee DANS LA PENSEE
    coupe la generation avant qu'elle ne redige une reponse batie dessus.
    Le serveur, lui, continue d'emettre jusqu'a EOS dans le vide : c'est
    un cout CPU residuel, pas un cout de correction — et il est tres
    inferieur a la redaction complete qu'on evite.
    """
    if not _assurer():
        return None
    # PROFILS PARTAGES (aura/echantillonnage.py) : les valeurs ne sont plus
    # ecrites ici. Elles l'etaient, et le serveur — le chemin ACTIF — avait
    # deja diverge de l'in-process : aucune penalite, pas de disjoncteur
    # anti-boucle, pas de profil de section. Le DRY de llama.cpp (qui
    # penalise les SEQUENCES repetitives longues) reste propre au serveur :
    # c'est le seul ecart, et il est declare comme tel dans le profil.
    corps: dict = {"messages": messages, "max_tokens": max_tokens,
                   **echantillonnage.pour_serveur(echantillonnage.NORMAL)}
    try:
        with _verrou:  # vs activer_mixte() : jamais les deux a la fois
            if publier is None:
                brut = _poster(corps)
                if brut is None:
                    return None
                # DISJONCTEUR ANTI-BOUCLE, PARITE AVEC L'IN-PROCESS : une
                # boucle nait -> UNE relance (penalite 1,25 / temp 0,85),
                # et si elle derape encore, on livre le texte COUPE avant la
                # repetition. La boucle ne doit jamais sortir vers
                # l'utilisateur, quel que soit le chemin pris.
                if not _en_boucle(brut):
                    return brut
                LOG.warning("[serveur_lora] boucle detectee -> relance")
                alt = _poster({**corps,
                               **echantillonnage.pour_serveur(
                                   echantillonnage.RELANCE)})
                if alt and not _en_boucle(alt):
                    return alt
                return _couper_boucle(brut) or brut
            return _chat_stream(corps, publier, interrompre)
    except Exception as e:
        LOG.info("[serveur_lora] chat impossible : %s", e)
        return None


def _poster(corps: dict) -> str | None:
    """Un aller-retour /v1/chat/completions (hors flux). None = echec."""
    _oublier_timings()
    req = urllib.request.Request(
        _url("/v1/chat/completions"), method="POST",
        data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        sortie = json.loads(r.read().decode())
    _memoriser_timings(sortie)
    return (sortie.get("choices") or [{}])[0] \
        .get("message", {}).get("content", "").strip() or None


def _en_boucle(texte: str) -> bool:
    """Detection de boucle — IMPLEMENTATION UNIQUE (llama_cerveau).

    Import paresseux : `llama_cerveau` importe ce module, donc un import de
    module a module ferait un cycle.
    """
    from . import llama_cerveau
    return llama_cerveau._en_boucle(texte)


def _couper_boucle(texte: str) -> str:
    """Tronque avant la repetition — IMPLEMENTATION UNIQUE (llama_cerveau)."""
    from . import llama_cerveau
    return llama_cerveau._couper_boucle(texte)


def _chat_stream(corps: dict, publier: Callable[[str], None],
                 interrompre: Callable[[], bool] | None = None,
                 ) -> str | None:
    """Reponse en flux SSE depuis llama-server (data: {...} par delta).

    Chaque delta est publie immediatement ; on renvoie le texte complet
    reconstitue (l'appelant en a besoin pour separer_reflexion/garde-fous).
    `interrompre()` coupe la lecture (cf. chat()).
    Appele avec _verrou deja pris (hot-swap interdit pendant le flux).
    """
    corps = dict(corps, stream=True)
    _oublier_timings()
    req = urllib.request.Request(
        _url("/v1/chat/completions"), method="POST",
        data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"})
    morceaux: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for ligne in r:                       # SSE : "data: {json}\n\n"
            texte = ligne.decode("utf-8", "replace").strip()
            if not texte.startswith("data:"):
                continue
            donnees = texte[5:].strip()
            if donnees == "[DONE]":
                break
            try:
                ev = json.loads(donnees)
            except ValueError:
                continue
            _memoriser_timings(ev)   # le DERNIER ev du flux porte les timings
            choix = ev.get("choices") or [{}]
            delta = (choix[0].get("delta") or {}).get("content") or ""
            if delta:
                morceaux.append(delta)
                try:
                    publier(delta)
                except Exception:             # abonne GUI parti : jamais bloquant
                    LOG.debug("[serveur_lora] abonne flux en erreur")
                if interrompre is not None:
                    try:
                        if interrompre():
                            LOG.info("[serveur_lora] flux coupe a la demande "
                                     "(%d signes) — auto-correction",
                                     sum(len(m) for m in morceaux))
                            break
                    except Exception:      # jamais bloquant
                        pass
    return _sans_boucle_ou_relance(corps, "".join(morceaux).strip())


def _sans_boucle_ou_relance(corps: dict, texte: str) -> str | None:
    """Disjoncteur anti-boucle POUR LE FLUX — parite avec le chemin non streame.

    Defaut reel comblé ici : `chat()` protegeait la boucle quand on ne
    streamait pas, et pas quand on streamait. Or c'est en streaming que
    passent la GUI et l'API — donc le chemin le PLUS visible etait le moins
    protege. Une reponse qui partait en repetition s'affichait en boucle a
    l'ecran.

    Le texte deja publie ne peut pas etre « repris » ; la relance produit
    donc une reponse NEUVE, publiee a son tour. Le client voit la reponse
    hative, puis la version corrigee — jamais la boucle seule.
    """
    from . import llama_cerveau

    if not texte or not llama_cerveau._en_boucle(texte):
        return texte or None
    LOG.warning("[serveur_lora] boucle detectee dans le flux -> relance")
    alt = _poster({**corps, **echantillonnage.pour_serveur(
        echantillonnage.RELANCE)})
    if alt and not llama_cerveau._en_boucle(alt):
        return alt.strip() or None
    return llama_cerveau._couper_boucle(texte) or texte


def disponible() -> bool:
    """Diagnostic externe : le chemin serveur est-il utilisable ?"""
    return _actifs() and _assurer()


def statut() -> dict:
    """Diagnostic : adresse, process, mix actif, etat du serveur."""
    return {"actives": _actifs(), "adresse": _ADRESSE,
            "serveur_vivant": _existe_serveur(),
            "adaptateur": dict(_actif) if _actif else None,
            "routage_top_k": TOP_K, "mlock": _mlock(),
            "indisponible": _indispo,
            "timings": derniers_timings()}


def arreter() -> None:
    """Arrete le serveur lance par RontoGPT (laisse vivre un serveur externe)."""
    global _process, _actif
    if _process is not None:
        _process.terminate()
        _process = None
    _actif = None
