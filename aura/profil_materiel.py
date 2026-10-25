"""Profil materiel : regarde SUR QUOI le systeme tourne, puis auto-optimise.

Trois usages :
1. DETECTION : cpu (modele, coeurs, AVX2), RAM totale, OS, disk libre.
2. AUTO-TUNING : calcule n_threads / n_ctx / n_batch optimaux pour LA machine
   (regles mesurees sur le PC de reference : tous les coeurs = +26%, batch
   768 et contexte borne a 4096 tokens). Sur une autre machine, les reglages suivent.
3. EMPREINTE : un hash court de la machine, stocke dans le .aef forge.
   Au boot, le kernel compare : config compilee = config validee sur CE pc.

Aucune dependance externe : lecture directe de l'API Windows / /proc.
"""
import hashlib
import json
import logging
import os
import platform
import sys

LOG = logging.getLogger(__name__)


def _cpu_windows() -> dict:
    """Modele + coeurs physiques via l'API Windows (zero dependance).

    Le controle de plateforme est ecrit avec `sys.platform`, et NON `os.name` :
    mypy ne LIT meme pas ce qui suit quand la plateforme analysee n'est pas
    Windows, alors que les stubs Linux n'ont ni `ctypes.windll` ni `winreg` —
    les `attr-defined` qui en decoulaient rougissaient le job `mypy` de la CI
    (11 erreurs mesurees le 16 oct. 2026) pendant que le meme `mypy aura/`
    rendait 0 sur un poste Windows. Le comportement ne change pas : `profiler()`
    n'appelle cette fonction que sous Windows (`_cpu_linux` sinon).
    """
    if sys.platform != "win32":
        return {"modele": platform.processor() or "CPU", "coeurs_physiques": 0}
    info = {"modele": platform.processor() or "CPU", "coeurs_physiques": 0}
    try:
        import ctypes

        class _SYSTEM_INFO(ctypes.Structure):
            _fields_ = [("wProcessorArchitecture", ctypes.c_ushort),
                        ("wReserved", ctypes.c_ushort),
                        ("dwPageSize", ctypes.c_ulong),
                        ("lpMinimumApplicationAddress", ctypes.c_void_p),
                        ("lpMaximumApplicationAddress", ctypes.c_void_p),
                        ("dwActiveProcessorMask", ctypes.c_size_t),
                        ("dwNumberOfProcessors", ctypes.c_ulong),
                        ("dwProcessorType", ctypes.c_ulong),
                        ("dwAllocationGranularity", ctypes.c_ulong),
                        ("wProcessorLevel", ctypes.c_ushort),
                        ("wProcessorRevision", ctypes.c_ushort)]

        si = _SYSTEM_INFO()
        ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
        info["coeurs_logiques"] = si.dwNumberOfProcessors

        # nom lisible du CPU : registre (fiable, rapide)
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            info["modele"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()

        # coeurs physiques : GetLogicalProcessorInformation est lourd ; on
        # approxime par la regle habituelle hyperthreading 2:1 si Le look.
        # Pas critique : n_threads optimal = logiques dans tous les cas mesures.
    except Exception:
        info["coeurs_logiques"] = os.cpu_count() or 4
    info.setdefault("coeurs_logiques", os.cpu_count() or 4)
    return info


def _cpu_linux() -> dict:
    info = {"modele": platform.processor() or "CPU",
            "coeurs_logiques": os.cpu_count() or 4, "coeurs_physiques": 0}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for ligne in f:
                if ligne.startswith("model name"):
                    info["modele"] = ligne.split(":", 1)[1].strip()
                    break
        # coeurs physiques = ids uniques (physical id, core id)
        ids = set()
        physique = coeur = None
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for ligne in f:
                if ":" not in ligne:
                    continue
                cle, val = (p.strip() for p in ligne.split(":", 1))
                if cle == "physical id":
                    physique = val
                elif cle == "core id":
                    coeur = val
                    ids.add((physique, coeur))
        info["coeurs_physiques"] = len(ids) or info["coeurs_logiques"]
    except Exception as e:
        # On retombe sur les coeurs logiques : moins juste, mais jamais faux
        # au point de faire deborder la memoire.
        LOG.debug("coeurs physiques non detailles : %s", e)
    return info


def ram_octets() -> int:
    """RAM totale en octets (zero dependance)."""
    # `sys.platform` (que mypy comprend) et non `os.name` : ce bloc n'est lu
    # que sous Windows, ou `ctypes.windll` existe.
    if sys.platform == "win32":
        # GlobalMemoryStatusEx via ctypes
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        except Exception:
            return 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ligne in f:
                if ligne.startswith("MemTotal"):
                    return int(ligne.split()[1]) * 1024
    except Exception as e:
        # 0 = "RAM inconnue" pour tout le module (marge memoire prudente).
        LOG.debug("RAM non lue dans /proc/meminfo : %s", e)
    return 0


def disk_libre_octets(path: str | None = None) -> int:
    try:
        import shutil
        return shutil.disk_usage(path or os.getcwd()).free
    except Exception:
        return 0


def _a_avx2() -> bool:
    """Support AVX2 (accélération llama.cpp) — via cpuid sur Windows."""
    # `sys.platform` : sous Linux, mypy ne lit donc pas le `winreg` qui suit.
    if sys.platform != "win32":
        try:
            return "avx2" in open("/proc/cpuinfo", encoding="utf-8",
                                  errors="replace").read()
        except Exception:
            return False
    try:
        # llama-cpp-python embarque le sien ; on sonde le CPU directement.

        # fonction CPUID inaccessible depuis Python pur ; heuristique :
        # tout CPU >= Haswell (2013) a AVX2. On lit la famille du registre.
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            ident = winreg.QueryValueEx(k, "Identifier")[0]
        famille = int(ident.split("Family")[1].split(" ")[0], 16) \
            if "Family" in ident else 6
        modele = int(ident.split("Model ")[-1].split(" ")[0], 16) \
            if "Model" in ident else 0
        # Haswell = family 6, model >= 0x3C
        return famille > 6 or (famille == 6 and modele >= 0x3C)
    except Exception:
        return True   # hypothese prudente : les GGUF Q4 exigent AVX2 quand meme


def profiler() -> dict:
    """Profil complet de la machine (dict JSON-serialisable)."""
    cpu = _cpu_windows() if os.name == "nt" else _cpu_linux()
    ram = ram_octets()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_modele": cpu["modele"],
        "cpu_coeurs_logiques": cpu["coeurs_logiques"],
        "cpu_coeurs_physiques": cpu.get("coeurs_physiques") or 0,
        "avx2": _a_avx2(),
        "ram_go": round(ram / 1024**3, 2),
        "disk_libre_go": round(disk_libre_octets() / 1024**3, 2),
        "python": platform.python_version(),
    }


# ── AUTO-TUNING : les regles mesurees, appliquees a n'importe quelle machine ──

def reglages_optimaux(profil: dict | None = None) -> dict:
    """Calcule n_threads / n_ctx / n_batch pour LA machine detectee.

    Regles (issues des benchmarks reels de ce projet) :
      - n_threads = coeurs LOGIQUES (+26% vs physiques, mesure)
      - RAM >= 6 Go  : ctx 4096 + batch 768 (marge pour prompts usuels,
                        moins de prefill et de RAM reservee ; KV q8_0)
        RAM <  6 Go  : ctx 1024 + batch 512 (boot garantit, latence egale)
      - RAM <  3 Go ou GGUF 3B+ : le kernel doit refuser le boot (place).
    """
    profil = profil or profiler()
    logiques = max(profil.get("cpu_coeurs_logiques") or 4, 1)
    ram_go = profil.get("ram_go") or 4

    threads = logiques
    # FENETRE : le cerveau par defaut (Gemma 3 1B) accepte jusqu'a 32 K tokens ;
    # la figer a 4096 bornait inutilement la memoire de conversation. Le cache
    # KV q8_0 d'un modele de ~1 Md de parametres coute de l'ordre de 15 Ko par
    # token : 8192 tokens ~ 120 Mo, 32768 ~ 0,5 Go. Seul le prefill du prompt
    # REELLEMENT envoye coute du temps, pas la taille de la fenetre.
    if ram_go >= 8:
        # Gemma 3 1B sait traiter 32 K : avec le KV q8_0 cela reste de
        # l'ordre d'un demi-Go. C'est le meilleur levier de memoire utile sur
        # une machine normale, sans rendre les petits prompts plus lents.
        ctx, batch = 32768, 768
    elif ram_go >= 6:
        ctx, batch = 8192, 768
    else:
        ctx, batch = 1024, 512
    # Surcharge explicite (AURA_N_CTX=16384, 32768...) : l'utilisateur sait
    # ce que sa machine encaisse. Borne : 512 a 131072, jamais un negatif.
    brut = os.environ.get("AURA_N_CTX", "").strip()
    if brut.isdigit():
        ctx = max(512, min(int(brut), 131072))

    return {
        "n_threads": threads,
        "n_threads_batch": threads,
        "n_ctx": ctx,
        "n_batch": batch,
        "flash_attn": True,       # impl CPU llama.cpp, jamais nuisible
        "type_kv": 8,             # q8_0 : RAM cache / 2
        "ram_go": ram_go,
        "avx2": profil.get("avx2", True),
    }


def tours_fenetre(n_ctx: int | None = None) -> int:
    """Nombre d'ECHANGES (question + reponse) gardes bruts dans la fenetre.

    Agrandir `n_ctx` ne change RIEN a ce que le cerveau se rappelle tant que la
    fenetre glissante reste a 4 echanges : c'est elle qui decide de ce qui est
    reenvoye. Regle : 1 echange par 1024 tokens de fenetre, entre 4 (l'ancien
    comportement, fenetre 4096) et 12. AURA_FENETRE force une valeur (2 a 32).
    """
    brut = os.environ.get("AURA_FENETRE", "").strip()
    if brut.isdigit():
        return max(2, min(int(brut), 32))
    try:
        n = int(n_ctx) if n_ctx else int(reglages_optimaux()["n_ctx"])
    except Exception:  # noqa: BLE001 — jamais bloquant au demarrage
        return 4
    return max(4, min(n // 1024, 12))


def empreinte(profil: dict | None = None) -> str:
    """Hash court et stable de la machine (verifie a l'ouverture du .aef)."""
    profil = profil or profiler()
    cle = json.dumps(profil, sort_keys=True).encode("utf-8")
    return hashlib.sha256(cle).hexdigest()[:12]


def verifier_place(gguf_octets: int, marge: float = 0.45) -> tuple[bool, str]:
    """Le GGUF tient-il en RAM avec la marge systeme ? (refus avant crash OOM)"""
    ram = ram_octets()
    if not ram:
        return True, "RAM inconnue : verif ignoree"
    besoin = gguf_octets * 2.2          # poids + KV cache + activations
    if besoin <= ram * marge:
        return True, f"OK ({besoin / 1024**3:.1f} Go necessaires / {ram / 1024**3:.1f} Go)"
    return False, (f"RAM insuffisante : {besoin / 1024**3:.1f} Go necessaires, "
                   f"{ram / 1024**3:.1f} Go presentes")


if __name__ == "__main__":
    p = profiler()
    print(json.dumps(p, indent=2, ensure_ascii=False))
    print("reglages :", json.dumps(reglages_optimaux(p)))
    print("empreinte :", empreinte(p))
