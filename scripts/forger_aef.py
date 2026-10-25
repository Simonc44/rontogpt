"""Supercompilateur : forge le fichier monolithique .aef (RontoGPT Executable Format) v2.

Format binaire (little-endian) :

  SECTION HEADER (148 octets, 16 champs)
    magic          4s   b"AURA"
    version        3B   majeur, mineur, correctif
    reserved       B    (alignement)
    build          8s   hash court du commit git
    flags          I    bit 0 : poids compressee (0 = GGUF brut, chargement direct)
    cfg_ctx        I    n_ctx compile
    cfg_batch      I    n_batch compile
    cfg_threads    I    n_threads compile
    taille_config  I    octets de la section config (JSON compresse LZMA)
    taille_code    I    octets de la section code (ZIP du paquet aura + tests)
    taille_gguf    Q    octets du GGUF brut (copie exacte, byte-for-byte)
    sha256_config  32s  integrite config
    sha256_code    32s  integrite code
    sha256_gguf    32s  integrite poids (le boot refuse tout fichier altere)

  SECTION CONFIG  (JSON compresse LZMA : routeur, niveau 0, reglages)
  SECTION CODE    (archive ZIP du paquet `aura/` : le systeme lui-meme)
  SECTION GGUF    (poids bruts : extraits dans le cache avant chargement)

Pourquoi les poids ne sont PAS compresses (mesure sur ce GGUF) :
  LZMA preset 9 ne gagne que 1.6% sur les tenseurs Q4 (haute entropie),
  contre ~7 min de forge et ~2 min de decompression au boot. Le GGUF brut
  limite le temps de forge ; le kernel extrait les poids dans `.cache_aef/`
  car llama.cpp charge un chemin de fichier GGUF.

Usage :
  python scripts/forger_aef.py                    # forge aura_system.aef
  python -m aura.kernel aura_system.aef "question"  # boot + question
"""
import hashlib
import json
import lzma
import os
import struct
import sys
import tomllib
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import chiffrement                    # noqa: E402
from aura import profil_materiel as pm          # noqa: E402

MAGIC = b"AURA"
VERSION = (0, 8, 0)
HEADER_FORMAT = "<4sBBBB8sIIIIIQQ32s32s32s"   # 16 champs, 148 octets
HEADER_TAILLE = struct.calcsize(HEADER_FORMAT)


def _build_court() -> str:
    """Hash court du commit git courant (8 car.), 'unknown' hors git."""
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=RACINE,
            capture_output=True, text=True, timeout=5,
            check=True).stdout.strip()
    except Exception:
        return "unknown"

def _gguf_configure() -> Path:
    """Cerveau selectionne par l'env ou aura.toml, sans repli silencieux."""
    try:
        with (RACINE / "aura.toml").open("rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        cfg = {}
    llama_cfg = cfg.get("llama") or {}
    chemin = os.environ.get("AURA_GGUF_CHEMIN") or llama_cfg.get("chemin")
    if chemin:
        p = Path(chemin).expanduser()
        return p if p.is_absolute() else (RACINE / p).resolve()

    cerveau = (os.environ.get("AURA_GGUF") or llama_cfg.get("gguf")
               or "GEMMA").upper()
    # Source unique des noms de cerveaux : les chemins utilises par le runtime.
    from aura.llama_cerveau import _CHEMINS
    if cerveau not in _CHEMINS:
        raise ValueError(f"cerveau inconnu dans aura.toml/AURA_GGUF : {cerveau}")
    return Path(_CHEMINS[cerveau])


GGUF_DEFAUT = _gguf_configure()
SORTIE = RACINE / "aura_system.aef"


def _sha256_stream(f, taille: int) -> bytes:
    h = hashlib.sha256()
    reste = taille
    while reste > 0:
        bloc = f.read(min(1024 * 1024, reste))
        if not bloc:
            raise IOError("fichier tronque pendant le hachage")
        h.update(bloc)
        reste -= len(bloc)
    return h.digest()


def _pack_code() -> bytes:
    """Archive ZIP du paquet aura/ (le systeme complet, sans les caches)."""
    import io
    buf = io.BytesIO()
    exclusions = {"__pycache__", ".cache_routeur", ".cache_reponses.jsonl",
                  ".conversation.jsonl", ".graphe_faits.jsonl"}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for chemin in sorted((RACINE / "aura").rglob("*")):
            if chemin.is_file() and not (set(chemin.parts) & exclusions):
                z.write(chemin, chemin.relative_to(RACINE))
    return buf.getvalue()


def _quantification(nom: str) -> str:
    """Extrait le niveau GGUF du nom de fichier, sinon le signale."""
    import re
    m = re.search(r"Q\d(?:_[Kk](?:_[A-Z])?)?(?:_\d+)?", nom)
    return m.group(0).upper() if m else "inconnue"


def _cle_cerveau(gguf: Path) -> str:
    """Identifiant runtime correspondant au fichier embarque."""
    from aura.llama_cerveau import _CHEMINS
    cible = gguf.resolve()
    for cle, chemin in _CHEMINS.items():
        if Path(chemin).resolve() == cible:
            return cle
    nom = gguf.name.lower()
    if "deepseek-r1" in nom:
        return "R1_DISTILL"
    return "CUSTOM"


def _pack_config(gguf: Path | None = None,
                 profil: dict | None = None,
                 reg: dict | None = None) -> bytes:
    """Config compilee : reglages AUTO-TUNES pour la machine de forge
    (profil materiel detecte) + identite du cerveau + empreinte machine."""
    profil = profil or pm.profiler()
    reg = reg or pm.reglages_optimaux(profil)
    cfg = {
        "n_ctx": reg["n_ctx"], "n_batch": reg["n_batch"],
        "n_threads": reg["n_threads"], "n_threads_batch": reg["n_threads_batch"],
        "flash_attn": reg["flash_attn"], "type_kv": reg["type_kv"],
        "modele": gguf.name if gguf else "inconnu",
        "cle_cerveau": _cle_cerveau(gguf) if gguf else "CUSTOM",
        "taille_gguf_octets": gguf.stat().st_size if gguf else 0,
        "quantification": _quantification(gguf.name) if gguf else "inconnue",
        "niveau0_seuil": 0.85,
        "max_tokens_faits": 120, "max_tokens_explications": 320,
        "build": ".".join(map(str, VERSION)),
        "machine": pm.empreinte(profil),          # provenance (pc de forge)
        "machine_desc": f"{profil['cpu_modele']} / {profil['ram_go']} Go RAM",
        "machine_specs": {k: profil.get(k) for k in (
            "cpu_modele", "cpu_coeurs_logiques", "ram_go", "avx2", "os")},
    }
    return lzma.compress(json.dumps(cfg).encode("utf-8"), preset=6)


def forger(gguf: Path = GGUF_DEFAUT, sortie: Path = SORTIE,
           secret: str | None = None) -> Path:
    if not gguf.exists():
        raise FileNotFoundError(
            f"GGUF introuvable : {gguf} (scripts/telecharger_llama.py)")

    print("== AURA FORGE ==")
    profil = pm.profiler()
    reg = pm.reglages_optimaux(profil)        # n_ctx/n_batch de CETTE machine
    print(f"  machine : {profil['cpu_modele']} | {profil['ram_go']} Go RAM "
          f"| empreinte {pm.empreinte(profil)}")
    config = _pack_config(gguf, profil, reg)
    code = _pack_code()
    taille_gguf = gguf.stat().st_size
    print(f"  config : {len(config):,} o | code : {len(code):,} o | "
          f"poids : {taille_gguf:,} o")

    with open(gguf, "rb") as f:
        sha_gguf = _sha256_stream(f, taille_gguf)

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC, VERSION[0], VERSION[1], VERSION[2], 0,
        _build_court().encode("ascii")[:8].ljust(8, b"\x00"),
        0,                       # flags : poids brutes (chargement direct)
        reg["n_ctx"], reg["n_batch"], reg["n_threads"],
        len(config), len(code), taille_gguf,
        hashlib.sha256(config).digest(),
        hashlib.sha256(code).digest(),
        sha_gguf,
    )

    print("  ecriture monolithique...")
    with open(sortie, "wb") as out, open(gguf, "rb") as src:
        out.write(header)
        out.write(config)
        out.write(code)
        reste = taille_gguf
        while reste > 0:
            bloc = src.read(min(4 * 1024 * 1024, reste))
            if not bloc:
                raise IOError("GGUF tronque pendant la forge")
            out.write(bloc)
            reste -= len(bloc)

    taille_mo = sortie.stat().st_size / (1024 * 1024)
    print(f"== forge OK : {sortie.name} ({taille_mo:.1f} Mo) ==")

    # Signature Ed25519 automatique : si la cle privee est a cote, le .aef
    # est signe (provenance authentifiee). Un seul .sig sert au .aef clair
    # ET au .aef.chiffre : il porte le SHA-256 du payload clair commun.
    cle_privee = RACINE / "cle_privee.pem"
    if cle_privee.exists():
        from aura import signature
        cle = signature.charger_cle_privee(cle_privee)
        path_sig = signature.ecrire_signature(sortie, cle, identite="Simonc44")
        empreinte = signature.empreinte_publique(
            signature.cle_privee_vers_publique_brute(cle))
        print(f"  signature Ed25519 -> {path_sig.name} (cle {empreinte})")
    else:
        print("  (non signe : cle_privee.pem absent — generate via aura.signature)")

    # Option chiffrement : conteneur AES-256-GCM pour publication
    if secret:
        chemin_chiffre = sortie.with_suffix(".aef.enc")
        print(f"  chiffrement AES-256-GCM -> {chemin_chiffre.name} ...")
        with open(sortie, "rb") as f:
            clair = f.read()
        chemin_chiffre.write_bytes(chiffrement.chiffrer(clair, secret))
        del clair
        print(f"== chiffre OK : {chemin_chiffre.name} "
              f"({chemin_chiffre.stat().st_size / (1024 * 1024):.1f} Mo) ==")
        return chemin_chiffre
    return sortie


if __name__ == "__main__":
    gguf = Path(sys.argv[1]) if len(sys.argv) > 1 and not \
        sys.argv[1].startswith("--") else GGUF_DEFAUT
    secret = None
    if "--chiffrer" in sys.argv:
        import getpass
        secret = os.environ.get("AURA_SECRET") or getpass.getpass(
            "Secret de chiffrement (ne le partage qu'avec toi) : ")
    sys.exit(forger(gguf, secret=secret) and 0)
