"""Classificateur flou : petit MLP (numpy) sur vecteurs MiniLM.

Historique et choix
-------------------
Ce brique donnait le 2e avis du routeur : le routeur historique classe sur des
n-grams de caracteres (TF-IDF), il reagit aux FORMULES ; celui-ci classe sur le
SENS (vecteurs 384-d) et reagit aux PARAPHRASES.

Le classifieur etait un LinearSVC. Mesure du 8 oct. 2026
(`scripts/experiment_routeur.py`, 150 exemples reels, validation croisee
10 plis x 3 graines) :

    LogisticRegression 0,878 +/- 0,008   (ancien regime < 300 exemples)
    LinearSVC          0,864 +/- 0,014   (ancien regime >= 300 exemples)
    MLP ReLU           0,900 +/- 0,011   <- NOUVEAU defaut (+2,2 pts)
    MLP GLU            0,889 +/- 0,003

Le classifieur lineaire est donc remplace par `aura.petit_mlp.ClassifieurMLP`
(ReLU), mais **DESACTIVE PAR DEFAUT**. Raison honnete : sur la tache de
routage isolee, le MLP gagne ; en revanche, branche dans le pipeline complet,
il modifie le NOMBRE d'etapes du mode PoT (1 appel au lieu de 4 sur le puzzle
d'age du banc `tests/test_mecanismes.py`). Ce n'est pas demontre comme une
amelioration de bout en bout — donc on ne l'impose pas. LinearSVC reste le
defaut ; le MLP s'active explicitement apres validation.

Interrupteurs
-------------
- AURA_FLOU=0   : coupe (le routeur garde son avis TF-IDF) ;
- AURA_FLOU=mlp : active le petit MLP (mesure meilleur hors-ligne) ;
- sinon         : LinearSVC (defaut historique, comportement inchange) ;
- MiniLM absent -> probas() renvoie None (idem).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from . import embeddings, integrite
from .routeur import _probas, exemples_entrainement

LOG = logging.getLogger("aura.classificateur_flou")

# meme dossier que le cache du routeur (gitignore : aura/.cache_routeur/)
_DOSSIER = Path(__file__).resolve().parent / ".cache_routeur"
_FICHIER = _DOSSIER / "classif_flou.joblib"
_FORMAT = 2                      # bump : invalide un cache d'avant le MLP

#: Repli : le classifieur lineaire historique (regime >= 300 exemples).
_H_MLP = 48

_modele: Any = None              # ClassifieurMLP ou estimateur sklearn
_backend: str | None = None      # "mlp" ou "svm" (backend reellement actif)
_classes: list[str] = []
_cle: str | None = None          # empreinte du savoir actuellement appris
_indispo = False                 # echec (pas de MiniLM) : on n'insiste pas


def _backend_voulu() -> str:
    """Backend demande : "mlp" si AURA_FLOU=mlp, sinon "svm" (defaut).

    Le defaut reste le LinearSVC : le MLP, meilleur sur la tache isolee,
    change le comportement du pipeline PoT et n'a pas ete valide de bout en
    bout (cf. docstring du module).
    """
    return "mlp" if os.environ.get("AURA_FLOU", "1").strip().lower() == "mlp" \
        else "svm"


def actif() -> bool:
    """AURA_FLOU != 0 et encodeur MiniLM disponible."""
    return os.environ.get("AURA_FLOU", "1") != "0" and embeddings.actif()


def _empreinte(questions: list[str]) -> str:
    return hashlib.sha1("\n".join(questions).encode("utf-8")).hexdigest()[:16]


def _entrainer_mlp(X, labels):
    """Petit MLP (numpy). None si numpy/MLP indisponible."""
    try:
        from .petit_mlp import ClassifieurMLP
        clf = ClassifieurMLP(X.shape[1], len(set(labels)), H=_H_MLP,
                             glu=False, graine=0)
        clf.fit(X, labels)
        return clf
    except Exception as e:  # noqa: BLE001  # repli sur SVM, jamais bloquant
        LOG.info("[flou] MLP indisponible (%s) -> LinearSVC", e)
        return None


def _entrainer_svm(X, labels):
    """Classifieur lineaire historique (LinearSVC)."""
    from sklearn.svm import LinearSVC
    return LinearSVC(C=1.0, dual="auto", random_state=0).fit(X, labels)


def _entrainer() -> bool:
    """Entraine le classifieur du savoir partage du routeur."""
    global _modele, _backend, _classes, _cle, _indispo
    questions, labels = exemples_entrainement()
    X = embeddings.encoder(questions)
    if X is None:
        _indispo = True
        return False

    backend = _backend_voulu()
    modele = _entrainer_mlp(X, labels) if backend == "mlp" else None
    if modele is None:
        backend = "svm"
        modele = _entrainer_svm(X, labels)
    _modele, _backend = modele, backend
    _brutes = getattr(modele, "classes_", None)      # MLP = list, SVM = ndarray
    _classes = list(_brutes) if _brutes is not None else sorted(set(labels))
    _cle = _empreinte(questions)
    try:
        import joblib
        _DOSSIER.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": _FORMAT, "cle": _cle, "backend": backend,
                     "classes": _classes, "modele": modele, "n": len(questions)},
                    _FICHIER)
    except Exception:  # noqa: BLE001  # cache best-effort
        LOG.debug("[flou] cache joblib non ecrit")
    LOG.info("[flou] %s entraine sur vecteurs MiniLM (%d exemples)",
             type(modele).__name__, len(questions))
    return True


def _charger() -> bool:
    """Modele en memoire (cache disque si le savoir n'a pas bouge)."""
    global _modele, _backend, _classes, _cle, _indispo
    if _modele is not None:
        return True
    if _indispo or not actif():
        return False
    questions, _labels = exemples_entrainement()
    empreinte = _empreinte(questions)
    try:
        if _FICHIER.exists():
            # Sceau AVANT deserialisation : un .joblib est du pickle.
            donnees = integrite.charger_joblib(_FICHIER)
            if (donnees.get("cle") == empreinte
                    and donnees.get("format") == _FORMAT
                    and donnees.get("backend") == _backend_voulu()):
                _modele = donnees["modele"]
                _backend = donnees["backend"]
                _classes = list(donnees.get("classes", []))
                _cle = empreinte
                LOG.info("[flou] modele charge depuis le cache (%s)", _backend)
                return True
    except integrite.IntegriteRompue as e:
        LOG.error("[flou] modele REFUSE (sceau rompu) : %s -> "
                  "re-entrainement", e)
    except Exception:  # noqa: BLE001
        pass
    return _entrainer()


def probas(question: str) -> dict[str, float] | None:
    """Probabilites {math, web, general} sur vecteurs MiniLM, ou None.

    None = coupe (AURA_FLOU=0), modele indisponible, ou echec d'encodage :
    l'appelant (routeur) reste sur son avis TF-IDF historique.
    """
    if not (question or "").strip() or not actif():
        return None
    if not _charger():
        return None
    vec = embeddings.encoder1(question)
    if vec is None:
        return None
    try:
        if _backend == "mlp" and hasattr(_modele, "probas1"):
            p = _modele.probas1(vec)
            return {c: float(v) for c, v in zip(_classes, p)}
        brutes = _probas(_modele, vec[None, :])
        return {str(k): float(v) for k, v in brutes.items()}
    except Exception as e:  # noqa: BLE001
        LOG.info("[flou] score impossible : %s", e)
        return None
