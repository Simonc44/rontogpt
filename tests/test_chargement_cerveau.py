"""Chargement du cerveau : UN seul modele, meme appele par plusieurs threads,
et liberation explicite (RAM rendue, rechargement possible).

Sans le verrou de _charger(), huit appels simultanes construisaient jusqu'a
huit Llama (~600 Mo chacun) : la RAM explosait au premier usage concurrent
(API + agents).
"""
import threading

import pytest

from aura import chargement_cerveau as lc


class _FauxLlama:
    """Remplace llama_cpp.Llama : compte les constructions et les fermetures."""

    construits = 0
    fermes = 0
    _verrou = threading.Lock()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.draft_model = None
        self.ferme = False
        with _FauxLlama._verrou:
            _FauxLlama.construits += 1

    def close(self):
        self.ferme = True
        with _FauxLlama._verrou:
            _FauxLlama.fermes += 1


@pytest.fixture
def cerveau_isole(monkeypatch, tmp_path):
    faux_gguf = tmp_path / "cerveau.gguf"
    faux_gguf.write_bytes(b"gguf")
    _FauxLlama.construits = 0
    _FauxLlama.fermes = 0
    # Les valeurs par defaut de _charger viennent de llama_cerveau
    from aura import llama_cerveau
    monkeypatch.setattr(llama_cerveau, "_CHEMIN_GGUF", str(faux_gguf))
    monkeypatch.setattr(llama_cerveau, "_reglages_materiel", lambda: {
        "n_ctx": 8192, "n_batch": 768, "n_threads": 4,
        "n_threads_batch": 4, "flash_attn": True, "type_kv": 8,
        "n_gpu_layers": 0, "use_mlock": False,
    })
    # Le _llm global et _charger sont dans chargement_cerveau
    monkeypatch.setattr("llama_cpp.Llama", _FauxLlama)
    monkeypatch.setattr(lc, "_llm", None)
    yield
    lc._llm = None


def test_chargement_concurrent_construit_un_seul_modele(cerveau_isole):
    rendus: list = []
    erreurs: list = []
    verrou = threading.Lock()
    depart = threading.Barrier(8)

    def charger():
        try:
            depart.wait(timeout=10)
            modele = lc._charger()
        except Exception as e:                       # remonte dans le test
            with verrou:
                erreurs.append(e)
            return
        with verrou:
            rendus.append(modele)

    fils = [threading.Thread(target=charger) for _ in range(8)]
    for f in fils:
        f.start()
    for f in fils:
        f.join(timeout=30)

    assert not erreurs, f"chargement en echec : {erreurs[0]!r}"
    assert len(rendus) == 8, "les huit threads doivent obtenir un modele"
    assert _FauxLlama.construits == 1, (
        f"un seul modele doit etre construit, obtenu {_FauxLlama.construits}")
    assert len({id(m) for m in rendus}) == 1, "tous les appels rendent le MEME objet"


def test_decharger_ferme_le_modele_et_rend_la_ram(cerveau_isole):
    modele = lc._charger()
    lc.decharger()

    assert modele.ferme is True, "le modele doit etre ferme"
    assert _FauxLlama.fermes == 1
    assert lc._llm is None, "le cerveau doit etre oublie apres dechargement"

    # Et le rechargement doit fonctionner (2e modele, propre).
    recharg = lc._charger()
    assert recharg is not None and recharg is not modele
    assert _FauxLlama.construits == 2
    lc.decharger()


def test_decharger_sans_cerveau_ne_leve_pas(cerveau_isole):
    lc.decharger()          # jamais charge : no-op
    assert lc._llm is None


def test_une_lecture_ratee_ne_devient_pas_definitive(monkeypatch):
    """Un en-tete illisible UNE fois ne doit pas condamner tout le processus.

    `gguf_meta.famille()` ne leve jamais : elle rend '' quand la lecture
    echoue. La memoriser comme un succes rendait la famille — et donc les
    balises de fin de tour — definitivement fausses : le cerveau s'appelait
    « inconnu » et Gemma recevait les balises generiques.

    Mesure du 16 oct. 2026 : sous charge, la premiere lecture a echoue une
    fois et 5 tests sur 1151 sont tombes, TOUS avec la meme valeur
    `'inconnu-1b'` — la trace d'un vide resservi a tout le monde.
    """
    from aura import llama_cerveau as lc
    from aura import gguf_meta

    monkeypatch.setattr(lc, "_FAMILLE_CACHE", None)
    monkeypatch.setattr(gguf_meta, "famille", lambda _chemin: "")

    assert lc.famille() == ""
    assert lc._FAMILLE_CACHE is None, "un echec ne se memorise pas"

    # La lecture suivante doit VRAIMENT relire, et aboutir.
    monkeypatch.setattr(gguf_meta, "famille", lambda _chemin: "gemma3")
    assert lc.famille() == "gemma3"
    assert lc._FAMILLE_CACHE == "gemma3", "le succes, lui, se memorise"

    # Et une fois memorise, on ne relit plus le fichier (cout d'un os.open).
    monkeypatch.setattr(gguf_meta, "famille",
                        lambda _chemin: pytest.fail("relu inutilement"))
    assert lc.famille() == "gemma3"
