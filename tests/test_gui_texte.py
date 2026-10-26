"""Rendu texte de la GUI : la seule partie verifiable sans ecran.

`aura/gui.py` est a 0 % de couverture parce qu'il s'importe devant une
fenetre. `_envelopper` n'a besoin de rien : il decide ou le texte passe a la
ligne dans les bulles, et une erreur la couperait au milieu d'un mot.
"""
import pytest

pytest.importorskip("tkinter")      # gui.py importe tkinter au chargement

from aura.gui import _envelopper    # noqa: E402  (apres le skip)

NL = chr(10)


def test_respecte_la_largeur():
    rendu = _envelopper(" ".join(["mot"] * 40), 20)
    assert all(len(ligne) <= 20 for ligne in rendu.split(NL))


def test_ne_coupe_pas_un_mot():
    assert _envelopper("algorithmes neuronaux", 10).split(NL) == [
        "algorithmes", "neuronaux"]


def test_mot_plus_long_que_la_largeur_reste_entier():
    assert _envelopper("x" * 30, 10) == "x" * 30


def test_paragraphes_conserves():
    source = NL.join(["un deux", "trois quatre"])
    assert _envelopper(source, 100) == source


def test_texte_vide():
    assert _envelopper("", 10) == ""
