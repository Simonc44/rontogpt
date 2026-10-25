"""Tests de la couche overlay ENGRAFT (`.pleo`), sans modele Engram.

Ce que ces tests verrouillent :
- le format binaire `.pleo` (magic, en-tete, taille exacte, refus des
  fichiers tronques/avec octets en trop, doublons, NaN) ;
- l'adressage des n-grammes, miroir de `llm_graph_input_ple::set_input`
  (contexte, EOS, multiplicateurs, modulo par tete, offsets globaux) ;
- l'index d'overlay : quelles positions lisent une ligne remplacee ;
- le verdict de capacite : Gemma-3 n'a pas de table Engram, et le moteur
  actuel ne peut pas appliquer un `.pleo` — le module le dit au lieu de
  simuler une injection.

Aucun test ne charge de modele ni ne modifie `llama_cerveau`.
"""
import struct

import numpy as np
import pytest

from aura import memoire_engraft as me


def _config() -> me.ConfigEngram:
    """Petite table synthetique : 4 tetes, bigramme + trigramme."""
    return me.ConfigEngram(
        eos_token_id=99,
        ngram_size=3,
        heads_per_ngram=2,
        layer_multipliers=(1, 2, 3),
        head_vocab_sizes=(5, 7, 11, 13),
        head_offsets=(0, 5, 12, 23),
    )


def _valeurs(n: int, graine: int = 0) -> np.ndarray:
    return np.random.default_rng(graine).standard_normal(
        (n, me.ROW_LEN)).astype(np.float32)


# ── 1. Format `.pleo` ───────────────────────────────────────────────────

class TestFormatPleo:
    def test_aller_retour_et_entete(self, tmp_path):
        chemin = tmp_path / "overlay.pleo"
        lignes = np.array([27, 5, 8], dtype=np.int64)
        valeurs = _valeurs(3)
        ecrit = me.ecrire_overlay(chemin, lignes, valeurs)
        assert ecrit.n == 3

        brut = chemin.read_bytes()
        assert brut[:4] == b"PLEO"
        n, dim = struct.unpack("<II", brut[4:12])
        assert (n, dim) == (3, me.ROW_LEN)
        assert len(brut) == 12 + 4 * 3 + 4 * 3 * me.ROW_LEN

        relu = me.lire_overlay(chemin)
        np.testing.assert_array_equal(relu.lignes, lignes)
        np.testing.assert_array_equal(relu.valeurs, valeurs)
        assert relu.chemin == str(chemin)

    def test_overlay_vide(self, tmp_path):
        chemin = tmp_path / "vide.pleo"
        me.ecrire_overlay(chemin, np.empty(0, dtype=np.int64),
                          np.empty((0, me.ROW_LEN), dtype=np.float32))
        relu = me.lire_overlay(chemin)
        assert relu.n == 0
        assert chemin.stat().st_size == 12

    def test_ecriture_refuse_forme_incoherente(self, tmp_path):
        with pytest.raises(ValueError):
            me.ecrire_overlay(tmp_path / "x.pleo", [1, 2],
                              np.zeros((2, me.ROW_LEN - 1), dtype=np.float32))

    def test_ecriture_refuse_doublon(self, tmp_path):
        with pytest.raises(ValueError, match="deux fois"):
            me.ecrire_overlay(tmp_path / "x.pleo", [4, 4], _valeurs(2))

    def test_ecriture_refuse_nan(self, tmp_path):
        valeurs = _valeurs(1)
        valeurs[0, 7] = np.nan
        with pytest.raises(ValueError, match="non fini"):
            me.ecrire_overlay(tmp_path / "x.pleo", [10], valeurs)

    def test_magic_invalide(self, tmp_path):
        chemin = tmp_path / "faux.pleo"
        chemin.write_bytes(b"NOPE" + struct.pack("<II", 0, me.ROW_LEN))
        with pytest.raises(ValueError, match="magic"):
            me.lire_overlay(chemin)

    def test_dim_inconnue(self, tmp_path):
        chemin = tmp_path / "dim.pleo"
        chemin.write_bytes(b"PLEO" + struct.pack("<II", 0, 128))
        with pytest.raises(ValueError, match="dim"):
            me.lire_overlay(chemin)

    def test_fichier_tronque(self, tmp_path):
        chemin = tmp_path / "tronque.pleo"
        me.ecrire_overlay(chemin, [1], _valeurs(1))
        chemin.write_bytes(chemin.read_bytes()[:-4])
        with pytest.raises(ValueError, match="tronque"):
            me.lire_overlay(chemin)

    def test_octets_en_trop(self, tmp_path):
        chemin = tmp_path / "trop.pleo"
        me.ecrire_overlay(chemin, [1], _valeurs(1))
        chemin.write_bytes(chemin.read_bytes() + b"\x00")
        with pytest.raises(ValueError, match="octets en trop"):
            me.lire_overlay(chemin)

    def test_n_enorme_refuse_sans_allocation(self, tmp_path):
        """Un en-tete qui annonce 1e9 lignes ne doit pas allouer 640 Go."""
        chemin = tmp_path / "mensonge.pleo"
        chemin.write_bytes(b"PLEO" + struct.pack("<II", 1_000_000_000,
                                                 me.ROW_LEN))
        with pytest.raises(ValueError, match="taille"):
            me.lire_overlay(chemin)

    def test_doublon_lu_refuse(self, tmp_path):
        chemin = tmp_path / "doublon.pleo"
        lignes = np.array([7, 7], dtype="<i4")
        valeurs = _valeurs(2)
        chemin.write_bytes(b"PLEO" + struct.pack("<II", 2, me.ROW_LEN)
                           + lignes.tobytes() + valeurs.tobytes())
        with pytest.raises(ValueError, match="deux fois"):
            me.lire_overlay(chemin)


# ── 2. Adressage des n-grammes ──────────────────────────────────────────

class TestAdressage:
    def test_valeurs_calculees_a_la_main(self):
        """Petit cas verifie a la main, multiplicateurs 1/2/3 compris."""
        attendu = np.array([
            [4, 3, 10, 0],      # t=0 : ctx [1, EOS, EOS]
            [0, 0, 0, 11],      # t=1 : ctx [2, 1, EOS]
            [2, 0, 4, 4],       # t=2 : ctx [3, 2, 1]
        ], dtype=np.int64)
        np.testing.assert_array_equal(
            me.adresses_locales([1, 2, 3], _config()), attendu)

    def test_offsets_globaux_par_tete(self):
        attendu = np.array([
            [4, 8, 22, 23],
            [0, 5, 12, 34],
            [2, 5, 16, 27],
        ], dtype=np.int64)
        np.testing.assert_array_equal(
            me.adresses_globales([1, 2, 3], _config()), attendu)

    def test_eos_courant_ne_coupe_pas_son_contexte(self):
        """Un EOS en position t est le token courant : il ne se coupe pas."""
        seul = me.adresses_locales([99], _config())[0]
        debut = me.adresses_locales([99, 1, 2], _config())[0]
        np.testing.assert_array_equal(seul, debut)

    def test_eos_precedent_coupe_le_contexte(self):
        avec_eos = me.adresses_locales([99, 1], _config())[1]
        sans_eos = me.adresses_locales([42, 1], _config())[1]
        assert not np.array_equal(avec_eos, sans_eos)

    def test_predecesseur_manquant_egal_debut_de_sequence(self):
        np.testing.assert_array_equal(
            me.adresses_locales([1], _config())[0],
            me.adresses_locales([1, 2, 3], _config())[0])

    def test_token_negatif_refuse(self):
        with pytest.raises(ValueError, match="negatif"):
            me.adresses_locales([-1], _config())

    def test_config_refuse_nombre_de_tetes_incoherent(self):
        with pytest.raises(ValueError, match="attendu"):
            me.ConfigEngram(
                eos_token_id=0, ngram_size=3, heads_per_ngram=2,
                layer_multipliers=(1, 2, 3),
                head_vocab_sizes=(5, 7, 11),
                head_offsets=(0, 5, 12))

    def test_config_refuse_chevauchement_de_tetes(self):
        with pytest.raises(ValueError, match="chevauchent"):
            me.ConfigEngram(
                eos_token_id=0, ngram_size=3, heads_per_ngram=2,
                layer_multipliers=(1, 2, 3),
                head_vocab_sizes=(5, 7, 11, 13),
                head_offsets=(0, 5, 12, 20))

    def test_config_refuse_ngram_trop_petit(self):
        with pytest.raises(ValueError, match="ngram_size"):
            me.ConfigEngram(
                eos_token_id=0, ngram_size=1, heads_per_ngram=2,
                layer_multipliers=(1,),
                head_vocab_sizes=(5,), head_offsets=(0,))


# ── 3. Index d'overlay et couverture ────────────────────────────────────

class TestIndexOverlay:
    def test_positions_retrouvees_meme_non_triees(self):
        overlay = me.Overlay(lignes=np.array([27, 3, 8], dtype=np.int64),
                             valeurs=_valeurs(3))
        trouvees = me.positions_dans_overlay(
            overlay, np.array([8, 3, 99, 27]))
        assert trouvees.tolist() == [2, 1, -1, 0]

    def test_overlay_vide_ne_touche_rien(self):
        overlay = me.Overlay(lignes=np.empty(0, dtype=np.int64),
                             valeurs=np.empty((0, me.ROW_LEN),
                                              dtype=np.float32))
        assert me.positions_dans_overlay(
            overlay, np.array([1, 2])).tolist() == [-1, -1]
        assert me.lignes_lues([1, 2, 3], _config(), overlay).tolist() == \
            [0, 0, 0]

    def test_positions_touchees_et_lignes_lues(self, tmp_path):
        config = _config()
        globales = me.adresses_globales([1, 2, 3], config)
        ligne = int(globales[2, 3])                 # 27, unique dans ce cas
        overlay = me.ecrire_overlay(tmp_path / "un.pleo", [ligne], _valeurs(1))
        assert me.positions_touchees(
            [1, 2, 3], config, overlay).tolist() == [False, False, True]
        assert me.lignes_lues([1, 2, 3], config, overlay).tolist() == [0, 0, 1]

    def test_index_reutilisable(self):
        config = _config()
        globales = me.adresses_globales([1, 2, 3], config)
        overlay = me.Overlay(
            lignes=np.array([int(globales[0, 0]), int(globales[1, 2])]),
            valeurs=_valeurs(2))
        index = me.indexer(overlay)
        assert me.lignes_lues([1, 2, 3], config, overlay, index).tolist() == \
            [1, 1, 0]

    def test_tokens_vides(self):
        assert me.lignes_lues([], _config(), me.fusionner([])).tolist() == []


# ── 4. Fusion d'overlays ────────────────────────────────────────────────

class TestFusion:
    def test_dedup_et_tri(self):
        v = _valeurs(1)[0]
        a = me.Overlay(lignes=np.array([9], dtype=np.int64),
                       valeurs=v.reshape(1, -1))
        b = me.Overlay(lignes=np.array([3, 9], dtype=np.int64),
                       valeurs=np.stack([v, v]))     # meme ligne 9, meme vecteur
        fusion = me.fusionner([a, b])
        assert fusion.lignes.tolist() == [3, 9]

    def test_conflit_de_vecteur_refuse(self):
        a = me.Overlay(lignes=np.array([9], dtype=np.int64),
                       valeurs=_valeurs(1, graine=1))
        b = me.Overlay(lignes=np.array([9], dtype=np.int64),
                       valeurs=_valeurs(1, graine=2))
        with pytest.raises(ValueError, match="deux vecteurs differents"):
            me.fusionner([a, b])

    def test_fusion_vide(self):
        assert me.fusionner([]).n == 0


# ── 5. Capacite Engram et garde-fou moteur ──────────────────────────────

_META_GEMMA = {
    "general.architecture": "gemma3",
    "gemma3.block_count": 26,
    "gemma3.embedding_length": 1152,
}

_META_ENGRAM = {
    "general.architecture": "qwen4exp",
    "qwen4exp.ple.eos_token_id": 99,
    "qwen4exp.ple.image_token_id": 98,
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.heads_per_ngram": 2,
    "qwen4exp.ple.layer_multipliers": [1, 2, 3],
    "qwen4exp.ple.head_vocab_sizes": [5, 7, 11, 13],
    "qwen4exp.ple.head_offsets": [0, 5, 12, 23],
}


class TestCapacite:
    def test_gemma_sans_table_engram(self):
        verdict = me.capacite_depuis_meta(_META_GEMMA)
        assert verdict.disponible is False
        assert verdict.famille == "gemma3"
        assert "aucune table Engram" in verdict.raison

    def test_table_engram_complete(self):
        verdict = me.capacite_depuis_meta(_META_ENGRAM)
        assert verdict.disponible is True
        assert verdict.config is not None
        assert verdict.config.n_heads == 4
        assert verdict.config.n_lignes_par_position == 4
        assert verdict.config.image_token_id == 98

    def test_table_incomplete(self):
        meta = dict(_META_ENGRAM)
        del meta["qwen4exp.ple.head_offsets"]
        verdict = me.capacite_depuis_meta(meta)
        assert verdict.disponible is False
        assert "incomplete" in verdict.raison

    def test_metadonnees_incoherentes(self):
        meta = dict(_META_ENGRAM)
        meta["qwen4exp.ple.head_vocab_sizes"] = [5, 7, 11]
        verdict = me.capacite_depuis_meta(meta)
        assert verdict.disponible is False
        assert "incoherentes" in verdict.raison

    def test_architecture_absente(self):
        verdict = me.capacite_depuis_meta({})
        assert verdict.disponible is False
        assert "architecture absente" in verdict.raison

    def test_gguf_absent(self, tmp_path):
        verdict = me.capacite_engram(tmp_path / "nexiste_pas.gguf")
        assert verdict.disponible is False
        assert "introuvable" in verdict.raison

    def test_moteur_actuel_ne_sait_pas_appliquer_un_pleo(self):
        ok, raison = me.verifier_moteur()
        assert ok is False
        assert raison                     # jamais un refus muet

    def test_cerveau_configure_n_a_pas_de_table_engram(self):
        from aura import llama_cerveau
        chemin = llama_cerveau.chemin_gguf()
        from pathlib import Path
        if not Path(chemin).is_file():
            pytest.skip("GGUF du cerveau absent : rien a mesurer")
        verdict = me.capacite_engram(chemin)
        assert verdict.disponible is False
        assert verdict.famille == "gemma3"
        ok, raison = me.verifier_moteur(chemin)
        assert ok is False and "aucune table Engram" in raison
