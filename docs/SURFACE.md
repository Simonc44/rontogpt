# Surface du paquet `aura`

82 modules : 74 ATTEIGNABLE, 4 SCRIPTS, 3 TESTS, 1 ORPHELIN.

Points d'entree suivis : __init__, __main__, api, coucou, gui, kernel, superviseur, tui, winmain.

## Orphelins (1) : personne ne les importe : archivables

| module | lignes | tests | scripts |
|---|---|---|---|
| `instantane` | 132 | - | - |

## Tests seulement (3) : non atteignables du produit ; les deplacer casse ces tests

| module | lignes | tests | scripts |
|---|---|---|---|
| `memoire_engraft` | 487 | test_memoire_engraft.py | - |
| `rlss` | 159 | test_ameliorations.py | - |
| `verrou` | 75 | test_ameliorations.py | - |

## Scripts seulement (4) : utilises par scripts/, pas par le produit

| module | lignes | tests | scripts |
|---|---|---|---|
| `chiffrement` | 83 | test_tourner_secret.py | forger_aef.py, tourner_secret.py |
| `concepts` | 426 | test_ameliorations_v4.py, test_concepts.py | extraire_concepts.py, recalibrer_concepts.py |
| `intercepteur_entropie` | 39 | test_intercepteur_entropie.py | mesure_marge.py |
| `signature` | 214 | test_signature_boot.py | forger_aef.py |

## Fichiers de cache / memoire nommes par plusieurs modules

| fichier | modules |
|---|---|
| `.cache_corrections.jsonl` | autoamelioration, rlss, verrou |
| `.graphe_faits.jsonl` | graphe_faits, rlss |

## Tous les fichiers de cache / memoire

| fichier | module(s) |
|---|---|
| `.cache_corrections.jsonl` | autoamelioration, rlss, verrou |
| `.cache_reponses.jsonl` | filtre_instantane |
| `.conversation.jsonl` | memoire_conversation |
| `.graphe_faits.jsonl` | graphe_faits, rlss |
| `feedback.jsonl` | api |
| `index.jsonl` | documents |
| `vecteurs.npy` | documents |
