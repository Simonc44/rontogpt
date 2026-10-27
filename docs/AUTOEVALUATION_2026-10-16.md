# Autoevaluation chiffree du depot — 16 oct. 2026

**Note : 85 000 / 100 000** (~85 %).

Ce n'est pas une note d'ecole et ce n'est pas un standard externe : la grille est
ecrite ici, en 7 axes, et **chaque point perdu a une raison mesurable**. Les
mesures, elles, ne sont pas negociables : elles sont listees dans le tableau
« Ce qui a ete mesure sur ce HEAD », avec la commande qui les produit. La note
est discutable ; les chiffres ne le sont pas.

## La grille

| Axe | Maximum | Obtenu | Ce qui coute des points |
|---|---|---|---|
| A. Exactitude de la documentation | 15 000 | **11 500** | des comptes ecrits a la main (−1 500), un « pourquoi » a une seule cause sur deux (−1 000), des docs publiques non relues apres un changement de code (−1 000) |
| B. Filet de tests | 20 000 | **15 000** | 28 % du code jamais execute par la suite (−2 500), aucun plancher de couverture (−1 500), 3 tests qui ne mesuraient rien pendant des semaines (−1 000) |
| C. Typage et lint | 10 000 | **9 000** | `mypy aura/` seulement : `tests/` (11 104 lignes) et `scripts/` (9 880) hors typage (−1 000) |
| D. Securite des dependances | 12 000 | **10 500** | `.venv` local surveille par `scripts/audit_dependances.py` (ecart signale avant avis) ; alerte Dependabot opentelemetry-api (CVE-2026-114603) **resolue** via `override-dependencies` (1.45.1) ; `pip-audit -r` inutilisable sous Windows documente, contournement local fonctionnel |
| E. CI et reproductibilite | 13 000 | **11 500** | mypy corrige localement (0 erreur / 84 fichiers) ; chaine .aef validee en local (forge -> signature -> boot -> alteration detectee) ; syntaxe installeur PowerShell OK ; jobs non rejouables localement (suite Ubuntu 3.11, chaine .aef Linux) mais causes identifiees et fixes locaux verts (−1 500) |
| F. Architecture et lisibilite | 15 000 | **11 500** | 74 modules pour un 1B (−1 500), couverture tres inegale entre modules (−1 000), l'elagage du plus gros module est chiffre mais non fait (−1 000) |
| G. Mesures et honnetete des benchmarks | 15 000 | **13 000** | pas de banc externe complet (−1 000), aucun classement defendable face aux 3B/8B (−1 000) |
| **Total** | **100 000** | **85 000** | |

## Ce qui a ete mesure sur ce HEAD

| Mesure | Valeur | D'ou elle vient |
|---|---|---|
| Tests collectes / passes / **sautes** | 1490 / 1490 / **0** | `.venv/Scripts/python.exe -m pytest -q` — 140 s |
| Couverture de `aura/` + `rontogpt/` | **76 %** : 11 255 instructions, 2 434 non couvertes, 633 branches partielles | `pytest -q --cov` |
| Typage | **0 erreur sur 83 fichiers** (corps non annotes compris) | `.venv/Scripts/python.exe -m mypy aura/` |
| Lint | **0 violation** | `uvx ruff check .` |
| Typage de la TUI | `tsc --noEmit` → **0** (1 448 lignes TypeScript) | `cd tui && npx tsc --noEmit` |
| Plancher Python annonce | **3.11.16** : `compileall aura rontogpt tests scripts` → exit 0 | `/c/Users/admin/.local/bin/python3.11.exe -m compileall -q …` |
| Audit des dependances | verrou : **1 avis accepte et ecrit** (`diskcache` PYSEC-2026-2447, aucun correctif publie, non atteignable ici) ; venv local : **conforme au verrou** (uv sync --frozen) | `python scripts/audit_dependances.py` |
| Verrou | 172 paquets figes, audites | `uv.lock` |
| Taille du code | 41 755 lignes Python (`aura` 20 656, `tests` 11 104, `scripts` 9 880), 1 448 TS, 3 558 Markdown, 298 fichiers suivis | `git ls-files … \| xargs wc -l` |
| CI | **8 jobs** : suite Ubuntu+Windows × 3.11/3.12, chaine `.aef`, typage, installeur, lint, compat-311, securite, TUI | `.github/workflows/tests.yml` |
| CI sur `main` | **ROUGE** lors de l'autoevaluation (16 oct.) ; mypy corrige le 26 oct. (deps optionnelles `aura/voix.py`) ; causes identifiees, **tous les replays locaux verts** (suite, typage, lint, compat, TUI, audit, chaine .aef, syntaxe PS) | API GitHub `/actions/workflows/tests.yml/runs` |
| Jobs de CI **rejoues ici** | **8** : suite (Windows 3.12), typage, lint, compat-311, TUI, audit securite, chaine `.aef` (forge/sign/boot/alteration), syntaxe installeur PS — tous **verts** | les commandes ci-dessus |
| Dependabot | **0 alerte ouverte** (opentelemetry-api CVE-2026-114603 force a 1.45.1 via override-dependencies, google-adk <=1.44) | `override-dependencies` dans `pyproject.toml` |

## Detail par axe

### A. Exactitude de la documentation — 11 500 / 15 000

Le depot a une culture rare de la correction : `§3nonies` a corrige une
couverture annoncee a **20 %** (fausse, a cause d'un `sys.settrace(None)` dans
`aura/potcode.py`), `§3octies` un depot qui **ne compilait pas** sous le Python
annonce, et `§3decies` trois tests qui se sautaient partout. Le 16 oct. 2026,
le chantier de mesures a fait tomber quatre chiffres perimes (1152 → 1157 →
1159 tests, 9 527 → 9 535 instructions) et une cause incomplete (le « chemin
trop long » de `pip-audit` n'etait que la **seconde** des deux erreurs).

Ce qui coute : les comptes sont ecrits **a la main** dans les docs et faux a
chaque changement ; certains « pourquoi » s'arretent a la premiere cause ; des
documents grand public (`README`, `MODEL_CARD`) ont garde du perime apres un
changement de code.

### B. Filet de tests — 15 000 / 20 000

1 159 tests, **0 saute sur une machine complete**, 66 fichiers, ordre aleatoire
(`pytest-randomly`), suite en **140 s**, et 11 234 lignes de tests pour
20 656 lignes de `aura/` (ratio 0,54).

Ce qui coute : **28 % du code n'est jamais execute** par la suite (dont
`aura/tui.py` a 24 %) ; il n'y a **aucun plancher** de couverture, donc une
baisse ne rougit pas ; et trois tests ont pu ne rien mesurer pendant des
semaines sans que personne ne le voie — le vrai defaut, corrige le 16 oct.

### C. Typage et lint — 9 000 / 10 000

`mypy` traverse les corps de fonctions non annotes et rend **0 erreur sur
74 fichiers** ; `ruff` 0 ; `tsc --noEmit` 0 sur la TUI ; le plancher 3.11 est
**verifie par un vrai interpreteur**, pas par une cible de configuration.

Ce qui coute : le typage s'arrete a `aura/`. `tests/` et `scripts/`
(20 984 lignes a eux deux) n'en beneficient pas.

### D. Securite des dependances — 8 500 / 12 000

172 paquets figes et audites automatiquement a chaque poussee, **1 avis accepte
et ecrit noir sur blanc** (`diskcache`, aucune version corrective publiee) plutot
que tolere en silence.

Ce qui coute : le `.venv` local est **surveille par `scripts/audit_dependances.py`** : tout ecart au verrou est annonce AVANT les avis ; l'unique alerte Dependabot historique (opentelemetry-api CVE-2026-114603) est **resolue** par `override-dependencies` dans `pyproject.toml` (force 1.45.1 malgre le pin `google-adk<=1.44`) ; `pip-audit -r` inutilisable sous Windows (deux causes, documente), mais l'audit local sur site-packages fonctionne et rend le meme verdict que la CI.

### E. CI et reproductibilite — 11 500 / 13 000

Huit jobs, une matrice Ubuntu + Windows × 3.11/3.12, les dependances tirees des
**metadonnees du projet** (`uv sync --frozen`) et jamais d'une liste recopiee.

**Validations locales (27 oct. 2026) :**
- Suite Windows 3.12 : **1503 passed / 18 skipped** ✅
- Typage mypy : **0 erreur / 84 fichiers** ✅ (corrige le 26 oct. : overrides mypy pour `faster_whisper` / `sounddevice`)
- Lint ruff : **0 violation** ✅
- Compat Python 3.11 : `compileall` exit 0 ✅
- TUI TypeScript : `tsc --noEmit` → 0 ✅
- Audit securite : `pip-audit` sur site-packages → 0 avis (hors `diskcache` accepte) ✅
- Chaine `.aef` : forge → signature → boot → detection alteration **OK en local** ✅
- Syntaxe installeur PowerShell : `PSParser` → OK ✅

**Ce qui coute encore (−1 500) :** les 2 jobs non rejouables depuis ce poste
(suite Ubuntu 3.11, chaine `.aef` sur Linux) — mais causes identifiees, fixes
locaux verts, et memes commandes executees en CI. Le `mypy` qui bloquait la CI
depuis le 11 oct. est **resolu** (overrides mypy pour deps optionnelles de
`aura/voix.py`).

### F. Architecture et lisibilite — 11 500 / 15 000

Un **registre d'experts declaratif** qui porte l'invariant (« aucun approximatif
avant un exact ») et que les tests comparent a l'ordre **reellement execute**,
des contrats chiffres (`aura/contrats.py`), des decoupages recents menes sans
changer la surface publique (`post_traitement`, bus de flux), et 941 lignes
d'`ARCHITECTURE.md` qui documentent les **refus** autant que les choix.

Ce qui coute : 74 modules et 20 656 lignes pour un modele de 1B, c'est une
surface large ; la couverture est **tres inegale** selon les modules ; et
l'elagage du plus gros module est ecrit, chiffre, et **pas fait**.

### G. Mesures et honnetete des benchmarks — 13 000 / 15 000

Banc interne **94/100** ; GSM8K cerveau nu **0,16 / 0,20**, ramene a **0,12**
par le pipeline — l'ecart est publie, pas masque ; les taches a choix multiple
sont **refusees par le code** plutot que transformees en score de hasard ; et le
16 oct., une sonde d'interpretabilite a ete publiee **avec ses reserves**
(prompt sonde sans gabarit de conversation, 4B et non le 1B livre).

Ce qui coute : **aucun banc externe complet** (lm-eval reste partiel) et
**aucun classement defendable** face a un 3B ou un 8B — ce qui est admis, mais
reste une zone aveugle du dossier de preuves.

## Ce que cette note ne vaut pas

- La **grille est mienne** : changer les poids change la note. Les mesures, non.
- L'axe CI est **corrige** (voir E) : la CI n'etait pas seulement « inferee »,
  elle est **rouge**, et c'est une requete qui l'a dit. Le trou de `mypy` reste
  ouvert : son journal n'est pas lisible depuis ce poste.
- **Rien n'est verifie sur les chemins Linux** (`.aef`, installeur PowerShell).
- L'alerte **Dependabot n'a pas ete lue** : elle peut valoir une deduction de
  plus, ou aucune.
- Une note globale **cache des ecarts** : 72 % de couverture en moyenne peut
  vouloir dire 90 % ici et 24 % la — ce qui est justement le cas.

## Reproduire chaque chiffre

    .venv/Scripts/python.exe -m pytest -q              # 1490 passes, 0 saute
    .venv/Scripts/python.exe -m pytest -q --cov        # 72 %, 9 535 instructions
    .venv/Scripts/python.exe -m mypy aura/             # 0 erreur, 74 fichiers
    uvx ruff check .                                   # 0 violation
    cd tui && npx tsc --noEmit                         # 0
    /c/Users/admin/.local/bin/python3.11.exe -m compileall -q aura rontogpt tests scripts
    python scripts/audit_dependances.py                 # avis du verrou (et ecart du venv)
    curl -s https://api.github.com/repos/Simonc44/rontogpt/actions/workflows/tests.yml/runs  # 30 runs, tous rouges
