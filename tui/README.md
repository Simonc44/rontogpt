# RontoGPT — interface terminal (OpenTUI)

Interface TUI du cerveau RontoGPT. Elle **n'implemente aucune logique** : elle
parle au processus Python par l'API locale (`aura/api.py`, `127.0.0.1:8760`) en
HTTP + SSE. Le cerveau, les experts, la memoire et le RAG restent donc
exactement la ou ils etaient.

```
tui/  (Bun 1.4.3 + @opentui/core 0.6.0)
   │  POST /v1/ask/stream   → morceau / final / fin
   ▼
aura/api.py  (Python, 127.0.0.1:8760)
   │
   ▼
aura/orchestrateur.py  (cerveau, experts, cache, web…)
```

## Habillage « Codex » + emblème RontoGPT

L'écran reprend la mise en page de l'interface Codex, avec l'emblème RontoGPT :

- en-tête `>_ RontoGPT (vX.Y.Z)` et dossier courant dessous ;
- l'emblème (`src/logo.ts`, points braille tirés du logo officiel) très discret,
  centré, tant que le fil est vide ; il se masque sur un terminal trop petit ;
- ta question dans une bande grise précédée de `›`, la réponse précédée de `•` ;
  aucune ligne de méta sous la réponse ;
- le markdown de la réponse est rendu : `**gras**`, `*italique*`, `` `code` ``,
  liens, titres, listes à puces / numérotées, cases à cocher, citations et
  règles (les identifiants comme `rag_binaire` ou `__init__` restent intacts) ;
- attente : puce clignotante et `Réflexion (12 s • Échap pour interrompre)` ;
- barre de saisie grise de 3 lignes et deux lignes de pied de page ;
- une reponse qui contient du code ou des sources : chaque bloc ```langue prend un
  cadre arrondi qui porte son etiquette de langage, et son corps est colore par la
  grammaire quand elle est embarquee ; la ligne finale `Sources : [titre](url)`
  devient une section `🔗 Sources` — un titre et son lien par source. Un bloc
  jamais referme (le 1B recopie parfois l'echafaudage de sa consigne) est cadre
  quand meme : c'est la seule regle que le TUI ajoute a `aura/gui.py`.

Format de `src/logo.ts` : une ligne de texte par rangée, chaque cellule braille
vaut 2x4 points, et les cases vides sont de vrais espaces (pas U+2800).

## Ce que l'ecran apporte par rapport au `toi>` du CLI

| Avant (`python -m aura --chat`) | Maintenant |
|---|---|
| `toi> ` brut, une reponse collee a la suite | fil de conversation encadre, chaque tour separe, question en vert, reponse en clair |
| Aucun retour pendant l'attente (18 s de vide) | minuteur + animation d'attente (`⠹ reflexion… 6,2 s`) des le depart |
| `[0.0s]` apres coup, sans dire qui a repondu | ligne de meta : `RontoGPT · niveau0-instantane · 0.0 s · experts : instantane` |
| `clear` / `aide` partaient au cerveau comme de vraies questions (et declenchaient une recherche web !) | commandes reconnues, avec ou sans `/` |
| Pas de moyen d'oublier la conversation sans tuer le processus | `/oubli` (nouvelle route `POST /v1/reset`) |
| Une reponse avec du code arrivait brute (``` et `Sources : [titre](url)` compris) | blocs de code encadres, etiquetes et colores, section `🔗 Sources` titree (`src/affichage.ts`) |
| Etat du cerveau invisible | en-tete vivant : version, profil, cerveau en memoire, llama.cpp, GPU, budget |

## Commandes

| Command | Effect |
|---|---|
| *(text)* | send the question |
| `/help` or `/aide` | this screen |
| `/clear` or `/effacer` | clear the **displayed** thread (the brain memory is intact) |
| `/forget` or `/oubli` | the brain forgets the conversation (`POST /v1/reset`) |
| `/memory` | what the brain keeps in mind, with context fill and compaction mode (`GET /v1/memoire`) |
| `/init` | load the brain now (`POST /v1/init`) |
| `/mcp` | Coucou MCP bridge and project MCP servers |
| `/plan` | toggle plan mode (plan first, no execution) |
| `/compact` | drop the raw window, keep durable facts (`POST /v1/compact`) |
| `/cloud` | Gemini: show/set the key, on/off, fallback mode, long-form writing (`GET/POST /v1/cloud`) |
| `/status` or `/etat` | version, profile, brain, llama.cpp, GPU, budget |
| `/quit`, `exit`, `Ctrl+C` | exit (Ctrl+C asks for confirmation) |
| `Esc` | interrupt the current answer (AbortController) |

## Installation de A à Z avec uv

Prérequis : [uv](https://docs.astral.sh/uv/), Git et Node.js (Node ne sert qu'à
poser Bun et OpenTUI en local via `npm install`).

```powershell
# 1. récupérer le dépôt
git clone <URL-DU-DEPOT> aura-1b
cd aura-1b

# 2. environnement Python + dépendances (crée .venv, lit pyproject.toml / uv.lock)
uv sync

# 3. modèles locaux (hors git, voir scripts/)
uv run python scripts/telecharger_llama.py
uv run python scripts/telecharger_minilm.py

# 4. interface terminal : Bun + @opentui/core (une seule fois)
npm --prefix tui install

# 5. lancer : démarre l'API dans le même processus et ouvre le TUI
uv run rontogpt
```

À l'étape 5, si `tui/node_modules` manque, `rontogpt` lance lui-même
`npm install` : l'étape 4 est donc facultative.

Variantes utiles :

```powershell
uv run python -m aura.api                 # API seule (autre terminal)
uv run rontogpt "Combien font 6 * 7 ?"    # une question, sans écran
uv run rontogpt --console                 # ancienne console texte
uv run pytest                             # tests Python
```

## Lancer

PowerShell (recommande) :

```powershell
powershell -ExecutionPolicy Bypass -File tui\lancer_tui.ps1
```

| Option | Effet |
|---|---|
| *(aucune)* | ouvre l'interface |
| `-Smoke` | joue les deux verifications automatiques et sort avec leur code |
| `-SansApi` | ne demarre pas l'API, dit seulement si elle repond |
| `-ApiUrl` | autre adresse de cerveau (defaut `http://127.0.0.1:8760`) |

Alternative, depuis l'explorateur : double-clic sur
`tui\lancer_tui.bat` (meme travail, sans options).

Les deux lanceurs verifient `/v1/health` ; si l'API ne repond pas, ils la
demarrent (`python -m aura.api`, fenetre reduite), attendent jusqu'a 40 s, puis
ouvrent le TUI. A la main :

```cmd
.venv\Scripts\python.exe -m aura.api          rem terminal 1
cd tui && node_modules\bun\bin\bun.exe src\index.ts   rem terminal 2
```

Prerequis, deja installes mais a refaire apres un `git clone` :

```cmd
cd tui && npm install
```

`npm install` pose **Bun 1.4.3 en local** dans `tui/node_modules` (aucune
installation systeme) et `@opentui/core`. OpenTUI exige Bun >= 1.3.14, ou
Node >= 26.4 avec `--experimental-ffi` : le Node 22 de la machine ne suffit
pas, d'ou le Bun local.

## Verification

```cmd
cd tui
npm run smoke                            rem client : 6 controles contre l'API reelle
npm run smoke:attente                    rem attente : 3 controles sur la ligne d'attente
npm run smoke:design                     rem design : 16 controles sur la frame peinte
node_modules\bun\bin\bun.exe src\index.ts --smoke   rem ecran complet, sans humain
```

Ou, depuis la racine du projet :

```powershell
powershell -ExecutionPolicy Bypass -File tui\lancer_tui.ps1 -Smoke
```

- `src/smoke.ts` exerce le client (sante, `/v1/status`, deux questions reelles,
  fin de flux, `/v1/reset`) et sort en code 1 au premier echec.
- `src/index.ts --smoke` monte vraiment l'ecran, joue une question, verifie
  qu'une reponse non vide a ete affichee, puis sort en 0.

Les trois scripts partagent le meme harnais (`src/smoke_commun.ts`) : compteur
et ligne `OK`/`ECHEC`, message final, code de sortie. Il vivait **recopie** dans
les trois ; il n'existe plus qu'une fois, et les messages finaux restent ceux
ci-dessus (ils sont cites plus bas).

## Ce qui est verifie, et ce qui ne l'est pas

Verifie en machine (11 oct. 2026) :

- `bun run src/smoke.ts` → **6/6 OK**, code de sortie 0 ;
- `bun run src/smoke:attente` → **3/3 OK**, code de sortie 0 : pas de compteur
  avant le premier morceau, compte reel des le premier, progression ensuite ;
- `bun run src/smoke:design` → **16/16 OK**, code de sortie 0 : le decoupage (deux
  blocs de code avec leur langage, corps repris tel quel, section de deux sources,
  repli « ligne brute » quand aucun lien n'est trouve, bloc jamais referme cadre),
  puis la frame reellement peinte — cadres `╭─ python ─╮`, etiquettes, code sur ses
  lignes, section `🔗 Sources`, coloration de `const`/`number`, et plus aucun ```
  ni `Sources : [` — y compris sur la fuite d'echafaudage observee en vrai ;
- `bun src/index.ts --smoke` → code de sortie 0, image reelle capturee :
  cadre `RontoGPT`, cadre `conversation`, barre d'aide, champ de saisie,
  `vous › Combien font 6 * 7 ?` et la reponse `42` peinte dans la couleur du
  corps (les octets de rendu ont ete relus pour le confirmer) ;
- `tsc --noEmit` → 0 erreur (strict) ;
- en-tete mis a jour en direct : `v0.1.0 · profil qualite · cerveau en memoire
  · llama.cpp actif · GPU 1 · budget 180 s`.

Pas verifie :

- **le rendu final a l'oeil** : le design a ete relu dans les frames capturees
  (caracteres et couleurs), pas sur un vrai ecran. Confort, couleurs et
  lisibilite sous ton terminal restent a juger par toi ;
- **la coloration des langages sans grammaire embarquee** : `@opentui/core` ne
  livre que javascript, typescript, typescriptreact, markdown et zig. Un bloc
  ```python ou ```bash garde son cadre et son etiquette, mais son corps reste en
  clair (`#d8dee9`) — c'est verifie comme tel, pas contourne ;
- le **streaming de tokens** (`morceau` avec sa phase) : le code le gere, mais
  sur cette machine les questions de test partent toutes aux couches
  deterministes (`niveau0-instantane`, `rag-binaire`, `extraction-web`) et
  arrivent donc d'un bloc. Le chemin `morceau` n'a **jamais ete pris** ici ;
- l'`Echap` (interruption) n'a pas ete exerce en test automatise ;
- le TUI dans un **vrai** terminal Windows : la capture a ete faite sans PTY
  (winpty de Git Bash refuse de lancer `bun.exe`) ; le passage a l'ecran
  alternatif et les sequences de capacite montrent que le renderer s'installe,
  mais un terminal natif reste a confirmer.
