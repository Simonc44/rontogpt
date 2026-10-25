# Étude de marché et architecture cible — octobre 2026

> Périmètre : assistants IA locaux sur ordinateur personnel, usage individuel.
> Chiffres RontoGPT **mesurés** sur la machine de développement (HP ProBook 440 G10,
> Intel U300, 16 Go, iGPU Vulkan) ; chiffres marché issus des sources citées en fin
> de document. Les pondérations de besoins sont des **hypothèses** à valider, pas
> des résultats d'enquête.

---

## 1. Le marché en quatre couches

| Couche | Acteurs | Ce qu'ils vendent | Limite |
|---|---|---|---|
| Assistants cloud | ChatGPT, Claude, Gemini, Le Chat | La capacité maximale, sans installation | Vie privée, coût récurrent, dépendance réseau |
| Runtimes | llama.cpp, Ollama, vLLM | Le moteur et l'API | Pas d'interface soignée (Ollama = terminal) |
| Apps tout-en-un | LM Studio, Jan, GPT4All, AnythingLLM, Msty | Moteur + gestionnaire de modèles + chat | Peu ou pas de contrôle sur la **qualité** des réponses |
| Mobile | PocketPal, MLC Chat, Private LLM, Locally AI | Le même usage, dans la poche | Modèles plus petits, pas de bureau |

**Constat structurant** : tout le monde utilise le même moteur (llama.cpp + GGUF).
La différence ne se joue donc **pas** sur l'inférence, mais sur trois choses :
le confort d'installation, les capacités autour du modèle (documents, outils,
voix) et la confiance dans la réponse.

---

## 2. Tableau concurrentiel

| | Ollama | LM Studio | Jan | GPT4All | PocketPal |
|---|---|---|---|---|---|
| Interface | Terminal + API | GUI desktop | GUI desktop | GUI desktop | Mobile |
| Modèles | 4 500+ (bibliothèque) | Tout GGUF Hugging Face | ~50 + HF | ~50 | HF (GGUF) |
| API OpenAI | Oui (:11434) | Oui (:1234) | Oui (:1337) | Oui (:4891) | Non (local) |
| Documents locaux (RAG) | Non (via Open WebUI) | Partiel | Extensions | **LocalDocs** | Non |
| Outils / agents | Non | Non | Extensions | Non | **Talents** |
| Voix (TTS) | Non | Non | Non | Non | **Oui** |
| Télémétrie | Aucune | Analytics par défaut | **Aucune** | Opt-in | Opt-in |
| Licence | MIT | Closed source | MIT | MIT | MIT |
| Positionnement | Développeur | Débutant | Vie privée | Documents | Mobile |

Enseignements :
- **Ollama** gagne par l'écosystème (4 500 modèles, API) et non par l'UX.
- **LM Studio** gagne par la finition : navigateur de modèles, chat, serveur local.
- **Jan** gagne sur la promesse « zéro télémétrie, tout open source ».
- **GPT4All** a compris le besoin n°1 après le chat : **parler à ses documents**.
- **PocketPal** montre que les fonctions qui font revenir (personas, voix, outils)
  sont des fonctions **d'usage**, pas de performance brute.

---

## 3. Ce qui marche, ce qui ne marche pas

### Table stakes (sans ça, l'app est écartée)
1. Installation < 5 minutes, sans compiler.
2. Gestionnaire de modèles : télécharger, voir la taille et la quantification,
   choisir selon sa RAM.
3. Chat en streaming, historique, prompt système, export.
4. API OpenAI-compatible locale.
5. Accélération GPU avec repli propre.
6. Fonctionnement hors ligne après le téléchargement.

### Différenciateurs (ce qui fait préférer)
1. **Documents personnels** : PDF, notes, code — avec citations.
2. **Outils** : calcul, date/heure, rendu, recherche — boucle d'agent.
3. **Personas** : un assistant par usage, avec son prompt et son modèle.
4. **Voix** : lire les réponses, dicter les questions.
5. **Benchmark** : tok/s, mémoire, et surtout **qualité** mesurée.
6. **Souveraineté** : open source, zéro télémétrie, auditable.

### Ce qui fait désinstaller
1. **Latence > 10 s** perçue sur une question simple.
2. Réponse fausse dite avec assurance, sans source ni refus.
3. Configuration technique exposée (contexte, threads, quantifications).
4. Modèle trop gros pour la RAM → plantage ou swap.
5. Télémétrie surprenante.
6. Fonctions qui coûtent du temps sans bénéfice visible.

---

## 4. Le vrai besoin d'un utilisateur

Classement en « jobs to be done », avec un poids indicatif (hypothèse) :

| Besoin | Poids | Ce que l'utilisateur juge |
|---|---|---|
| Obtenir une réponse fiable, vite | 30 % | justesse, latence, refus honnête |
| Comprendre **ses** documents | 25 % | import, recherche, citation |
| Rédiger / corriger en français | 20 % | style, structure, longueur |
| Calculer / raisonner juste | 10 % | exactitude vérifiable |
| Coder | 10 % | exécution, erreurs |
| Confidentialité / hors ligne | 5 % | aucune fuite, aucun compte |

Deux conséquences :
- La **confidentialité seule ne suffit pas** : elle est un prérequis, pas un usage.
- La **qualité perçue** se joue sur les 55 % « documents + rédaction », que
  RontoGPT couvre aujourd'hui mal (pas d'import de documents) ou lentement
  (rédaction ~30 s).

---

## 5. Diagnostic RontoGPT

### Forces mesurées
- **Exactitude symbolique** : maths, logique, code, taux — réponse exacte sans
  génération (banc : 4/4 exact, 1/1 logique).
- **Anti-hallucination** : ancrage web, graphe de faits, refus honnête,
  anti-boucle, anti-contamination. Aucun concurrent local ne le fait à ce niveau.
- **Français évalué** : banc dédié, 12/12 rapide, 2/2 rédaction.
- **Ingénierie** : 1 233 tests, ruff propre, mesures datées, doc abondante.
- **Sobriété** : 1,1 Go de modèle, tourne sur un iGPU.

### Faiblesses mesurées
- **Latence** : 32 s pour 1 600 caractères, ~11 tok/s en Q8. Meta 0,4 s,
  fait ancré 0,5 s, fait hors-ligne 15 s.
- **Pas de RAG documents** : le besoin n°2 du marché n'est pas couvert.
- **Pas de gestionnaire de modèles** : téléchargement et choix laissés au terminal.
- **Surface** : 76 modules, 66 atteignables, 10 non atteignables ; plusieurs
  mémoires et plusieurs cache qui se recouvrent.
- **Doc en retard** : `ARCHITECTURE.md` décrit encore le cerveau Heretic Q4 supprimé.
- **Produit** : pas d'installeur grand public, pas de premier lancement guidé,
  pas de streaming partout, pas de page benchmark.

### Positionnement actuel
RontoGPT est un **très bon moteur** (74/100) et un **produit moyen** (~55/100).
Il ne doit pas concurrencer LM Studio sur le nombre de modèles : il perdrait.
Il doit gagner sur une phrase : **« l'assistant local français qui prouve ses
réponses »**.

---

## 6. Architecture cible

### Principe directeur
> Aucune étape ne reste dans le chemin de réponse si elle ne peut pas prouver
> son gain. Chaque étape a un **budget de temps** et un **effet mesurable** sur
> le banc. C'est la réponse directe au problème « des fonctions qui coûtent
> sans rien apporter ».

### Cinq couches

```
┌──────────────────────────────────────────────────────────────┐
│ 5. ÉVALUATION   banc qualité, tests, télémétrie locale        │
├──────────────────────────────────────────────────────────────┤
│ 4. INTERFACES   API OpenAI, GUI web, GUI desktop, TUI, voix   │
├──────────────────────────────────────────────────────────────┤
│ 3. CAPACITÉS    experts exacts · documents · web · outils     │
│                 mémoire · personas                            │
├──────────────────────────────────────────────────────────────┤
│ 2. NOYAU        routeur → plan (si prouvé) → génération       │
│                 → vérification → garde-fous                   │
├──────────────────────────────────────────────────────────────┤
│ 1. RUNTIME      modèle, quantifications, serveur, adaptateurs │
└──────────────────────────────────────────────────────────────┘
```

### Correspondance avec le code actuel

| Couche cible | Modules actuels | Action |
|---|---|---|
| `runtime/` | `llama_cerveau`, `serveur_lora`, `profil_materiel`, `gguf_meta`, `adaptateurs`, `echantillonnage`, `verrou` | Regrouper + ajouter un **gestionnaire de modèles** |
| `noyau/` | `pipeline_rontogpt`, `orchestrateur`, `routeur`, `aiguilleur`, `detection_demande`, `synthese_rontogpt`, `regles_reponse`, `verification_rontogpt`, `clarification` | Garder, instrumenter chaque étape en temps |
| `capacites/` | `expert_symbolique`, `experts_rontogpt`, `calcul_*`, `logique`, `potcode`, `syllogismes`, `tri_transitif`, `memoire_web`, `graphe_faits`, `rag_binaire`, `concepts`, `embeddings` | Ajouter **documents locaux** ; fusionner les mémoires |
| `interfaces/` | `api`, `gui_web`, `gui`, `tui`, `__main__`, `winmain`, `coucou`, `pal`, `typographie` | Ajouter **streaming partout** + onboarding |
| `eval/` | `scripts/bench_qualite`, `scripts/sous_banc`, `tests/`, `resultats/` | Ajouter **30 questions** et un score latence/qualité |
| `experimental/` | `selfplay`, `rlss`, `intercepteur_entropie`, `memoire_engraft`, `gemini`, `arbre_reflexion` | Archiver hors du chemin runtime |
| `packaging/` | `kernel`, `signature`, `chiffrement` | Isoler : ce n'est pas du raisonnement |

### Décisions d'architecture
1. **Un seul cerveau par défaut** : Gemma-3-1B Q4_K_M (~800 Mo), estimé 1,5 à
   2× plus rapide que le Q8 sur ce CPU (à mesurer avant adoption) ; le Q8 reste
   sélectionnable. Le choix se fait selon la RAM au premier lancement.
2. **Budget de latence par classe** : méta < 1 s, fait ancré < 5 s,
   rédaction < 30 s, hors-ligne < 10 s. Une échéance est propagée (déjà fait via
   `ContextVar`) et coupe les étapes facultatives.
3. **Une seule mémoire** : conversation + faits + documents dans un même
   index, avec provenance. Les caches séparés (`filtre_instantane`,
   `cache_reponses`, `instantane`) fusionnent ou disparaissent.
4. **Documents d'abord** : import PDF/MD/TXT, découpage, embeddings MiniLM
   (déjà présent), réponse avec citations. C'est le besoin n°2 du marché et la
   plus grosse lacune actuelle.
5. **Outils déclarés** : les experts exacts existent déjà ; il manque la
   boucle qui laisse le modèle les appeler (calcul, date/heure, documents).
6. **Zéro étape sans preuve** : chaque étape du pipeline logue son temps et son
   apport sur le banc. Une étape sans gain mesuré est supprimée.

---

## 7. Feuille de route

### P0 — combler les manques de marché (2 à 3 semaines)
| Chantier | Impact | Effort |
|---|---|---|
| Gestionnaire de modèles + premier lancement (Q4/Q8 selon RAM) | ++ | M |
| RAG documents avec citations (PDF/MD/TXT) | +++ | L |
| Budgets de latence + instrumentation par étape | ++ | S |
| `python -m aura.config` : diagnostic et doc à jour | + | S |

> **État au 24 octobre 2026** : le RAG documents est implémenté
> (`aura/documents.py`, CLI `python -m aura.documents`, injection dans le
> pipeline avec citations). Le gestionnaire de modèles et les budgets
> instrumentés restent à faire.

### P1 — confort et rétention
| Chantier | Impact | Effort |
|---|---|---|
| Boucle d'outils générique (Talents) | ++ | M |
| TTS central + streaming partout | + | M |
| Page benchmark tok/s + mémoire + qualité | + | S |
| Installeur Windows / archive portable | ++ | M |

### P2 — à ne faire que si P0/P1 tiennent
Multi-utilisateur, marketplace de personas, application mobile.
**Ne pas** ajouter d'experts supplémentaires avant que les documents et la
latence soient réglés : la surface est déjà le principal risque.

---

## 8. KPI de réussite

| Indicateur | Seuil cible | Mesure actuelle |
|---|---|---|
| TTFT (premier token) | < 2 s | non mesuré |
| Question méta | < 1 s | 0,4 s ✅ |
| Fait ancré | < 5 s | 0,5 s ✅ |
| Rédaction 1 500 car | < 30 s | 32 s ⚠️ |
| Réponses fausses sur 30 questions | 0 | non mesuré (14 questions) |
| Installation → premier chat | < 5 min | non mesuré (CLI) |
| Tests verts | 100 % | 1 233 ✅ |

---

## 9. Conclusion

Le marché local est mature sur l'**inférence** et encore faible sur la
**confiance**. Ollama, LM Studio et Jan ont déjà gagné la course à l'installation
et au catalogue de modèles ; les concurrencer sur ce terrain serait perdu
d'avance. En revanche, aucun ne garantit qu'une réponse est exacte, ancrée ou
refusée quand elle ne peut pas l'être.

RontoGPT doit donc se concentrer sur trois choses, dans cet ordre :
1. **Répondre à ses documents** (le besoin le plus mal couvert).
2. **Prouver ses réponses** (ce qu'il sait déjà faire, à rendre visible).
3. **Rester sous 30 s** sur une rédaction, et sous 5 s sur un fait.

Le reste — plus de modules, plus d'experts, une app mobile — est du bruit tant
que ces trois points ne sont pas tenus.

---

## Sources

- PromptQuorum, *Ollama vs LM Studio vs Jan AI vs GPT4All: 2026 Guide* —
  interfaces, ports, licences, télémétrie, LocalDocs.
- NeuralChainAI, *Ollama vs LM Studio vs GPT4All: Best Local LLM Tool (2026)*.
- SitePoint, *Guide to Local LLMs in 2026: Privacy, Tools & Hardware*.
- Nature, *How to search the literature with local AI systems* (12 oct. 2026) —
  privacy, reproducibility, cost.
- GitHub, *a-ghorbani/pocketpal-ai* — Pals, TTS, Talents, HF, benchmark, 11 langues.
- Mesures internes : `scripts/bench_qualite.py`, `docs/BASELINE_2026-10-11.md`,
  suite `pytest` (1 233 tests), 24 octobre 2026.
