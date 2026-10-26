/**
 * RontoGPT — interface terminal (OpenTUI), habillage « Codex ».
 *
 * Ce fichier ne fait QUE l'affichage et la saisie : le cerveau vit dans le
 * processus Python (`aura/api.py`) et on lui parle en HTTP/SSE sur
 * 127.0.0.1:8760. Aucun import de `aura.*` ici, aucune duplication de logique.
 *
 * Langage visuel (celui de l'écran d'accueil de Codex, avec l'emblème RontoGPT) :
 *   - fond quasi noir, aucun cadre autour de la conversation ;
 *   - en-tête `>_ RontoGPT (vX.Y.Z)` et dossier courant en dessous ;
 *   - emblème RontoGPT très discret, centré, tant que le fil est vide ;
 *   - ta question : bande grise pleine largeur, précédée de `›` ;
 *   - la réponse : markdown rendu (gras, italique, code, liens, titres,
 *     listes…) précédé de `•`, un cadre par bloc de code et une section
 *     « 🔗 Sources » — aucune ligne de méta sous la réponse (src/affichage.ts) ;
 *   - attente : puce clignotante + « Réflexion (12 s • Échap pour interrompre) » ;
 *   - barre de saisie grise de 3 lignes, et deux lignes de pied de page.
 *
 * Règle d'interface : UNE action principale, visible en moins de 2 secondes.
 *   - un seul bouton fort, à droite de la saisie : « Envoyer ⏎ » (devient
 *     « Arrêter ■ » pendant une réponse : même place, même geste) ;
 *   - tout le reste est caché : `/help` liste les commandes, `Échap`
 *     interrompt, `/commande` ouvre les actions secondaires.
 *   (Le badge de preuve et le « Demander à Gemini » du backend ne sont pas
 *   affichés dans cet écran : `provenance` et `repli_propose` arrivent déjà
 *   dans l'événement `final`, mais le TUI ne garde la ligne de méta que pour
 *   les échecs et les interruptions.)
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import {
  BoxRenderable,
  InputRenderable,
  InputRenderableEvents,
  ScrollBoxRenderable,
  TextRenderable,
  bold,
  createCliRenderer,
  fg,
  t,
} from "@opentui/core";
import {
  BASE, api, cloud, compacter, configurerCloud, etat, formaterDuree, initialiser,
  memoire, editerMemoire, documents, editerDocuments, reglerLangue, reinitialiser, repondre, sante,
} from "./client";
import type { EtatContexte, InfoCloud, InfoEtat } from "./client";
import type { InfoApi } from "./client";
import { LOGO, LOGO_HAUTEUR, LOGO_LARGEUR } from "./logo";
import { TH, creerCorps, creerStyleCode, libelleAttente } from "./affichage";
import type { Corps, OptionsPeinture } from "./affichage";

/** Dossier courant, avec ~ pour le dossier personnel (comme Codex). */
function dossier(): string {
  const cwd = process.cwd();
  const home = homedir();
  return home && cwd.toLowerCase().startsWith(home.toLowerCase())
    ? `~${cwd.slice(home.length)}`
    : cwd;
}

// -- squelette de l'écran ---------------------------------------------------
// exitOnCtrlC: false est INDISPENSABLE : sinon OpenTUI detruit la scene au
// premier Ctrl+C, avant notre gestionnaire — donc pas de copie et pas de
// confirmation. C'est nous qui decidons quand quitter.
const renderer = await createCliRenderer({
  exitOnCtrlC: false,
  backgroundColor: TH.fond,
});

// Le style de coloration des blocs de code : créé une fois, lié à cet écran.
const styleCode = creerStyleCode(renderer);

// Passe à true dès qu'on quitte : les timers en vol (sondage d'état, réponse
// en cours) ne doivent plus écrire dans une scène native détruite.
let arret = false;
let suiviEtat: ReturnType<typeof setInterval> | null = null;

const racine = new BoxRenderable(renderer, {
  width: "100%",
  height: "100%",
  flexDirection: "column",
  backgroundColor: TH.fond,
});
renderer.root.add(racine);

// En-tête : `>_ RontoGPT (v0.1.0)` puis le dossier courant.
const entete = new BoxRenderable(renderer, {
  width: "100%",
  height: 3,
  flexDirection: "column",
  paddingLeft: 1,
  paddingTop: 1,
  backgroundColor: TH.fond,
});
const enteteTitre = new TextRenderable(renderer, { content: "", fg: TH.texte });
const enteteDossier = new TextRenderable(renderer, { content: "", fg: TH.discret });

/** Dossier courant, tronqué à gauche s'il dépasse la largeur (jamais sur 2 lignes). */
function majDossier(): void {
  const plein = dossier();
  const max = Math.max(10, renderer.width - 6);
  enteteDossier.content = `   ${plein.length > max ? `…${plein.slice(plein.length - max + 1)}` : plein}`;
}
majDossier();
renderer.on("resize", majDossier);
entete.add(enteteTitre);
entete.add(enteteDossier);
racine.add(entete);

function majTitre(version: string | null): void {
  if (arret || renderer.isDestroyed) return;
  enteteTitre.content = version
    ? t`${fg(TH.accent)(">_")} ${bold(fg("#ffffff")("RontoGPT"))} ${fg(TH.discret)(`(v${version})`)}`
    : t`${fg(TH.accent)(">_")} ${bold(fg("#ffffff")("RontoGPT"))}`;
}
majTitre(null);

// Écran d'accueil : l'emblème, tant que le fil est vide.
const accueil = new BoxRenderable(renderer, {
  width: "100%",
  flexGrow: 1,
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
  gap: 1,
  backgroundColor: TH.fond,
});
// L'accueil ne porte QUE l'embleme : aucune phrase sous le logo. La phrase
// d'identite n'apparait que dans la reponse a une question sur le modele.
const logo = new TextRenderable(renderer, { content: LOGO.join("\n"), fg: TH.logo });
accueil.add(logo);
racine.add(accueil);

/** L'emblème ne s'affiche que si l'écran est assez grand pour le contenir. */
function ajusterAccueil(): void {
  logo.visible = renderer.height >= LOGO_HAUTEUR + 12 && renderer.width >= LOGO_LARGEUR + 4;
}
ajusterAccueil();
renderer.on("resize", ajusterAccueil);

// Le fil de conversation (masqué à l'accueil), sans bordure ni cadre.
const fil = new ScrollBoxRenderable(renderer, {
  width: "100%",
  flexGrow: 1,
  stickyScroll: true,
  stickyStart: "bottom",
  backgroundColor: TH.fond,
  paddingBottom: 1,
});
fil.visible = false;
racine.add(fil);

// Barre de saisie : bande grise de 3 lignes, `›` à gauche, texte sur la ligne du milieu.
const composer = new BoxRenderable(renderer, {
  width: "100%",
  height: 3,
  flexDirection: "row",
  alignItems: "center",
  backgroundColor: TH.bande,
  paddingLeft: 1,
});
const invite = new TextRenderable(renderer, { content: "›", fg: TH.discret, width: 2 });
const saisie = new InputRenderable(renderer, {
  id: "question",
  width: "100%",
  placeholder: "Posez une question ou déposez un document",
  backgroundColor: TH.bande,
  focusedBackgroundColor: TH.bande,
  textColor: TH.texte,
  placeholderColor: TH.discret,
  cursorColor: TH.texte,
});
saisie.flexGrow = 1;

// L'ACTION PRINCIPALE : le seul élément fort de l'écran (couleur d'accent pleine,
// texte gras, collé à droite de la saisie). Il change de rôle selon l'état
// (Envoyer / Arrêter) au lieu d'ajouter un second bouton.
const bouton = new TextRenderable(renderer, {
  content: " Envoyer ⏎ ",
  fg: TH.fond,
  bg: TH.accent,
  flexShrink: 0,
  marginLeft: 1,
  marginRight: 1,
});

function majBouton(): void {
  const enCours = enVol !== false;
  bouton.content = enCours ? " Arrêter ■ " : " Envoyer ⏎ ";
  bouton.bg = enCours ? TH.alerte : TH.accent;
}
composer.add(invite);
composer.add(saisie);
composer.add(bouton);
racine.add(composer);
saisie.focus();

// Pied de page : ligne « modèle » puis raccourcis.
const pied = new BoxRenderable(renderer, {
  width: "100%",
  height: 2,
  flexDirection: "column",
  paddingLeft: 3,
  backgroundColor: TH.fond,
});
// Ligne d'etat : volontairement VIDE quand tout va bien. Pas de
// « RontoGPT qualite » permanent dans le pied de page. Elle sert aussi de
// message flash (copie reussie) : le texte s'efface seul apres 1,5 s.
const piedEtat = new TextRenderable(renderer, { content: "connexion…", fg: TH.discret });
const piedAide = new TextRenderable(renderer, {
  content: t`${fg(TH.doux)("⋯")} ${fg(TH.discret)("/help  commands")}`,
  fg: TH.discret,
});
pied.add(piedEtat);
pied.add(piedAide);
racine.add(pied);

// -- helpers d'affichage ----------------------------------------------------
function descendre(): void {
  const hauteur = (fil as unknown as { scrollHeight?: number }).scrollHeight;
  if (typeof hauteur === "number") fil.scrollTo({ x: 0, y: hauteur });
}

/** Passe de l'écran d'accueil au fil dès qu'il y a quelque chose à montrer. */
function montrerFil(): void {
  if (!fil.visible) {
    accueil.visible = false;
    fil.visible = true;
  }
}

function montrerAccueil(): void {
  fil.visible = false;
  accueil.visible = true;
}

/**
 * Un bloc « puce + texte » : la puce est posée en absolu dans la marge, le
 * texte garde donc une retraite fixe et se replie proprement (comme Codex).
 */
function blocAvecPuce(
  glyphe: string,
  couleur: string,
  fond?: string,
): { bloc: BoxRenderable; puce: TextRenderable } {
  // marginTop explicite : le `gap` d'un ScrollBox ne sépare pas ses enfants.
  const bloc = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    marginTop: 1,
    paddingLeft: 3,
    paddingRight: 1,
    ...(fond ? { backgroundColor: fond } : {}),
  });
  const puce = new TextRenderable(renderer, {
    content: glyphe,
    fg: couleur,
    position: "absolute",
    left: 1,
    top: 0,
  });
  bloc.add(puce);
  return { bloc, puce };
}

/** Note hors conversation (aide, état, erreur) : même mise en page qu'une réponse. */
function note(texte: string, couleur: string): TextRenderable {
  montrerFil();
  const { bloc } = blocAvecPuce("•", couleur);
  const corps = new TextRenderable(renderer, {
    content: texte,
    fg: couleur,
    wrapMode: "word",
    width: "100%",
  });
  bloc.add(corps);
  fil.add(bloc);
  descendre();
  return corps;
}

const AIDE = [
  "Commands",
  "  /help      this screen (alias /aide)",
  "  /clear     clear the displayed thread (alias /effacer)",
  "  /forget    the brain forgets the conversation (alias /oubli)",
  "  /memory    what the brain keeps in mind",
  "  /memory set <key> = <value>   edit durable memory",
  "  /memory delete <key>           remove a durable memory",
  "  /docs add <path>               add a document (paste/drop its path)",
  "  /docs remove <name>            remove a local document",
  "  /voice on|off                  read completed answers locally",
  "  /langue fr|en|auto             response language",
  "  /context   is the context window full? (gauge + advice)",
  "  /init      load the brain now",
  "  /mcp       MCP bridge and servers",
  "  /api       local API: address, key, routes",
  "  /plan      toggle plan mode (plan first, no execution)",
  "  /plan <goal>   create a checklist; /plan done marks the next task",
  "  /compact   drop the raw window, keep durable facts",
  "  /cloud     Gemini: key, on/off, fallback, long-form writing",
  "  /status    profile, llama.cpp, brain budget (alias /etat)",
  "  /quit      exit (alias /quitter)",
  "",
  "  Esc        interrupt the current answer",
  "  Ctrl+C     copy the selection; otherwise ask before quitting",
].join("\n");

// -- mode plan ---------------------------------------------------------------
// Quand il est actif, chaque question demande un PLAN avant toute execution.
let modePlan = false;
let langueChoisie: "auto" | "fr" | "en" = "auto";

interface TachePlan { texte: string; faite: boolean }
let planActif: TachePlan[] = [];

function afficherPlan(): void {
  if (planActif.length === 0) {
    note("Aucun plan en cours. Utilisez /plan <objectif>.", TH.doux);
    return;
  }
  const prochaine = planActif.findIndex((tache) => !tache.faite);
  const lignes = ["Plan en cours", ...planActif.map((tache, i) =>
    `  ${tache.faite ? "[x]" : "[ ]"} ${i + 1}. ${tache.texte}`)];
  lignes.push(prochaine === -1
    ? "Terminé."
    : `Étape en cours : ${prochaine + 1}. ${planActif[prochaine]!.texte}`);
  lignes.push("/plan fait lorsque cette étape est terminée.");
  note(lignes.join("\n"), prochaine === -1 ? TH.ok : TH.doux);
}

function enregistrerPlan(texte: string): void {
  const taches = texte.split("\n")
    .map((ligne) => ligne.match(/^\s*(?:[-*]|\d+[.)])\s*\[\s*\]\s+(.+)$/))
    .filter((m): m is RegExpMatchArray => m !== null)
    .map((m) => ({ texte: m[1]!.trim(), faite: false }))
    .slice(0, 12);
  if (taches.length >= 2) {
    planActif = taches;
    afficherPlan();
  }
}

// -- en-tête / pied de page -------------------------------------------------
// Message flash dans la ligne d'etat : « Sélection copiée ✓ » reste 1,5 s,
// puis la ligne redevient vide. Le sondage d'etat ne doit pas l'effacer avant
// la fin : `flashActif` le fait patienter.
let flashActif = false;
let flashMinuteur: ReturnType<typeof setTimeout> | null = null;

function flash(texte: string, couleur: string, duree = 1500): void {
  if (arret || renderer.isDestroyed) return;
  flashActif = true;
  piedEtat.content = texte;
  piedEtat.fg = couleur;
  renderer.requestRender();
  if (flashMinuteur !== null) clearTimeout(flashMinuteur);
  flashMinuteur = setTimeout(() => {
    flashMinuteur = null;
    flashActif = false;
    if (arret || renderer.isDestroyed) return;
    piedEtat.content = "";
    piedEtat.fg = TH.discret;
    renderer.requestRender();
  }, duree);
}

function majEntete(info: InfoEtat | null, nota = ""): void {
  if (arret || renderer.isDestroyed) return;
  if (info === null) {
    majTitre(null);
    if (flashActif) return;          // un flash en cours garde la ligne
    piedEtat.content = `hors ligne${nota}`;
    piedEtat.fg = TH.erreur;
    renderer.requestRender();
    return;
  }
  majTitre(info.version);
  if (flashActif) return;
  piedEtat.content = info.chargement ? "chargement…" : (modePlan ? "plan mode" : "");
  piedEtat.fg = TH.alerte;
  renderer.requestRender();
}

// -- un tour de conversation ------------------------------------------------
let enVol: false | AbortController = false;
let voixActive = false;

/** Lecture SAPI locale sous Windows : aucune reponse ne quitte la machine. */
function direLocalement(texte: string): void {
  if (!voixActive || !texte.trim()) return;
  if (process.platform !== "win32") {
    flash("Voice output is currently available on Windows.", TH.alerte, 2500);
    return;
  }
  const script = "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(" + JSON.stringify(texte.slice(0, 6000)) + ")";
  const encoded = Buffer.from(script, "utf16le").toString("base64");
  const p = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
    { stdio: "ignore", windowsHide: true });
  p.on("error", () => flash("Local voice could not start.", TH.erreur, 2500));
}

interface Tour {
  puce: TextRenderable;
  attente: TextRenderable;
  corps: Corps;
  meta: TextRenderable;
}

function tour(question: string): Tour {
  montrerFil();

  // Question : bande grise pleine largeur, `›` dans la marge.
  const utilisateur = blocAvecPuce("›", TH.doux, TH.bande);
  utilisateur.bloc.add(new TextRenderable(renderer, {
    content: question,
    fg: TH.texte,
    wrapMode: "word",
    width: "100%",
  }));
  fil.add(utilisateur.bloc);

  // Réponse : `•` dans la marge (il clignote pendant l'attente).
  const reponse = blocAvecPuce("•", TH.discret);
  const attente = new TextRenderable(renderer, { content: "", fg: TH.doux });
  const corps = creerCorps(renderer, styleCode);
  const meta = new TextRenderable(renderer, { content: "", fg: TH.trait, width: "100%" });
  corps.bloc.visible = false;  // pas de ligne vide tant qu'il n'y a rien à dire
  meta.visible = false;
  reponse.bloc.add(attente);
  reponse.bloc.add(corps.bloc);
  reponse.bloc.add(meta);
  fil.add(reponse.bloc);
  descendre();
  return { puce: reponse.puce, attente, corps, meta };
}

function habillerMeta(texte: string): string {
  const largeur = Math.max(12, renderer.width - 10);
  return `─ ${texte} `.padEnd(largeur, "─");
}

/** Message lisible quand le contexte est sature (aura/contexte.py). */
function phraseContexte(c: EtatContexte): string {
  const retires = c.compacte ?? 0;
  const detail = retires > 0
    ? `${retires} older message(s) compacted, ${c.gardes ?? 0} kept`
    : c.raison === "fenetre minimale"
      ? "nothing older to compact — the next answer may be truncated"
      : c.raison === "echec compaction"
        ? "compaction failed — the next answer may be truncated"
        : c.raison === "hors memoire"
          ? "memory is empty, but the system/web context alone is still full"
          : "no message removed";
  const apres = c.apres !== undefined ? `, now ${c.apres}%` : "";
  return `Context ${c.pourcentage}% full (${c.tokens}/${c.budget} tokens, ${c.mode ?? "importance"}) — ${detail}${apres}.`;
}

// -- envoi ------------------------------------------------------------------
async function envoyer(question: string): Promise<string> {
  if (enVol !== false) {
    note("Un tour est déjà en cours — Échap pour l'interrompre.", TH.alerte);
    return "";
  }
  const { puce, attente, corps, meta } = tour(question);
  // Le fil affiche la question REELLE ; le cerveau recoit la consigne de
  // plan en plus, uniquement quand le mode plan est actif.
  const requete = modePlan
    ? "Avant de planifier, demande une à trois précisions si l'objectif, le résultat attendu, "
      + "les contraintes ou les informations nécessaires manquent. Sinon fais un plan exécutable. "
      + "Réponds uniquement par une liste de 3 à 12 étapes au format exact `- [ ] action`, "
      + "dans l'ordre nécessaire pour achever l'objectif. Chaque action doit être concrète et vérifiable. "
      + "N'exécute aucune étape.\n\nObjectif : " + question
    : question;
  const controleur = new AbortController();
  enVol = controleur;

  const debut = Date.now();
  let phase = "Réflexion";
  let texte = "";        // morceaux cumulés (flux du cerveau)
  let affiche = "";      // ce qui est réellement à l'écran
  let morceaux = 0;
  // NB : `content` d'un TextRenderable est un StyledText, pas une chaîne : on
  // ne l'ADDITIONNE jamais (`meta.content += …` affichait « [object Object] »),
  // on l'affecte toujours avec un texte déjà complet.

  attente.content = `${phase} (0 s • Échap pour interrompre)`;
  const tic = setInterval(() => {
    const s = (Date.now() - debut) / 1000;
    // Pas de total connu : on affiche le compte de morceaux reçus (indicateur
    // de progression réel) plutôt qu'une barre dont la fin serait inventée.
    attente.content = libelleAttente(phase, s, morceaux);
    attente.fg = TH.doux;
    puce.content = Math.floor(s * 2) % 2 === 0 ? "•" : "◦";   // puce clignotante
    puce.fg = TH.alerte;
  }, 250);

  // Une seule fois : `final` ET le `finally` appellent cette fonction.
  let attenteFinie = false;
  const finirAttente = (): void => {
    if (attenteFinie) return;
    attenteFinie = true;
    clearInterval(tic);
    attente.content = "";
    attente.visible = false;
    // `requestRender` explicite : sans lui, la ligne d'attente déjà peinte
    // pouvait rester à l'écran (« Réflexion (0 s …) » juste au-dessus de la
    // réponse) quand autre chose avait repeint par-dessus. Détruire le noeud
    // marche aussi mais fait perdre la puce de la réponse (mesuré).
    puce.content = "•";
    puce.fg = TH.discret;
    renderer.requestRender();
  };
  const poser = (contenu: string, options: OptionsPeinture = {}): void => {
    corps.peindre(contenu, options);        // riche par défaut : cadres + sources
  };
  const poserMeta = (contenu: string, couleur: string): void => {
    meta.content = contenu;
    meta.fg = couleur;
    meta.visible = true;
  };

  try {
    for await (const e of repondre(requete, { signal: controleur.signal })) {
      if (e.type === "etape") {
        phase = e.libelle;                  // annonce du cerveau : recherche, lecture…
      } else if (e.type === "morceau") {
        morceaux += 1;
        phase = e.phase === "reflexion" ? "Réflexion" : "Rédaction";
        texte += e.texte;
        affiche = texte;
        poser(affiche, { riche: false });   // le design complet arrive au `final`
        descendre();
      } else if (e.type === "final") {
        finirAttente();
        let couleur: string = TH.texte;
        if (e.reponse && e.reponse.trim() !== "") {
          affiche = e.reponse;
        } else if (affiche === "") {
          affiche = "(réponse vide)";
          couleur = TH.alerte;
        }
        if (e.clarification) {
          affiche += `\n\nÀ préciser : ${e.clarification}`;
        }
        poser(affiche, { couleur });
        if (modePlan && e.ok) enregistrerPlan(affiche);
        // PAS de bandeau de provenance sous la réponse : « Base vérifiée ·
        // verified · base de faits locale relue » est une méta que le backend
        // fournit (`e.provenance`) mais qui n'apprend rien sous une réponse
        // ordinaire, et qui coupait le fil par un trait pleine largeur à
        // chaque tour. La méta reste réservée aux ÉCHECS et interruptions.
        if (e.contexte?.sature) note(phraseContexte(e.contexte), TH.alerte);
        if (!e.ok && e.erreur) note(`Le cerveau signale : ${e.erreur}`, TH.erreur);
        direLocalement(affiche);
      } else if (e.type === "erreur") {
        finirAttente();
        affiche = e.erreur;
        poser(affiche, { couleur: TH.erreur });
        poserMeta(habillerMeta(`échec après ${formaterDuree((Date.now() - debut) / 1000)}`), TH.erreur);
      }
    }
  } catch (e) {
    const interrompu = (e as Error)?.name === "AbortError";
    const raison = interrompu ? "interrompu" : (e as Error).message;
    if (affiche === "") {
      affiche = `(${raison})`;
      if (!interrompu) poser(affiche);      // une interruption ne se dit qu'une fois (méta)
    } else {
      poser(affiche);                       // le flux s'arrête : la réponse prend sa forme
    }
    poserMeta(habillerMeta(raison), TH.alerte);
  } finally {
    finirAttente();
    enVol = false;
    descendre();
    // Le cerveau vient de charger (premier tour) : on rafraichit tout de
    // suite le pied de page, au lieu d'attendre le sondage de 5 s.
    void etat().then((info) => majEntete(info));
  }
  return affiche;
}

// -- MCP --------------------------------------------------------------------
/** Etat local du pont MCP et des serveurs declares pour le projet. */
function rapportMcp(): string {
  const racine = resolve(import.meta.dir, "..", "..");
  const pont = join(racine, "scripts", "mcp_coucou.py");
  const config = join(racine, ".agents", "mcp_config.json");
  const lignes = ["MCP", `  Coucou bridge   ${existsSync(pont) ? "available" : "missing"} (scripts/mcp_coucou.py)`];
  if (!existsSync(config)) {
    lignes.push("  project config  none (.agents/mcp_config.json)");
    return lignes.join("\n");
  }
  try {
    const brut = JSON.parse(readFileSync(config, "utf8")) as { mcpServers?: Record<string, unknown> };
    const noms = Object.keys(brut.mcpServers ?? {});
    lignes.push(`  project config  ${noms.length > 0 ? noms.join(", ") : "no server"}`);
  } catch {
    lignes.push("  project config  unreadable");
  }
  return lignes.join("\n");
}

// -- saisie -----------------------------------------------------------------
function normaliser(brut: string): string {
  return brut.trim().replace(/^\//, "").toLowerCase();
}

// -- Gemini (/cloud) -----------------------------------------------------------
function libelleMode(mode: string): string {
  if (mode === "oui") return "auto (sends without asking)";
  if (mode === "non") return "never";
  return "ask (default)";
}

function resumeCloud(info: InfoCloud): string {
  return [
    "Gemini",
    `  key       ${info.cle_presente
      ? `set (${info.cle_source || "unknown source"})`
      : "missing — /cloud key <key>"}`,
    `  enabled   ${info.actif ? "yes" : "no (AURA_GEMINI=0)"}`,
    `  model     ${info.modele}`,
    `  fallback  ${libelleMode(info.mode)}`,
    `  writing   ${info.redaction ? "Gemini first" : "local"}`,
    info.erreur ? `  last error ${info.erreur}` : "",
    info.pause ? "  paused after an error — set a valid key to resume" : "",
    "",
    "  /cloud key <key>            save the Google AI key to .env",
    "  /cloud on | off             enable or disable Gemini",
    "  /cloud mode ask|auto|never  when a weak local answer may escalate",
    "  /cloud writing on | off     long-form drafts go to Gemini",
    "  /cloud model <name>         default gemini-2.5-flash",
  ].filter((l) => l !== "").join("\n");
}

async function traiterCloud(arguments_: string): Promise<void> {
  const brut = arguments_.trim();
  if (brut === "") {
    const info = await cloud();
    note(info === null ? `API unreachable on ${BASE}` : resumeCloud(info),
         info === null ? TH.erreur : TH.doux);
    return;
  }
  const [tete = "", ...reste] = brut.split(/\s+/);
  const mot = tete.toLowerCase();
  const valeur = reste.join(" ").trim();
  let champs: Parameters<typeof configurerCloud>[0] | null = null;
  if (mot === "on" || mot === "off") {
    champs = { actif: mot === "on" };
  } else if (mot === "key" || mot === "cle") {
    // La cle garde sa casse : on reprend le texte brut, pas la valeur normalisée.
    const cle = brut.replace(/^(?:key|cle)\s+/i, "").trim();
    if (cle === "") {
      note("Usage: /cloud key <Google AI key>", TH.alerte);
      return;
    }
    champs = { cle };
  } else if (mot === "mode") {
    if (!["ask", "auto", "never", "demander", "oui", "non"].includes(valeur.toLowerCase())) {
      note("Usage: /cloud mode ask|auto|never", TH.alerte);
      return;
    }
    champs = { mode: valeur.toLowerCase() };
  } else if (["writing", "redaction", "rédaction"].includes(mot)) {
    const v = valeur.toLowerCase();
    if (!["on", "off", "oui", "non", "1", "0"].includes(v)) {
      note("Usage: /cloud writing on|off", TH.alerte);
      return;
    }
    champs = { redaction: ["on", "oui", "1"].includes(v) };
  } else if (["model", "modele", "modèle"].includes(mot)) {
    if (valeur === "") {
      note("Usage: /cloud model <name>", TH.alerte);
      return;
    }
    champs = { modele: valeur };
  } else {
    note("Unknown /cloud option. Run /cloud to see the current setup.", TH.alerte);
    return;
  }
  const r = await configurerCloud(champs);
  if (!r.ok || r.etat === null) {
    note(`Cloud configuration failed: ${r.detail}`, TH.erreur);
    return;
  }
  note(resumeCloud(r.etat), TH.ok);
}

async function traiter(brut: string): Promise<void> {
  const texte = brut.trim();
  if (texte === "") return;
  const cmd = normaliser(texte);

  // Les mots nus marchent aussi : `clear`, `reset`, `exit` (l'ancien CLI les
  // acceptait, et `clear` partait sinon au cerveau comme une vraie question).
  if (["aide", "help", "?"].includes(cmd)) {
    note(AIDE, TH.doux);
    return;
  }
  if (["effacer", "clear", "cls"].includes(cmd)) {
    for (const enfant of [...fil.getChildren()]) enfant.destroy();
    montrerAccueil();
    return;
  }
  if (["oubli", "forget", "reset", "oublier"].includes(cmd)) {
    const r = await reinitialiser();
    note(r.ok ? "Brain: conversation forgotten." : `Could not forget: ${r.detail}`,
         r.ok ? TH.ok : TH.erreur);
    return;
  }
  const memCmd = texte.match(/^\/?(?:mémoire|memoire|memory)\b\s*(.*)$/i);
  if (memCmd) {
    const args = (memCmd[1] ?? "").trim();
    if (/^(set|definir)\s+/i.test(args)) {
      const pair = args.replace(/^(set|definir)\s+/i, "").split(/\s*=\s*/, 2);
      if (pair.length !== 2 || !pair[0] || !pair[1]) {
        note("Usage: /memory set <key> = <value>", TH.alerte); return;
      }
      const r = await editerMemoire(pair[0], pair[1]);
      note(r.ok ? `Memory saved: ${pair[0]} = ${pair[1]}` : `Memory edit failed: ${r.erreur}`,
           r.ok ? TH.ok : TH.erreur);
      return;
    }
    if (/^(delete|remove|supprimer)\s+/i.test(args)) {
      const cle = args.replace(/^(delete|remove|supprimer)\s+/i, "").trim();
      const r = await editerMemoire(cle);
      note(r.ok ? `Memory removed: ${cle}` : `Memory edit failed: ${r.erreur}`,
           r.ok ? TH.ok : TH.erreur);
      return;
    }
    if (args) { note("Usage: /memory [set <key> = <value> | delete <key>]", TH.alerte); return; }
    const m = await memoire();
    if (m === null) {
      note(`API unreachable on ${BASE}`, TH.erreur);
      return;
    }
    if (m.messages === 0) {
      note("Memory\n  The brain has no conversation in mind.", TH.doux);
      return;
    }
    const etatUtilisateur = Object.entries(m.etat).map(([k, v]) => `${k}=${v}`).join(", ");
    const derniere = [...m.fenetre].reverse().find((t) => t.role === "user");
    const ctx = m.contexte;
    note([
      "Memory",
      `  ${m.messages} message(s) · ${m.echanges} exchange(s)`,
      `  user state: ${etatUtilisateur || "(empty)"}`,
      ctx ? `  context: ${ctx.pourcentage}% (${ctx.tokens}/${ctx.budget} tokens)`
        + ` · compaction: ${ctx.mode ?? "importance"}`
        + (ctx.sature ? " — SATURATED" : "") : "",
      derniere ? `  last question: ${derniere.contenu}` : "",
    ].filter((l) => l !== "").join("\n"), ctx?.sature ? TH.alerte : TH.doux);
    return;
  }
  const docsCmd = texte.match(/^\/?(?:docs?|documents)\b\s*(.*)$/i);
  if (docsCmd) {
    const args = (docsCmd[1] ?? "").trim();
    if (/^(add|ajouter)\s+/i.test(args)) {
      const chemin = args.replace(/^(add|ajouter)\s+/i, "").trim().replace(/^"|"$/g, "");
      const r = await editerDocuments({ ajouter: [chemin] });
      note(r.ok ? `Document added (${r.ajoutes ?? 0}). Its citations will appear in answers.`
        : `Could not add document: ${r.erreur ?? r.erreurs?.map((e) => e.erreur).join(", ")}`,
      r.ok ? TH.ok : TH.erreur);
      return;
    }
    if (/^(remove|delete|supprimer)\s+/i.test(args)) {
      const nom = args.replace(/^(remove|delete|supprimer)\s+/i, "").trim();
      const r = await editerDocuments({ supprimer: nom });
      note(r.ok ? `${r.supprimes ?? 0} document passage(s) removed.` : `Could not remove: ${r.erreur}`,
           r.ok ? TH.ok : TH.erreur);
      return;
    }
    const r = await documents();
    if (r === null) { note(`API unreachable on ${BASE}`, TH.erreur); return; }
    note(r.ok ? ["Local documents", ...(r.fichiers.length ? r.fichiers.map((f) => `  ${f.nom} · ${f.morceaux} passage(s)`) : ["  No document. Drop or paste a path: /docs add C:\\file.pdf"])].join("\n")
      : `Documents unavailable: ${r.erreur}`, r.ok ? TH.doux : TH.erreur);
    return;
  }
  const voixCmd = texte.match(/^\/?(?:voice|voix)\b\s*(.*)$/i);
  if (voixCmd) {
    const option = (voixCmd[1] ?? "").trim().toLowerCase();
    if (option === "on" || option === "oui") voixActive = true;
    else if (option === "off" || option === "non") voixActive = false;
    else { note(`Local voice is ${voixActive ? "on" : "off"}. Use /voice on|off.`, TH.doux); return; }
    note(`Local voice ${voixActive ? "enabled — answers are read aloud locally." : "disabled."}`, TH.ok);
    return;
  }
  const langueCmd = texte.match(/^\/?(?:langue|language)\b\s*(.*)$/i);
  if (langueCmd) {
    const option = (langueCmd[1] ?? "").trim().toLowerCase();
    if (["", "?", "statut"].includes(option)) {
      note(`Langue des réponses : ${langueChoisie === "auto" ? "automatique" : langueChoisie === "fr" ? "français" : "anglais"}.\n/langue fr · /langue en · /langue auto`, TH.doux);
      return;
    }
    if (["fr", "français", "francais"].includes(option)) langueChoisie = "fr";
    else if (["en", "english", "anglais"].includes(option)) langueChoisie = "en";
    else if (["auto", "automatique"].includes(option)) langueChoisie = "auto";
    else { note("Usage : /langue fr | en | auto", TH.alerte); return; }
    const r = await reglerLangue(langueChoisie);
    note(r.ok
      ? `Langue des réponses : ${langueChoisie === "auto" ? "automatique" : langueChoisie === "fr" ? "français" : "anglais"}.`
      : `Impossible de changer la langue : ${r.erreur}`, r.ok ? TH.ok : TH.erreur);
    return;
  }
  // Le glisser-déposer d'un terminal colle le chemin dans la saisie : il est
  // indexe directement, sans demander de connaitre une commande.
  const cheminDepose = texte.replace(/^"|"$/g, "");
  if (existsSync(cheminDepose)) {
    const r = await editerDocuments({ ajouter: [cheminDepose] });
    note(r.ok ? "Document ajouté. Vous pouvez maintenant lui poser une question."
      : `Impossible d'ajouter ce document : ${r.erreur ?? r.erreurs?.map((e) => e.erreur).join(", ")}`,
    r.ok ? TH.ok : TH.erreur);
    return;
  }
  if (["init", "initialiser"].includes(cmd)) {
    if (enVol !== false) {
      note("A turn is already running — press Esc to interrupt it.", TH.alerte);
      return;
    }
    note("Loading the brain…", TH.doux);
    const debut = Date.now();
    const r = await initialiser();
    if (!r.ok) {
      note(`Initialization failed: ${r.detail}`, TH.erreur);
      return;
    }
    note(r.deja
      ? "The brain was already loaded."
      : `Brain loaded in ${((Date.now() - debut) / 1000).toFixed(1)} s.`, TH.ok);
    void etat().then((info) => majEntete(info));
    return;
  }
  if (cmd === "mcp") {
    note(rapportMcp(), TH.doux);
    return;
  }
  const planCmd = texte.match(/^\/?(?:plan|planifier)\b\s*(.*)$/i);
  if (planCmd) {
    const argument = (planCmd[1] ?? "").trim();
    if (["fait", "done", "suivant"].includes(argument.toLowerCase())) {
      const prochaine = planActif.find((tache) => !tache.faite);
      if (prochaine) prochaine.faite = true;
      afficherPlan();
      return;
    }
    if (["annuler", "abandonner", "stop"].includes(argument.toLowerCase())) {
      planActif = [];
      modePlan = false;
      note("Plan annulé.", TH.doux);
      return;
    }
    if (argument !== "") {
      modePlan = true;
      await envoyer(argument);
      return;
    }
    if (planActif.length > 0) { afficherPlan(); return; }
    modePlan = !modePlan;
    note(modePlan
      ? "Plan mode on — the next answers are step-by-step plans, with no execution. Type /plan again to turn it off."
      : "Plan mode off.", modePlan ? TH.ok : TH.doux);
    void etat().then((info) => majEntete(info));
    return;
  }
  if (["compact", "compacter"].includes(cmd)) {
    const r = await compacter();
    if (!r.ok) {
      note(`Compaction failed: ${r.detail}`, TH.erreur);
      return;
    }
    note(r.compacte > 0
      ? `Compacted ${r.compacte} message(s) — durable facts were kept.`
      : "Nothing to compact — the brain has no conversation in mind.",
      TH.doux);
    return;
  }
  if (["context", "contexte", "ctx"].includes(cmd)) {
    await afficherContexte();
    return;
  }
  // `/api`, `/api stop`, `/api start` : on ne passe a traiterApi QUE ce qui
  // suit le mot (avant, `/api` entier arrivait la et tombait dans « inconnue »).
  // Le slash est exige, sauf pour le mot nu `api` : une vraie question comme
  // « api rest c'est quoi ? » part au cerveau.
  if (/^(?:\/api\b|api\s*$)/i.test(texte)) {
    await traiterApi(texte.replace(/^\/?api\b/i, "").trim());
    return;
  }
  // `/cloud` explicite, ou mot nu suivi d'une sous-commande connue : une
  // vraie question qui commence par « cloud computing… » part au cerveau.
  const cmdCloud = texte.match(/^(\/)?(?:cloud|nuage)\b\s*(.*)$/i);
  if (cmdCloud) {
    const arguments_ = (cmdCloud[2] ?? "").trim();
    const tete = (arguments_.split(/\s+/)[0] ?? "").toLowerCase();
    const sous = ["", "key", "cle", "on", "off", "mode", "writing",
                  "redaction", "rédaction", "model", "modele", "modèle"];
    if (cmdCloud[1] === "/" || sous.includes(tete)) {
      await traiterCloud(arguments_);
      return;
    }
  }
  if (["etat", "status", "statut"].includes(cmd)) {
    const info = await etat();
    majEntete(info);
    if (info === null) {
      note(`API unreachable on ${BASE}`, TH.erreur);
      return;
    }
    note(
      `version ${info.version} · profile ${info.profil} · model ${info.prete ? "loaded" : "not loaded"}`
      + `\nllama.cpp ${info.serveur_llama ? "active" : "stopped"} · GPU ${info.gpu} · budget ${info.budget_s} s`
      + (info.erreur ? `\nerror: ${info.erreur}` : ""),
      TH.doux,
    );
    return;
  }
  if (["quitter", "exit", "quit", "q", "bye"].includes(cmd)) {
    if (quitterDemande) quitter();   // seconde demande = confirmation
    else demanderQuitter();
    return;
  }

  await envoyer(texte);
}

/** Jauge texte : `██████░░░░░░░░░░░░░░` pour 30 %. */
function jauge(pourcent: number, largeur = 20): string {
  const p = Math.max(0, Math.min(100, pourcent));
  const plein = Math.round((p / 100) * largeur);
  return "█".repeat(plein) + "░".repeat(largeur - plein);
}

/** `/context` : la fenêtre de contexte est-elle pleine ? (GET /v1/memoire) */
async function afficherContexte(): Promise<void> {
  const m = await memoire();
  if (m === null) {
    note(`API unreachable on ${BASE}`, TH.erreur);
    return;
  }
  const c = m.contexte;
  if (!c) {
    note("Context\n  The brain is not loaded yet, so nothing is in the window. "
         + "Ask a question or run /init.", TH.doux);
    return;
  }
  const systemTokens = c.systeme_tokens ?? 428;
  const conversationTokens = c.conversation_tokens ?? Math.max(0, c.tokens - systemTokens);
  const conversationBudget = c.budget_conversation ?? Math.max(1, c.budget - systemTokens);
  const conversationPercent = Math.round((conversationTokens / conversationBudget) * 100);
  const restant = Math.max(0, conversationBudget - conversationTokens);
  const seuil = c.seuil ?? 90;
  const verdict = c.sature
    ? "FULL — the next message pushes the oldest exchanges out of the window"
    : c.pourcentage >= 60
      ? "filling up"
      : "plenty of room";
  note([
    `Conversation  ${jauge(conversationPercent)}  ${conversationPercent}% used`,
    `  ${verdict}`,
    `  ${conversationTokens} / ${conversationBudget} tokens · ${restant} remaining · ${m.messages} message(s) in the window`,
    `  auto-compaction at ${seuil}% (mode ${c.mode ?? "importance"})`,
    c.sature
      ? "  /forget starts fresh · /compact keeps only durable facts · older turns stay searchable"
      : "",
  ].filter((l) => l !== "").join("\n"),
  c.sature ? TH.alerte : c.pourcentage >= 60 ? TH.alerte : TH.ok);
}

/** `/api` : l'API locale (adresse, version, clé, routes) — `argument` suit le mot. */
async function traiterApi(argument: string): Promise<void> {
  const mot = argument.split(/\s+/)[0]?.toLowerCase() ?? "";
  if (mot === "stop" || mot === "arreter" || mot === "disable") {
    const r = await api("disable");
    if (r === null) {
      note(`API unreachable on ${BASE}`, TH.erreur);
      return;
    }
    note(r.ok === false ? `Workshop server: ${r.erreur ?? "could not stop"}` : "Workshop server stopped.",
         r.ok === false ? TH.alerte : TH.doux);
    return;
  }
  if (mot === "start" || mot === "demarrer" || mot === "deploy") {
    const r = await api("deploy");
    note(r === null ? `API unreachable on ${BASE}` : `Workshop server: ${r.erreur ?? "not available"}`,
         r === null ? TH.erreur : TH.alerte);
    return;
  }
  if (mot !== "") {
    note("Unknown /api option. Run /api for the address and routes.", TH.alerte);
    return;
  }
  const i = await api("status");
  if (i === null) {
    note(`API unreachable on ${BASE}`, TH.erreur);
    return;
  }
  if (i.ok === false) {
    note(`API error: ${i.erreur ?? "unknown"}`, TH.erreur);
    return;
  }
  const atelier = i.atelier;
  note([
    "Local API",
    `  address   ${i.serveur}`,
    `  version   ${i.version}${i.build ? ` (build ${i.build})` : ""}`,
    `  key       ${i.cle_requise ? "required (X-RontoGPT-Cle header)" : "none — local only (127.0.0.1)"}`,
    atelier
      ? `  workshop  ${atelier.disponible
          ? (atelier.actif ? `running on ${atelier.serveur}` : "stopped")
          : `unavailable (${atelier.erreur ?? "module missing"})`}`
      : "",
    "",
    ...(i.routes ?? []).map((r) => `  ${r}`),
  ].filter((l, k, tout) => l !== "" || tout[k - 1] !== "").join("\n"), TH.doux);
}

saisie.on(InputRenderableEvents.ENTER, (valeur: string) => {
  saisie.value = "";
  void traiter(valeur);
});

// -- copier / quitter --------------------------------------------------------
// Ctrl+C ne quitte plus tout de suite : une sélection copie, sinon il faut
// confirmer par un second Ctrl+C. Aucune sortie silencieuse.
let quitterDemande = false;
let quitterMinuteur: ReturnType<typeof setTimeout> | null = null;

/** Écrit dans le presse-papiers : OSC 52 (terminal) puis outil système. */
function copier(texte: string): void {
  try { renderer.copyToClipboardOSC52(texte); } catch { /* terminal sans OSC 52 */ }
  const candidats = process.platform === "win32"
    ? [["clip"]]
    : process.platform === "darwin"
      ? [["pbcopy"]]
      : [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]];
  for (const [commande, ...args] of candidats) {
    try {
      const p = spawn(commande!, args, { stdio: ["pipe", "ignore", "ignore"] });
      p.on("error", () => { /* outil absent : OSC 52 a déjà été tenté */ });
      p.stdin?.on("error", () => { /* idem */ });
      p.stdin?.end(texte);
      return;
    } catch {
      continue;                       // outil suivant
    }
  }
}

function annulerQuitter(dire: boolean = false): void {
  if (!quitterDemande) return;
  quitterDemande = false;
  if (quitterMinuteur !== null) clearTimeout(quitterMinuteur);
  quitterMinuteur = null;
  if (dire) note("Quit cancelled.", TH.discret);
}

function demanderQuitter(): void {
  quitterDemande = true;
  // Un vrai message dans le fil : une ligne de pied de page peut être
  // tronquée et ne laisse alors aucun moyen de comprendre la demande.
  note("Quit RontoGPT? Press Ctrl+C again to confirm, or Esc to cancel.",
       TH.alerte);
  if (quitterMinuteur !== null) clearTimeout(quitterMinuteur);
  quitterMinuteur = setTimeout(() => annulerQuitter(false), 8000);
}

function quitter(): void {
  arret = true;                 // plus aucun timer ne touche à la scène
  if (quitterMinuteur !== null) clearTimeout(quitterMinuteur);
  if (flashMinuteur !== null) clearTimeout(flashMinuteur);
  if (suiviEtat !== null) clearInterval(suiviEtat);
  renderer.destroy();
  process.exit(0);
}

renderer.keyInput.on("keypress", (key) => {
  if (key.name === "escape") {
    if (quitterDemande) {
      annulerQuitter(true);      // Échap annule la demande de sortie…
      return;
    }
    if (enVol !== false) enVol.abort();   // …sinon il interrompt le tour
    return;
  }
  if (key.ctrl && key.name === "c") {
    const selection = renderer.getSelection()?.getSelectedText() ?? "";
    if (selection.trim() !== "") {
      copier(selection);         // copier d'abord : on ne quitte pas
      renderer.clearSelection();
      flash("Sélection copiée ✓", TH.ok);   // 1,5 s puis disparaît
      return;
    }
    if (quitterDemande) quitter();
    else demanderQuitter();
  }
});

// -- démarrage --------------------------------------------------------------
const santeInfo = await sante();
if (santeInfo === null) {
  majEntete(null, ` — lance :  uv run rontogpt`);
} else {
  majEntete(await etat());
}

// Le pied de page suit l'état du cerveau (chargement au premier appel, llama.cpp…).
// `suiviEtat` est arrêté explicitement dans quitter() : sinon un tick peut
// écrire dans la scène native déjà détruite (« Native scene is destroyed »).
suiviEtat = setInterval(() => {
  if (arret || renderer.isDestroyed) return;
  void etat().then((info) => majEntete(info));
}, 5000);

// -- mode --smoke : parcours complet sans humain ----------------------------
// Monte vraiment l'écran, joue une question, puis ressort sur le code 0.
// C'est ce qui rend la mise en page vérifiable en machine (voir README).
if (process.argv.includes("--smoke")) {
  const question = process.env.AURA_TUI_SMOKE_Q ?? "Combien font 6 * 7 ?";
  const reponse = await envoyer(question);
  await new Promise((r) => setTimeout(r, 1500));   // laisse partir la frame finale
  const blocs = fil.getChildren().length;
  arret = true;
  if (suiviEtat !== null) clearInterval(suiviEtat);
  renderer.destroy();
  console.log(`[smoke] écran monté · ${blocs} bloc(s) dans le fil · question : ${question}`
              + ` · réponse affichée : ${JSON.stringify(reponse.slice(0, 120))}`);
  process.exit(blocs >= 2 && reponse.trim() !== "" ? 0 : 1);
}
