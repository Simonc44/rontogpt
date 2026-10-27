/**
 * RontoGPT v2 — TUI moderne (architecture split-footer inspirée d'OpenCode).
 * Point d'entrée principal.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { renderer, initializeRenderer, destroyRenderer } from "./renderer-ref";
import {
  BASE, api, cloud, compacter, configurerCloud, demanderGemini, etat,
  formaterDuree, initialiser, memoire, editerMemoire, documents, editerDocuments,
  reglerLangue, reinitialiser, repondre, sante,
} from "./client";
import type { EtatContexte, InfoCloud, InfoEtat } from "./client";
import { LOGO, LOGO_HAUTEUR, LOGO_LARGEUR } from "./logo";
import { DEFAULT_THEME, SEMANTIC_COLORS } from "./theme";
import { Scrollback } from "./scrollback";
import { Footer } from "./footer";

let arret = false;
let suiviEtat: ReturnType<typeof setInterval> | null = null;
let flashMinuteur: ReturnType<typeof setTimeout> | null = null;
let flashActif = false;

let etatBase = "";
let couleurEtat = SEMANTIC_COLORS.statusIdle;
let jaugeContexte = "";
let modePlan = false;
let langueChoisie: "auto" | "fr" | "en" = "auto";
let voixActive = false;
let derniereQuestion = "";
let quitterDemande = false;
let quitterMinuteur: ReturnType<typeof setTimeout> | null = null;

interface TachePlan { texte: string; faite: boolean }
let planActif: TachePlan[] = [];

const scrollback = new Scrollback(DEFAULT_THEME);
let footer: Footer;

async function main(): Promise<void> {
  await initializeRenderer();

  footer = new Footer({
    onSubmit: envoyer,
    onInterrupt: () => { if (enVol !== false) enVol.abort(); },
    onExit: quitter,
    onCycleVariant: () => { /* TODO */ },
    onCycleModel: () => { /* TODO */ },
    onCycleAgent: () => { /* TODO */ },
    onSettings: () => { /* TODO */ },
    onEditorOpen: async () => undefined,
    onFindFiles: async () => [],
    onQueuedPromptAction: async () => { /* TODO */ },
  }, DEFAULT_THEME);

  // Layout racine
  const { BoxRenderable, TextRenderable } = await import("@opentui/core");
  const racine = new BoxRenderable(renderer, {
    width: "100%", height: "100%", flexDirection: "column", backgroundColor: DEFAULT_THEME.background,
  });
  renderer.root.add(racine);

  // En-tête
  const entete = new BoxRenderable(renderer, {
    width: "100%", height: 3, flexDirection: "column", paddingLeft: 1, paddingTop: 1, backgroundColor: DEFAULT_THEME.background,
  });
  const enteteTitre = new TextRenderable(renderer, { content: "", fg: DEFAULT_THEME.text });
  const enteteDossier = new TextRenderable(renderer, { content: "", fg: DEFAULT_THEME.textDim });
  entete.add(enteteTitre); entete.add(enteteDossier); racine.add(entete);

  function dossier(): string {
    const cwd = process.cwd(); const home = homedir();
    return home && cwd.toLowerCase().startsWith(home.toLowerCase()) ? `~${cwd.slice(home.length)}` : cwd;
  }
  function majDossier(): void {
    const plein = dossier(); const max = Math.max(10, renderer.width - 6);
    enteteDossier.content = `   ${plein.length > max ? `…${plein.slice(plein.length - max + 1)}` : plein}`;
  }
  majDossier(); renderer.on("resize", majDossier);

  function majTitre(version: string | null): void {
    if (arret || renderer.isDestroyed) return;
    enteteTitre.content = version ? `>_ RontoGPT ${version}` : `>_ RontoGPT`;
  }
  majTitre(null);

  // Écran d'accueil
  const accueil = new BoxRenderable(renderer, {
    width: "100%", flexGrow: 1, flexDirection: "column", justifyContent: "center", alignItems: "center", gap: 1, backgroundColor: DEFAULT_THEME.background,
  });
  const logo = new TextRenderable(renderer, { content: LOGO.join("\n"), fg: DEFAULT_THEME.logo });
  accueil.add(logo); racine.add(accueil);

  function ajusterAccueil(): void { logo.visible = renderer.height >= LOGO_HAUTEUR + 12 && renderer.width >= LOGO_LARGEUR + 4; }
  ajusterAccueil(); renderer.on("resize", ajusterAccueil);

  // Scrollback
  scrollback.element.visible = false; racine.add(scrollback.element);

  // Footer
  racine.add(footer.element);

  function montrerFil(): void { if (!scrollback.element.visible) { accueil.visible = false; scrollback.element.visible = true; } }
  function montrerAccueil(): void { scrollback.element.visible = false; accueil.visible = true; }

  function flash(texte: string, couleur: string, duree = 1500): void {
    if (arret || renderer.isDestroyed) return;
    flashActif = true; footer.flash(texte, couleur, duree);
    if (flashMinuteur !== null) clearTimeout(flashMinuteur);
    flashMinuteur = setTimeout(() => { flashMinuteur = null; flashActif = false; }, duree);
  }

  function copier(texte: string): void {
    try { renderer.copyToClipboardOSC52(texte); } catch { }
    const candidats = process.platform === "win32" ? [["clip"]] : process.platform === "darwin" ? [["pbcopy"]] : [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]];
    for (const [commande, ...args] of candidats) {
      try { const p = spawn(commande!, args, { stdio: ["pipe", "ignore", "ignore"] }); p.on("error", () => {}); p.stdin?.on("error", () => {}); p.stdin?.end(texte); return; } catch { continue; }
    }
  }

  function annulerQuitter(dire = false): void { if (!quitterDemande) return; quitterDemande = false; if (quitterMinuteur !== null) clearTimeout(quitterMinuteur); quitterMinuteur = null; if (dire) footer.flash("Quit cancelled.", DEFAULT_THEME.textDim); }
  function demanderQuitter(): void { quitterDemande = true; scrollback.append({ type: "system", content: "Quit RontoGPT? Press Ctrl+C again to confirm, or Esc to cancel.", meta: { timestamp: new Date() } }); if (quitterMinuteur !== null) clearTimeout(quitterMinuteur); quitterMinuteur = setTimeout(() => annulerQuitter(false), 8000); }
  function quitter(): void { arret = true; if (quitterMinuteur !== null) clearTimeout(quitterMinuteur); if (flashMinuteur !== null) clearTimeout(flashMinuteur); if (suiviEtat !== null) clearInterval(suiviEtat); destroyRenderer(); process.exit(0); }

  const AIDE = `Commands
  /help      this screen (alias /aide)
  /clear     clear the displayed thread (alias /effacer)
  /forget    the brain forgets the conversation (alias /oubli)
  /memory    what the brain keeps in mind
  /memory set <key> = <value>   edit durable memory
  /memory add <sujet> | <relation> | <objet>   add a fact
  /memory delete <key>           remove a durable memory
  /docs add <path>               add a document (paste/drop its path)
  /docs remove <name>            remove a local document
  /voice on|off                  read completed answers locally
  /langue fr|en|auto             response language
  /context   is the context window full? (gauge + advice)
  /init      load the brain now
  /mcp       MCP bridge and servers
  /api       local API: address, key, routes
  /plan      toggle plan mode (plan first, no execution)
  /plan <goal>   create a checklist; /plan done marks the next task
  /compact   drop the raw window, keep durable facts
  /cloud     Gemini: key, on/off, fallback, long-form writing
  /gemini    send the last question to Gemini (shows it first; /gemini oui)
  /status    profile, llama.cpp, brain budget (alias /etat)
  /quit      exit (alias /quitter)

  Esc        interrupt the current answer
  Ctrl+C     copy the selection; otherwise ask before quitting
  ↑/↓        navigate history (when input focused)
  @          mention files/agents
  /          slash commands`;

  function normaliser(brut: string): string { return brut.trim().replace(/^\//, "").toLowerCase(); }

  let enVol: false | AbortController = false;

  async function envoyer(question: string): Promise<string> {
    if (enVol !== false) { scrollback.append({ type: "system", content: "Un tour est déjà en cours — Échap pour l'interrompre.", meta: { timestamp: new Date() } }); return ""; }
    const q = question.trim(); if (q) footer.addToHistory(q); derniereQuestion = q;
    scrollback.append({ type: "user", content: q, meta: { timestamp: new Date() } }); montrerFil();
    const requete = modePlan ? "Avant de planifier, demande une à trois précisions si l'objectif, le résultat attendu, les contraintes ou les informations nécessaires manquent. Sinon fais un plan exécutable. Réponds uniquement par une liste de 3 à 12 étapes au format exact \`- [ ] action\`, dans l'ordre nécessaire pour achever l'objectif. Chaque action doit être concrète et vérifiable. N'exécute aucune étape.\n\nObjectif : " + question : question;
    const controleur = new AbortController(); enVol = controleur;
    const debut = Date.now(); let phase = "Réflexion"; let texte = ""; let affiche = ""; let morceaux = 0; let thinkingEntryId: string | null = null; let finalEntryId: string | null = null;
    footer.setStatus("thinking", `${phase} (0 s • Échap pour interrompre)`); footer.setPromptVisible(false);
    const tic = setInterval(() => { const s = (Date.now() - debut) / 1000; footer.setStatus("thinking", `⚡ ${phase} (${s.toFixed(1)} s • ${morceaux} morceaux • Échap pour interrompre)`); }, 250);
    let attenteFinie = false; const finirAttente = (): void => { if (attenteFinie) return; attenteFinie = true; clearInterval(tic); footer.setPromptVisible(true); footer.focus(); };

    try {
      for await (const e of repondre(requete, { signal: controleur.signal })) {
        if (e.type === "etape") { phase = e.libelle; footer.setStatus("thinking", `⚡ ${phase} (${((Date.now() - debut) / 1000).toFixed(1)} s • ${morceaux} morceaux)`); }
        else if (e.type === "morceau") { morceaux += 1; phase = e.phase === "reflexion" ? "Réflexion" : "Rédaction"; texte += e.texte; affiche = texte;
          if (thinkingEntryId) scrollback.update(thinkingEntryId, { content: affiche });
          else thinkingEntryId = scrollback.append({ type: "thinking", content: affiche, meta: { timestamp: new Date(), phase } });
        } else if (e.type === "final") { finirAttente();
          if (thinkingEntryId) { scrollback.update(thinkingEntryId, { type: "assistant", content: e.reponse || affiche || "(réponse vide)" }); thinkingEntryId = null; }
          let couleur = DEFAULT_THEME.text;
          if (e.reponse && e.reponse.trim() !== "") affiche = e.reponse; else if (affiche === "") { affiche = "(réponse vide)"; couleur = DEFAULT_THEME.error; }
          if (e.clarification) affiche += `\n\nÀ préciser : ${e.clarification}`;
          finalEntryId = scrollback.append({ type: "assistant", content: affiche, meta: { timestamp: new Date(), provenance: e.provenance, clarification: e.clarification } });
          if (modePlan && e.ok) enregistrerPlan(affiche);
          const pv = e.provenance;
          if (pv && e.ok) {
            const durationSec = ((Date.now() - debut) / 1000).toFixed(1);
            // Determine model label from pv.cerveau
            let modelLabel = "Build RontoGPT SYSTEM";
            const cerveau = (pv as any).cerveau || "";
            if (pv.quitte_la_machine || cerveau === "gemini") modelLabel = "modèle gemini en ligne";
            else if (cerveau.includes("gemma") || pv.niveau === "local") modelLabel = "Gemma 3";
            else if (cerveau.includes("nemotron")) modelLabel = "Nemotron 3 Ultra";
            else if (pv.niveau === "base") modelLabel = "Base de faits locale";
            const grey = DEFAULT_THEME.textDim;
            scrollback.append({
              type: "system",
              content: `${modelLabel} · généré en ${durationSec} s`,
              meta: { timestamp: new Date(), provenance: pv, color: grey }
            });
          }
          if (e.repli_propose) { scrollback.append({ type: "system", content: "💡 /gemini pour demander à Gemini (quitte l'ordinateur)", meta: { timestamp: new Date() } }); }
          if (e.contexte?.sature) { scrollback.append({ type: "system", content: phraseContexte(e.contexte), meta: { timestamp: new Date() } }); }
          if (!e.ok && e.erreur) { scrollback.append({ type: "error", content: `Le cerveau signale : ${e.erreur}`, meta: { timestamp: new Date() } }); }
          majJauge(e.contexte); direLocalement(affiche);
        } else if (e.type === "erreur") { finirAttente(); affiche = e.erreur; scrollback.append({ type: "error", content: e.erreur, meta: { timestamp: new Date(), duration: (Date.now() - debut) } }); footer.flash(`Échec : ${e.erreur}`, DEFAULT_THEME.error); }
      }
    } catch (err) { const interrompu = (err as Error)?.name === "AbortError"; const raison = interrompu ? "interrompu" : (err as Error).message; if (affiche === "") { affiche = `(${raison})`; if (!interrompu) scrollback.append({ type: "error", content: raison, meta: { timestamp: new Date() } }); } else { if (finalEntryId) scrollback.update(finalEntryId, { content: affiche }); } footer.flash(raison, DEFAULT_THEME.error); }
    finally { finirAttente(); enVol = false; void etat().then((info) => majEntete(info)); void memoire().then((m) => majJauge(m?.contexte)); }
    return affiche;
  }

  async function traiter(brut: string | undefined): Promise<void> {
    const texte = (brut ?? "").trim(); if (texte === "") return; const cmd = normaliser(texte);
    if (["aide", "help", "?"].includes(cmd)) { scrollback.append({ type: "system", content: AIDE, meta: { timestamp: new Date() } }); return; }
    if (["effacer", "clear", "cls"].includes(cmd)) { scrollback.clear(); montrerAccueil(); return; }
    if (["oubli", "forget", "reset", "oublier"].includes(cmd)) { const r = await reinitialiser(); scrollback.append({ type: "system", content: r.ok ? "Brain: conversation forgotten." : `Could not forget: ${r.detail}`, meta: { timestamp: new Date() } }); return; }
    const memCmd = texte.match(/^\/?(?:mémoire|memoire|memory)\b\s*(.*)$/i);
    if (memCmd) { const args = (memCmd[1] ?? "").trim(); await handleMemory(args); return; }
    const docsCmd = texte.match(/^\/?(?:docs?|documents)\b\s*(.*)$/i);
    if (docsCmd) { const args = (docsCmd[1] ?? "").trim(); await handleDocs(args); return; }
    const voixCmd = texte.match(/^\/?(?:voice|voix)\b\s*(.*)$/i);
    if (voixCmd) { const option = (voixCmd[1] ?? "").trim().toLowerCase(); if (option === "on" || option === "oui") voixActive = true; else if (option === "off" || option === "non") voixActive = false; else { scrollback.append({ type: "system", content: `Local voice is ${voixActive ? "on" : "off"}. Use /voice on|off.`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: `Local voice ${voixActive ? "enabled — answers are read aloud locally." : "disabled."}`, meta: { timestamp: new Date() } }); return; }
    const langueCmd = texte.match(/^\/?(?:langue|language)\b\s*(.*)$/i);
    if (langueCmd) { const option = (langueCmd[1] ?? "").trim().toLowerCase(); if (["", "?", "statut"].includes(option)) { scrollback.append({ type: "system", content: `Langue des réponses : ${langueChoisie === "auto" ? "automatique" : langueChoisie === "fr" ? "français" : "anglais"}.\n/langue fr · /langue en · /langue auto`, meta: { timestamp: new Date() } }); return; } if (["fr", "français", "francais"].includes(option)) langueChoisie = "fr"; else if (["en", "english", "anglais"].includes(option)) langueChoisie = "en"; else if (["auto", "automatique"].includes(option)) langueChoisie = "auto"; else { scrollback.append({ type: "error", content: "Usage : /langue fr | en | auto", meta: { timestamp: new Date() } }); return; } const r = await reglerLangue(langueChoisie); scrollback.append({ type: "system", content: r.ok ? `Langue des réponses : ${langueChoisie === "auto" ? "automatique" : langueChoisie === "fr" ? "français" : "anglais"}.` : `Impossible de changer la langue : ${r.erreur}`, meta: { timestamp: new Date() } }); return; }
    const cheminDepose = texte.replace(/^"|"$/g, ""); if (existsSync(cheminDepose)) { const r = await editerDocuments({ ajouter: [cheminDepose] }); scrollback.append({ type: "system", content: r.ok ? "Document ajouté. Vous pouvez maintenant lui poser une question." : `Impossible d'ajouter ce document : ${r.erreur ?? r.erreurs?.map((e) => e.erreur).join(", ")}`, meta: { timestamp: new Date() } }); return; }
    if (["init", "initialiser"].includes(cmd)) { if (enVol !== false) { scrollback.append({ type: "system", content: "A turn is already running — press Esc to interrupt it.", meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: "Loading the brain…", meta: { timestamp: new Date() } }); const debut = Date.now(); const r = await initialiser(); if (!r.ok) { scrollback.append({ type: "error", content: `Initialization failed: ${r.detail}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: r.deja ? "The brain was already loaded." : `Brain loaded in ${((Date.now() - debut) / 1000).toFixed(1)} s.`, meta: { timestamp: new Date() } }); void etat().then((info) => majEntete(info)); return; }
    if (cmd === "mcp") { scrollback.append({ type: "system", content: rapportMcp(), meta: { timestamp: new Date() } }); return; }
    const planCmd = texte.match(/^\/?(?:plan|planifier)\b\s*(.*)$/i);
    if (planCmd) { const argument = (planCmd[1] ?? "").trim(); await handlePlan(argument); return; }
    if (["compact", "compacter"].includes(cmd)) { const r = await compacter(); if (!r.ok) { scrollback.append({ type: "error", content: `Compaction failed: ${r.detail}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: r.compacte > 0 ? `Compacted ${r.compacte} message(s) — durable facts were kept.` : "Nothing to compact — the brain has no conversation in mind.", meta: { timestamp: new Date() } }); return; }
    if (["context", "contexte", "ctx"].includes(cmd)) { await afficherContexte(); return; }
    if (/^(?:\/api\b|api\s*$)/i.test(texte)) { await traiterApi(texte.replace(/^\/?api\b/i, "").trim()); return; }
    if (/^\/gemini\b/i.test(texte)) { await traiterGemini(texte.replace(/^\/gemini\b/i, "").trim()); return; }
    const cmdCloud = texte.match(/^(\/)?(?:cloud|nuage)\b\s*(.*)$/i);
    if (cmdCloud) { const arguments_ = (cmdCloud[2] ?? "").trim(); const tete = (arguments_.split(/\s+/)[0] ?? "").toLowerCase(); const sous = ["", "key", "cle", "on", "off", "mode", "writing", "redaction", "rédaction", "model", "modele", "modèle"]; if (cmdCloud[1] === "/" || sous.includes(tete)) { await traiterCloud(arguments_); return; } }
    if (["etat", "status", "statut"].includes(cmd)) { const info = await etat(); majEntete(info); if (info === null) { scrollback.append({ type: "error", content: `API unreachable on ${BASE}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: `version ${info.version} · profile ${info.profil} · model ${info.prete ? "loaded" : "not loaded"}\nllama.cpp ${info.serveur_llama ? "active" : "stopped"} · GPU ${info.gpu} · budget ${info.budget_s} s` + (info.erreur ? `\nerror: ${info.erreur}` : ""), meta: { timestamp: new Date() } }); return; }
    if (["quitter", "exit", "quit", "q", "bye"].includes(cmd)) { if (quitterDemande) quitter(); else demanderQuitter(); return; }
    await envoyer(texte);
  }

  // --- Helpers déplacés hors de main pour lisibilité ---

  async function handleMemory(args: string): Promise<void> {
    if (/^(set|definir)\s+/i.test(args)) { const pair = args.replace(/^(set|definir)\s+/i, "").split(/\s*=\s*/, 2); if (pair.length !== 2 || !pair[0] || !pair[1]) { scrollback.append({ type: "error", content: "Usage: /memory set <key> = <value>", meta: { timestamp: new Date() } }); return; } const r = await editerMemoire(pair[0], pair[1]); scrollback.append({ type: "system", content: r.ok ? `Memory saved: ${pair[0]} = ${pair[1]}` : `Memory edit failed: ${r.erreur}`, meta: { timestamp: new Date() } }); return; }
    if (/^(delete|remove|supprimer)\s+/i.test(args)) { const cle = args.replace(/^(delete|remove|supprimer)\s+/i, "").trim(); const r = await editerMemoire(cle); scrollback.append({ type: "system", content: r.ok ? `Memory removed: ${cle}` : `Memory edit failed: ${r.erreur}`, meta: { timestamp: new Date() } }); return; }
    if (/^(add|a|jouter)\s+/i.test(args)) { const faits = args.replace(/^(add|a|jouter)\s+/i, "").split("|").map((s) => s.trim()).filter(Boolean); if (faits.length !== 3 || faits.some((s) => s.length === 0)) { scrollback.append({ type: "error", content: "Usage: /memory add <sujet> | <relation> | <objet>", meta: { timestamp: new Date() } }); return; } const r = await editerMemoire(faits[0]!, faits[1]! + " | " + faits[2]!); scrollback.append({ type: "system", content: r.ok ? `Fact added: ${faits[0]} — ${faits[1]} — ${faits[2]}` : `Fact add failed: ${r.erreur}`, meta: { timestamp: new Date() } }); return; }
    if (args) { scrollback.append({ type: "error", content: "Usage: /memory [set <key> = <value> | add <sujet> | <relation> | <objet> | delete <key>]", meta: { timestamp: new Date() } }); return; }
    const m = await memoire(); if (m === null) { scrollback.append({ type: "error", content: `API unreachable on ${BASE}`, meta: { timestamp: new Date() } }); return; }
    if (m.messages === 0) { scrollback.append({ type: "system", content: "Memory\n  The brain has no conversation in mind.", meta: { timestamp: new Date() } }); return; }
    const etatUtilisateur = Object.entries(m.etat).map(([k, v]) => `${k}=${v}`).join(", ");
    const derniere = [...m.fenetre].reverse().find((t) => t.role === "user") ?? { contenu: "" };
    const ctx = m.contexte;
    scrollback.append({ type: "system", content: ["Memory", `  ${m.messages} message(s) · ${m.echanges} exchange(s)`, `  user state: ${etatUtilisateur || "(empty)"}`, ctx ? `  context: ${ctx.pourcentage}% (${ctx.tokens}/${ctx.budget} tokens) · compaction: ${ctx.mode ?? "importance"}` + (ctx.sature ? " — SATURATED" : "") : "", derniere ? `  last question: ${derniere.contenu}` : ""].filter((l) => l !== "").join("\n"), meta: { timestamp: new Date() } });
  }

  async function handleDocs(args: string): Promise<void> {
    if (/^(add|ajouter)\s+/i.test(args)) { const chemin = args.replace(/^(add|ajouter)\s+/i, "").trim().replace(/^"|"$/g, ""); const r = await editerDocuments({ ajouter: [chemin] }); scrollback.append({ type: "system", content: r.ok ? `Document added (${r.ajoutes ?? 0}). Its citations will appear in answers.` : `Could not add document: ${r.erreur ?? r.erreurs?.map((e) => e.erreur).join(", ")}`, meta: { timestamp: new Date() } }); return; }
    if (/^(remove|delete|supprimer)\s+/i.test(args)) { const nom = args.replace(/^(remove|delete|supprimer)\s+/i, "").trim(); const r = await editerDocuments({ supprimer: nom }); scrollback.append({ type: "system", content: r.ok ? `${r.supprimes ?? 0} document passage(s) removed.` : `Could not remove: ${r.erreur}`, meta: { timestamp: new Date() } }); return; }
    const r = await documents(); scrollback.append({ type: "system", content: r.ok ? ["Local documents", ...(r.fichiers.length ? r.fichiers.map((f) => `  ${f.nom} · ${f.morceaux} passage(s)`) : ["  No document. Drop or paste a path: /docs add C:\\file.pdf"])].join("\n") : `Documents unavailable: ${r.erreur}`, meta: { timestamp: new Date() } });
  }

  async function handlePlan(argument: string): Promise<void> {
    const arg = argument.toLowerCase();
    if (["fait", "done", "suivant"].includes(arg)) { const prochaine = planActif.find((tache) => !tache.faite); if (prochaine) prochaine.faite = true; afficherPlan(); return; }
    if (["annuler", "abandonner", "stop"].includes(arg)) { planActif = []; modePlan = false; scrollback.append({ type: "system", content: "Plan annulé.", meta: { timestamp: new Date() } }); return; }
    if (argument !== "") { modePlan = true; await envoyer(argument); return; }
    if (planActif.length > 0) { afficherPlan(); return; }
    modePlan = !modePlan; scrollback.append({ type: "system", content: modePlan ? "Plan mode on — the next answers are step-by-step plans, with no execution. Type /plan again to turn it off." : "Plan mode off.", meta: { timestamp: new Date() } }); void etat().then((info) => majEntete(info));
  }

  // Raccourcis clavier globaux
  renderer.keyInput.on("keypress", (key) => {
    if (key.name === "escape") { if (quitterDemande) { annulerQuitter(true); return; } if (enVol !== false) { enVol.abort(); return; } return; }
    if (key.ctrl && key.name === "c") { const selection = renderer.getSelection()?.getSelectedText() ?? ""; if (selection.trim() !== "") { copier(selection); renderer.clearSelection(); flash("Sélection copiée ✓", DEFAULT_THEME.success); return; } if (quitterDemande) quitter(); else demanderQuitter(); }
  });

  // Démarrage
  const santeInfo = await sante(); if (santeInfo === null) { majEntete(null, ` — lance :  uv run rontogpt`); } else { majEntete(await etat()); }

  suiviEtat = setInterval(() => { if (arret || renderer.isDestroyed) return; void etat().then((info) => majEntete(info)); if (enVol === false) void memoire().then((m) => majJauge(m?.contexte)); }, 5000);

  if (process.argv.includes("--smoke")) { const question = process.env.AURA_TUI_SMOKE_Q ?? "Combien font 6 * 7 ?"; const reponse = await envoyer(question); await new Promise((r) => setTimeout(r, 1500)); const blocs = scrollback.length; arret = true; if (suiviEtat !== null) clearInterval(suiviEtat); destroyRenderer(); console.log(`[smoke] écran monté · ${blocs} bloc(s) dans le fil · question : ${question} · réponse affichée : ${JSON.stringify(reponse.slice(0, 120))}`); process.exit(blocs >= 2 && reponse.trim() !== "" ? 0 : 1); }
}

// Wrappers pour les fonctions existantes (identiques à l'ancien index.ts)
function majEntete(info: InfoEtat | null, nota = ""): void { if (arret || renderer.isDestroyed) return; if (info === null) { majTitre(null); etatBase = `hors ligne${nota}`; couleurEtat = DEFAULT_THEME.error; footer.setStatus("error", etatBase); return; } majTitre(info.version); etatBase = info.chargement ? "chargement…" : (modePlan ? "plan mode" : ""); couleurEtat = DEFAULT_THEME.warning; footer.setStatus("idle", etatBase); }
function majJauge(contexte: EtatContexte | null | undefined): void { jaugeContexte = contexte ? `contexte ${contexte.pourcentage} %` : ""; footer.setContextPercent(contexte?.pourcentage ?? 0); }
function phraseContexte(c: EtatContexte): string { const retires = c.compacte ?? 0; const detail = retires > 0 ? `${retires} older message(s) compacted, ${c.gardes ?? 0} kept` : c.raison === "fenetre minimale" ? "nothing older to compact — the next answer may be truncated" : c.raison === "echec compaction" ? "compaction failed — the next answer may be truncated" : c.raison === "hors memoire" ? "memory is empty, but the system/web context alone is still full" : "no message removed"; const apres = c.apres !== undefined ? `, now ${c.apres}%` : ""; return `Context ${c.pourcentage}% full (${c.tokens}/${c.budget} tokens, ${c.mode ?? "importance"}) — ${detail}${apres}.`; }
function afficherPlan(): void { if (planActif.length === 0) { scrollback.append({ type: "system", content: "Aucun plan en cours. Utilisez /plan <objectif>.", meta: { timestamp: new Date() } }); return; } const prochaine = planActif.findIndex((tache) => !tache.faite); const lignes = ["Plan en cours", ...planActif.map((tache, i) => `  ${tache.faite ? "[x]" : "[ ]"} ${i + 1}. ${tache.texte}`)]; lignes.push(prochaine === -1 ? "Terminé." : `Étape en cours : ${prochaine + 1}. ${planActif[prochaine]!.texte}`); lignes.push("/plan fait lorsque cette étape est terminée."); scrollback.append({ type: "system", content: lignes.join("\n"), meta: { timestamp: new Date() } }); }
function enregistrerPlan(texte: string): void { const taches = texte.split("\n").map((ligne) => ligne.match(/^\s*(?:[-*]|\d+[.)])\s*\[\s*\]\s+(.+)$/)).filter((m): m is RegExpMatchArray => m !== null).map((m) => ({ texte: m[1]!.trim(), faite: false })).slice(0, 12); if (taches.length >= 2) { planActif = taches; afficherPlan(); } }
async function traiterGemini(argument: string): Promise<void> { const mot = argument.trim().toLowerCase(); const sansQuestion = derniereQuestion === ""; if (mot === "") { scrollback.append({ type: "system", content: sansQuestion ? "Rien à envoyer : posez d'abord une question locale." : `Envoyer à Gemini ?\n  question : ${derniereQuestion}\n  Cette question QUITTE votre ordinateur (API Google).\n  /gemini oui   l'envoyer\n  /gemini non   rester en local`, meta: { timestamp: new Date() } }); return; } if (mot !== "oui" && mot !== "yes") { scrollback.append({ type: "system", content: "Envoi annulé — rien n'a quitté votre ordinateur.", meta: { timestamp: new Date() } }); return; } if (sansQuestion) { scrollback.append({ type: "system", content: "Rien à envoyer : posez d'abord une question locale.", meta: { timestamp: new Date() } }); return; } if (enVol !== false) { scrollback.append({ type: "system", content: "Un tour est en cours — Échap pour l'interrompre.", meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: `Envoi à Gemini : « ${derniereQuestion} »`, meta: { timestamp: new Date() } }); const r = await demanderGemini(derniereQuestion); if (r.ok) { scrollback.append({ type: "system", content: `Gemini (${r.latence.toFixed(1)} s)\n${r.reponse}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "error", content: `Gemini n'a pas répondu : ${r.erreur ?? "erreur inconnue"}`, meta: { timestamp: new Date() } }); }
async function traiterCloud(arguments_: string): Promise<void> { const brut = arguments_.trim(); if (brut === "") { const info = await cloud(); scrollback.append({ type: "system", content: info === null ? `API unreachable on ${BASE}` : resumeCloud(info), meta: { timestamp: new Date() } }); return; } const [tete = "", ...reste] = brut.split(/\s+/); const mot = tete.toLowerCase(); const valeur = reste.join(" ").trim(); let champs: Parameters<typeof configurerCloud>[0] | null = null; if (mot === "on" || mot === "off") { champs = { actif: mot === "on" }; } else if (mot === "key" || mot === "cle") { const cle = brut.replace(/^(?:key|cle)\s+/i, "").trim(); if (cle === "") { scrollback.append({ type: "error", content: "Usage: /cloud key <Google AI key>", meta: { timestamp: new Date() } }); return; } champs = { cle }; } else if (mot === "mode") { if (!["ask", "auto", "never", "demander", "oui", "non"].includes(valeur.toLowerCase())) { scrollback.append({ type: "error", content: "Usage: /cloud mode ask|auto|never", meta: { timestamp: new Date() } }); return; } champs = { mode: valeur.toLowerCase() }; } else if (["writing", "redaction", "rédaction"].includes(mot)) { const v = valeur.toLowerCase(); if (!["on", "off", "oui", "non", "1", "0"].includes(v)) { scrollback.append({ type: "error", content: "Usage: /cloud writing on|off", meta: { timestamp: new Date() } }); return; } champs = { redaction: ["on", "oui", "1"].includes(v) }; } else if (["model", "modele", "modèle"].includes(mot)) { if (valeur === "") { scrollback.append({ type: "error", content: "Usage: /cloud model <name>", meta: { timestamp: new Date() } }); return; } champs = { modele: valeur }; } else { scrollback.append({ type: "error", content: "Unknown /cloud option. Run /cloud to see the current setup.", meta: { timestamp: new Date() } }); return; } const r = await configurerCloud(champs); if (!r.ok || r.etat === null) { scrollback.append({ type: "error", content: `Cloud configuration failed: ${r.detail}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: resumeCloud(r.etat), meta: { timestamp: new Date() } }); }
function resumeCloud(info: InfoCloud): string { const libelleMode = (mode: string) => mode === "oui" ? "auto (sends without asking)" : mode === "non" ? "never" : "ask (default)"; return ["Gemini", `  key       ${info.cle_presente ? `set (${info.cle_source || "unknown source"})` : "missing — /cloud key <key>"}`, `  enabled   ${info.actif ? "yes" : "no (AURA_GEMINI=0)"}`, `  model     ${info.modele}`, `  fallback  ${libelleMode(info.mode)}`, `  writing   ${info.redaction ? "Gemini first" : "local"}`, info.erreur ? `  last error ${info.erreur}` : "", info.pause ? "  paused after an error — set a valid key to resume" : "", "", "  /cloud key <key>            save the Google AI key to .env", "  /cloud on | off             enable or disable Gemini", "  /cloud mode ask|auto|never  when a weak local answer may escalate", "  /cloud writing on | off     long-form drafts go to Gemini", "  /cloud model <name>         default gemini-2.5-flash"].filter((l) => l !== "").join("\n"); }
async function traiterApi(argument: string): Promise<void> { const mot = argument.split(/\s+/)[0]?.toLowerCase() ?? ""; if (mot !== "") { scrollback.append({ type: "error", content: "Unknown /api option. Run /api for the address and routes.", meta: { timestamp: new Date() } }); return; } const i = await api("status"); if (i === null) { scrollback.append({ type: "error", content: `API unreachable on ${BASE}`, meta: { timestamp: new Date() } }); return; } if (i.ok === false) { scrollback.append({ type: "error", content: `API error: ${i.erreur ?? "unknown"}`, meta: { timestamp: new Date() } }); return; } scrollback.append({ type: "system", content: ["Local API", `  address   ${i.serveur}`, `  openai    ${i.openai ?? `${i.serveur}/v1`}   (Open WebUI, editors)`, `  version   ${i.version}${i.build ? ` (build ${i.build})` : ""}`, `  key       ${i.cle_requise ? "required (X-RontoGPT-Cle or Authorization: Bearer)" : "none — local only (127.0.0.1)"}`, ""].filter((l) => l !== "").join("\n"), meta: { timestamp: new Date() } }); }
async function afficherContexte(): Promise<void> { const m = await memoire(); if (m === null) { scrollback.append({ type: "error", content: `API unreachable on ${BASE}`, meta: { timestamp: new Date() } }); return; } const c = m.contexte; if (!c) { scrollback.append({ type: "system", content: "Context\n  The brain is not loaded yet, so nothing is in the window. Ask a question or run /init.", meta: { timestamp: new Date() } }); return; } const systemTokens = c.systeme_tokens ?? 428; const conversationTokens = c.conversation_tokens ?? Math.max(0, c.tokens - systemTokens); const conversationBudget = c.budget_conversation ?? Math.max(1, c.budget - systemTokens); const conversationPercent = Math.round((conversationTokens / conversationBudget) * 100); const restant = Math.max(0, conversationBudget - conversationTokens); const seuil = c.seuil ?? 90; const verdict = c.sature ? "FULL — the next message pushes the oldest exchanges out of the window" : c.pourcentage >= 60 ? "filling up" : "plenty of room"; const jauge = (p: number, w = 20) => "█".repeat(Math.round((p / 100) * w)) + "░".repeat(w - Math.round((p / 100) * w)); scrollback.append({ type: "system", content: [`Conversation  ${jauge(conversationPercent)}  ${conversationPercent}% used`, `  ${verdict}`, `  ${conversationTokens} / ${conversationBudget} tokens · ${restant} remaining · ${m.messages} message(s) in the window`, `  auto-compaction at ${seuil}% (mode ${c.mode ?? "importance"})`, c.sature ? "  /forget starts fresh · /compact keeps only durable facts · older turns stay searchable" : ""].filter((l) => l !== "").join("\n"), meta: { timestamp: new Date() } }); }
function rapportMcp(): string { const racine = resolve(import.meta.dir, "..", ".."); const pont = join(racine, "scripts", "mcp_coucou.py"); const config = join(racine, ".agents", "mcp_config.json"); const lignes = ["MCP", `  Coucou bridge   ${existsSync(pont) ? "available" : "missing"} (scripts/mcp_coucou.py)`]; if (!existsSync(config)) { lignes.push("  project config  none (.agents/mcp_config.json)"); return lignes.join("\n"); } try { const brut = JSON.parse(readFileSync(config, "utf8")) as { mcpServers?: Record<string, unknown> }; const noms = Object.keys(brut.mcpServers ?? {}); lignes.push(`  project config  ${noms.length > 0 ? noms.join(", ") : "no server"}`); } catch { lignes.push("  project config  unreadable"); } return lignes.join("\n"); }
function direLocalement(texte: string): void { if (!voixActive || !texte.trim()) return; if (process.platform !== "win32") { flash("Voice output is currently available on Windows.", DEFAULT_THEME.warning, 2500); return; } const script = "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(" + JSON.stringify(texte.slice(0, 6000)) + ")"; const encoded = Buffer.from(script, "utf16le").toString("base64"); const p = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], { stdio: "ignore", windowsHide: true }); p.on("error", () => flash("Local voice could not start.", DEFAULT_THEME.error, 2500)); }

main().catch((err) => { console.error("[TUI] Fatal error:", err); destroyRenderer(); process.exit(1); });