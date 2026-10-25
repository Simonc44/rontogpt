/**
 * Vérification du design des réponses — cadres de code et section « Sources » —
 * sans humain devant l'écran : on découpe des réponses connues, on en peint une
 * sur un écran de test, puis on relit la frame et les couleurs réellement
 * produites. Sort en code 1 au premier échec.
 *
 * Usage : tui\node_modules\bin\bun.cmd run src/smoke_design.ts
 */
import { RGBA } from "@opentui/core";
import type { CapturedSpan } from "@opentui/core";
import { createTestRenderer } from "@opentui/core/testing";
import { analyserInline, analyserMarkdown, decouper } from "./design";
import type { Segment } from "./design";
import { TH, creerCorps, creerStyleCode } from "./affichage";
import { controle, terminer } from "./smoke_commun";

// -- 1. le découpage (aucun terminal requis) --------------------------------
const REPONSE = [
  "Voici comment doubler un nombre.",
  "",
  "```python",
  "def double(x):",
  "    return x * 2",
  "```",
  "",
  "La même chose en TypeScript :",
  "",
  "```typescript",
  "const double = (x: number): number => x * 2;",
  "```",
  "",
  "Sources : [Documentation Python](https://docs.python.org/3/); [MDN](https://developer.mozilla.org/)",
].join("\n");

const segments = decouper(REPONSE);
const codes = segments.filter((s): s is Extract<Segment, { type: "code" }> => s.type === "code");
const textes = segments.filter((s): s is Extract<Segment, { type: "texte" }> => s.type === "texte");
const sections = segments.filter((s): s is Extract<Segment, { type: "sources" }> => s.type === "sources");
const section = sections[0];

/** La section « Sources » d'une réponse, pour les cas limites. */
function sectionDe(reponse: string): Extract<Segment, { type: "sources" }> | undefined {
  return decouper(reponse).find(
    (s): s is Extract<Segment, { type: "sources" }> => s.type === "sources");
}

controle("les deux blocs de code sont reconnus, avec leur langage",
  codes.length === 2 && codes[0]?.langue === "python" && codes[1]?.langue === "typescript",
  codes.map((c) => c.langue).join(", ") || "aucun");
controle("le corps du code est repris tel quel (indentation comprise)",
  codes[0]?.code === "def double(x):\n    return x * 2\n",
  JSON.stringify(codes[0]?.code ?? null));
controle("aucun marqueur ``` ne reste dans le texte affiché",
  !segments.some((s) => s.type === "texte" && s.texte.includes("```")),
  "les fences sont sorties du texte");
controle("la mention `Sources :` devient une section dédiée",
  sections.length === 1 && (section?.sources.length ?? 0) === 2
    && section?.sources[0]?.titre === "Documentation Python"
    && section?.sources[0]?.url === "https://docs.python.org/3/"
    && section?.sources[1]?.titre === "MDN",
  section?.sources.map((s) => `${s.titre} <${s.url}>`).join(" | ") ?? "aucune");
controle("le texte qui précède les sources est conservé",
  textes.some((t) => t.texte.includes("La même chose en TypeScript")),
  textes.map((t) => JSON.stringify(t.texte.slice(0, 24))).join(" "));
const urlSeule = sectionDe("Rien.\n\nSource : https://exemple.fr/a");
const sansLien = sectionDe("Rien.\n\nSources : voir le site");
controle("une URL seule devient une source, une mention sans lien garde sa ligne",
  urlSeule?.sources[0]?.url === "https://exemple.fr/a"
  && sansLien?.brut === "Sources : voir le site",
  `${urlSeule?.sources.length ?? 0} source(s) ; repli ${JSON.stringify(sansLien?.brut ?? null)}`);

// -- 2. le rendu réel, relu dans la frame -----------------------------------
const ecran = await createTestRenderer({ width: 78, height: 30 });
const corps = creerCorps(ecran.renderer, creerStyleCode(ecran.renderer));
ecran.renderer.root.add(corps.bloc);
corps.peindre(REPONSE);
await ecran.flush();

const frame = ecran.captureCharFrame();
controle("le bloc de code est encadré et porte son étiquette",
  frame.includes("╭") && frame.includes("╰") && frame.includes("python") && frame.includes("typescript"),
  "cadre arrondi + étiquettes peints");
controle("le code apparaît tel quel, sur ses propres lignes",
  frame.includes("def double(x):") && frame.includes("return x * 2")
  && frame.includes("const double = (x: number): number => x * 2;"),
  "corps du code présent dans la frame");
controle("la section « 🔗 Sources » remplace la ligne brute",
  frame.includes("🔗 Sources") && frame.includes("Documentation Python")
  && frame.includes("docs.python.org") && frame.includes("developer.mozilla.org"),
  "titres et liens peints");
controle("chaque source porte sa pastille de site (le favicon du terminal)",
  frame.includes(" P ") && frame.includes(" M ") && frame.includes("docs.python.org"),
  "initiales P (python.org) et M (mozilla.org) peintes");
controle("plus aucun marqueur brut à l'écran",
  !frame.includes("```") && !frame.includes("Sources : ["),
  "ni fence ni ligne markdown brute");

// Forme réelle d'une réponse web du 1B (12 oct. 2026) : le bloc de code de
// l'échafaudage est recopié sans être refermé.
const FUITE = [
  "C'est :",
  "",
  "```python",
  "def recherche_web(url):",
  "    # N'affirme que ce que les extraits appuient",
  "    return url",
  "",
  "Sources : [Tournoi de Wimbledon](https://fr.wikipedia.org/wiki/Tournoi_de_Wimbledon_2025)",
].join("\n");

const fuite = decouper(FUITE);
controle("un bloc de code jamais refermé est cadré quand même",
  fuite.some((s) => s.type === "code" && s.langue === "python")
  && !fuite.some((s) => s.type === "texte" && s.texte.includes("```")),
  fuite.map((s) => s.type).join(", "));
controle("un ``` au milieu d'une phrase reste du texte",
  decouper("Utilise ``` pour ouvrir un bloc.").every((s) => s.type === "texte"),
  "aucun cadre parasite");

// -- 1 bis. markdown : gras, italique, code, liens, titres, listes ----------
const MARKDOWN = [
  "# Titre principal",
  "",
  "Voici du **gras**, de l'*italique*, du `code inline`, ~~barre~~ et un [lien](https://exemple.fr).",
  "",
  "## Sous-titre",
  "",
  "- premier point",
  "- **point gras**",
  "  - sous-point",
  "1. etape une",
  "2. etape deux",
  "",
  "> une citation",
  "",
  "---",
  "",
  "Un identifiant rag_binaire reste intact.",
].join("\n");

const lignesMd = analyserMarkdown(MARKDOWN);
const inline = analyserInline("**gras** *italique* `code` [lien](https://ex.fr) rag_binaire _vraiment_");
controle("markdown : titres, puces, numeros, citation et regle sont reconnus",
  lignesMd.some((l) => l.genre === "titre" && l.niveau === 1)
  && lignesMd.some((l) => l.genre === "titre" && l.niveau === 2)
  && lignesMd.some((l) => l.genre === "puce" && l.marque === "•" && l.retrait === 1)
  && lignesMd.some((l) => l.genre === "numero" && l.marque === "1.")
  && lignesMd.some((l) => l.genre === "citation")
  && lignesMd.some((l) => l.genre === "regle"),
  lignesMd.map((l) => l.genre).join(", "));
controle("markdown : gras, italique, code et lien deviennent des styles",
  inline.some((s) => s.texte === "gras" && s.gras === true)
  && inline.some((s) => s.texte === "italique" && s.italique === true)
  && inline.some((s) => s.texte === "code" && s.code === true)
  && inline.some((s) => s.texte === "lien" && s.lien === "https://ex.fr")
  && inline.some((s) => s.texte === "vraiment" && s.italique === true),
  inline.map((s) => `${s.texte}${s.gras ? "(g)" : ""}${s.italique ? "(i)" : ""}${s.code ? "(c)" : ""}${s.lien ? "(l)" : ""}`).join(" "));
controle("markdown : un identifiant comme rag_binaire n'est pas casse",
  inline.some((s) => s.texte.includes("rag_binaire") && s.italique !== true)
  && !inline.some((s) => s.italique === true && s.texte.includes("binaire")));
const pieges = analyserInline("2 * 3 * 4, __init__ et __name__");
controle("markdown : multiplication et dunders restent du texte brut",
  pieges.every((s) => s.gras !== true && s.italique !== true)
  && pieges.map((s) => s.texte).join("").includes("2 * 3 * 4")
  && pieges.map((s) => s.texte).join("").includes("__init__"),
  pieges.map((s) => s.texte).join(""));

const spansDe = (): CapturedSpan[] => ecran.captureSpans().lines.flatMap((l) => l.spans);
const aLaCouleur = (liste: CapturedSpan[], hex: string): boolean => {
  const [r, g, b] = RGBA.fromHex(hex).toInts();
  return liste.some((s) => {
    const [sr, sg, sb] = s.fg.toInts();
    return sr === r && sg === g && sb === b;
  });
};

// La coloration vient du worker tree-sitter : elle n'est donc pas peinte dès la
// première frame. On laisse passer quelques frames avant de conclure.
let spans = spansDe();
let colorie = false;
for (let i = 0; i < 30 && !colorie; i += 1) {
  await new Promise((r) => setTimeout(r, 100));
  await ecran.renderOnce();
  spans = spansDe();
  colorie = aLaCouleur(spans, "#c678dd") && aLaCouleur(spans, "#e5c07b");
}
controle("le code TypeScript est coloré par sa grammaire",
  colorie, colorie ? "mots-clés et types colorés" : "aucune couleur de grammaire dans la frame");
controle("le code sans grammaire embarquée (python) reste lisible en clair",
  aLaCouleur(spans, TH.codeTexte),
  `couleur de repli ${TH.codeTexte}`);

corps.peindre(FUITE);
await ecran.flush();
const frameFuite = ecran.captureCharFrame();
controle("la fuite d'échafaudage s'affiche cadrée, sans marqueur à l'écran",
  frameFuite.includes("def recherche_web(url):") && frameFuite.includes("python")
  && !frameFuite.includes("```"),
  "bloc cadré et étiqueté");

corps.peindre(MARKDOWN);
await ecran.flush();
const frameMd = ecran.captureCharFrame();
const spansMd = ecran.captureSpans().lines.flatMap((l) => l.spans);
controle("markdown : aucun marqueur brut ne reste a l'ecran",
  !frameMd.includes("**") && !frameMd.includes("*italique*") && !frameMd.includes("`code")
  && !frameMd.includes("[lien]") && !frameMd.includes("## ") && !frameMd.includes("> une"),
  "marqueurs retires");
controle("markdown : le contenu reste lisible (titres, puces, citation)",
  frameMd.includes("Titre principal") && frameMd.includes("Sous-titre")
  && frameMd.includes("premier point") && frameMd.includes("sous-point")
  && frameMd.includes("etape une") && frameMd.includes("une citation")
  && frameMd.includes("rag_binaire"),
  "texte visible");
controle("markdown : puces et titres prennent la couleur d'accent",
  aLaCouleur(spansMd, TH.accent), `accent ${TH.accent}`);
controle("markdown : le code inline a un fond distinct",
  spansMd.some((s) => {
    const [r, g, b] = s.bg.toInts();
    const [br, bgc, bb] = RGBA.fromHex(TH.codeFond).toInts();
    return r === br && g === bgc && b === bb;
  }), `fond ${TH.codeFond}`);

await ecran.dispose();

terminer("TOUT EST OK");
