/**
 * Habillage de l'écran RontoGPT : la palette, et le rendu d'une réponse.
 *
 * Ce module ne parle ni à l'API ni au clavier : il reçoit du texte et pose des
 * renderables. Il est donc importable par la vérification automatique
 * (`src/smoke_design.ts`), qui monte un écran de test et relit la frame peinte.
 *
 * Design d'une réponse :
 *   - le texte courant rendu en markdown : gras, italique, code inline, liens,
 *     titres, listes, cases à cocher, citations et règles ;
 *   - un bloc de code `` ```langue `` : cadre arrondi, étiquette de langage sur
 *     le cadre, corps coloré par la grammaire quand elle est embarquée ;
 *   - une section « 🔗 Sources » : un titre et son lien par source, au lieu de
 *     la ligne brute `Sources : [titre](url)` (même partage que la GUI,
 *     `aura/gui.py::_bulle_aura`).
 */
import { spawn } from "node:child_process";
import {
  BoxRenderable,
  CodeRenderable,
  StyledText,
  SyntaxStyle,
  TextRenderable,
  bg,
  bold,
  fg,
  italic,
  link,
  strikethrough,
  t,
  underline,
} from "@opentui/core";
import type { RenderContext, TextChunk } from "@opentui/core";
import {
  analyserMarkdown, chemin, couleurSite, decouper, domaine, filetype, initiale,
  texteSurPastille,
} from "./design";
import type { InlineSpan, Segment } from "./design";

// -- palette (noir Codex) ---------------------------------------------------
export const TH = {
  fond: "#0d0d0d",
  bande: "#2b2b2b",        // barre de saisie + message utilisateur
  texte: "#e6e6e6",
  doux: "#b4b4b4",
  discret: "#7c7c7c",
  trait: "#4a4a4a",
  accent: "#5aa9ff",       // le « >_ » de l'en-tête
  modele: "#efe3b0",       // ligne « RontoGPT <profil> » du pied de page
  logo: "#34474f",         // emblème : volontairement très faible contraste
  alerte: "#e5c07b",
  erreur: "#f7768e",
  ok: "#7fd6a5",
  codeFond: "#151515",     // le bloc de code se détache à peine du fond
  codeBord: "#333333",
  codeTexte: "#d8dee9",
} as const;

/**
 * Couleurs de la coloration syntaxique. Les noms sont ceux des captures
 * tree-sitter (`@keyword`, `@string.escape`…) ; une capture pointée retombe sur
 * son nom de base (`@keyword.conditional` → `keyword`), donc ces entrées
 * couvrent aussi les variantes.
 */
const COULEURS_CODE: Record<string, { fg: string; bold?: boolean; italic?: boolean }> = {
  comment: { fg: "#6f7783", italic: true },
  keyword: { fg: "#c678dd" },
  string: { fg: "#98c379" },
  number: { fg: "#d19a66" },
  constant: { fg: "#d19a66" },
  boolean: { fg: "#d19a66" },
  function: { fg: "#61afef" },
  constructor: { fg: "#61afef" },
  "function.method": { fg: "#61afef" },
  type: { fg: "#e5c07b" },
  module: { fg: "#e5c07b" },
  label: { fg: "#e5c07b" },
  attribute: { fg: "#e5c07b" },
  operator: { fg: "#56b6c2" },
  punctuation: { fg: "#9aa4b2" },
  variable: { fg: TH.texte },
  "variable.member": { fg: "#e06c75" },
  "variable.parameter": { fg: "#d19a66" },
};

/**
 * Le style de coloration, à créer UNE fois par écran (il est lié au contexte
 * natif du renderer). `fromStyles` exige un propriétaire de ressources : la
 * scène du renderer est le seul qui partage le contexte des renderables.
 */
export function creerStyleCode(renderer: RenderContext): SyntaxStyle {
  return SyntaxStyle.fromStyles(COULEURS_CODE, renderer.nativeScene);
}

export interface OptionsPeinture {
  /** Couleur du texte : le corps écrit en `texte`, sauf réponse vide ou échec. */
  couleur?: string;
  /** `false` : tout d'un bloc, sans cadre — le chemin bon marché du flux. */
  riche?: boolean;
}

// -- markdown ---------------------------------------------------------------
/** Attributs appliqués à TOUS les morceaux d'une ligne (titre, citation…). */
interface AttributsLigne {
  gras?: boolean;
  italique?: boolean;
  souligne?: boolean;
}

/**
 * Traduit les morceaux markdown en chunks styles d'OpenTUI. Chaque marqueur
 * devient un attribut réel : `**gras**` → bold, `*italique*` → italic,
 * `\`code\`` → fond de code, `[titre](url)` → lien cliquable souligné.
 */
function morceauxStyles(
  spans: InlineSpan[],
  couleur: string,
  attributs: AttributsLigne = {},
): TextChunk[] {
  return spans.map((span) => {
    let chunk = fg(span.code === true || span.lien !== undefined ? TH.accent : couleur)(span.texte);
    if (span.code === true) chunk = bg(TH.codeFond)(chunk);
    if (span.lien !== undefined) chunk = link(span.lien)(underline(chunk));
    if (span.barre === true) chunk = strikethrough(chunk);
    if (span.italique === true || attributs.italique === true) chunk = italic(chunk);
    if (span.gras === true || attributs.gras === true) chunk = bold(chunk);
    if (attributs.souligne === true) chunk = underline(chunk);
    return chunk;
  });
}

/** Une ligne de texte riche, repliée sur la largeur disponible. */
function ligneStylee(
  renderer: RenderContext,
  spans: InlineSpan[],
  couleur: string,
  attributs: AttributsLigne = {},
): TextRenderable {
  return new TextRenderable(renderer, {
    content: new StyledText(morceauxStyles(spans, couleur, attributs)),
    fg: couleur,
    wrapMode: "word",
    width: "100%",
  });
}

/**
 * Un bloc de texte markdown : une ligne = un renderable. Les listes gardent
 * leur puce en accent et leur retrait ; les titres sont colorés ; les citations
 * portent une barre verticale. Le but est lisible ET compact — pas de cadre.
 */
function blocMarkdown(renderer: RenderContext, contenu: string, couleur: string): BoxRenderable {
  const boite = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    backgroundColor: TH.fond,
  });

  for (const ligne of analyserMarkdown(contenu)) {
    if (ligne.genre === "vide") {
      boite.add(new TextRenderable(renderer, { content: " ", fg: couleur, width: "100%" }));
      continue;
    }
    if (ligne.genre === "regle") {
      boite.add(new TextRenderable(renderer, {
        content: "─".repeat(48),
        fg: TH.trait,
        width: "100%",
      }));
      continue;
    }
    if (ligne.genre === "titre") {
      boite.add(ligneStylee(renderer, ligne.spans,
        ligne.niveau <= 2 ? TH.accent : TH.doux,
        { gras: true, souligne: ligne.niveau === 1 }));
      continue;
    }
    if (ligne.genre === "citation") {
      const rang = new BoxRenderable(renderer, {
        width: "100%", flexDirection: "row", backgroundColor: TH.fond,
      });
      rang.add(new TextRenderable(renderer, {
        content: t`${fg(TH.accent)("▎")} `, width: 2,
      }));
      const txt = ligneStylee(renderer, ligne.spans, TH.doux, { italique: true });
      txt.flexGrow = 1;
      rang.add(txt);
      boite.add(rang);
      continue;
    }
    if (ligne.genre === "puce" || ligne.genre === "numero") {
      const rang = new BoxRenderable(renderer, {
        width: "100%", flexDirection: "row", backgroundColor: TH.fond,
        paddingLeft: ligne.retrait * 2,
      });
      rang.add(new TextRenderable(renderer, {
        content: t`${fg(TH.accent)(ligne.marque)} `,
        width: ligne.marque.length + 1,
      }));
      const txt = ligneStylee(renderer, ligne.spans, couleur);
      txt.flexGrow = 1;
      rang.add(txt);
      boite.add(rang);
      continue;
    }
    boite.add(ligneStylee(renderer, ligne.spans, couleur));
  }
  return boite;
}

export interface Corps {
  /** Le conteneur à poser dans le fil ; `visible` le masque tant qu'il est vide. */
  bloc: BoxRenderable;
  peindre(texte: string, options?: OptionsPeinture): void;
}

/** Le texte courant d'une réponse. */
function texte(renderer: RenderContext, contenu: string, couleur: string): TextRenderable {
  return new TextRenderable(renderer, {
    content: contenu,
    fg: couleur,
    wrapMode: "word",
    width: "100%",
  });
}

/**
 * Un bloc de code : cadre arrondi portant l'étiquette de langage, corps
 * coloré. Le contenu n'est jamais réinterprété — c'est le code reçu, tel quel.
 */
function cadreCode(
  renderer: RenderContext,
  style: SyntaxStyle,
  segment: Extract<Segment, { type: "code" }>,
): BoxRenderable {
  const cadre = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    border: true,
    borderStyle: "rounded",
    borderColor: TH.codeBord,
    backgroundColor: TH.codeFond,
    title: ` ${segment.langue} `,
    titleColor: TH.accent,
    paddingLeft: 1,
    paddingRight: 1,
  });

  // Bouton "Copier" dans le coin haut-droit du cadre
  const btnCopier = new TextRenderable(renderer, {
    content: " 📋 ",
    fg: TH.discret,
    bg: TH.codeFond,
  });
  btnCopier.on("click", () => {
    // Copie via OSC 52 + fallback système
    const code = segment.code.replace(/\s+$/, "");
    // OSC 52 via renderer (si supporté)
    try { (renderer as any).copyToClipboardOSC52?.(code); } catch { /* terminal sans OSC 52 */ }
    const candidats = process.platform === "win32"
      ? [["clip"]]
      : process.platform === "darwin"
        ? [["pbcopy"]]
        : [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]];
    for (const [commande, ...args] of candidats) {
      try {
        const p = spawn(commande!, args, { stdio: ["pipe", "ignore", "ignore"] });
        p.stdin?.end(code);
        break;
      } catch { /* outil suivant */ }
    }
    // Feedback visuel temporaire
    const original = btnCopier.content;
    btnCopier.content = " ✓ ";
    btnCopier.fg = TH.ok;
    renderer.requestRender();
    setTimeout(() => {
      btnCopier.content = original;
      btnCopier.fg = TH.discret;
      renderer.requestRender();
    }, 1500);
  });
  btnCopier.on("mouseenter", () => { btnCopier.fg = TH.accent; renderer.requestRender(); });
  btnCopier.on("mouseleave", () => { btnCopier.fg = TH.discret; renderer.requestRender(); });

  // Barre de titre avec le bouton à droite
  const titreBarre = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    backgroundColor: TH.codeFond,
    paddingLeft: 1,
    paddingRight: 1,
  });
  titreBarre.add(new TextRenderable(renderer, {
    content: ` ${segment.langue} `,
    fg: TH.accent,
  }));
  titreBarre.add(btnCopier);

  cadre.add(titreBarre);
  cadre.add(new CodeRenderable(renderer, {
    content: segment.code.replace(/\s+$/, ""),
    filetype: filetype(segment.langue),
    syntaxStyle: style,
    fg: TH.codeTexte,
    width: "100%",
    selectable: true,
    conceal: false,
    drawUnstyledText: true,
  }));
  return cadre;
}

/**
 * La section « Sources » : un titre, puis un titre de lien et son URL par
 * source. `brut` sert de repli quand aucune URL n'a pu être extraite : mieux
 * vaut afficher la ligne telle quelle que de perdre l'information.
 */
function blocSources(
  renderer: RenderContext,
  segment: Extract<Segment, { type: "sources" }>,
): BoxRenderable {
  const section = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    backgroundColor: TH.fond,
  });
  section.add(new TextRenderable(renderer, { content: "🔗 Sources", fg: TH.doux, width: "100%" }));

  if (segment.sources.length === 0) {
    section.add(texte(renderer, segment.brut ?? "", TH.doux));
    return section;
  }
  // Une carte par source, façon résultat Google : pastille de site (le
  // « favicon » du terminal), titre cliquable, puis `domaine › chemin`.
  segment.sources.forEach((source, i) => {
    const hote = domaine(source.url);
    const fond = couleurSite(hote);
    const carte = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "row",
      marginTop: i === 0 ? 0 : 1,
      backgroundColor: TH.fond,
    });
    carte.add(new TextRenderable(renderer, {
      content: t`${bg(fond)(fg(texteSurPastille(fond))(bold(` ${initiale(hote)} `)))}`,
      width: 4,
    }));
    const colonne = new BoxRenderable(renderer, {
      flexGrow: 1,
      flexDirection: "column",
      backgroundColor: TH.fond,
    });
    colonne.add(new TextRenderable(renderer, {
      content: t`${link(source.url)(fg(TH.accent)(bold(`${i + 1}. ${source.titre}`)))}`,
      wrapMode: "word",
      width: "100%",
    }));
    const fil = chemin(source.url);
    colonne.add(new TextRenderable(renderer, {
      content: fil ? `${hote} › ${fil}` : hote,
      fg: TH.discret,
      wrapMode: "word",
      width: "100%",
    }));
    carte.add(colonne);
    section.add(carte);
  });
  return section;
}

/**
 * Le corps d'une réponse : un conteneur qu'on repeint. Pendant le flux
 * (`riche: false`) le texte est posé d'un seul tenant — reconstruire un cadre
 * de code à chaque morceau coûterait cher pour un état qui dure un instant.
 * Au `final` (`riche: true`, le défaut) la réponse prend sa forme définitive.
 */
/** Ligne d'attente pendant un tour : phase, secondes écoulées et, dès qu'un
 *  morceau a été reçu, le compte réel de morceaux (pas de total inventé). */
export function libelleAttente(phase: string, secondes: number, morceaux: number): string {
  const mot = morceaux > 1 ? "morceaux" : "morceau";
  const compte = morceaux > 0 ? ` • ${morceaux} ${mot}` : "";
  return `${phase} (${Math.floor(secondes)} s${compte} • Échap pour interrompre)`;
}

export function creerCorps(renderer: RenderContext, style: SyntaxStyle): Corps {
  // `gap`: un texte, un cadre de code et une section « Sources » restent des
  // morceaux distincts — l'air entre eux fait partie du dessin.
  const bloc = new BoxRenderable(renderer, {
    width: "100%",
    flexDirection: "column",
    gap: 1,
    backgroundColor: TH.fond,
  });

  const peindre = (contenu: string, options: OptionsPeinture = {}): void => {
    const { couleur = TH.texte, riche = true } = options;
    for (const enfant of [...bloc.getChildren()]) enfant.destroy();
    bloc.visible = contenu !== "";
    if (contenu === "") return;

    if (!riche) {
      const flux = texte(renderer, contenu, couleur);
      flux.bg = TH.fond;
      bloc.add(flux);
      return;
    }
    for (const segment of decouper(contenu)) {
      if (segment.type === "code") bloc.add(cadreCode(renderer, style, segment));
      else if (segment.type === "sources") bloc.add(blocSources(renderer, segment));
      else bloc.add(blocMarkdown(renderer, segment.texte, couleur));
    }
  };

  return { bloc, peindre };
}
