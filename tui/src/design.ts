/**
 * Découpage d'une réponse RontoGPT en morceaux affichables : texte courant,
 * blocs de code (```langue) et section « Sources ».
 *
 * VOLONTAIREMENT sans aucune dépendance à OpenTUI : c'est la partie risquée de
 * l'affichage (les expressions régulières), donc elle se vérifie seule, sans
 * terminal (`bun src/smoke_design.ts`).
 *
 * Les expressions viennent de la GUI (`aura/gui.py` : _RE_CODE_FENCE,
 * _RE_SOURCE_MD, _RE_SOURCE_TEXTE). Le cerveau écrit ces deux formes — un bloc
 * de code, une ligne de sources — et les deux interfaces doivent les lire
 * pareil : une seule définition du contrat, ici. Seule exception, documentée et
 * vérifiée : un bloc de code jamais refermé, que la GUI laisse brut (voir
 * `decouperOuvert`).
 */
export interface Source {
  titre: string;
  url: string;
}

export type Segment =
  | { type: "texte"; texte: string }
  | { type: "code"; langue: string; code: string }
  /** `brut` : la ligne d'origine, quand aucun lien n'a pu être extrait. */
  | { type: "sources"; sources: Source[]; brut: string | null };

// -- markdown : le texte enrichi d'une reponse ---------------------------------
// Le cerveau ecrit du markdown simple (**gras**, *italique*, `code`, [lien](url),
// titres, listes, citations). On le DECODE ici, sans OpenTUI : l'affichage ne
// recoit que des morceaux deja styles, et la logique se teste seule.

export interface InlineSpan {
  texte: string;
  gras?: boolean;
  italique?: boolean;
  code?: boolean;
  barre?: boolean;
  lien?: string;
}

export type LigneMarkdown =
  | { genre: "vide" }
  | { genre: "titre"; niveau: number; spans: InlineSpan[] }
  | { genre: "puce"; marque: string; retrait: number; spans: InlineSpan[] }
  | { genre: "numero"; marque: string; retrait: number; spans: InlineSpan[] }
  | { genre: "citation"; spans: InlineSpan[] }
  | { genre: "regle" }
  | { genre: "texte"; spans: InlineSpan[] };

const RE_CODE_INLINE = /^`([^`\n]+)`/;
const RE_LIEN_INLINE = /^\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/;
// Les marqueurs doivent encadrer du texte SANS espace collé : sans ça,
// « 2 * 3 * 4 » deviendrait de l'italique. (Le `__gras__` n'est pas reconnu :
// il casserait `__init__` et `__name__`, trop fréquents dans les réponses.)
const RE_GRAS = /^\*\*(?!\s)([\s\S]+?)(?<!\s)\*\*/;
const RE_BARRE = /^~~(?!\s)([\s\S]+?)(?<!\s)~~/;
const RE_ITALIQUE = /^\*(?!\s)([^*\n]+?)(?<!\s)\*/;
const RE_ITALIQUE_SOULIGNE = /^_(?!_|\s)([^_\n]+?)(?<![\s_])_(?![\w_])/;
const RE_URL_NUE = /^https?:\/\/[^\s)\]]+/;

/**
 * Decoupe une ligne en morceaux styles (gras, italique, code, lien, barre).
 * Les marqueurs sont retires : l'ecran ne montre jamais `**` ni `[texte](url)`.
 * Volontairement simple : une imbrication est relue recursivement, mais les
 * cas tordus (marqueurs echappes, tableaux) restent du texte.
 */
export function analyserInline(texte: string, base: Partial<InlineSpan> = {}): InlineSpan[] {
  const spans: InlineSpan[] = [];
  let tampon = "";
  let i = 0;
  const vider = (): void => {
    if (tampon !== "") {
      spans.push({ ...base, texte: tampon });
      tampon = "";
    }
  };

  while (i < texte.length) {
    const reste = texte.slice(i);
    let m: RegExpExecArray | null;

    if ((m = RE_CODE_INLINE.exec(reste)) !== null) {
      vider();
      spans.push({ ...base, texte: m[1] ?? "", code: true });
      i += m[0].length;
      continue;
    }
    if ((m = RE_LIEN_INLINE.exec(reste)) !== null) {
      vider();
      spans.push({ ...base, texte: m[1] ?? "", lien: m[2] ?? "" });
      i += m[0].length;
      continue;
    }
    if ((m = RE_GRAS.exec(reste)) !== null) {
      vider();
      spans.push(...analyserInline(m[1] ?? "", { ...base, gras: true }));
      i += m[0].length;
      continue;
    }
    if ((m = RE_BARRE.exec(reste)) !== null) {
      vider();
      spans.push(...analyserInline(m[1] ?? "", { ...base, barre: true }));
      i += m[0].length;
      continue;
    }
    if ((m = RE_ITALIQUE.exec(reste)) !== null) {
      vider();
      spans.push(...analyserInline(m[1] ?? "", { ...base, italique: true }));
      i += m[0].length;
      continue;
    }
    // `_italique_` seulement s'il ouvre un mot : sinon `rag_binaire` deviendrait
    // de l'italique. C'est le piege des identifiants techniques.
    const debutMot = i === 0 || !/[\w_]/.test(texte[i - 1] ?? "");
    if (debutMot && (m = RE_ITALIQUE_SOULIGNE.exec(reste)) !== null) {
      vider();
      spans.push(...analyserInline(m[1] ?? "", { ...base, italique: true }));
      i += m[0].length;
      continue;
    }
    if ((m = RE_URL_NUE.exec(reste)) !== null) {
      vider();
      const brut = m[0];
      const url = brut.replace(/[.,;:!?]+$/, "");
      spans.push({ ...base, texte: url, lien: url });
      tampon += brut.slice(url.length);      // la ponctuation reste du texte
      i += brut.length;
      continue;
    }
    tampon += texte[i] ?? "";
    i += 1;
  }
  vider();
  return spans;
}

/**
 * Decoupe une reponse en lignes typées : titres, puces, numeros, citations,
 * regles, lignes vides et paragraphes. Les blocs de code sont deja sortis par
 * `decouper` : ce analyseur ne voit que le texte courant.
 */
export function analyserMarkdown(texte: string): LigneMarkdown[] {
  return texte.replace(/\r\n?/g, "\n").split("\n").map((ligne): LigneMarkdown => {
    if (ligne.trim() === "") return { genre: "vide" };

    const titre = /^\s{0,3}(#{1,6})\s+(.*)$/.exec(ligne);
    if (titre !== null) {
      return { genre: "titre", niveau: (titre[1] ?? "#").length, spans: analyserInline((titre[2] ?? "").trim()) };
    }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(ligne)) return { genre: "regle" };

    const puce = /^(\s*)[-*+]\s+(.*)$/.exec(ligne);
    if (puce !== null) {
      let contenu = (puce[2] ?? "").trim();
      let marque = "•";
      const tache = /^\[([ xX])\]\s+(.*)$/.exec(contenu);
      if (tache !== null) {
        marque = (tache[1] ?? " ").toLowerCase() === "x" ? "☑" : "☐";
        contenu = tache[2] ?? "";
      }
      return { genre: "puce", marque, retrait: Math.min(6, Math.floor((puce[1] ?? "").length / 2)), spans: analyserInline(contenu) };
    }

    const numero = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(ligne);
    if (numero !== null) {
      return { genre: "numero", marque: `${numero[2] ?? "1"}.`, retrait: Math.min(6, Math.floor((numero[1] ?? "").length / 2)), spans: analyserInline(numero[3] ?? "") };
    }

    const citation = /^\s*>\s?(.*)$/.exec(ligne);
    if (citation !== null) return { genre: "citation", spans: analyserInline(citation[1] ?? "") };

    return { genre: "texte", spans: analyserInline(ligne) };
  });
}

/** ```langue\n…``` — même expression que `aura/gui.py` (_RE_CODE_FENCE). */
const RE_CODE = /```(\w*)\s*\n([\s\S]*?)```/g;

/**
 * `Sources : …` — même expression que la GUI (_RE_SOURCE_TEXTE). Le point ne
 * franchit pas le saut de ligne : on ne prend que la ligne de la mention.
 */
const RE_ENTETE_SOURCES = /(?:Source|Sources)\s*:\s*(.+)/i;

/** `[titre](https://url)` — _RE_SOURCE_MD. */
const RE_SOURCE_MD = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;

/** URL nue : certaines réponses citent un lien sans lui donner de titre. */
const RE_URL = /https?:\/\/[^\s)\]>,;]+/g;

/** Au plus ce nombre de sources : au-delà, la réponse se noie dans les liens. */
export const SOURCES_MAX = 5;

// -- identité visuelle d'une source (façon résultats Google) ----------------
// Un terminal ne peut pas charger le vrai favicon d'un site : on dessine son
// équivalent — une pastille colorée portant l'initiale du site, la couleur
// étant stable pour un domaine donné (ou celle de la marque connue).

/** Nom de domaine lisible : sans `www.`, en minuscules. */
export function domaine(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./i, "").toLowerCase();
  } catch {
    return url.replace(/^https?:\/\/(www\.)?/i, "").split(/[/?#]/)[0]?.toLowerCase() ?? url;
  }
}

/** Chemin de l'URL façon fil d'Ariane Google : `wiki › Tournoi_de_Wimbledon`. */
export function chemin(url: string, max = 48): string {
  let morceaux: string[] = [];
  try {
    morceaux = new URL(url).pathname.split("/").filter(Boolean).map((m) => decodeURIComponent(m));
  } catch {
    return "";
  }
  const texte = morceaux.join(" › ");
  return texte.length > max ? `${texte.slice(0, max - 1)}…` : texte;
}

const MARQUES: Array<[RegExp, string]> = [
  [/(^|\.)wikipedia\.org$/, "#636466"],
  [/(^|\.)wikidata\.org$/, "#339966"],
  [/(^|\.)github\.com$/, "#6e40c9"],
  [/(^|\.)stackoverflow\.com$/, "#f48024"],
  [/(^|\.)python\.org$/, "#3776ab"],
  [/(^|\.)mozilla\.org$/, "#1c6fd1"],
  [/(^|\.)lemonde\.fr$/, "#2a2a2a"],
  [/(^|\.)youtube\.com$/, "#e52d27"],
  [/(^|\.)reddit\.com$/, "#ff4500"],
  [/(^|\.)gouv\.fr$/, "#3a55b4"],
];
const PASTILLES = ["#4285f4", "#ea4335", "#34a853", "#ab47bc", "#00acc1", "#ff7043", "#5c6bc0", "#26a69a"];

/** Couleur de la pastille : celle de la marque connue, sinon dérivée du domaine. */
export function couleurSite(hote: string): string {
  for (const [motif, couleur] of MARQUES) if (motif.test(hote)) return couleur;
  let h = 0;
  for (const c of hote) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PASTILLES[h % PASTILLES.length] ?? "#4285f4";
}

/** Initiale du site : `wikipedia.org` → `W`, `docs.python.org` → `P`. */
export function initiale(hote: string): string {
  const parts = hote.split(".");
  const nom = parts.length >= 2 ? (parts[parts.length - 2] ?? hote) : hote;
  return (nom[0] ?? "?").toUpperCase();
}

/** Texte noir ou blanc selon la luminance du fond de la pastille. */
export function texteSurPastille(fond: string): string {
  const n = parseInt(fond.slice(1), 16);
  const lum = 0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  return lum > 170 ? "#0d0d0d" : "#ffffff";
}

/** Titre de repli quand l'URL n'est pas présentée par un `[titre]`. */
const TITRE_SANS_NOM = "Source web";

/**
 * Les sources citées par `queue` : les liens markdown d'abord (c'est la forme
 * que produit `memoire_web.ajouter_citations`), les URL nues ensuite.
 */
function sourcesDe(queue: string): Source[] {
  const vues = new Set<string>();
  const sources: Source[] = [];
  const ajouter = (titre: string, url: string): void => {
    const propre = url.replace(/[.,;:]+$/, "");      // ponctuation de fin de phrase
    if (vues.has(propre) || sources.length >= SOURCES_MAX) return;
    vues.add(propre);
    sources.push({ titre: titre.trim() || TITRE_SANS_NOM, url: propre });
  };

  for (const m of queue.matchAll(RE_SOURCE_MD)) {
    ajouter(m[1] ?? TITRE_SANS_NOM, m[2] ?? "");
  }
  if (sources.length === 0) {
    for (const m of queue.matchAll(RE_URL)) ajouter(TITRE_SANS_NOM, m[0] ?? "");
  }
  return sources;
}

/**
 * Un bloc de code jamais refermé : `` ```langue `` sans fermeture. Le 1B en
 * laisse parfois un, en recopiant l'échafaudage de sa consigne (mesuré le
 * 12 oct. 2026 sur une question web). Le cadre est plus honnête que le
 * marqueur brut ; la GUI, elle, n'en tient pas compte.
 *
 * Le marqueur n'est repris que s'il ouvre une ligne : au milieu d'une phrase,
 * `` ``` `` reste du texte.
 */
function decouperOuvert(texte: string): Segment[] {
  if (texte === "") return [];
  const i = texte.lastIndexOf("```");
  if (i === -1) return [{ type: "texte", texte }];

  const avant = texte.slice(0, i);
  if (avant !== "" && !avant.endsWith("\n")) return [{ type: "texte", texte }];

  const apres = texte.slice(i + 3);
  const fin = apres.indexOf("\n");
  const langue = (fin === -1 ? apres : apres.slice(0, fin)).trim();
  const corps = (fin === -1 ? "" : apres.slice(fin + 1)).replace(/\s+$/, "");
  if (corps === "") return [{ type: "texte", texte }];      // marqueur seul : rien à cadrer

  const morceaux: Segment[] = [];
  if (avant.trim() !== "") morceaux.push({ type: "texte", texte: avant.trim() });
  morceaux.push({
    type: "code",
    langue: /^\w+$/.test(langue) ? langue : "code",
    code: `${corps}\n`,
  });
  return morceaux;
}

/** Un morceau de texte : la mention de sources en sort, le reste reste du texte. */
function decouperTexte(texte: string): Segment[] {
  if (texte.trim() === "") return [];
  const entete = RE_ENTETE_SOURCES.exec(texte);
  if (entete === null) return decouperOuvert(texte.trim());

  const avant = texte.slice(0, entete.index ?? 0).trim();
  const queue = texte.slice(entete.index ?? 0);
  const sources = sourcesDe(queue);

  const morceaux = decouperOuvert(avant);
  morceaux.push({
    type: "sources",
    sources,
    brut: sources.length > 0 ? null : queue.trim().slice(0, 160),
  });
  return morceaux;
}

/**
 * Découpe une réponse complète. Les morceaux vides sont écartés : la disposition
 * (espacements, cadres) est décidée par l'affichage, pas par le texte reçu.
 */
export function decouper(reponse: string): Segment[] {
  const segments: Segment[] = [];
  let dernier = 0;
  for (const m of reponse.matchAll(RE_CODE)) {
    const debut = m.index ?? 0;
    segments.push(...decouperTexte(reponse.slice(dernier, debut)));
    segments.push({
      type: "code",
      langue: (m[1] ?? "").trim() || "code",
      code: m[2] ?? "",
    });
    dernier = debut + m[0].length;
  }
  segments.push(...decouperTexte(reponse.slice(dernier)));
  return segments;
}

/**
 * Étiquettes de langage écrites par le cerveau → types de fichier connus de la
 * coloration syntaxique. L'étiquette affichée reste celle de la réponse ; cette
 * table ne sert qu'à choisir la grammaire.
 *
 * Limite assumée : `@opentui/core` ne livre que quelques grammaires
 * (javascript, typescript, typescriptreact, markdown, zig). Les autres langages
 * gardent le cadre et l'étiquette, mais restent en clair (§ README).
 */
const FILETYPES: Record<string, string> = {
  js: "javascript", javascript: "javascript", mjs: "javascript", cjs: "javascript",
  jsx: "javascriptreact", javascriptreact: "javascriptreact",
  ts: "typescript", typescript: "typescript",
  tsx: "typescriptreact", typescriptreact: "typescriptreact",
  md: "markdown", markdown: "markdown",
  zig: "zig",
};

/** Type de fichier pour la coloration, ou `undefined` (cadre + étiquette seuls). */
export function filetype(langue: string): string | undefined {
  return FILETYPES[langue.trim().toLowerCase()];
}
