/**
 * Thèmes RontoGPT v2 — inspirés d'OpenCode v2.
 * Couleurs, tokens, et utilitaires de style centralisés.
 */
export const THEME_TOKENS = {
  // Fond & texte
  background: "#0d0d0d",        // quasi noir
  surface: "#141414",           // bande de saisie, cartes
  surfaceHover: "#1a1a1a",
  border: "#2a2a2a",
  text: "#e8e8e8",              // texte principal
  textMuted: "#888888",         // discret
  textDim: "#555555",           // très discret

  // Accent (identité RontoGPT)
  accent: "#00d4aa",            // vert menthe signature
  accentHover: "#00e8bb",
  accentDim: "#00b38f",

  // États
  success: "#00d47e",
  warning: "#ffb800",
  error: "#ff4757",
  info: "#3498db",

  // Code / technique
  codeBg: "#0f0f0f",
  codeBorder: "#333333",
  comment: "#6a737d",
  keyword: "#c678dd",
  string: "#98c379",
  number: "#d19a66",
  function: "#61afef",
  type: "#e5c07b",

  // UI spécifique
  promptBg: "#141414",
  promptBorder: "#2a2a2a",
  selection: "rgba(0, 212, 170, 0.2)",
  cursor: "#00d4aa",

  // Logo
  logo: "#00d4aa",
  logoDim: "#00a382",

  // Scrollbar
  scrollbarThumb: "#333333",
  scrollbarTrack: "#0d0d0d",
} as const;

export type ThemeTokens = typeof THEME_TOKENS;

export const DEFAULT_THEME: ThemeTokens = THEME_TOKENS;

export function createTheme(overrides: Partial<ThemeTokens> = {}): ThemeTokens {
  return { ...DEFAULT_THEME, ...overrides };
}

export function resolveColor(token: keyof ThemeTokens | string, theme: ThemeTokens = DEFAULT_THEME): string {
  return (theme as Record<string, string>)[token] ?? token;
}

export const SEMANTIC_COLORS = {
  // Badges de provenance
  verified: THEME_TOKENS.success,
  local: THEME_TOKENS.textMuted,
  cloud: THEME_TOKENS.warning,
  unknown: THEME_TOKENS.error,

  // Phases
  thinking: THEME_TOKENS.accent,
  drafting: THEME_TOKENS.info,
  toolCall: THEME_TOKENS.warning,
  toolResult: THEME_TOKENS.success,

  // Pied de page
  statusIdle: THEME_TOKENS.textMuted,
  statusBusy: THEME_TOKENS.accent,
  statusError: THEME_TOKENS.error,

  // Jauge contexte
  contextLow: THEME_TOKENS.success,
  contextMedium: THEME_TOKENS.warning,
  contextHigh: THEME_TOKENS.error,
} as const;