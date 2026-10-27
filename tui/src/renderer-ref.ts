/**
 * Référence partagée au renderer OpenTUI.
 * Évite les imports circulaires entre index.ts, scrollback.ts, footer.tsx.
 */
import { createCliRenderer, type CliRenderer } from "@opentui/core";
import { DEFAULT_THEME } from "./theme";

let _renderer: CliRenderer | null = null;

export function getRenderer(): CliRenderer {
  if (!_renderer) {
    throw new Error("Renderer not initialized. Call initializeRenderer() first.");
  }
  return _renderer;
}

export async function initializeRenderer(): Promise<CliRenderer> {
  if (_renderer) return _renderer;

  _renderer = await createCliRenderer({
    exitOnCtrlC: false,
    backgroundColor: DEFAULT_THEME.background,
    targetFps: 60,
    gatherStats: false,
    useKittyKeyboard: {},
    autoFocus: false,
    openConsoleOnError: false,
    useMouse: true,
    consoleOptions: {
      keyBindings: [{ name: "y", ctrl: true, action: "copy-selection" }],
    },
  });

  return _renderer;
}

export function destroyRenderer(): void {
  if (_renderer) {
    _renderer.destroy();
    _renderer = null;
  }
}

export const renderer = new Proxy({} as CliRenderer, {
  get(_target, prop) {
    const r = getRenderer();
    const value = (r as any)[prop];
    return typeof value === "function" ? value.bind(r) : value;
  },
});