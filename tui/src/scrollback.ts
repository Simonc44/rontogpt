/**
 * Scrollback immuable (append-only) — inspiré d'OpenCode mini.
 * Le transcript ne fait qu'accumuler ; seul le footer se repaint.
 */
import { renderer } from "./renderer-ref";
import { BoxRenderable, ScrollBoxRenderable, TextRenderable, type StyledText } from "@opentui/core";
import { DEFAULT_THEME, type ThemeTokens } from "./theme";

export interface ScrollbackEntry {
  id: string;
  type: "user" | "assistant" | "system" | "tool" | "thinking" | "source" | "error";
  content: StyledText | string;
  meta?: {
    timestamp: Date;
    tokens?: number;
    duration?: number;
    provenance?: { level: string; badge: string; verified: boolean };
    toolName?: string;
    toolArgs?: Record<string, unknown>;
    toolResult?: unknown;
    sourceTitle?: string;
    sourceUrl?: string;
  };
  visible: boolean;
}

export class Scrollback {
  private container: ScrollBoxRenderable;
  private entries: ScrollbackEntry[] = [];
  private theme: ThemeTokens;
  private pendingFlush: ScrollbackEntry[] = [];
  private flushScheduled = false;

  constructor(theme: ThemeTokens = DEFAULT_THEME) {
    this.theme = theme;
    this.container = new ScrollBoxRenderable(renderer, {
      width: "100%",
      flexGrow: 1,
      stickyScroll: true,
      stickyStart: "bottom",
      backgroundColor: theme.background,
      paddingBottom: 1,
    });
    this.container.visible = false; // caché jusqu'au premier message
  }

  get element(): ScrollBoxRenderable {
    return this.container;
  }

  /** Affiche le scrollback (masque l'écran d'accueil). */
  show(): void {
    this.container.visible = true;
  }

  /** Ajoute une entrée (mise en file pour flush groupé). */
  append(entry: Omit<ScrollbackEntry, "id" | "visible">): string {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const fullEntry: ScrollbackEntry = { ...entry, id, visible: true };
    this.pendingFlush.push(fullEntry);
    this.scheduleFlush();
    return id;
  }

  /** Planifie le flush au prochain microtask. */
  private scheduleFlush(): void {
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    queueMicrotask(() => this.flush());
  }

  /** Vide la file d'attente et rend les entrées. */
  private flush(): void {
    this.flushScheduled = false;
    if (this.pendingFlush.length === 0) return;

    const entries = this.pendingFlush.splice(0);
    this.entries.push(...entries);

    for (const entry of entries) {
      this.renderEntry(entry);
    }

    // Auto-scroll si on était déjà en bas
    this.scrollToBottom();
    renderer.requestRender();
  }

  private renderEntry(entry: ScrollbackEntry): void {
    const { bloc, puce } = this.createBlock(entry);
    this.container.add(bloc);
  }

  private createBlock(entry: ScrollbackEntry): { bloc: BoxRenderable; puce: TextRenderable } {
    const isUser = entry.type === "user";
    const isThinking = entry.type === "thinking";
    const isTool = entry.type === "tool";
    const isSource = entry.type === "source";
    const isError = entry.type === "error";

    const glyph = isUser ? "›" : isThinking ? "◆" : isTool ? "▸" : isSource ? "▸" : "•";
    const color = isUser
      ? this.theme.textDim
      : isThinking
      ? this.theme.accent
      : isTool
      ? this.theme.warning
      : isSource
      ? this.theme.info
      : isError
      ? this.theme.error
      : this.theme.textDim;

    const bg = isUser ? this.theme.surface : undefined;

    const bloc = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "column",
      marginTop: 1,
      paddingLeft: 3,
      paddingRight: 1,
      ...(bg ? { backgroundColor: bg } : {}),
    });

    const puce = new TextRenderable(renderer, {
      content: glyph,
      fg: color,
      position: "absolute",
      left: 1,
      top: 0,
    });

    // Puce clignotante pour thinking
    if (isThinking) {
      let toggle = true;
      const interval = setInterval(() => {
        if (renderer.isDestroyed) {
          clearInterval(interval);
          return;
        }
        puce.content = toggle ? "◆" : "◇";
        toggle = !toggle;
        renderer.requestRender();
      }, 400);
      // Nettoyage quand l'entrée change de type
      (entry as any)._blinkInterval = interval;
    }

    bloc.add(puce);

    // Contenu principal
    const content = new TextRenderable(renderer, {
      content: entry.content,
      fg: isError ? this.theme.error : this.theme.text,
      wrapMode: "word",
      width: "100%",
    });
    bloc.add(content);

    // Métadonnées (tokens, durée, tool) — provenance gérée séparément dans index-v2.tsx
    if (entry.meta) {
      const metaParts: string[] = [];
      if (entry.meta.tokens) metaParts.push(`${entry.meta.tokens} tok`);
      if (entry.meta.duration) metaParts.push(`${(entry.meta.duration / 1000).toFixed(1)}s`);
      if (entry.meta.toolName) metaParts.push(`tool: ${entry.meta.toolName}`);

      if (metaParts.length > 0) {
        const meta = new TextRenderable(renderer, {
          content: `─ ${metaParts.join(" · ")} `,
          fg: this.theme.textDim,
          wrapMode: "word",
          width: "100%",
        });
        bloc.add(meta);
      }
    }

    // Source citation
    if (isSource && entry.meta?.sourceTitle) {
      const source = new TextRenderable(renderer, {
        content: `🔗 ${entry.meta.sourceTitle}${entry.meta.sourceUrl ? ` — ${entry.meta.sourceUrl}` : ""}`,
        fg: this.theme.info,
        wrapMode: "word",
        width: "100%",
      });
      bloc.add(source);
    }

    // Résultat d'outil
    if (isTool && entry.meta?.toolResult !== undefined) {
      const result = new TextRenderable(renderer, {
        content: typeof entry.meta.toolResult === "string"
          ? entry.meta.toolResult
          : JSON.stringify(entry.meta.toolResult, null, 2),
        fg: this.theme.textMuted,
        wrapMode: "word",
        width: "100%",
      });
      bloc.add(result);
    }

    return { bloc, puce };
  }

  /** Met à jour une entrée existante (ex: thinking → final). */
  update(id: string, patch: Partial<Omit<ScrollbackEntry, "id">>): boolean {
    const idx = this.entries.findIndex((e) => e.id === id);
    if (idx === -1) return false;

    const old = this.entries[idx]!;
    const updated = { ...old, ...patch };
    this.entries[idx] = updated;

    // Si on passe de thinking à final, arrêter le blink
    if (old.type === "thinking" && updated.type !== "thinking") {
      const interval = (old as any)._blinkInterval;
      if (interval) clearInterval(interval);
    }

    // Re-render complet (simple mais efficace)
    this.rebuild();
    return true;
  }

  /** Reconstruit tout le scrollback (pour thème, mise à jour massive). */
  rebuild(): void {
    // Nettoyer les enfants existants
    for (const child of [...this.container.getChildren()]) {
      child.destroy();
    }
    // Re-render toutes les entrées
    for (const entry of this.entries) {
      this.renderEntry(entry);
    }
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    const scrollHeight = (this.container as any).scrollHeight;
    if (typeof scrollHeight === "number") {
      this.container.scrollTo({ x: 0, y: scrollHeight });
    }
  }

  /** Nettoie tout. */
  clear(): void {
    this.entries = [];
    this.pendingFlush = [];
    for (const child of [...this.container.getChildren()]) {
      child.destroy();
    }
    this.container.visible = false;
  }

  /** Nombre d'entrées. */
  get length(): number {
    return this.entries.length;
  }

  /** Dernière entrée. */
  get last(): ScrollbackEntry | undefined {
    return this.entries[this.entries.length - 1];
  }
}