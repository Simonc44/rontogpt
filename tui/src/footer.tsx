/**
 * Footer mutable — zone de contrôle (prompt, status, permission, form).
 * Architecture split-footer : seul le footer se repaint.
 */
import { renderer } from "./renderer-ref";
import {
  BoxRenderable,
  InputRenderable,
  InputRenderableEvents,
  TextRenderable,
  TextareaRenderable,
  type KeyEvent,
  type PasteEvent,
  type StyledText,
} from "@opentui/core";
import { DEFAULT_THEME, type ThemeTokens, SEMANTIC_COLORS } from "./theme";

export type FooterView = "prompt" | "status" | "permission" | "form" | "subagent" | "panel";

export interface FooterState {
  view: FooterView;
  phase: "idle" | "thinking" | "drafting" | "tool" | "error";
  statusText: string;
  statusColor: string;
  promptVisible: boolean;
  promptPlaceholder: StyledText | string;
  promptValue: string;
  promptRows: number;
  maxPromptRows: number;
  autocompleteOptions: AutocompleteOption[];
  autocompleteSelected: number;
  autocompleteMode: "none" | "mention" | "slash";
  showExitHint: boolean;
  showCopyHint: boolean;
  contextPercent: number;
  modelLabel: string;
  variantLabel: string | undefined;
}

export interface AutocompleteOption {
  kind: "mention" | "slash";
  label: string;
  value: string;
  description?: string;
  icon?: string;
  action?: "editor" | "settings";
}

export interface FooterCallbacks {
  onSubmit: (prompt: string) => void | Promise<void>;
  onInterrupt: () => void;
  onExit: () => void;
  onCycleVariant: () => void;
  onCycleModel: () => void;
  onCycleAgent: () => void;
  onSettings: () => void;
  onEditorOpen: (input: { value: string }) => Promise<string | undefined>;
  onFindFiles: (query: string) => Promise<string[]>;
  onQueuedPromptAction: (action: "accept" | "dismiss", inboxId: string) => Promise<void>;
}

const FOOTER_MENU_ROWS = 6;
const TEXTAREA_MIN_ROWS = 1;
const TEXTAREA_MAX_ROWS = 6;
const PROMPT_MAX_ROWS = TEXTAREA_MAX_ROWS + FOOTER_MENU_ROWS;

export class Footer {
  private container: BoxRenderable;
  private theme: ThemeTokens;
  private callbacks: FooterCallbacks;

  // État réactif
  private state: FooterState = {
    view: "prompt",
    phase: "idle",
    statusText: "",
    statusColor: SEMANTIC_COLORS.statusIdle,
    promptVisible: true,
    promptPlaceholder: "Posez une question ou déposez un document",
    promptValue: "",
    promptRows: TEXTAREA_MIN_ROWS,
    maxPromptRows: PROMPT_MAX_ROWS,
    autocompleteOptions: [],
    autocompleteSelected: 0,
    autocompleteMode: "none",
    showExitHint: false,
    showCopyHint: false,
    contextPercent: 0,
    modelLabel: "RontoGPT",
    variantLabel: undefined,
  };

  // Composants UI
  private promptContainer: BoxRenderable;
  private promptInput: TextareaRenderable;
  private promptPrefix: TextRenderable;
  private autocompleteBox: BoxRenderable;
  private statusLine: TextRenderable;
  private modelLine: TextRenderable;
  private helpLine: TextRenderable;
  private exitHint: TextRenderable;
  private copyHint: TextRenderable;

  // Historique de saisie
  private history: string[] = [];
  private historyIndex = -1;
  private historyMax = 500;

  // Callbacks externes
  private findFilesDebounce: ReturnType<typeof setTimeout> | null = null;

  constructor(callbacks: FooterCallbacks, theme: ThemeTokens = DEFAULT_THEME) {
    this.theme = theme;
    this.callbacks = callbacks;
    this.build();
    this.bindKeys();
  }

  private build(): void {
    // Container principal (footer)
    this.container = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "column",
      backgroundColor: this.theme.background,
    });

    // === PROMPT COMPOSER ===
    this.promptContainer = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "column",
      backgroundColor: this.theme.promptBg,
      borderTop: 1,
      borderColor: this.theme.promptBorder,
    });

    // Ligne de saisie : préfixe + textarea
    const inputRow = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "row",
      alignItems: "flex-start",
      paddingLeft: 1,
      paddingRight: 1,
      paddingTop: 0,
      paddingBottom: 0,
    });

    this.promptPrefix = new TextRenderable(renderer, {
      content: "›",
      fg: this.theme.textDim,
      width: 2,
      marginTop: 0,
    });

    this.promptInput = new TextareaRenderable(renderer, {
      width: "100%",
      flexGrow: 1,
      minRows: TEXTAREA_MIN_ROWS,
      maxRows: TEXTAREA_MAX_ROWS,
      placeholder: this.state.promptPlaceholder,
      placeholderColor: this.theme.textDim,
      textColor: this.theme.text,
      cursorColor: this.theme.cursor,
      backgroundColor: this.theme.promptBg,
      focusedBackgroundColor: this.theme.promptBg,
      selectColor: this.theme.selection,
    });
    this.promptInput.flexGrow = 1;

    inputRow.add(this.promptPrefix);
    inputRow.add(this.promptInput);
    this.promptContainer.add(inputRow);

    // Autocomplete dropdown
    this.autocompleteBox = new BoxRenderable(renderer, {
      width: "100%",
      flexDirection: "column",
      backgroundColor: this.theme.surface,
      borderTop: 1,
      borderColor: this.theme.border,
      visible: false,
    });
    this.promptContainer.add(this.autocompleteBox);

    this.container.add(this.promptContainer);

    // === STATUS LINE (modèle, variante, jauge contexte) ===
    this.statusLine = new TextRenderable(renderer, {
      content: "",
      fg: this.theme.textMuted,
      width: "100%",
      paddingLeft: 3,
      paddingRight: 1,
    });
    this.container.add(this.statusLine);

    // === MODEL/VARIANT LINE ===
    this.modelLine = new TextRenderable(renderer, {
      content: "",
      fg: this.theme.accent,
      width: "100%",
      paddingLeft: 3,
    });
    this.container.add(this.modelLine);

    // === HELP LINE (raccourcis contextuels) ===
    this.helpLine = new TextRenderable(renderer, {
      content: "⋯ /help  commands",
      fg: this.theme.textDim,
      width: "100%",
      paddingLeft: 3,
      paddingBottom: 1,
    });
    this.container.add(this.helpLine);

    // === HINTS (exit, copy) ===
    this.exitHint = new TextRenderable(renderer, {
      content: "Press Ctrl+C again to quit, Esc to cancel",
      fg: this.theme.warning,
      width: "100%",
      paddingLeft: 3,
      visible: false,
    });
    this.container.add(this.exitHint);

    this.copyHint = new TextRenderable(renderer, {
      content: "Sélection copiée ✓",
      fg: this.theme.success,
      width: "100%",
      paddingLeft: 3,
      visible: false,
    });
    this.container.add(this.copyHint);

    // Liaison événements
    this.promptInput.on(InputRenderableEvents.ENTER, () => this.handleSubmit());
    this.promptInput.on(InputRenderableEvents.CHANGE, () => this.handleInputChange());
    this.promptInput.on(InputRenderableEvents.KEY_DOWN, (e: KeyEvent) => this.handleKeyDown(e));
    this.promptInput.on(InputRenderableEvents.PASTE, (e: PasteEvent) => this.handlePaste(e));
  }

  get element(): BoxRenderable {
    return this.container;
  }

  // --- API publique ---

  /** Affiche/masque le footer prompt. */
  setPromptVisible(visible: boolean): void {
    this.state.promptVisible = visible;
    this.promptContainer.visible = visible;
    if (visible) this.promptInput.focus();
    renderer.requestRender();
  }

  /** Met à jour le statut (phase, texte, couleur). */
  setStatus(phase: FooterState["phase"], text: string, color?: string): void {
    this.state.phase = phase;
    this.state.statusText = text;
    this.state.statusColor = color ?? this.getPhaseColor(phase);
    this.renderStatus();
  }

  /** Met à jour le modèle/variante affiché. */
  setModel(modelLabel: string, variantLabel?: string): void {
    this.state.modelLabel = modelLabel;
    this.state.variantLabel = variantLabel;
    this.renderModelLine();
  }

  /** Met à jour la jauge de contexte. */
  setContextPercent(percent: number): void {
    this.state.contextPercent = Math.max(0, Math.min(100, percent));
    this.renderStatus();
  }

  /** Affiche un flash (copie, etc.). */
  flash(text: string, color: string, duration = 1500): void {
    this.copyHint.content = text;
    this.copyHint.fg = color;
    this.copyHint.visible = true;
    renderer.requestRender();
    setTimeout(() => {
      this.copyHint.visible = false;
      renderer.requestRender();
    }, duration);
  }

  /** Affiche l'indice de sortie. */
  showExitHint(show: boolean): void {
    this.state.showExitHint = show;
    this.exitHint.visible = show;
    renderer.requestRender();
  }

  /** Ajoute à l'historique. */
  addToHistory(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (this.history.length === 0 || this.history[this.history.length - 1] !== trimmed) {
      this.history.push(trimmed);
      if (this.history.length > this.historyMax) this.history.shift();
    }
    this.historyIndex = -1;
  }

  /** Définit la valeur du prompt (pour navigation historique). */
  setPromptValue(value: string): void {
    this.state.promptValue = value;
    this.promptInput.value = value;
    this.adjustRows();
    renderer.requestRender();
  }

  /** Focus sur la saisie. */
  focus(): void {
    this.promptInput.focus();
  }

  /** Met à jour les options d'autocomplétion. */
  setAutocompleteOptions(options: AutocompleteOption[], mode: "mention" | "slash"): void {
    this.state.autocompleteOptions = options;
    this.state.autocompleteMode = mode;
    this.state.autocompleteSelected = 0;
    this.renderAutocomplete();
  }

  /** Masque l'autocomplétion. */
  hideAutocomplete(): void {
    this.state.autocompleteOptions = [];
    this.state.autocompleteMode = "none";
    this.autocompleteBox.visible = false;
    renderer.requestRender();
  }

  // --- Gestionnaires internes ---

  private handleSubmit(): void {
    const text = this.promptInput.value.trim();
    if (!text) return;

    this.addToHistory(text);
    this.promptInput.value = "";
    this.state.promptValue = "";
    this.state.promptRows = TEXTAREA_MIN_ROWS;
    this.adjustRows();
    this.hideAutocomplete();
    this.callbacks.onSubmit(text);
  }

  private handleInputChange(): void {
    const value = this.promptInput.value;
    this.state.promptValue = value;
    this.adjustRows();

    // Déclencher autocomplétion @ et /
    this.checkAutocompleteTriggers(value);
    renderer.requestRender();
  }

  private checkAutocompleteTriggers(value: string): void {
    const mentionIndex = value.lastIndexOf("@");
    const slashIndex = value.lastIndexOf("/");

    // @ mention (fichiers, agents, skills)
    if (mentionIndex > slashIndex && mentionIndex !== -1) {
      const query = value.slice(mentionIndex + 1).trim();
      if (query.length > 0) {
        this.debouncedFindFiles(query);
        this.state.autocompleteMode = "mention";
        return;
      }
    }

    // / slash command
    if (slashIndex > mentionIndex && slashIndex !== -1 && slashIndex === value.length - 1 - (value.endsWith(" ") ? 1 : 0)) {
      // Slash au début ou après espace
      if (slashIndex === 0 || value[slashIndex - 1] === " ") {
        this.state.autocompleteMode = "slash";
        this.renderSlashAutocomplete();
        return;
      }
    }

    this.hideAutocomplete();
  }

  private debouncedFindFiles(query: string): void {
    if (this.findFilesDebounce) clearTimeout(this.findFilesDebounce);
    this.findFilesDebounce = setTimeout(async () => {
      try {
        const files = await this.callbacks.onFindFiles(query);
        const options: AutocompleteOption[] = files.slice(0, 10).map((f) => ({
          kind: "mention",
          label: f,
          value: `@${f}`,
          icon: "📄",
        }));
        this.setAutocompleteOptions(options, "mention");
      } catch {
        this.hideAutocomplete();
      }
    }, 150);
  }

  private renderSlashAutocomplete(): void {
    const commands = [
      { name: "help", description: "Show help", icon: "❓" },
      { name: "clear", description: "Clear conversation", icon: "🗑️" },
      { name: "memory", description: "Manage memory", icon: "🧠" },
      { name: "docs", description: "Manage documents", icon: "📄" },
      { name: "voice", description: "Toggle local voice", icon: "🔊" },
      { name: "langue", description: "Set response language", icon: "🌐" },
      { name: "init", description: "Load brain", icon: "🧠" },
      { name: "mcp", description: "MCP status", icon: "🔌" },
      { name: "plan", description: "Plan mode", icon: "📋" },
      { name: "compact", description: "Compact context", icon: "📦" },
      { name: "context", description: "Context gauge", icon: "📊" },
      { name: "api", description: "Local API info", icon: "🌐" },
      { name: "gemini", description: "Send to Gemini", icon: "☁️" },
      { name: "cloud", description: "Cloud settings", icon: "☁️" },
      { name: "status", description: "System status", icon: "📈" },
      { name: "quit", description: "Exit RontoGPT", icon: "🚪" },
    ];

    const options: AutocompleteOption[] = commands.map((c) => ({
      kind: "slash",
      label: `/${c.name}`,
      value: `/${c.name}`,
      description: c.description,
      icon: c.icon,
    }));

    this.setAutocompleteOptions(options, "slash");
  }

  private handleKeyDown(event: KeyEvent): void {
    const { name, ctrl, shift, alt } = event;

    // Navigation historique ↑/↓
    if ((name === "up" || name === "down") && this.promptInput.focused) {
      if (this.history.length === 0) return;

      if (name === "up") {
        const next = this.historyIndex === -1
          ? this.history.length - 1
          : Math.min(this.historyIndex + 1, this.history.length - 1);
        if (next !== this.historyIndex) {
          this.historyIndex = next;
          this.setPromptValue(this.history[this.history.length - 1 - this.historyIndex]);
        }
        return;
      }
      if (name === "down") {
        if (this.historyIndex === -1) return;
        const next = Math.max(this.historyIndex - 1, -1);
        if (next !== this.historyIndex) {
          this.historyIndex = next;
          this.setPromptValue(this.historyIndex === -1 ? "" : this.history[this.history.length - 1 - this.historyIndex]);
        }
        return;
      }
    }

    // Autocomplete navigation
    if (this.state.autocompleteMode !== "none" && this.state.autocompleteOptions.length > 0) {
      if (name === "up") {
        this.state.autocompleteSelected = Math.max(0, this.state.autocompleteSelected - 1);
        this.renderAutocomplete();
        return;
      }
      if (name === "down") {
        this.state.autocompleteSelected = Math.min(
          this.state.autocompleteOptions.length - 1,
          this.state.autocompleteSelected + 1
        );
        this.renderAutocomplete();
        return;
      }
      if (name === "enter" || name === "tab") {
        const opt = this.state.autocompleteOptions[this.state.autocompleteSelected];
        if (opt) {
          this.applyAutocomplete(opt);
          return;
        }
      }
      if (name === "escape") {
        this.hideAutocomplete();
        return;
      }
    }

    // Raccourcis globaux
    if (name === "escape") {
      this.callbacks.onInterrupt();
      return;
    }

    if (ctrl && name === "c") {
      const selection = renderer.getSelection()?.getSelectedText() ?? "";
      if (selection.trim()) {
        this.copyToClipboard(selection);
        this.flash("Sélection copiée ✓", this.theme.success);
        return;
      }
      if (this.state.showExitHint) {
        this.callbacks.onExit();
      } else {
        this.showExitHint(true);
        setTimeout(() => this.showExitHint(false), 5000);
      }
      return;
    }

    // Ctrl+Entrée = nouvelle ligne dans textarea
    if (ctrl && name === "enter") {
      const cursor = this.promptInput.cursorPosition ?? this.promptInput.value.length;
      const before = this.promptInput.value.slice(0, cursor);
      const after = this.promptInput.value.slice(cursor);
      this.promptInput.value = before + "\n" + after;
      this.promptInput.cursorPosition = cursor + 1;
      this.handleInputChange();
      return;
    }
  }

  private applyAutocomplete(option: AutocompleteOption): void {
    const value = this.promptInput.value;
    let newValue = value;

    if (option.kind === "mention") {
      const atIndex = value.lastIndexOf("@");
      if (atIndex !== -1) {
        newValue = value.slice(0, atIndex + 1) + option.value.slice(1) + " ";
      }
    } else if (option.kind === "slash") {
      const slashIndex = value.lastIndexOf("/");
      if (slashIndex !== -1 && (slashIndex === 0 || value[slashIndex - 1] === " ")) {
        newValue = value.slice(0, slashIndex) + option.value + " ";
      }
    }

    this.setPromptValue(newValue);
    this.hideAutocomplete();
  }

  private handlePaste(event: PasteEvent): void {
    // Gestion collage images/fichiers
    if (event.files.length > 0) {
      // TODO: upload et attachement
      event.preventDefault();
    }
  }

  private adjustRows(): void {
    const lines = this.promptInput.value.split("\n").length;
    const newRows = Math.min(Math.max(lines, TEXTAREA_MIN_ROWS), TEXTAREA_MAX_ROWS);
    if (newRows !== this.state.promptRows) {
      this.state.promptRows = newRows;
      this.promptInput.maxRows = newRows;
      renderer.requestRender();
    }
  }

  private getPhaseColor(phase: FooterState["phase"]): string {
    switch (phase) {
      case "thinking": return SEMANTIC_COLORS.thinking;
      case "drafting": return SEMANTIC_COLORS.drafting;
      case "tool": return SEMANTIC_COLORS.toolCall;
      case "error": return SEMANTIC_COLORS.error;
      default: return SEMANTIC_COLORS.statusIdle;
    }
  }

  private renderStatus(): void {
    const gauge = this.renderContextGauge(this.state.contextPercent);
    const parts = [this.state.statusText, gauge].filter(Boolean);
    this.statusLine.content = parts.join("  ·  ");
    this.statusLine.fg = this.state.statusColor;
    renderer.requestRender();
  }

  private renderContextGauge(percent: number): string {
    const width = 16;
    const filled = Math.round((percent / 100) * width);
    const bar = "█".repeat(filled) + "░".repeat(width - filled);
    const color =
      percent >= 90 ? this.theme.error :
      percent >= 70 ? this.theme.warning :
      this.theme.success;
    return `ctx ${bar} ${percent}%`;
  }

  private renderModelLine(): void {
    const parts = [this.state.modelLabel];
    if (this.state.variantLabel) parts.push(this.state.variantLabel);
    if (this.state.contextPercent > 0) parts.push(`${this.state.contextPercent}% ctx`);
    this.modelLine.content = parts.join(" · ");
    renderer.requestRender();
  }

  private renderAutocomplete(): void {
    const { autocompleteOptions, autocompleteSelected } = this.state;
    if (autocompleteOptions.length === 0) {
      this.autocompleteBox.visible = false;
      renderer.requestRender();
      return;
    }

    // Nettoyer
    for (const child of [...this.autocompleteBox.getChildren()]) child.destroy();

    // Rendu options
    for (let i = 0; i < autocompleteOptions.length; i++) {
      const opt = autocompleteOptions[i];
      const isSelected = i === autocompleteSelected;
      const row = new BoxRenderable(renderer, {
        width: "100%",
        flexDirection: "row",
        alignItems: "center",
        paddingLeft: 2,
        paddingRight: 1,
        backgroundColor: isSelected ? this.theme.accent : undefined,
      });

      const icon = new TextRenderable(renderer, {
        content: opt.icon ?? "",
        fg: isSelected ? this.theme.background : this.theme.textMuted,
        width: 2,
      });

      const label = new TextRenderable(renderer, {
        content: opt.label,
        fg: isSelected ? this.theme.background : this.theme.text,
      });
      label.flexGrow = 1;

      const desc = opt.description
        ? new TextRenderable(renderer, {
            content: opt.description,
            fg: isSelected ? this.theme.background : this.theme.textDim,
            width: 30,
          })
        : null;

      row.add(icon);
      row.add(label);
      if (desc) row.add(desc);
      this.autocompleteBox.add(row);
    }

    this.autocompleteBox.visible = true;
    renderer.requestRender();
  }

  // --- Raccourcis clavier globaux ---
  private bindKeys(): void {
    renderer.keyInput.on("keypress", (key) => {
      // Géré dans handleKeyDown pour les événements du textarea
    });
  }

  private copyToClipboard(text: string): void {
    try {
      renderer.copyToClipboardOSC52(text);
    } catch {
      // fallback système géré par l'app principale
    }
  }

  // --- Cycle callbacks ---
  triggerCycleVariant(): void {
    this.callbacks.onCycleVariant();
  }

  triggerCycleModel(): void {
    this.callbacks.onCycleModel();
  }

  triggerCycleAgent(): void {
    this.callbacks.onCycleAgent();
  }

  triggerSettings(): void {
    this.callbacks.onSettings();
  }
}