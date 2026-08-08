/**
 * An async gate the generation step loops await on between steps. Pausing doesn't touch any tensor state —
 * the loop just parks on a promise until resumed, so whatever's currently in memory (the working image,
 * optimizer state, etc.) stays exactly as it is and picks back up where it left off.
 */
export class PauseController {
  private paused = false;
  private resumeSignal: (() => void) | null = null;

  isPaused(): boolean {
    return this.paused;
  }

  pause(): void {
    this.paused = true;
  }

  resume(): void {
    this.paused = false;
    this.resumeSignal?.();
    this.resumeSignal = null;
  }

  /** Resolves immediately if not paused. Otherwise blocks until resume() — or the abort signal fires. */
  async waitIfPaused(abortSignal?: AbortSignal): Promise<void> {
    if (!this.paused) return;
    await new Promise<void>((resolve) => {
      this.resumeSignal = resolve;
      abortSignal?.addEventListener('abort', () => resolve(), { once: true });
    });
  }
}
