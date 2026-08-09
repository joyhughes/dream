const FPS = 24;
const HOLD_SECONDS = 2;

export function isMovieRecordingSupported(): boolean {
  return (
    typeof HTMLCanvasElement !== 'undefined' &&
    'captureStream' in HTMLCanvasElement.prototype &&
    typeof MediaRecorder !== 'undefined'
  );
}

function pickSupportedMimeType(): string | undefined {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Records a movie of a live canvas as it's redrawn during generation: two seconds held on
 * whatever's on the canvas when recording starts, then a continuous mirror of the canvas at a
 * steady frame rate, then two seconds held on the final frame.
 *
 * Captures from a dedicated off-screen canvas (mirrored via requestAnimationFrame) rather than
 * the live display canvas directly, so recording is decoupled from the display canvas being
 * resized between octaves or swapped for a persisted <img> once generation finishes.
 */
export class MovieRecorder {
  private recordingCanvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sourceCanvas: HTMLCanvasElement;
  private stream: MediaStream;
  private recorder: MediaRecorder;
  private chunks: Blob[] = [];
  private rafHandle: number | null = null;
  private mirroring = false;

  constructor(sourceCanvas: HTMLCanvasElement) {
    this.sourceCanvas = sourceCanvas;

    this.recordingCanvas = document.createElement('canvas');
    this.recordingCanvas.width = sourceCanvas.width;
    this.recordingCanvas.height = sourceCanvas.height;
    // Kept off-screen (not display:none) so browsers don't throttle a canvas they consider hidden.
    this.recordingCanvas.style.position = 'fixed';
    this.recordingCanvas.style.top = '-99999px';
    this.recordingCanvas.style.left = '-99999px';
    document.body.appendChild(this.recordingCanvas);

    const ctx = this.recordingCanvas.getContext('2d');
    if (!ctx) throw new Error('Could not create a 2D context for movie recording.');
    this.ctx = ctx;

    this.stream = this.recordingCanvas.captureStream(FPS);
    const mimeType = pickSupportedMimeType();
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
  }

  private drawCurrentFrame() {
    this.ctx.drawImage(this.sourceCanvas, 0, 0, this.recordingCanvas.width, this.recordingCanvas.height);
  }

  private mirrorLoop = () => {
    if (!this.mirroring) return;
    this.drawCurrentFrame();
    this.rafHandle = requestAnimationFrame(this.mirrorLoop);
  };

  /** Draws the current canvas contents, starts the recorder, and holds for the lead-in period. */
  async start(): Promise<void> {
    this.drawCurrentFrame();
    this.recorder.start();
    this.mirroring = true;
    this.rafHandle = requestAnimationFrame(this.mirrorLoop);
    await sleep(HOLD_SECONDS * 1000);
  }

  /** Holds on the final frame, then stops the recorder and resolves with the assembled video. */
  async finish(): Promise<Blob> {
    await sleep(HOLD_SECONDS * 1000);
    this.mirroring = false;
    if (this.rafHandle !== null) cancelAnimationFrame(this.rafHandle);

    const blob = await new Promise<Blob>((resolve) => {
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: this.recorder.mimeType || 'video/webm' }));
      this.recorder.stop();
    });

    this.cleanup();
    return blob;
  }

  /** Stops recording immediately and discards the result, e.g. when generation is cancelled. */
  abort(): void {
    this.mirroring = false;
    if (this.rafHandle !== null) cancelAnimationFrame(this.rafHandle);
    if (this.recorder.state !== 'inactive') {
      try {
        this.recorder.stop();
      } catch {
        // already stopped
      }
    }
    this.cleanup();
  }

  private cleanup() {
    this.stream.getTracks().forEach((track) => track.stop());
    this.recordingCanvas.remove();
  }
}
