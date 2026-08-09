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
 * Records a movie of a canvas as it's redrawn during generation: two seconds held on whatever's
 * on the canvas when recording starts, then one video frame per generation step as it completes,
 * then two seconds held on the final frame.
 *
 * Captures in manual mode (`captureStream(0)`, advanced only via `track.requestFrame()`) rather
 * than sampling continuously at a fixed rate — frames land in the recording exactly when a step
 * actually completes, not on a wall-clock timer, so a slow step doesn't get padded with extra
 * real-time frames of the same (possibly stale) image, and the two hold periods are exact.
 *
 * Captures from a dedicated off-screen canvas rather than the live display canvas directly, so
 * recording is decoupled from the display canvas being resized between octaves or swapped for a
 * persisted <img> once generation finishes.
 */
export class MovieRecorder {
  private recordingCanvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sourceCanvas: HTMLCanvasElement;
  private stream: MediaStream;
  private track: CanvasCaptureMediaStreamTrack;
  private recorder: MediaRecorder;
  private chunks: Blob[] = [];

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

    // frameRequestRate 0 = manual mode: a frame is captured only on an explicit requestFrame() call.
    this.stream = this.recordingCanvas.captureStream(0);
    this.track = this.stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;

    const mimeType = pickSupportedMimeType();
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
  }

  private captureCurrentFrame() {
    this.ctx.drawImage(this.sourceCanvas, 0, 0, this.recordingCanvas.width, this.recordingCanvas.height);
    this.track.requestFrame();
  }

  /** Captures the current canvas contents, starts the recorder, and holds for the lead-in period. */
  async start(): Promise<void> {
    this.recorder.start();
    this.captureCurrentFrame();
    await sleep(HOLD_SECONDS * 1000);
  }

  /** Captures one frame from the current canvas contents — call once per completed generation step. */
  captureStep(): void {
    this.captureCurrentFrame();
  }

  /** Captures the final frame, holds on it for the trailing period, then stops and resolves the video. */
  async finish(): Promise<Blob> {
    this.captureCurrentFrame();
    await sleep(HOLD_SECONDS * 1000);
    this.captureCurrentFrame();

    // requestFrame() only queues the capture — stopping immediately after can cut the recording
    // off before the browser has actually processed it, silently dropping this last frame and
    // truncating the video right before the trailing hold it was supposed to establish.
    await sleep(150);

    const blob = await new Promise<Blob>((resolve) => {
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: this.recorder.mimeType || 'video/webm' }));
      this.recorder.stop();
    });

    this.cleanup();
    return blob;
  }

  /** Stops recording immediately and discards the result, e.g. when generation is cancelled. */
  abort(): void {
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
