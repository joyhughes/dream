const HOLD_SECONDS = 2;
const STEP_FRAME_MS = 100; // playback pace through the step sequence — fixed, not real generation time

export function isMovieRecordingSupported(): boolean {
  return (
    typeof HTMLCanvasElement !== 'undefined' &&
    'captureStream' in HTMLCanvasElement.prototype &&
    typeof MediaRecorder !== 'undefined' &&
    typeof createImageBitmap === 'function'
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
 * Records a movie of a canvas across a generation run, at a pace independent of how long
 * generation actually took: two seconds held on the starting image, then the sequence of
 * rendered steps played back at a fixed pace (one step per frame, not real time), then two
 * seconds held on the final image.
 *
 * A run can take anywhere from seconds to minutes depending on hardware/backend and settings,
 * and step timing varies a lot between octaves — encoding live off real capture timestamps would
 * make the movie exactly as slow and uneven as the run itself. So instead, this collects a cheap
 * snapshot (an ImageBitmap, not a video frame) of the canvas at each step during generation, and
 * defers actually encoding the video until `finish()`, which replays the collected snapshots onto
 * a dedicated off-screen canvas at a fixed rate and captures that.
 */
export class MovieRecorder {
  private sourceCanvas: HTMLCanvasElement;
  private recordingCanvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private initialFrame: ImageBitmap | null = null;
  private stepFrames: ImageBitmap[] = [];

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
  }

  /** Snapshots the current canvas contents as the movie's starting frame. */
  async start(): Promise<void> {
    this.initialFrame = await createImageBitmap(this.sourceCanvas);
  }

  /** Snapshots the current canvas contents — call once per completed, rendered generation step. */
  async captureStep(): Promise<void> {
    this.stepFrames.push(await createImageBitmap(this.sourceCanvas));
  }

  /**
   * Snapshots the final canvas contents, then encodes the whole movie by replaying every
   * collected snapshot onto the recording canvas at a fixed pace and capturing that — so encoding
   * itself takes a short, predictable amount of time regardless of how long generation took.
   */
  async finish(): Promise<Blob> {
    const finalFrame = await createImageBitmap(this.sourceCanvas);

    const stream = this.recordingCanvas.captureStream(0);
    const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
    const mimeType = pickSupportedMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    const draw = (bitmap: ImageBitmap) => {
      this.ctx.drawImage(bitmap, 0, 0, this.recordingCanvas.width, this.recordingCanvas.height);
      track.requestFrame();
    };

    recorder.start();

    if (this.initialFrame) draw(this.initialFrame);
    await sleep(HOLD_SECONDS * 1000);

    for (const frame of this.stepFrames) {
      draw(frame);
      await sleep(STEP_FRAME_MS);
    }

    draw(finalFrame);
    await sleep(HOLD_SECONDS * 1000);
    draw(finalFrame);
    // requestFrame() only queues the capture — stopping immediately after can drop it before the
    // browser processes it, truncating the recording right before this trailing hold.
    await sleep(150);

    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || 'video/webm' }));
      recorder.stop();
    });

    finalFrame.close();
    this.cleanup();
    return blob;
  }

  /** Discards everything collected so far without encoding, e.g. when generation is cancelled. */
  abort(): void {
    this.cleanup();
  }

  private cleanup() {
    this.initialFrame?.close();
    this.initialFrame = null;
    this.stepFrames.forEach((frame) => frame.close());
    this.stepFrames = [];
    this.recordingCanvas.remove();
  }
}
