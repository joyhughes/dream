import { encodeFrameSequence, type TimedFrame } from './frameEncoding';

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

/**
 * Records a movie of a canvas across a generation run, at a pace independent of how long
 * generation actually took: two seconds held on whatever's on the canvas when recording starts,
 * then one video frame per generation step as it completes, then two seconds held on the final
 * frame.
 *
 * A run can take anywhere from seconds to minutes depending on hardware/backend and settings,
 * and step timing varies a lot between octaves — encoding live off real capture timestamps would
 * make the movie exactly as slow and uneven as the run itself. So instead, this collects a cheap
 * snapshot (an ImageBitmap, not a video frame) of the canvas at each step during generation, and
 * defers actually encoding the video until `finish()`, which replays the collected snapshots at a
 * fixed rate via `encodeFrameSequence`.
 */
export class MovieRecorder {
  private sourceCanvas: HTMLCanvasElement;
  private initialFrame: ImageBitmap | null = null;
  private stepFrames: ImageBitmap[] = [];

  constructor(sourceCanvas: HTMLCanvasElement) {
    this.sourceCanvas = sourceCanvas;
  }

  /** Snapshots the current canvas contents as the movie's starting frame. */
  async start(): Promise<void> {
    this.initialFrame = await createImageBitmap(this.sourceCanvas);
  }

  /** Snapshots the current canvas contents — call once per completed, rendered generation step. */
  async captureStep(): Promise<void> {
    this.stepFrames.push(await createImageBitmap(this.sourceCanvas));
  }

  /** Snapshots the final canvas contents and encodes the whole movie. */
  async finish(): Promise<Blob> {
    const finalFrame = await createImageBitmap(this.sourceCanvas);

    const frames: TimedFrame[] = [];
    if (this.initialFrame) frames.push({ bitmap: this.initialFrame, holdMs: HOLD_SECONDS * 1000 });
    for (const frame of this.stepFrames) frames.push({ bitmap: frame, holdMs: STEP_FRAME_MS });
    frames.push({ bitmap: finalFrame, holdMs: HOLD_SECONDS * 1000 });

    const blob = await encodeFrameSequence(frames, this.sourceCanvas.width, this.sourceCanvas.height);

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
  }
}
