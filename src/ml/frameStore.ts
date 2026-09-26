import { getDeviceLimits } from './deviceLimits';
import type { TimedFrame } from './frameEncoding';

/**
 * The frames of a sequence, held compressed until there is a video to encode them into.
 *
 * Holding finished frames as ImageBitmaps costs their full pixel count in RGBA bytes — 12.6 MB for a
 * single 2048x1536 frame — so a byte budget that keeps a phone's tab alive works out to only twenty-odd
 * frames of a big image, and an animation asked for a hundred and twenty would quietly get twenty-odd.
 * Encoded to WebP the same frame is well under a megabyte, which is the difference between the budget
 * binding on every desktop sweep and it never binding at all.
 *
 * The cost is a decode per frame at encoding time, which `encodeFrameSequence` absorbs by pacing each
 * frame's hold against an absolute schedule rather than sleeping a fixed time after the draw.
 */

const FALLBACK_TYPE = 'image/jpeg';
/** High enough that the sweep is not judged on the frame store's artifacts; the video codec is lossier. */
const QUALITY = 0.92;

let cachedType: string | null = null;

/**
 * WebP where the browser will encode it, JPEG otherwise. Both are a large win over raw bitmaps, and
 * a dream frame is exactly the high-frequency detail WebP keeps better at the same size.
 */
function frameImageType(): string {
  if (cachedType) return cachedType;

  const probe = document.createElement('canvas');
  probe.width = 1;
  probe.height = 1;
  // A browser that cannot encode the type asked for silently returns a PNG data URL instead.
  cachedType = probe.toDataURL('image/webp').startsWith('data:image/webp') ? 'image/webp' : FALLBACK_TYPE;
  return cachedType;
}

function canvasToFrameBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  const type = frameImageType();
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode an animation frame.'))),
      type,
      QUALITY,
    );
  });
}

export class CompressedFrameStore {
  private frames: Blob[] = [];
  private bytes = 0;
  private readonly budgetBytes: number;

  /**
   * Keep every Nth frame. Doubles each time the store fills, so a sequence that outgrows its budget
   * gets coarser rather than shorter — a caller stepping an index should skip the frames this rejects
   * instead of rendering work that is about to be thrown away.
   */
  stride = 1;

  constructor(budgetBytes: number = getDeviceLimits().frameStoreBudgetBytes) {
    this.budgetBytes = budgetBytes;
  }

  get length(): number {
    return this.frames.length;
  }

  /** True when this index survives the current stride, and so is worth the cost of rendering. */
  wants(index: number): boolean {
    return index % this.stride === 0;
  }

  async add(canvas: HTMLCanvasElement): Promise<void> {
    const blob = await canvasToFrameBlob(canvas);
    this.frames.push(blob);
    this.bytes += blob.size;

    if (this.bytes > this.budgetBytes) {
      this.thin();
    }
  }

  /**
   * Halves the store by dropping every other frame, and halves the capture rate to match.
   *
   * Discarding the newest frames would be cheaper, but it would end the sequence partway through —
   * and for a parameter sweep that means the video stops short of the range it was asked for, which
   * is the one outcome worth going to any length to avoid.
   */
  private thin(): void {
    const kept: Blob[] = [];
    let bytes = 0;

    this.frames.forEach((frame, i) => {
      if (i % 2 === 0) {
        kept.push(frame);
        bytes += frame.size;
      }
    });

    this.frames = kept;
    this.bytes = bytes;
    this.stride *= 2;
  }

  /**
   * The stored frames, decoded one at a time and closed as soon as the consumer has drawn each one,
   * so encoding costs one live bitmap rather than the whole sequence.
   */
  async *timed(holdMs: number): AsyncGenerator<TimedFrame> {
    for (const blob of this.frames) {
      const bitmap = await createImageBitmap(blob);
      try {
        yield { bitmap, holdMs };
      } finally {
        bitmap.close();
      }
    }
  }

  clear(): void {
    this.frames = [];
    this.bytes = 0;
    this.stride = 1;
  }
}
