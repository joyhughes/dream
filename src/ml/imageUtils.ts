import { tf } from './tfSetup';
import { getDeviceLimits } from './deviceLimits';

/** Longest side, in pixels, an uploaded image or video frame is worked on at. See `deviceLimits`. */
export function workingMaxDimension(): number {
  return getDeviceLimits().workingMaxDimension;
}

const HEIC_EXTENSION = /\.hei[cf]$/i;
const HEIC_MIME_TYPES = new Set(['image/heic', 'image/heif', 'image/heic-sequence', 'image/heif-sequence']);

export function isHeicFile(file: File): boolean {
  return HEIC_MIME_TYPES.has(file.type.toLowerCase()) || HEIC_EXTENSION.test(file.name);
}

/**
 * Chrome (unlike Safari) has no built-in HEIC/HEIF decoder, so `<img>` and canvas APIs can't read
 * photos straight off an iPhone. This transcodes to JPEG client-side before anything else touches the file.
 * The decoder is lazy-loaded (dynamic import) so non-HEIC uploads never pay for it.
 */
export async function ensureBrowserDecodableImage(file: File): Promise<File> {
  if (!isHeicFile(file)) {
    return file;
  }

  const { isHeic, heicTo } = await import('heic-to');

  // The name/MIME check above is just a cheap gate for whether to load the decoder at all;
  // heic-to's isHeic reads the file's actual byte signature, which is what we trust.
  if (!(await isHeic(file))) {
    return file;
  }

  const jpegBlob = await heicTo({ blob: file, type: 'image/jpeg', quality: 0.92 });
  const newName = file.name.replace(HEIC_EXTENSION, '.jpg') || 'converted.jpg';

  return new File([jpegBlob], newName, { type: 'image/jpeg' });
}

export function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}

export type FrameSource = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;

/** The intrinsic [height, width] of a frame source, which for images and videos is not its layout size. */
function sourceSize(img: FrameSource): [number, number] {
  if (img instanceof HTMLVideoElement) {
    return [img.videoHeight, img.videoWidth];
  }
  if (img instanceof HTMLImageElement) {
    return [img.naturalHeight || img.height, img.naturalWidth || img.width];
  }
  return [img.height, img.width];
}

// Two reusable canvases, ping-ponged between halving steps so a long downscale chain allocates two
// canvases in total rather than one per step. Safe to reuse across calls because the result is handed
// straight to `fromPixels` before anything else can ask for a downscale.
let scratchA: HTMLCanvasElement | null = null;
let scratchB: HTMLCanvasElement | null = null;

function scratch(second: boolean): HTMLCanvasElement {
  if (second) {
    scratchB ??= document.createElement('canvas');
    return scratchB;
  }
  scratchA ??= document.createElement('canvas');
  return scratchA;
}

/** Draws `src` into `canvas` at exactly `w`x`h`, with the best downscaling filter the browser offers. */
function drawTo(canvas: HTMLCanvasElement, src: CanvasImageSource, w: number, h: number): void {
  // Assigning the size also clears the canvas and resets its context state, so the smoothing hints
  // below have to be set after it, not once at creation.
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not create a 2D context to resize this image.');
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
}

/**
 * Draws `img` down to fit `maxDim` on its longest side, returning a canvas at the reduced size.
 *
 * The point of doing this in canvas-land rather than with `tf.image.resizeBilinear` is that it never
 * materializes the source at full resolution as a tensor. A 12-megapixel iPhone photo is ~146 MB as a
 * float32 RGB tensor, and `fromPixels().toFloat().div(255)` holds three of those at once — ~440 MB of
 * peak allocation just to *open* the photo, before any dreaming starts. That alone is enough to get an
 * iPhone tab killed. Drawing to a canvas first hands tf.js an image that is already small.
 *
 * The reduction is done by repeated halving rather than in one step: a single `drawImage` that shrinks
 * by more than ~2x point-samples on some browsers (Safari included) instead of averaging, which throws
 * away most of the detail and aliases hard. Halving stays in the range where the built-in filtering is
 * an honest box filter, so the last partial step lands on a properly averaged image.
 */
function downscaleSource(img: FrameSource, maxDim: number): FrameSource {
  const [h, w] = sourceSize(img);
  const longest = Math.max(h, w);

  if (longest === 0 || longest <= maxDim) {
    return img;
  }

  const scale = maxDim / longest;
  const targetH = Math.max(1, Math.round(h * scale));
  const targetW = Math.max(1, Math.round(w * scale));

  let src: CanvasImageSource = img;
  let curH = h;
  let curW = w;
  let second = false;

  // Runs while a >2x reduction still remains on either axis; the final `drawTo` below covers the rest,
  // which is by then at most a 2x step. Testing the axes independently matters for very lopsided
  // sources: a wide, short panorama reaches its target height long before its width, and requiring
  // both to still be oversized would exit early and leave the width to a single huge, aliased step.
  // `second` alternates so a step never draws a canvas onto itself.
  while (curH > targetH * 2 || curW > targetW * 2) {
    const nextH = Math.max(targetH, Math.round(curH / 2));
    const nextW = Math.max(targetW, Math.round(curW / 2));
    const canvas = scratch(second);

    drawTo(canvas, src, nextW, nextH);

    src = canvas;
    curH = nextH;
    curW = nextW;
    second = !second;
  }

  const out = scratch(second);
  drawTo(out, src, targetW, targetH);
  return out;
}

/** The [width, height] a source of this size ends up being worked on at, after the `workingMaxDimension()` cap. */
export function workingDimensions(width: number, height: number): [number, number] {
  const longest = Math.max(width, height);
  const max = workingMaxDimension();

  if (longest === 0 || longest <= max) {
    return [width, height];
  }

  const scale = max / longest;
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

/** Decodes an image (or a positioned video frame) into a float32 [0,1] HWC tensor, capped to `workingMaxDimension()` on its longest side. */
export function imageToWorkingTensor(img: FrameSource): tf.Tensor3D {
  const source = downscaleSource(img, workingMaxDimension());
  return tf.tidy(() => tf.browser.fromPixels(source).toFloat().div(255) as tf.Tensor3D);
}

export async function renderTensorToCanvas(image01: tf.Tensor3D, canvas: HTMLCanvasElement): Promise<void> {
  const clamped = tf.tidy(() => tf.clipByValue(image01, 0, 1) as tf.Tensor3D);
  try {
    await tf.browser.toPixels(clamped, canvas);
  } finally {
    clamped.dispose();
  }
}

export function cloneTensor(t: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => tf.clone(t));
}
