import { tf } from './tfSetup';

export const WORKING_MAX_DIMENSION = 2048;

/**
 * This MobileNetV2 build (TF-Hub classification signature, loaded via @tensorflow-models/mobilenet) declares
 * a fixed graph input shape of [-1, 224, 224, 3]. Unlike a raw fully-convolutional backbone, GraphModel.execute()
 * rejects any other spatial size outright — this is not a quality/speed tradeoff knob, it cannot be changed
 * without swapping in a different model.
 */
export const MODEL_INPUT_SIZE = 224;

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

/** Decodes an image (or a positioned video frame) into a float32 [0,1] HWC tensor, capped to WORKING_MAX_DIMENSION on its longest side. */
export function imageToWorkingTensor(img: FrameSource): tf.Tensor3D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(img).toFloat().div(255) as tf.Tensor3D;
    const [h, w] = pixels.shape;
    const longest = Math.max(h, w);

    if (longest <= WORKING_MAX_DIMENSION) {
      return tf.keep(pixels);
    }

    const scale = WORKING_MAX_DIMENSION / longest;
    const targetH = Math.round(h * scale);
    const targetW = Math.round(w * scale);
    return tf.keep(tf.image.resizeBilinear(pixels, [targetH, targetW]));
  });
}

/**
 * Resizes a [0,1] HWC tensor into the NHWC batch MobileNet expects.
 * This build of MobileNetV2 (TF-Hub, alpha 1.0) takes its input already in [0,1], so no rescale is needed.
 * Kept as a distinct differentiable step (not a no-op passthrough) so the gradient path is explicit.
 */
export function preprocessForMobilenet(image01: tf.Tensor3D): tf.Tensor4D {
  return tf.tidy(() => {
    const resized = tf.image.resizeBilinear(image01, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    return resized.expandDims(0) as tf.Tensor4D;
  });
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
