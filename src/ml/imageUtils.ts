import { tf } from './tfSetup';

export const WORKING_MAX_DIMENSION = 1024;
export const MODEL_INPUT_SIZE = 224;

const HEIC_EXTENSION = /\.hei[cf]$/i;
const HEIC_MIME_TYPES = new Set(['image/heic', 'image/heif', 'image/heic-sequence', 'image/heif-sequence']);

export function isHeicFile(file: File): boolean {
  return HEIC_MIME_TYPES.has(file.type.toLowerCase()) || HEIC_EXTENSION.test(file.name);
}

/**
 * Chrome (unlike Safari) has no built-in HEIC/HEIF decoder, so `<img>` and canvas APIs can't read
 * photos straight off an iPhone. This transcodes to JPEG client-side before anything else touches the file.
 */
export async function ensureBrowserDecodableImage(file: File): Promise<File> {
  if (!isHeicFile(file)) {
    return file;
  }

  const { default: heic2any } = await import('heic2any');
  const result = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
  const jpegBlob = Array.isArray(result) ? result[0] : result;
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

/** Decodes an image into a float32 [0,1] HWC tensor, capped to WORKING_MAX_DIMENSION on its longest side. */
export function imageToWorkingTensor(img: HTMLImageElement): tf.Tensor3D {
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
export function preprocessForMobilenet(image01: tf.Tensor3D, size = MODEL_INPUT_SIZE): tf.Tensor4D {
  return tf.tidy(() => {
    const resized = tf.image.resizeBilinear(image01, [size, size]);
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
