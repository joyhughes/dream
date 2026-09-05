import { tf } from './tfSetup';
import type { ImageRegularizers } from '../types';

/**
 * Regularizers for gradient-ascent image synthesis.
 *
 * Maximizing a network's activations with nothing holding the image back produces high-frequency
 * garbage: the optimum of an unconstrained ascent is an adversarial pattern, not a picture. Every
 * technique here is a prior on what a natural image looks like, pulling the result back toward one.
 * They come in two kinds, applied at different points in the loop:
 *
 *   - Gradient-space (`laplacianNormalize`, `totalVariationGradient`): change the direction of a step.
 *   - Image-space (`applyImageRegularizers`): change the image after a step has been taken.
 *
 * References: Mahendran & Vedaldi 2015 (total variation, "Understanding Deep Image Representations by
 * Inverting Them"); Yosinski et al. 2015 (blur and decay, "Understanding Neural Networks Through Deep
 * Visualization"); Mordvintsev et al. 2015 (Laplacian pyramid gradient normalization, the DeepDream
 * "lapnorm" notebooks); Olah et al. 2017 ("Feature Visualization") for the survey these sit in.
 */

/** Guards every division by a standard deviation, including the degenerate all-constant case. */
const EPSILON = 1e-8;

/** Scales a tensor to unit standard deviation, which is what makes a step size mean the same thing twice. */
function normalizeByStd(t: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const { variance } = tf.moments(t);
    return t.div(variance.sqrt().add(EPSILON)) as tf.Tensor3D;
  });
}

/**
 * Builds a depthwise convolution kernel that applies the same 2D stencil to every channel independently.
 * Colors are regularized separately on purpose: mixing them here would desaturate the image, which is a
 * change to its content rather than to its smoothness.
 */
function depthwiseKernel(values: number[], height: number, width: number, channels: number): tf.Tensor4D {
  return tf.tidy(() => tf.tensor4d(values, [height, width, 1, 1]).tile([1, 1, channels, 1]) as tf.Tensor4D);
}

/**
 * Extends an image outward by repeating its edge pixels.
 *
 * Every convolution below has to be told what lies outside the image. TensorFlow's 'same' padding answers
 * "black", which is a lie that shows: the blur drags a dark border inward, and the total-variation gradient
 * reads the jump from the edge pixel to that imaginary black as real detail and pushes the whole frame
 * toward it. Repeating the edge instead says "more of the same", so a flat image stays flat everywhere and
 * neither regularizer leaves a vignette behind.
 */
function padEdge(image: tf.Tensor3D, padY: number, padX: number): tf.Tensor3D {
  return tf.tidy(() => {
    let padded = image;

    if (padY > 0) {
      const [height, width, channels] = padded.shape;
      const top = padded.slice([0, 0, 0], [1, width, channels]).tile([padY, 1, 1]);
      const bottom = padded.slice([height - 1, 0, 0], [1, width, channels]).tile([padY, 1, 1]);
      padded = tf.concat([top, padded, bottom], 0) as tf.Tensor3D;
    }

    if (padX > 0) {
      const [height, width, channels] = padded.shape;
      const left = padded.slice([0, 0, 0], [height, 1, channels]).tile([1, padX, 1]);
      const right = padded.slice([0, width - 1, 0], [height, 1, channels]).tile([1, padX, 1]);
      padded = tf.concat([left, padded, right], 1) as tf.Tensor3D;
    }

    return padded;
  });
}

function gaussianKernel1D(sigma: number): number[] {
  // Past three standard deviations the remaining weight is under 0.3%, so the kernel stops there.
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const weights: number[] = [];
  let sum = 0;

  for (let i = -radius; i <= radius; i++) {
    const weight = Math.exp(-(i * i) / (2 * sigma * sigma));
    weights.push(weight);
    sum += weight;
  }

  return weights.map((weight) => weight / sum);
}

/**
 * Gaussian blur, as two 1D passes rather than one 2D kernel — the kernel is separable, so this is O(k)
 * work per pixel instead of O(k²) and matters once sigma (and with it the kernel) grows.
 */
export function gaussianBlur(image: tf.Tensor3D, sigma: number): tf.Tensor3D {
  if (sigma <= 0) {
    return image.clone();
  }

  return tf.tidy(() => {
    const weights = gaussianKernel1D(sigma);
    const radius = (weights.length - 1) / 2;
    const channels = image.shape[2];
    const vertical = depthwiseKernel(weights, weights.length, 1, channels);
    const horizontal = depthwiseKernel(weights, 1, weights.length, channels);

    const blurredRows = tf.depthwiseConv2d(padEdge(image, radius, 0), vertical, 1, 'valid');
    return tf.depthwiseConv2d(padEdge(blurredRows as tf.Tensor3D, 0, radius), horizontal, 1, 'valid') as tf.Tensor3D;
  });
}

/**
 * The gradient of the total-variation penalty Σ‖∇x‖², computed in closed form as a Laplacian rather than
 * through autodiff — the derivative of that sum is exactly −2∇²x, so a single depthwise convolution gives
 * what a tape over the whole image would, without building the tape.
 *
 * This is the β=2 variant from Mahendran & Vedaldi: differentiable everywhere, unlike the β=1 absolute
 * difference, which is why it is the one that suits gradient descent.
 */
export function totalVariationGradient(image: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const channels = image.shape[2];
    const laplacian = depthwiseKernel([0, -1, 0, -1, 4, -1, 0, -1, 0], 3, 3, channels);
    return tf.depthwiseConv2d(padEdge(image, 1, 1), laplacian, 1, 'valid').mul(2) as tf.Tensor3D;
  });
}

/**
 * Normalizes a gradient band by band across a Laplacian pyramid, instead of once over the whole thing.
 *
 * A raw ascent gradient is dominated by its lowest frequencies, so a plain normalization lets broad blobs
 * set the step size and leaves fine detail far below it — the pattern grows in smears. Splitting the
 * gradient into frequency bands, scaling each to unit variance, and recombining gives every scale a
 * comparable say, which is what produces DeepDream's characteristic detail at every size at once.
 *
 * `levels` of 1 (or fewer) is a plain whole-image normalization, i.e. no pyramid at all.
 */
export function laplacianNormalize(gradient: tf.Tensor3D, levels: number): tf.Tensor3D {
  if (levels <= 1) {
    return normalizeByStd(gradient);
  }

  return tf.tidy(() => {
    // Split into detail bands, coarsening as it goes: each level keeps what the next one down cannot hold.
    const bands: tf.Tensor3D[] = [];
    let residual = gradient as tf.Tensor3D;

    for (let level = 0; level < levels - 1; level++) {
      const [height, width] = residual.shape;
      // Below this a "half size" image is a couple of pixels, and a band of it carries no usable detail.
      if (height < 8 || width < 8) {
        break;
      }

      const half: [number, number] = [Math.floor(height / 2), Math.floor(width / 2)];
      const coarse = tf.image.resizeBilinear(residual, half) as tf.Tensor3D;
      const coarseUpsampled = tf.image.resizeBilinear(coarse, [height, width]) as tf.Tensor3D;

      bands.push(residual.sub(coarseUpsampled) as tf.Tensor3D);
      residual = coarse;
    }

    // Recombine coarse to fine, each band normalized to unit variance on the way.
    let combined = normalizeByStd(residual);
    for (let level = bands.length - 1; level >= 0; level--) {
      const band = normalizeByStd(bands[level]);
      const [bandHeight, bandWidth] = band.shape;
      const upsampled = tf.image.resizeBilinear(combined, [bandHeight, bandWidth]) as tf.Tensor3D;
      combined = upsampled.add(band) as tf.Tensor3D;
    }

    // Bands sum to more than unit variance, so a final pass keeps the step size meaning what it did before.
    return normalizeByStd(combined);
  });
}

/** Whether a blur should run on this step, given the interval the user set. `blurEvery` of 0 means never. */
export function shouldBlurOnStep(regularizers: ImageRegularizers, step: number): boolean {
  return regularizers.blurSigma > 0 && regularizers.blurEvery > 0 && (step + 1) % regularizers.blurEvery === 0;
}

/**
 * The image-space regularizers, applied to a finished step. Returns a new tensor the caller owns, or
 * `null` when nothing is enabled on this step — the common case, since these all default to off, and
 * returning null rather than a copy keeps a disabled panel from allocating a full image every step.
 *
 * L2 decay shrinks the image toward its own mean. The literature applies it in a mean-centered space
 * where decay pulls toward zero; working in [0, 1] as we do, the mean is where zero was. It bleeds off
 * the runaway extremes that ascent keeps pushing pixels into, at the cost of some contrast.
 */
export function applyImageRegularizers(
  image: tf.Tensor3D,
  regularizers: ImageRegularizers,
  step: number,
): tf.Tensor3D | null {
  const wantsBlur = shouldBlurOnStep(regularizers, step);
  const wantsDecay = regularizers.l2Decay > 0;

  if (!wantsBlur && !wantsDecay) {
    return null;
  }

  return tf.tidy(() => {
    let result = image as tf.Tensor3D;

    if (wantsDecay) {
      const mean = result.mean();
      result = mean.add(result.sub(mean).mul(1 - regularizers.l2Decay)) as tf.Tensor3D;
    }

    if (wantsBlur) {
      result = gaussianBlur(result, regularizers.blurSigma);
    }

    return tf.clipByValue(result, 0, 1) as tf.Tensor3D;
  });
}
