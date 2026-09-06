import { tf } from './tfSetup';

export type ColorSpace = 'rgb' | 'hsv';

/** Guards divisions by the chroma/value of a gray or black pixel, where hue and saturation are undefined. */
const EPSILON = 1e-8;

function splitChannels(image: tf.Tensor3D): [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D] {
  const [height, width] = image.shape;
  return [
    image.slice([0, 0, 0], [height, width, 1]) as tf.Tensor3D,
    image.slice([0, 0, 1], [height, width, 1]) as tf.Tensor3D,
    image.slice([0, 0, 2], [height, width, 1]) as tf.Tensor3D,
  ];
}

/**
 * HSV to RGB, written branchlessly so it is differentiable — this sits inside the gradient tape, since
 * in HSV mode the network still has to be shown RGB and the gradient has to find its way back.
 *
 * The usual six-sector case analysis is replaced by the standard closed form
 * `f(n) = v − v·s·clamp(min(k, 4−k), 0, 1)` with `k = (n + 6h) mod 6`, which is the same function with
 * the branches folded into a clamp.
 */
export function hsvToRgb(image: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const [hue, saturation, value] = splitChannels(image);
    const shifted = hue.mul(6);

    const component = (n: number): tf.Tensor3D => {
      const k = tf.mod(shifted.add(n), 6);
      const ramp = tf.clipByValue(tf.minimum(k, k.neg().add(4)), 0, 1);
      return value.sub(value.mul(saturation).mul(ramp)) as tf.Tensor3D;
    };

    return tf.concat([component(5), component(3), component(1)], 2) as tf.Tensor3D;
  });
}

/**
 * RGB to HSV. Only ever runs outside a gradient tape — it converts an image into the space the ascent
 * works in, and nothing differentiates back through it.
 *
 * Hue is undefined for a gray pixel and saturation for a black one; both collapse to 0 here, which is the
 * conventional choice and keeps the round trip exact for those pixels even though the value it invents
 * carries no information.
 */
export function rgbToHsv(image: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const [red, green, blue] = splitChannels(image);

    const value = tf.maximum(red, tf.maximum(green, blue));
    const minimum = tf.minimum(red, tf.minimum(green, blue));
    const chroma = value.sub(minimum);
    const safeChroma = chroma.add(EPSILON);

    // Which of the three is largest picks the 120°-wide sector hue falls in.
    const fromRed = green.sub(blue).div(safeChroma);
    const fromGreen = blue.sub(red).div(safeChroma).add(2);
    const fromBlue = red.sub(green).div(safeChroma).add(4);

    const sector = tf.where(value.equal(red), fromRed, tf.where(value.equal(green), fromGreen, fromBlue));

    // The red sector runs from −1 to 1, so a sixth of it lands below zero and has to wrap around.
    const scaled = sector.div(6) as tf.Tensor3D;
    const hue = scaled.sub(scaled.floor()) as tf.Tensor3D;

    return tf.concat(
      [
        tf.where(chroma.lessEqual(EPSILON), tf.zerosLike(hue), hue),
        chroma.div(value.add(EPSILON)),
        value,
      ],
      2,
    ) as tf.Tensor3D;
  });
}

/** Converts an image in `space` into RGB. A no-op clone when it is already RGB. */
export function toRgb(image: tf.Tensor3D, space: ColorSpace): tf.Tensor3D {
  return space === 'hsv' ? hsvToRgb(image) : image.clone();
}

/** Converts an RGB image into `space`. A no-op clone when that space is RGB. */
export function fromRgb(image: tf.Tensor3D, space: ColorSpace): tf.Tensor3D {
  return space === 'hsv' ? rgbToHsv(image) : image.clone();
}

/**
 * Puts every channel back in range after a step has moved it.
 *
 * Hue wraps rather than clamping: it is an angle, so 1.02 is 0.02 and clamping it to 1 would pile every
 * step past the end of the wheel onto red. Saturation and value are ordinary bounded quantities and clamp.
 */
export function clampToColorSpace(image: tf.Tensor3D, space: ColorSpace): tf.Tensor3D {
  if (space === 'rgb') {
    return tf.clipByValue(image, 0, 1) as tf.Tensor3D;
  }

  return tf.tidy(() => {
    const [hue, saturation, value] = splitChannels(image);
    return tf.concat(
      [hue.sub(hue.floor()), tf.clipByValue(saturation, 0, 1), tf.clipByValue(value, 0, 1)],
      2,
    ) as tf.Tensor3D;
  });
}

/**
 * The image's average saturation, as a scalar left on the GPU.
 *
 * Deliberately not read back to JavaScript: a per-step `dataSync` would stall the pipeline on a readback,
 * and readbacks are exactly what has been killing tabs on phones. The value is only ever used to build
 * another tensor, so it never has to leave the device.
 */
export function meanSaturation(image: tf.Tensor3D, space: ColorSpace): tf.Scalar {
  return tf.tidy(() => {
    if (space === 'hsv') {
      const [, saturation] = splitChannels(image);
      return saturation.mean() as tf.Scalar;
    }

    // (max − min) / max, the definition, without paying for a full conversion.
    const value = image.max(2);
    return value.sub(image.min(2)).div(value.add(EPSILON)).mean() as tf.Scalar;
  });
}

/**
 * Rescales saturation so the image's average matches `targetMean`, leaving hue and brightness alone.
 *
 * Unlike `preserveColor`, which pins each pixel's color to where it started, this fixes only the average:
 * the run stays free to make one region more vivid and another less, and only the drift of the whole image
 * is taken away. That drift is what shows up as washing out over a long run, and as saturation wandering
 * between the frames of a video, where each frame is its own run and would otherwise land somewhere
 * slightly different.
 *
 * In RGB this is done directly as `rgb' = V − (V − rgb)·k` with V the pixel's own maximum, which is the
 * transform that scales saturation while holding hue and value exactly — cheaper and more precise than a
 * round trip through HSV.
 */
export function normalizeSaturation(image: tf.Tensor3D, targetMean: tf.Scalar, space: ColorSpace): tf.Tensor3D {
  return tf.tidy(() => {
    const current = meanSaturation(image, space);

    // A fully desaturated image has no color left to rescale, and dividing by its zero would blow the
    // image out to fully saturated noise. The comparison stays on the GPU so no readback is needed.
    const scale = tf.where(current.greater(EPSILON), targetMean.div(current.add(EPSILON)), tf.onesLike(current));

    if (space === 'hsv') {
      const [hue, saturation, value] = splitChannels(image);
      return tf.concat([hue, tf.clipByValue(saturation.mul(scale), 0, 1), value], 2) as tf.Tensor3D;
    }

    const value = image.max(2, true);
    return tf.clipByValue(value.sub(value.sub(image).mul(scale)), 0, 1) as tf.Tensor3D;
  });
}

/**
 * The image's average brightness — the value channel, the largest of a pixel's three — as a scalar left on
 * the GPU, for the same reason `meanSaturation` is: a per-step readback would stall the pipeline.
 */
export function meanBrightness(image: tf.Tensor3D, space: ColorSpace): tf.Scalar {
  return tf.tidy(() => {
    if (space === 'hsv') {
      const [, , value] = splitChannels(image);
      return value.mean() as tf.Scalar;
    }
    return image.max(2).mean() as tf.Scalar;
  });
}

/**
 * Rescales brightness so the image's average matches `targetMean`, leaving hue and saturation alone.
 *
 * In RGB this is a plain multiply of all three channels. Scaling them together moves the value while
 * leaving hue untouched — the channels keep their ratios — and saturation too, since (max − min) / max is
 * unchanged by a common factor. That makes this exactly orthogonal to `normalizeSaturation`, which holds
 * value fixed in the same way, so both switches can be on without fighting each other.
 *
 * The counterpart to holding saturation: maximizing activations likes brightness and will climb toward it
 * given the chance, and a long run or a video's worth of separate runs drift upward as a result.
 */
export function normalizeBrightness(image: tf.Tensor3D, targetMean: tf.Scalar, space: ColorSpace): tf.Tensor3D {
  return tf.tidy(() => {
    const current = meanBrightness(image, space);

    // An all-black image has no brightness to rescale, and dividing by its zero would blow it out to white.
    const scale = tf.where(current.greater(EPSILON), targetMean.div(current.add(EPSILON)), tf.onesLike(current));

    if (space === 'hsv') {
      const [hue, saturation, value] = splitChannels(image);
      return tf.concat([hue, saturation, tf.clipByValue(value.mul(scale), 0, 1)], 2) as tf.Tensor3D;
    }

    return tf.clipByValue(image.mul(scale), 0, 1) as tf.Tensor3D;
  });
}

/**
 * Pulls an image's hue and saturation back toward a reference, leaving its brightness alone.
 *
 * Gradient ascent has no reason to respect the colors it started with, and in HSV it actively spends them:
 * raising value and lowering saturation both brighten a pixel, so a loss that likes brightness — which
 * activation-maximizing does — takes whichever is cheaper, and saturation drains away. This constrains the
 * result instead of hoping the parameterization will, and works in either working space, so a run can keep
 * its colors without giving up RGB's dynamics.
 *
 * `amount` of 0 leaves the image alone; 1 restores the reference's color exactly, so only brightness — the
 * structure the ascent is drawing — survives from the run.
 *
 * Hue is interpolated the short way around the wheel. A plain blend between 0.99 and 0.01 would travel
 * through the entire spectrum to cross a boundary the eye sees as no distance at all.
 */
export function preserveColor(
  image: tf.Tensor3D,
  referenceRgb: tf.Tensor3D,
  amount: number,
  space: ColorSpace,
): tf.Tensor3D {
  return tf.tidy(() => {
    const hsv = space === 'hsv' ? image : rgbToHsv(image);
    const referenceHsv = rgbToHsv(referenceRgb);

    const [hue, saturation, value] = splitChannels(hsv);
    const [referenceHue, referenceSaturation] = splitChannels(referenceHsv);

    // Shortest signed distance around the wheel, in [-0.5, 0.5).
    const offset = referenceHue.sub(hue).add(0.5) as tf.Tensor3D;
    const shortestPath = offset.sub(offset.floor()).sub(0.5) as tf.Tensor3D;

    const blendedHue = hue.add(shortestPath.mul(amount)) as tf.Tensor3D;
    const wrappedHue = blendedHue.sub(blendedHue.floor()) as tf.Tensor3D;
    const blendedSaturation = saturation.add(referenceSaturation.sub(saturation).mul(amount)) as tf.Tensor3D;

    const result = tf.concat([wrappedHue, blendedSaturation, value], 2) as tf.Tensor3D;
    return space === 'hsv' ? result : hsvToRgb(result);
  });
}

/**
 * Runs `consume` with an RGB view of an image held in `space`, and cleans up after it.
 *
 * This is how the live preview, and everything downstream of the canvas it draws to — the movie recorder,
 * "save current step", the crash-recovery snapshot — see a picture rather than raw channels during an HSV
 * run. In RGB the image is passed straight through, so the common path allocates nothing.
 *
 * The view is disposed as soon as `consume` returns, so callers must not retain it.
 */
export async function withRgbView<T>(
  image: tf.Tensor3D,
  space: ColorSpace,
  consume: (rgb: tf.Tensor3D) => T | Promise<T>,
): Promise<T> {
  if (space === 'rgb') {
    return consume(image);
  }

  const rgb = hsvToRgb(image);
  try {
    return await consume(rgb);
  } finally {
    rgb.dispose();
  }
}

/**
 * Resizes an image held in `space`, doing the interpolation in RGB.
 *
 * Hue cannot be interpolated where it wraps: averaging 0.98 and 0.02 — two neighboring reds — gives 0.5,
 * a cyan that was never in the image. Converting to RGB first sidesteps that entirely, and an octave
 * boundary is a cheap enough place to pay for two conversions.
 */
export function resizeInRgb(image: tf.Tensor3D, space: ColorSpace, size: [number, number]): tf.Tensor3D {
  return tf.tidy(() => {
    const rgb = toRgb(image, space);
    const resized = tf.image.resizeBilinear(rgb, size) as tf.Tensor3D;
    return fromRgb(resized, space);
  });
}
