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
