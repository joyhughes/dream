import { tf } from './tfSetup';
import type { BrushSettings } from '../types';

/**
 * Painting the effect in by hand.
 *
 * A whole-image run is a fixed pipeline: press Generate, wait, get a result. The brush turns the same
 * machinery into something continuous — a patch under the cursor is processed a few steps at a time and
 * blended back through a soft-edged mask, so holding still keeps building the effect up in that spot and
 * moving away leaves it behind. The image being painted lives here across the whole session rather than
 * being rebuilt per run, since every dab has to land on what the last one left.
 *
 * Only a patch is processed, not the image: a dab has to finish inside a frame to feel like a brush, and
 * that budget buys a few hundred pixels square, not a few thousand.
 */

/** Square patch of the image, clipped to its bounds. */
export interface PatchBounds {
  y: number;
  x: number;
  height: number;
  width: number;
}

/**
 * The patch a dab at (x, y) touches, clipped to the image. Returns null when the dab falls entirely
 * outside — the pointer can be dragged off the canvas mid-stroke.
 */
export function patchAround(
  centerX: number,
  centerY: number,
  radius: number,
  imageHeight: number,
  imageWidth: number,
): PatchBounds | null {
  const left = Math.max(0, Math.floor(centerX - radius));
  const top = Math.max(0, Math.floor(centerY - radius));
  const right = Math.min(imageWidth, Math.ceil(centerX + radius));
  const bottom = Math.min(imageHeight, Math.ceil(centerY + radius));

  const width = right - left;
  const height = bottom - top;
  if (width < 2 || height < 2) {
    return null;
  }

  return { y: top, x: left, height, width };
}

/**
 * A radial falloff mask for one dab, in the patch's own coordinates.
 *
 * `feather` is the fraction of the radius spent fading. At 0 the dab has a hard edge and shows its own
 * outline every time it is laid down; at 1 it fades from the center out and only ever tints. The ramp is
 * smoothstep rather than linear because a linear fade still leaves a visible crease where the gradient of
 * the blend changes abruptly at the two ends.
 */
export function buildDabMask(
  patch: PatchBounds,
  centerX: number,
  centerY: number,
  radius: number,
  feather: number,
): tf.Tensor3D {
  return tf.tidy(() => {
    const rows = tf.range(patch.y, patch.y + patch.height).add(0.5).sub(centerY);
    const cols = tf.range(patch.x, patch.x + patch.width).add(0.5).sub(centerX);

    const distance = rows
      .square()
      .reshape([patch.height, 1])
      .add(cols.square().reshape([1, patch.width]))
      .sqrt();

    // 1 at the center, reaching 0 exactly at the radius.
    const fade = Math.max(1e-3, feather) * radius;
    const t = tf.clipByValue(distance.neg().add(radius).div(fade), 0, 1);
    const smooth = t.square().mul(t.mul(-2).add(3));

    return smooth.reshape([patch.height, patch.width, 1]) as tf.Tensor3D;
  });
}

/**
 * Holds the image being painted on, and applies dabs to it.
 *
 * The tensor is owned here for the lifetime of the session's painting, so `dispose` must be called when
 * the image is replaced or the app is done with it.
 */
export class DreamBrush {
  private image: tf.Tensor3D;
  private cachedMask: { key: string; mask: tf.Tensor3D } | null = null;

  constructor(initial: tf.Tensor3D) {
    this.image = tf.tidy(() => tf.keep(initial.clone()));
  }

  /** The painted image. Borrowed, not owned by the caller — clone it to keep it past the next dab. */
  get current(): tf.Tensor3D {
    return this.image;
  }

  get shape(): [number, number] {
    const [height, width] = this.image.shape;
    return [height, width];
  }

  /**
   * Runs one dab: crop the patch, hand it to `process`, and blend what comes back through the mask.
   *
   * `process` is given the patch and returns a processed copy of it — whatever the current mode does to an
   * image. Keeping that a callback is what lets the brush stay ignorant of DeepDream and style transfer
   * both, and pick up their settings for free.
   */
  async dab(
    centerX: number,
    centerY: number,
    settings: BrushSettings,
    process: (patch: tf.Tensor3D) => Promise<tf.Tensor3D>,
  ): Promise<boolean> {
    const [height, width] = this.shape;
    const patch = patchAround(centerX, centerY, settings.radius, height, width);
    if (!patch) {
      return false;
    }

    const cropped = tf.tidy(
      () => tf.keep(this.image.slice([patch.y, patch.x, 0], [patch.height, patch.width, 3])) as tf.Tensor3D,
    );

    let processed: tf.Tensor3D;
    try {
      processed = await process(cropped);
    } finally {
      cropped.dispose();
    }

    const mask = this.maskFor(patch, centerX, centerY, settings);

    const blended = tf.tidy(() => {
      const existing = this.image.slice([patch.y, patch.x, 0], [patch.height, patch.width, 3]) as tf.Tensor3D;
      const mixed = existing.add(processed.sub(existing).mul(mask)) as tf.Tensor3D;

      // Written back by rebuilding the rows and columns around the patch. tf has no scatter-into-a-slice,
      // and concat of the untouched parts is cheaper than materializing a full-image mask per dab.
      const above = patch.y > 0 ? this.image.slice([0, 0, 0], [patch.y, width, 3]) : null;
      const below =
        patch.y + patch.height < height
          ? this.image.slice([patch.y + patch.height, 0, 0], [height - patch.y - patch.height, width, 3])
          : null;

      const left = patch.x > 0 ? this.image.slice([patch.y, 0, 0], [patch.height, patch.x, 3]) : null;
      const right =
        patch.x + patch.width < width
          ? this.image.slice(
              [patch.y, patch.x + patch.width, 0],
              [patch.height, width - patch.x - patch.width, 3],
            )
          : null;

      const row = tf.concat([left, mixed, right].filter((t): t is tf.Tensor3D => t !== null), 1) as tf.Tensor3D;
      return tf.keep(
        tf.concat([above, row, below].filter((t): t is tf.Tensor3D => t !== null), 0),
      ) as tf.Tensor3D;
    });

    processed.dispose();
    this.image.dispose();
    this.image = blended;
    return true;
  }

  /** Dabs repeat at one spot with the same geometry, so the mask is built once and kept until it changes. */
  private maskFor(patch: PatchBounds, centerX: number, centerY: number, settings: BrushSettings): tf.Tensor3D {
    const key = `${patch.y}:${patch.x}:${patch.height}:${patch.width}:${centerX}:${centerY}:${settings.radius}:${settings.feather}`;
    if (this.cachedMask?.key === key) {
      return this.cachedMask.mask;
    }

    this.cachedMask?.mask.dispose();
    const mask = tf.tidy(
      () => tf.keep(buildDabMask(patch, centerX, centerY, settings.radius, settings.feather)) as tf.Tensor3D,
    );
    this.cachedMask = { key, mask };
    return mask;
  }

  dispose() {
    this.image.dispose();
    this.cachedMask?.mask.dispose();
    this.cachedMask = null;
  }
}
