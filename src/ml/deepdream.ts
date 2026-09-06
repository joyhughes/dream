import { tf } from './tfSetup';
import type { FeatureModel } from './featureModel';
import { computeOctaveShapes } from './octaves';
import { computeTiledGradient } from './tiledGradient';
import {
  applyImageRegularizers,
  hasActiveImageRegularizer,
  laplacianNormalize,
  totalVariationGradient,
} from './regularizers';
import {
  clampToColorSpace,
  fromRgb,
  hsvToRgb,
  meanBrightness,
  meanSaturation,
  normalizeBrightness,
  normalizeSaturation,
  preserveColor,
  resizeInRgb,
  toRgb,
  withRgbView,
} from './colorSpace';
import type { PauseController } from './pauseController';
import type { ColorSpace, DreamParams, DreamPreset } from '../types';

export interface DeepDreamProgress {
  octave: number;
  totalOctaves: number;
  step: number;
  totalStepsInOctave: number;
  image: tf.Tensor3D;
}

export interface RunDeepDreamOptions {
  featureModel: FeatureModel;
  preset: DreamPreset;
  params: DreamParams;
  previewEvery?: number;
  onProgress?: (progress: DeepDreamProgress) => void | Promise<void>;
  signal?: AbortSignal;
  pauseController?: PauseController;
}

function computeLoss(
  tile: tf.Tensor3D,
  featureModel: FeatureModel,
  preset: DreamPreset,
  colorSpace: ColorSpace,
): tf.Scalar {
  // The tile arrives in whatever space the ascent works in; the network only ever sees RGB. This
  // conversion is inside the tape, so the gradient comes back in the working space's coordinates.
  const batched = featureModel.preprocess(colorSpace === 'hsv' ? hsvToRgb(tile) : tile);
  const nodeNames = preset.layers.map((l) => l.nodeName);
  const activations = featureModel.activations(batched, nodeNames);

  const terms = activations.map((act, i) => act.mean().mul(preset.layers[i].weight) as tf.Scalar);
  const total = terms.reduce((acc, t) => acc.add(t) as tf.Scalar, tf.scalar(0));
  return total;
}

export async function runDeepDream(baseImage: tf.Tensor3D, options: RunDeepDreamOptions): Promise<tf.Tensor3D> {
  const { featureModel, preset, params, previewEvery = 5, onProgress, signal, pauseController } = options;

  if (preset.layers.length === 0) {
    throw new Error('Preset has no target layers.');
  }

  const [h, w] = baseImage.shape;

  // Pattern scale works by drawing at a coarser resolution and enlarging the result. The network draws
  // features at a size fixed by its own input, so the only way to make them come out bigger in the final
  // image is to give it fewer pixels to draw on — the same lever octaves pull, held at one setting instead
  // of swept across several. Octaves still spread detail across scales; this sets where that range sits.
  const scale = Math.max(1, params.patternScale);
  const workH = Math.max(8, Math.round(h / scale));
  const workW = Math.max(8, Math.round(w / scale));

  const shapes = computeOctaveShapes(workH, workW, params.octaves, params.octaveScale);

  const { colorSpace } = params;

  // From here to the end of the run `current` is held in `colorSpace`, not RGB. Everything that touches
  // it — the loss, the regularizers, the preview — converts at its own boundary.
  let current = tf.tidy(() => {
    const resized = tf.image.resizeBilinear(baseImage, shapes[0]) as tf.Tensor3D;
    return tf.keep(fromRgb(resized, colorSpace)) as tf.Tensor3D;
  });

  // The starting image at the current octave's size, kept only when color preservation needs something to
  // restore toward.
  let reference: tf.Tensor3D | null = null;

  // A single scalar, measured once from the input and held on the GPU for the whole run. Average
  // saturation barely moves with scale, so one measurement serves every octave.
  const targetSaturation = params.normalizeSaturation
    ? (tf.tidy(() => tf.keep(meanSaturation(baseImage, 'rgb'))) as tf.Scalar)
    : null;
  const targetBrightness = params.normalizeBrightness
    ? (tf.tidy(() => tf.keep(meanBrightness(baseImage, 'rgb'))) as tf.Scalar)
    : null;

  // Callers always get RGB back, whatever space the run worked in — including on an abort, which can
  // land anywhere in the loop below.
  const finish = (): tf.Tensor3D => {
    const rgb = tf.tidy(() => {
      const asRgb = toRgb(current, colorSpace);
      // Back to the caller's resolution. At scale 1 the sizes already agree and this is a no-op.
      return asRgb.shape[0] === h && asRgb.shape[1] === w
        ? tf.keep(asRgb)
        : (tf.keep(tf.image.resizeBilinear(asRgb, [h, w])) as tf.Tensor3D);
    });
    current.dispose();
    reference?.dispose();
    targetSaturation?.dispose();
    targetBrightness?.dispose();
    return rgb;
  };

  for (let octave = 0; octave < shapes.length; octave++) {
    const [targetH, targetW] = shapes[octave];

    const upscaled = tf.tidy(
      () => tf.keep(resizeInRgb(current, colorSpace, [targetH, targetW])) as tf.Tensor3D,
    );
    current.dispose();
    current = upscaled;

    if (params.colorPreservation > 0) {
      reference?.dispose();
      reference = tf.tidy(
        () => tf.keep(tf.image.resizeBilinear(baseImage, [targetH, targetW])) as tf.Tensor3D,
      );
    }

    for (let step = 0; step < params.stepsPerOctave; step++) {
      if (signal?.aborted) {
        return finish();
      }

      await pauseController?.waitIfPaused(signal);

      if (signal?.aborted) {
        return finish();
      }

      const gradient = await computeTiledGradient(current, params.tileSize, (tile) =>
        computeLoss(tile, featureModel, preset, colorSpace),
      );

      const updated = tf.tidy(() => {
        // Both terms are normalized to unit standard deviation before they are combined, so `stepSize`
        // means the same thing regardless of the loss's raw scale and `tvWeight` reads as the share of
        // the step spent smoothing rather than ascending.
        const ascent = laplacianNormalize(gradient, params.lapLevels);

        // Total variation is a penalty, so its gradient is subtracted from the direction we ascend in.
        // It is computed on the whole image rather than per tile: smoothness across a tile boundary is
        // exactly what a per-tile version would be blind to.
        const direction =
          params.tvWeight > 0
            ? (ascent.sub(laplacianNormalize(totalVariationGradient(current), 1).mul(params.tvWeight)) as tf.Tensor3D)
            : ascent;

        const stepped = current.add(direction.mul(params.stepSize)) as tf.Tensor3D;
        return tf.keep(clampToColorSpace(stepped, colorSpace)) as tf.Tensor3D;
      });
      gradient.dispose();

      current.dispose();
      current = updated;

      // Regularizers are defined on natural images, so they are applied in RGB regardless of the space
      // being optimized in: a blur of the hue channel would smear across the red wrap, and a decay of it
      // would pull every color toward whatever hue happens to sit at zero rather than toward gray.
      if (hasActiveImageRegularizer(params.regularizers, step)) {
        const regularized = tf.tidy(() => {
          const rgb = toRgb(current, colorSpace);
          return tf.keep(fromRgb(applyImageRegularizers(rgb, params.regularizers, step), colorSpace)) as tf.Tensor3D;
        });
        current.dispose();
        current = regularized;
      }

      // These two constrain whatever the step and the regularizers between them produced, so they run
      // last — and normalization runs after preservation, so it has the final say on the average.
      if (reference) {
        const preserved = tf.tidy(
          () => tf.keep(preserveColor(current, reference!, params.colorPreservation, colorSpace)) as tf.Tensor3D,
        );
        current.dispose();
        current = preserved;
      }

      if (targetSaturation) {
        const normalized = tf.tidy(
          () => tf.keep(normalizeSaturation(current, targetSaturation, colorSpace)) as tf.Tensor3D,
        );
        current.dispose();
        current = normalized;
      }

      // After saturation: the two are orthogonal — one holds value fixed, the other holds hue and
      // saturation fixed — so neither undoes the other, and this one ends up with the last word on
      // brightness where clipping makes them disagree at all.
      if (targetBrightness) {
        const normalized = tf.tidy(
          () => tf.keep(normalizeBrightness(current, targetBrightness, colorSpace)) as tf.Tensor3D,
        );
        current.dispose();
        current = normalized;
      }

      if (onProgress && (step % previewEvery === 0 || step === params.stepsPerOctave - 1)) {
        await withRgbView(current, colorSpace, (image) =>
          onProgress({
            octave,
            totalOctaves: shapes.length,
            step,
            totalStepsInOctave: params.stepsPerOctave,
            image,
          }),
        );
      }

      await tf.nextFrame();
    }
  }

  return finish();
}
