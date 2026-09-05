import { tf } from './tfSetup';
import type { FeatureModel } from './featureModel';
import { computeOctaveShapes } from './octaves';
import { computeTiledGradient } from './tiledGradient';
import { applyImageRegularizers, laplacianNormalize, totalVariationGradient } from './regularizers';
import type { PauseController } from './pauseController';
import type { DreamParams, DreamPreset } from '../types';

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

function computeLoss(tile: tf.Tensor3D, featureModel: FeatureModel, preset: DreamPreset): tf.Scalar {
  const batched = featureModel.preprocess(tile);
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
  const shapes = computeOctaveShapes(h, w, params.octaves, params.octaveScale);

  let current = tf.tidy(() => tf.keep(tf.image.resizeBilinear(baseImage, shapes[0])) as tf.Tensor3D);

  for (let octave = 0; octave < shapes.length; octave++) {
    const [targetH, targetW] = shapes[octave];

    const upscaled = tf.tidy(() => tf.keep(tf.image.resizeBilinear(current, [targetH, targetW])) as tf.Tensor3D);
    current.dispose();
    current = upscaled;

    for (let step = 0; step < params.stepsPerOctave; step++) {
      if (signal?.aborted) {
        return current;
      }

      await pauseController?.waitIfPaused(signal);

      if (signal?.aborted) {
        return current;
      }

      const gradient = await computeTiledGradient(current, params.tileSize, (tile) =>
        computeLoss(tile, featureModel, preset),
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

        return tf.keep(tf.clipByValue(current.add(direction.mul(params.stepSize)), 0, 1)) as tf.Tensor3D;
      });
      gradient.dispose();

      current.dispose();
      current = updated;

      const regularized = applyImageRegularizers(current, params.regularizers, step);
      if (regularized) {
        current.dispose();
        current = regularized;
      }

      if (onProgress && (step % previewEvery === 0 || step === params.stepsPerOctave - 1)) {
        await onProgress({
          octave,
          totalOctaves: shapes.length,
          step,
          totalStepsInOctave: params.stepsPerOctave,
          image: current,
        });
      }

      await tf.nextFrame();
    }
  }

  return current;
}
