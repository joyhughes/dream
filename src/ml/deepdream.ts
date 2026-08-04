import { tf } from './tfSetup';
import { preprocessForMobilenet } from './imageUtils';
import { getActivations, type FeatureModel } from './mobilenetFeatures';
import { computeOctaveShapes } from './octaves';
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
  modelInputSize?: number;
  previewEvery?: number;
  onProgress?: (progress: DeepDreamProgress) => void | Promise<void>;
  signal?: AbortSignal;
}

function computeLoss(
  imgVar: tf.Tensor3D,
  featureModel: FeatureModel,
  preset: DreamPreset,
  modelInputSize: number,
): tf.Scalar {
  const batched = preprocessForMobilenet(imgVar, modelInputSize);
  const nodeNames = preset.layers.map((l) => l.nodeName);
  const activations = getActivations(featureModel.graphModel, batched, nodeNames);

  const terms = activations.map((act, i) => act.mean().mul(preset.layers[i].weight) as tf.Scalar);
  const total = terms.reduce((acc, t) => acc.add(t) as tf.Scalar, tf.scalar(0));
  return total;
}

export async function runDeepDream(baseImage: tf.Tensor3D, options: RunDeepDreamOptions): Promise<tf.Tensor3D> {
  const { featureModel, preset, params, modelInputSize = 224, previewEvery = 5, onProgress, signal } = options;

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

      const updated = tf.tidy(() => {
        const lossFn = (x: tf.Tensor) => computeLoss(x as tf.Tensor3D, featureModel, preset, modelInputSize);
        const gradFn = tf.grad(lossFn);
        const g = gradFn(current) as tf.Tensor3D;

        const { variance } = tf.moments(g);
        const std = variance.sqrt();
        const normalized = g.div(std.add(1e-8)) as tf.Tensor3D;

        return tf.keep(tf.clipByValue(current.add(normalized.mul(params.stepSize)), 0, 1)) as tf.Tensor3D;
      });

      current.dispose();
      current = updated;

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
