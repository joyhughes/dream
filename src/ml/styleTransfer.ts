import { tf } from './tfSetup';
import { preprocessForMobilenet } from './imageUtils';
import { getActivations, type DiscoveredLayer, type FeatureModel } from './mobilenetFeatures';
import { computeOctaveShapes } from './octaves';
import type { StyleParams } from '../types';

export interface StyleTransferProgress {
  octave: number;
  totalOctaves: number;
  step: number;
  totalStepsInOctave: number;
  image: tf.Tensor3D;
}

export interface RunStyleTransferOptions {
  featureModel: FeatureModel;
  params: StyleParams;
  onProgress?: (progress: StyleTransferProgress) => void | Promise<void>;
  signal?: AbortSignal;
}

const STYLE_LAYER_FRACTIONS = [0.05, 0.2, 0.4, 0.6, 0.8];
const CONTENT_LAYER_FRACTION = 0.5;

function pickLayer(sorted: DiscoveredLayer[], fraction: number): DiscoveredLayer {
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round(fraction * (sorted.length - 1))));
  return sorted[idx];
}

function gramMatrix(activation: tf.Tensor4D): tf.Tensor2D {
  return tf.tidy(() => {
    const [, h, w, c] = activation.shape;
    const reshaped = activation.reshape([h * w, c]) as tf.Tensor2D;
    const gram = reshaped.transpose().matMul(reshaped) as tf.Tensor2D;
    return gram.div(h * w);
  });
}

function totalVariationLoss(img: tf.Tensor3D): tf.Scalar {
  return tf.tidy(() => {
    const [h, w] = img.shape;
    const dx = img.slice([0, 0, 0], [h - 1, w, 3]).sub(img.slice([1, 0, 0], [h - 1, w, 3]));
    const dy = img.slice([0, 0, 0], [h, w - 1, 3]).sub(img.slice([0, 1, 0], [h, w - 1, 3]));
    return (dx.square().mean().add(dy.square().mean())) as tf.Scalar;
  });
}

/** One Adam step at the generated variable's current resolution. Mutates `generated` in place. */
function optimizeStep(
  generated: tf.Variable,
  optimizer: tf.AdamOptimizer,
  featureModel: FeatureModel,
  contentLayer: DiscoveredLayer,
  styleLayers: DiscoveredLayer[],
  contentTarget: tf.Tensor,
  styleGramTargets: tf.Tensor2D[],
  params: StyleParams,
): void {
  optimizer.minimize(() => {
    return tf.tidy(() => {
      const batched = preprocessForMobilenet(generated as unknown as tf.Tensor3D, params.imageSize);

      const contentAct = getActivations(featureModel.graphModel, batched, [contentLayer.nodeName])[0];
      const contentLoss = contentAct.sub(contentTarget).square().mean() as tf.Scalar;

      const styleActs = getActivations(featureModel.graphModel, batched, styleLayers.map((l) => l.nodeName));
      const styleLosses = styleActs.map((act, i) => {
        const gram = gramMatrix(act as tf.Tensor4D);
        return gram.sub(styleGramTargets[i]).square().mean() as tf.Scalar;
      });
      const styleLoss = styleLosses.reduce((acc, l) => acc.add(l) as tf.Scalar, tf.scalar(0)).div(
        styleLosses.length,
      ) as tf.Scalar;

      const tvLoss = totalVariationLoss(generated as unknown as tf.Tensor3D);

      return contentLoss
        .mul(params.contentWeight)
        .add(styleLoss.mul(params.styleWeight))
        .add(tvLoss.mul(params.totalVariationWeight)) as tf.Scalar;
    });
  }, false, [generated]);

  tf.tidy(() => {
    generated.assign(tf.clipByValue(generated, 0, 1));
  });
}

export async function runStyleTransfer(
  contentImage: tf.Tensor3D,
  styleImage: tf.Tensor3D,
  options: RunStyleTransferOptions,
): Promise<tf.Tensor3D> {
  const { featureModel, params, onProgress, signal } = options;

  const sorted = [...featureModel.layers].sort((a, b) => a.depthIndex - b.depthIndex);
  const contentLayer = pickLayer(sorted, CONTENT_LAYER_FRACTION);

  const styleLayerNames = new Set<string>();
  const styleLayers: DiscoveredLayer[] = [];
  for (const fraction of STYLE_LAYER_FRACTIONS) {
    const layer = pickLayer(sorted, fraction);
    if (!styleLayerNames.has(layer.nodeName)) {
      styleLayerNames.add(layer.nodeName);
      styleLayers.push(layer);
    }
  }

  const contentTarget = tf.tidy(() => {
    const batched = preprocessForMobilenet(contentImage, params.imageSize);
    const [act] = getActivations(featureModel.graphModel, batched, [contentLayer.nodeName]);
    return tf.keep(act.clone());
  });

  const styleGramTargets: tf.Tensor2D[] = tf.tidy(() => {
    const batched = preprocessForMobilenet(styleImage, params.imageSize);
    const acts = getActivations(featureModel.graphModel, batched, styleLayers.map((l) => l.nodeName));
    return acts.map((act) => tf.keep(gramMatrix(act as tf.Tensor4D)));
  });

  const [h, w] = contentImage.shape;
  const shapes = computeOctaveShapes(h, w, params.octaves, params.octaveScale);

  let generated = tf.variable(
    tf.tidy(() => tf.image.resizeBilinear(contentImage, shapes[0]) as tf.Tensor3D),
    true,
    'dream-style-generated',
  );
  let optimizer = tf.train.adam(params.learningRate);

  try {
    octaveLoop: for (let octave = 0; octave < shapes.length; octave++) {
      const [targetH, targetW] = shapes[octave];

      if (octave > 0) {
        const upscaled = tf.tidy(() => tf.image.resizeBilinear(generated, [targetH, targetW]).clone());
        generated.dispose();
        generated = tf.variable(upscaled, true, 'dream-style-generated');

        // Adam's momentum accumulators are keyed by variable name, not identity — since `generated` is
        // recreated at each octave's resolution, a fresh optimizer avoids reusing wrong-shaped moment tensors.
        optimizer.dispose();
        optimizer = tf.train.adam(params.learningRate);
      }

      for (let step = 0; step < params.stepsPerOctave; step++) {
        if (signal?.aborted) {
          break octaveLoop;
        }

        optimizeStep(generated, optimizer, featureModel, contentLayer, styleLayers, contentTarget, styleGramTargets, params);

        if (onProgress && (step % 5 === 0 || step === params.stepsPerOctave - 1)) {
          await onProgress({
            octave,
            totalOctaves: shapes.length,
            step,
            totalStepsInOctave: params.stepsPerOctave,
            image: generated as unknown as tf.Tensor3D,
          });
        }

        await tf.nextFrame();
      }
    }

    return tf.tidy(() => tf.keep(generated.clone())) as tf.Tensor3D;
  } finally {
    generated.dispose();
    contentTarget.dispose();
    styleGramTargets.forEach((g) => g.dispose());
    optimizer.dispose();
  }
}
