import { tf } from './tfSetup';
import { preprocessForMobilenet } from './imageUtils';
import { getActivations, type DiscoveredLayer, type FeatureModel } from './mobilenetFeatures';
import { computeOctaveShapes } from './octaves';
import { computeTiledGradient, type TileSpec } from './tiledGradient';
import type { PauseController } from './pauseController';
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
  pauseController?: PauseController;
}

const STYLE_LAYER_FRACTIONS = [0.05, 0.2, 0.4, 0.6, 0.8];
const CONTENT_LAYER_FRACTION = 0.5;
const STYLE_CROP_SAMPLES = 4;

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
    if (h < 2 || w < 2) {
      return tf.scalar(0);
    }
    const dx = img.slice([0, 0, 0], [h - 1, w, 3]).sub(img.slice([1, 0, 0], [h - 1, w, 3]));
    const dy = img.slice([0, 0, 0], [h, w - 1, 3]).sub(img.slice([0, 1, 0], [h, w - 1, 3]));
    return (dx.square().mean().add(dy.square().mean())) as tf.Scalar;
  });
}

/**
 * Style statistics (Gram matrices) are translation-invariant texture stats, so a single global target works
 * fine per tile. Content, on the other hand, is spatially specific — comparing a tile against a single global
 * content activation would push every tile toward the same content, so each tile is compared against the
 * matching crop of the content image at this octave's resolution instead.
 */
function makeTiledStyleLoss(
  featureModel: FeatureModel,
  contentLayer: DiscoveredLayer,
  styleLayers: DiscoveredLayer[],
  contentImageAtOctave: tf.Tensor3D,
  styleGramTargets: tf.Tensor2D[],
  params: StyleParams,
) {
  return (tile: tf.Tensor3D, spec: TileSpec): tf.Scalar => {
    const batched = preprocessForMobilenet(tile);

    const contentAct = getActivations(featureModel.graphModel, batched, [contentLayer.nodeName])[0];
    const contentCrop = contentImageAtOctave.slice([spec.y, spec.x, 0], [spec.h, spec.w, 3]);
    const contentTargetBatched = preprocessForMobilenet(contentCrop);
    const contentTargetAct = getActivations(featureModel.graphModel, contentTargetBatched, [contentLayer.nodeName])[0];
    const contentLoss = contentAct.sub(contentTargetAct).square().mean() as tf.Scalar;

    const styleActs = getActivations(featureModel.graphModel, batched, styleLayers.map((l) => l.nodeName));
    const styleLosses = styleActs.map((act, i) => {
      const gram = gramMatrix(act as tf.Tensor4D);
      return gram.sub(styleGramTargets[i]).square().mean() as tf.Scalar;
    });
    const styleLoss = styleLosses.reduce((acc, l) => acc.add(l) as tf.Scalar, tf.scalar(0)).div(
      styleLosses.length,
    ) as tf.Scalar;

    const tvLoss = totalVariationLoss(tile);

    return contentLoss
      .mul(params.contentWeight)
      .add(styleLoss.mul(params.styleWeight))
      .add(tvLoss.mul(params.totalVariationWeight)) as tf.Scalar;
  };
}

/**
 * Content tiles only ever show the network a `tileSize`-pixel crop of the (often much larger)
 * content image — but if style statistics come from the *whole* template squished down to the
 * network's 224x224 input, the network is matching "the whole template" against "a small crop of
 * the content," at two different real-world scales. The optimizer then reproduces the pattern at
 * whatever size makes those two 224x224 views match statistically, which — since the content crop
 * covers far fewer native pixels than the whole template did — comes out as many small repeats
 * instead of one style-sized motif. Sampling several `tileSize`-ish crops of the template instead
 * (matching the same real-world scale content tiles are seen at, and averaging for stability)
 * keeps the reproduced pattern close to its size in the original template.
 */
function computeStyleGramTargets(
  featureModel: FeatureModel,
  styleImage: tf.Tensor3D,
  styleLayers: DiscoveredLayer[],
  tileSize: number,
): tf.Tensor2D[] {
  return tf.tidy(() => {
    const [styleH, styleW] = styleImage.shape;
    const cropDim = Math.min(tileSize, styleH, styleW);
    const maxY = styleH - cropDim;
    const maxX = styleW - cropDim;

    const samplesPerLayer: tf.Tensor2D[][] = styleLayers.map(() => []);

    for (let i = 0; i < STYLE_CROP_SAMPLES; i++) {
      const y = maxY > 0 ? Math.floor(Math.random() * (maxY + 1)) : 0;
      const x = maxX > 0 ? Math.floor(Math.random() * (maxX + 1)) : 0;
      const crop = styleImage.slice([y, x, 0], [cropDim, cropDim, 3]) as tf.Tensor3D;
      const batched = preprocessForMobilenet(crop);
      const acts = getActivations(featureModel.graphModel, batched, styleLayers.map((l) => l.nodeName));
      acts.forEach((act, li) => {
        samplesPerLayer[li].push(gramMatrix(act as tf.Tensor4D));
      });
    }

    return samplesPerLayer.map((samples) => tf.keep(tf.stack(samples).mean(0) as tf.Tensor2D));
  });
}

/** One Adam step at the generated variable's current resolution, using a tiled gradient. Mutates `generated`. */
async function optimizeStep(
  generated: tf.Variable,
  optimizer: tf.AdamOptimizer,
  featureModel: FeatureModel,
  contentLayer: DiscoveredLayer,
  styleLayers: DiscoveredLayer[],
  contentImageAtOctave: tf.Tensor3D,
  styleGramTargets: tf.Tensor2D[],
  params: StyleParams,
): Promise<void> {
  const lossFn = makeTiledStyleLoss(featureModel, contentLayer, styleLayers, contentImageAtOctave, styleGramTargets, params);
  const gradient = await computeTiledGradient(generated as unknown as tf.Tensor3D, params.tileSize, lossFn);

  // Tiling means accumulating several separate tf.grad() calls into one gradient tensor, which
  // optimizer.minimize()'s built-in autodiff can't do — applyGradients() lets us hand it a precomputed one.
  optimizer.applyGradients([{ name: generated.name, tensor: gradient }]);
  gradient.dispose();

  tf.tidy(() => {
    generated.assign(tf.clipByValue(generated, 0, 1));
  });
}

export async function runStyleTransfer(
  contentImage: tf.Tensor3D,
  styleImage: tf.Tensor3D,
  options: RunStyleTransferOptions,
): Promise<tf.Tensor3D> {
  const { featureModel, params, onProgress, signal, pauseController } = options;

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

  // Style statistics don't depend on octave resolution, so these are computed once up front.
  const styleGramTargets = computeStyleGramTargets(featureModel, styleImage, styleLayers, params.tileSize);

  const [h, w] = contentImage.shape;
  const shapes = computeOctaveShapes(h, w, params.octaves, params.octaveScale);

  let generated = tf.variable(
    tf.tidy(() => tf.image.resizeBilinear(contentImage, shapes[0]) as tf.Tensor3D),
    true,
    'dream-style-generated',
  );
  let optimizer = tf.train.adam(params.learningRate);
  let contentImageAtOctave = tf.tidy(
    () => tf.keep(tf.image.resizeBilinear(contentImage, shapes[0])) as tf.Tensor3D,
  );

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

        contentImageAtOctave.dispose();
        contentImageAtOctave = tf.tidy(
          () => tf.keep(tf.image.resizeBilinear(contentImage, [targetH, targetW])) as tf.Tensor3D,
        );
      }

      for (let step = 0; step < params.stepsPerOctave; step++) {
        if (signal?.aborted) {
          break octaveLoop;
        }

        await pauseController?.waitIfPaused(signal);

        if (signal?.aborted) {
          break octaveLoop;
        }

        await optimizeStep(
          generated,
          optimizer,
          featureModel,
          contentLayer,
          styleLayers,
          contentImageAtOctave,
          styleGramTargets,
          params,
        );

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
    contentImageAtOctave.dispose();
    styleGramTargets.forEach((g) => g.dispose());
    optimizer.dispose();
  }
}
