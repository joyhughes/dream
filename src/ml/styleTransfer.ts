import { tf } from './tfSetup';
import type { DiscoveredLayer, FeatureModel } from './featureModel';
import { computeOctaveShapes } from './octaves';
import { computeTiledGradient, computeTileGrid, effectiveTileSize, rollImage, type TileSpec } from './tiledGradient';
import { applyImageRegularizers, hasActiveImageRegularizer } from './regularizers';
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

function tileKey(spec: TileSpec): string {
  return `${spec.y}:${spec.x}:${spec.h}:${spec.w}`;
}

/**
 * The content activation each tile is pulled toward, for one step's jitter.
 *
 * The tiled gradient shifts the image before cutting tiles, so a tile at grid position (y, x) holds the
 * content that was at (y − shiftY, x − shiftX). Cropping the reference at (y, x) instead compares each
 * region of the result against a displaced piece of the photo, redrawn every step — the content term then
 * pulls nowhere in particular and blurs out the structure it exists to hold. Rolling the content image by
 * the same shift puts the two back in register.
 *
 * That makes this per-step work rather than per-octave, since the shift is new each step. It stays outside
 * the gradient tape, which is the expensive place: inside, every intermediate of this pass would be held
 * for a backward pass that never uses it.
 */
function computeContentTargets(
  featureModel: FeatureModel,
  contentLayer: DiscoveredLayer,
  contentImageAtOctave: tf.Tensor3D,
  tileSize: number,
  shiftY: number,
  shiftX: number,
): Map<string, tf.Tensor> {
  const [h, w] = contentImageAtOctave.shape;
  const specs = computeTileGrid(h, w, effectiveTileSize(tileSize, h, w));
  const targets = new Map<string, tf.Tensor>();

  const rolled = tf.tidy(() => tf.keep(rollImage(contentImageAtOctave, shiftY, shiftX)) as tf.Tensor3D);

  try {
    for (const spec of specs) {
      // A scope per tile, so only the one activation being kept outlives each pass.
      tf.tidy(() => {
        const crop = rolled.slice([spec.y, spec.x, 0], [spec.h, spec.w, 3]) as tf.Tensor3D;
        const batched = featureModel.preprocess(crop);
        const act = featureModel.activations(batched, [contentLayer.nodeName])[0];
        targets.set(tileKey(spec), tf.keep(act));
      });
    }
  } finally {
    rolled.dispose();
  }

  return targets;
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
  readContentTargets: () => Map<string, tf.Tensor>,
  styleGramTargets: tf.Tensor2D[],
  params: StyleParams,
) {
  // Content and style activations come out of a single pass over the tile. Asking for them separately
  // ran the whole network twice per tile, since a feature model executes the graph once per call.
  const requestNames = Array.from(new Set([contentLayer.nodeName, ...styleLayers.map((l) => l.nodeName)]));
  const indexOfName = new Map(requestNames.map((name, index) => [name, index]));

  return (tile: tf.Tensor3D, spec: TileSpec): tf.Scalar => {
    // The tile is in the working color space; the network only ever sees RGB. Converting inside the tape
    // is what brings the gradient back in the working space's coordinates.
    const rgbTile = params.colorSpace === 'hsv' ? hsvToRgb(tile) : tile;
    const batched = featureModel.preprocess(rgbTile);
    const acts = featureModel.activations(batched, requestNames);

    const contentTarget = readContentTargets().get(tileKey(spec));
    if (!contentTarget) {
      // Both grids come from computeTileGrid on the same dimensions, so this can only mean they've
      // drifted apart in code — worth saying plainly rather than failing inside an arithmetic op.
      throw new Error(`No content target for tile ${tileKey(spec)}; tile grid and content targets disagree.`);
    }

    const contentAct = acts[indexOfName.get(contentLayer.nodeName)!];
    const contentLoss = contentAct.sub(contentTarget).square().mean() as tf.Scalar;

    const styleLosses = styleLayers.map((layer, i) => {
      const act = acts[indexOfName.get(layer.nodeName)!];
      const gram = gramMatrix(act as tf.Tensor4D);
      return gram.sub(styleGramTargets[i]).square().mean() as tf.Scalar;
    });
    const styleLoss = styleLosses.reduce((acc, l) => acc.add(l) as tf.Scalar, tf.scalar(0)).div(
      styleLosses.length,
    ) as tf.Scalar;

    // Smoothness is a property of the visible image, so this is measured on the RGB the network saw
    // rather than on the raw channels — a hue that steps across the red wrap is not a rough edge.
    const tvLoss = totalVariationLoss(rgbTile);

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
  const [styleH, styleW] = styleImage.shape;
  const cropDim = Math.min(tileSize, styleH, styleW);
  const maxY = styleH - cropDim;
  const maxX = styleW - cropDim;

  const samplesPerLayer: tf.Tensor2D[][] = styleLayers.map(() => []);

  for (let i = 0; i < STYLE_CROP_SAMPLES; i++) {
    const y = maxY > 0 ? Math.floor(Math.random() * (maxY + 1)) : 0;
    const x = maxX > 0 ? Math.floor(Math.random() * (maxX + 1)) : 0;

    // One scope per sample, rather than one around the whole loop: the Gram matrices are all that
    // outlive a sample, but a shared scope would hold every crop's activations — a full set of network
    // intermediates each — alive until the last sample was done.
    tf.tidy(() => {
      const crop = styleImage.slice([y, x, 0], [cropDim, cropDim, 3]) as tf.Tensor3D;
      const batched = featureModel.preprocess(crop);
      const acts = featureModel.activations(batched, styleLayers.map((l) => l.nodeName));
      acts.forEach((act, li) => {
        samplesPerLayer[li].push(tf.keep(gramMatrix(act as tf.Tensor4D)));
      });
    });
  }

  return samplesPerLayer.map((samples) => {
    const mean = tf.tidy(() => tf.keep(tf.stack(samples).mean(0) as tf.Tensor2D));
    samples.forEach((sample) => sample.dispose());
    return mean;
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
  // Rebuilt below, once the tiled gradient has chosen this step's jitter.
  let contentTargets = new Map<string, tf.Tensor>();

  const lossFn = makeTiledStyleLoss(
    featureModel,
    contentLayer,
    styleLayers,
    () => contentTargets,
    styleGramTargets,
    params,
  );

  let gradient: tf.Tensor3D;
  try {
    gradient = await computeTiledGradient(
      generated as unknown as tf.Tensor3D,
      params.tileSize,
      lossFn,
      (shiftY, shiftX) => {
        contentTargets = computeContentTargets(
          featureModel,
          contentLayer,
          contentImageAtOctave,
          params.tileSize,
          shiftY,
          shiftX,
        );
      },
    );
  } finally {
    contentTargets.forEach((target) => target.dispose());
  }

  // Tiling means accumulating several separate tf.grad() calls into one gradient tensor, which
  // optimizer.minimize()'s built-in autodiff can't do — applyGradients() lets us hand it a precomputed one.
  optimizer.applyGradients([{ name: generated.name, tensor: gradient }]);
  gradient.dispose();

  tf.tidy(() => {
    generated.assign(clampToColorSpace(generated as unknown as tf.Tensor3D, params.colorSpace));
  });
}

export async function runStyleTransfer(
  contentImage: tf.Tensor3D,
  styleImage: tf.Tensor3D,
  options: RunStyleTransferOptions,
): Promise<tf.Tensor3D> {
  const { featureModel, params, onProgress, signal, pauseController } = options;

  const sorted = [...featureModel.layers].sort((a, b) => a.depthIndex - b.depthIndex);
  const byName = new Map(sorted.map((layer) => [layer.nodeName, layer]));

  // A network that names its own style/content layers (VGG-19 has a well-established set) is taken at its
  // word; anything else falls back to sampling the discovered layer list by depth.
  const contentLayer =
    (featureModel.contentLayerName ? byName.get(featureModel.contentLayerName) : undefined) ??
    pickLayer(sorted, CONTENT_LAYER_FRACTION);

  const curated = featureModel.styleLayerNames
    ?.map((name) => byName.get(name))
    .filter((layer): layer is DiscoveredLayer => layer !== undefined);

  const styleLayers: DiscoveredLayer[] = [];
  const seen = new Set<string>();
  for (const layer of curated?.length ? curated : STYLE_LAYER_FRACTIONS.map((f) => pickLayer(sorted, f))) {
    if (!seen.has(layer.nodeName)) {
      seen.add(layer.nodeName);
      styleLayers.push(layer);
    }
  }

  // Style statistics don't depend on octave resolution, so these are computed once up front.
  const styleGramTargets = computeStyleGramTargets(featureModel, styleImage, styleLayers, params.tileSize);

  const [h, w] = contentImage.shape;

  // Pattern scale works the same way it does for DeepDream: optimize at a coarser resolution and enlarge
  // the result. Sizing the template's crop instead — matching a C-wide view of the template against a
  // T-wide content tile to reproduce motifs at M·T/C — is the tempting derivation and does not hold. The
  // optimizer can only draw what the network's fixed-size view can represent, which pins the largest
  // feature it can express to the tile's own scale; asking for a motif beyond that returns mush rather
  // than a bigger motif. Measured, that route moved the output by 0.3% to 4% with no monotone trend.
  const scale = Math.max(1, params.patternScale);
  const workH = Math.max(8, Math.round(h / scale));
  const workW = Math.max(8, Math.round(w / scale));

  const shapes = computeOctaveShapes(workH, workW, params.octaves, params.octaveScale);

  /**
   * Wraps a tensor in a variable and releases the tensor.
   *
   * `tf.variable` shares its argument's data rather than copying it, and refcounts that data — but it
   * leaves the original tensor alive and owned by whoever made it. Handing it a temporary and walking away
   * therefore leaks one full-resolution tensor, once at the start and once at every octave boundary. It
   * cost two per run at the default settings, which is the sort of thing that takes a phone tab out after
   * a handful of them.
   */
  const asVariable = (value: tf.Tensor3D): tf.Variable => {
    const variable = tf.variable(value, true, 'dream-style-generated');
    value.dispose();
    return variable;
  };

  // `generated` is held in the working color space from here on; `contentImageAtOctave` stays RGB, since
  // it only ever feeds the network.
  let generated = asVariable(
    tf.tidy(() => fromRgb(tf.image.resizeBilinear(contentImage, shapes[0]) as tf.Tensor3D, params.colorSpace)),
  );
  let optimizer = tf.train.adam(params.learningRate);
  let contentImageAtOctave = tf.tidy(
    () => tf.keep(tf.image.resizeBilinear(contentImage, shapes[0])) as tf.Tensor3D,
  );
  // Measured from the content image, which is what the result is meant to still look like the color of.
  const targetSaturation = params.normalizeSaturation
    ? (tf.tidy(() => tf.keep(meanSaturation(contentImage, 'rgb'))) as tf.Scalar)
    : null;
  const targetBrightness = params.normalizeBrightness
    ? (tf.tidy(() => tf.keep(meanBrightness(contentImage, 'rgb'))) as tf.Scalar)
    : null;

  try {
    octaveLoop: for (let octave = 0; octave < shapes.length; octave++) {
      const [targetH, targetW] = shapes[octave];

      if (octave > 0) {
        const upscaled = tf.tidy(() =>
          resizeInRgb(generated as unknown as tf.Tensor3D, params.colorSpace, [targetH, targetW]),
        );
        generated.dispose();
        generated = asVariable(upscaled);

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

        // Image-space regularizers act on the variable in place. Adam's moment estimates are keyed by
        // variable name and survive the assignment, so its momentum carries across the change rather
        // than being reset by it.
        if (hasActiveImageRegularizer(params.regularizers, step)) {
          // Applied in RGB whatever space the run is in: blurring the hue channel would smear across the
          // red wrap, and decaying it would pull every color toward whichever hue sits at zero.
          const regularized = tf.tidy(() => {
            const rgb = toRgb(generated as unknown as tf.Tensor3D, params.colorSpace);
            return tf.keep(fromRgb(applyImageRegularizers(rgb, params.regularizers, step), params.colorSpace));
          });
          generated.assign(regularized);
          regularized.dispose();
        }

        // The content image at this octave is already the reference we want, and already RGB.
        if (params.colorPreservation > 0) {
          const preserved = tf.tidy(() =>
            tf.keep(
              preserveColor(
                generated as unknown as tf.Tensor3D,
                contentImageAtOctave,
                params.colorPreservation,
                params.colorSpace,
              ),
            ),
          );
          generated.assign(preserved);
          preserved.dispose();
        }

        // After preservation, so it has the final say on the average.
        if (targetSaturation) {
          const normalized = tf.tidy(() =>
            tf.keep(normalizeSaturation(generated as unknown as tf.Tensor3D, targetSaturation, params.colorSpace)),
          );
          generated.assign(normalized);
          normalized.dispose();
        }

        // Orthogonal to the saturation pass above, so the order only matters where clipping bites.
        if (targetBrightness) {
          const normalized = tf.tidy(() =>
            tf.keep(normalizeBrightness(generated as unknown as tf.Tensor3D, targetBrightness, params.colorSpace)),
          );
          generated.assign(normalized);
          normalized.dispose();
        }

        if (onProgress && (step % 5 === 0 || step === params.stepsPerOctave - 1)) {
          // Converted for display: `generated` holds the working color space, and painting HSV channels
          // as if they were RGB shows colors that are nowhere in the image.
          await withRgbView(generated as unknown as tf.Tensor3D, params.colorSpace, (image) =>
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

    // Callers always get RGB back, whatever space the run worked in, and at the size they handed in.
    return tf.tidy(() => {
      const rgb = toRgb(generated as unknown as tf.Tensor3D, params.colorSpace);
      return rgb.shape[0] === h && rgb.shape[1] === w
        ? (tf.keep(rgb) as tf.Tensor3D)
        : (tf.keep(tf.image.resizeBilinear(rgb, [h, w])) as tf.Tensor3D);
    });
  } finally {
    generated.dispose();
    contentImageAtOctave.dispose();
    targetSaturation?.dispose();
    targetBrightness?.dispose();
    styleGramTargets.forEach((g) => g.dispose());
    optimizer.dispose();
  }
}
