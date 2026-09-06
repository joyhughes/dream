export type Mode = 'deepdream' | 'style';

import type { ColorSpace } from './ml/colorSpace';

export type { ColorSpace };

export interface DreamLayerTarget {
  nodeName: string;
  label: string;
  weight: number;
}

export interface DreamPreset {
  id: string;
  name: string;
  description: string;
  layers: DreamLayerTarget[];
}

/**
 * Regularizers applied to the image itself once a step has been taken, rather than to the gradient that
 * produced it. Both algorithms apply these identically, so they live in one shape. See `ml/regularizers`.
 */
export interface ImageRegularizers {
  /** Per-step shrink toward the image's own mean, bleeding off runaway pixels. 0 disables it. */
  l2Decay: number;
  /** Standard deviation of the periodic Gaussian blur, in pixels. 0 disables it. */
  blurSigma: number;
  /** Blur every N steps. 0 disables it; 1 blurs every step. */
  blurEvery: number;
}

export interface DreamParams {
  octaves: number;
  octaveScale: number;
  stepsPerOctave: number;
  stepSize: number;
  tileSize: number;
  /** Strength of the total-variation smoothing term, as a fraction of each step. 0 disables it. */
  tvWeight: number;
  /** Frequency bands the ascent gradient is normalized across. 1 is a plain whole-image normalization. */
  lapLevels: number;
  /** The space the optimization steps in. See `ml/colorSpace`. */
  colorSpace: ColorSpace;
  /** How strongly the original hue and saturation are restored after each step. 0 disables it. */
  colorPreservation: number;
  /** Holds the image's average saturation at the source's, so it cannot drift over a run. */
  normalizeSaturation: boolean;
  /** Holds the image's average brightness at the source's, so it cannot drift over a run. */
  normalizeBrightness: boolean;
  /**
   * How large the drawn patterns come out, as a multiplier. 1 is the finest the network can draw; higher
   * values give bigger, coarser motifs. Independent of `octaves`, which spreads detail across many scales
   * at once — this sets the scale itself. See `patternScale` handling in each algorithm.
   */
  patternScale: number;
  regularizers: ImageRegularizers;
}

export interface StyleParams {
  contentWeight: number;
  styleWeight: number;
  totalVariationWeight: number;
  learningRate: number;
  octaves: number;
  octaveScale: number;
  stepsPerOctave: number;
  tileSize: number;
  /** The space the optimization steps in. See `ml/colorSpace`. */
  colorSpace: ColorSpace;
  /** How strongly the content image's hue and saturation are restored after each step. 0 disables it. */
  colorPreservation: number;
  /** Holds the image's average saturation at the source's, so it cannot drift over a run. */
  normalizeSaturation: boolean;
  /** Holds the image's average brightness at the source's, so it cannot drift over a run. */
  normalizeBrightness: boolean;
  /**
   * How large the drawn patterns come out, as a multiplier. 1 is the finest the network can draw; higher
   * values give bigger, coarser motifs. Independent of `octaves`, which spreads detail across many scales
   * at once — this sets the scale itself. See `patternScale` handling in each algorithm.
   */
  patternScale: number;
  regularizers: ImageRegularizers;
}

/** How the brush lays the effect down. Sizes are in working-image pixels. */
export interface BrushSettings {
  /** Radius of the dab, in pixels of the image being painted. */
  radius: number;
  /** Fraction of the radius spent fading out, so the dab blends into what is behind it. 0 is a hard edge. */
  feather: number;
  /** Iterations run per tick while the brush is held down. More is faster to build up, coarser to control. */
  stepsPerDab: number;
}

/** Which tool the pointer drives on the image. `none` leaves the canvas inert. */
export type ToolId = 'none' | 'paint' | 'wand' | 'bucket' | 'lasso' | 'select-brush';

/** Settings shared by the selection tools. Sizes are in working-image pixels. */
export interface SelectionSettings {
  /** How different a pixel may be from the one clicked and still be taken. 0 is an exact color match. */
  tolerance: number;
  /** Whether the wand only takes pixels connected to the one clicked, or every match in the image. */
  contiguous: boolean;
  /** How far the selection's edge fades, in pixels either side of it. 0 is a hard edge. */
  feather: number;
  /** Radius of the selection brush. */
  brushRadius: number;
}

export interface ProgressUpdate {
  step: number;
  totalSteps: number;
  canvas: HTMLCanvasElement;
}

export type EngineStatus =
  | { phase: 'idle' }
  | { phase: 'loading-model' }
  | { phase: 'ready' }
  | { phase: 'running'; step: number; totalSteps: number }
  | { phase: 'done' }
  | { phase: 'error'; message: string };
