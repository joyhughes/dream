export type Mode = 'deepdream' | 'style';

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
  regularizers: ImageRegularizers;
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
