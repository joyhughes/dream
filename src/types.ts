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
  /**
   * Present on a blend of several presets: each one's layers, with its share of the push. The loss scales
   * each group to the same size before weighting it, so a share means a share rather than whichever layer
   * happens to respond most strongly. See `blendPresets`.
   */
  groups?: { weight: number; layers: DreamLayerTarget[] }[];
}

/** A further preset amplified alongside the selected one, with its own share of the push. */
export interface PresetBlendEntry {
  /** Stable, so an animation track on this entry's weight stays with it. */
  id: string;
  presetId: string;
  weight: number;
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
  /**
   * Which handful of each target layer's channels to amplify. 0 amplifies the whole layer, the classic
   * effect; any other number names one reproducible handful. See `ml/channelFocus`.
   */
  focusSeed: number;
  /**
   * How many channels a nonzero `focusSeed` picks per layer. Fewer gives purer, stranger forms. A fraction
   * fades the next channel in, so a sweep of it is smooth.
   */
  focusChannels: number;
  /** The variation `morph` moves toward. 0 is the whole layer, as for `focusSeed`. */
  morphSeed: number;
  /** 0 is `focusSeed` alone, 1 is `morphSeed` alone; between, a continuous mix of the two. */
  morph: number;
  /** How activations are scored. See `DreamObjective`. */
  objective: DreamObjective;
  /** The selected preset's share of the push when others are blended in. */
  presetWeight: number;
  /** Consecutive steps the main dream takes each time round the mix. See `MixIngredient`. */
  mainTurns: number;
  /** Further dreams and styles taking turns with the main dream. Empty is plain DeepDream. */
  mix: MixIngredient[];
  /** Further presets amplified at the same time as the selected one. Empty is the preset alone. */
  presetBlend: PresetBlendEntry[];
}

/**
 * `mean` rewards every activation equally, so the ascent spreads its effort across the whole layer.
 * `squared` rewards strong activations disproportionately, so whatever the network already half-sees gets
 * pushed hardest — fewer, bolder, more fully formed shapes in place of an even carpet of them.
 */
export type DreamObjective = 'mean' | 'squared';

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
  /** Consecutive steps the main template takes each time round the mix. See `MixIngredient`. */
  templateTurns: number;
  /** Further styles and dreams taking turns with the template. Empty is plain style transfer. */
  mix: MixIngredient[];
}

/**
 * One entry in a run's mix, on either tab. A run cycles through the tab's main dream or template and then each
 * ingredient in order, each taking `turns` consecutive steps on the one shared image before handing it on, and
 * `stepsPerOctave` counts all of those steps together. Every ingredient carries all of its own settings; what it
 * takes from the tab is only the frame of the run — octaves, tiling, and what is done to the image after each
 * step.
 */
export type MixIngredient = MixStyleIngredient | MixDreamIngredient;

interface MixIngredientBase {
  /** Stable across edits and reorders, so an animation track and a loaded image stay attached to it. */
  id: string;
  turns: number;
}

/** A style image, matched with its own settings and its own optimizer state. */
export interface MixStyleIngredient extends MixIngredientBase {
  kind: 'style';
  /** The built-in template it uses, or null for the user's own file, which a saved picture cannot carry. */
  templateId: string | null;
  name: string;
  styleWeight: number;
  contentWeight: number;
  learningRate: number;
  totalVariationWeight: number;
}

/** A DeepDream preset with its own tuning, taking plain ascent steps on its turns. */
export interface MixDreamIngredient extends MixIngredientBase {
  kind: 'dream';
  presetId: string;
  presetWeight: number;
  /** Further presets blended in, exactly as `DreamParams.presetBlend`. */
  presetBlend: PresetBlendEntry[];
  name: string;
  stepSize: number;
  focusSeed: number;
  focusChannels: number;
  morphSeed: number;
  morph: number;
  objective: DreamObjective;
  lapLevels: number;
  tvWeight: number;
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
