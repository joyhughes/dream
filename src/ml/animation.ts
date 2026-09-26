import { getDeviceLimits } from './deviceLimits';
import type { DreamParams, Mode, StyleParams } from '../types';

/**
 * Sweeping a parameter across a range and keeping every frame.
 *
 * A slider tells you what one setting does at one value. Watching it move tells you what it does — where a
 * pattern stops growing and starts dissolving, which weight a style stops fighting the photo at. This runs
 * the whole pipeline once per frame with the chosen settings interpolated, and hands back a video.
 *
 * Several settings can move at once, each on its own range, which is how you sweep a diagonal through the
 * parameter space rather than one axis at a time.
 */

/** Matches the cap the tile slider uses, so an animation cannot sweep into a size that is silently clamped. */
const MAX_TILE_SIZE = getDeviceLimits().maxTileSize;

export interface AnimatableParam {
  /** Where the value lives, either a field name or `regularizers.<field>`. */
  path: string;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Rounded on the way out — octaves and step counts have no meaning between whole numbers. */
  integer?: boolean;
}

export const DREAM_ANIMATABLE: AnimatableParam[] = [
  { path: 'octaves', label: 'Octaves', min: 1, max: 7, step: 1, integer: true },
  { path: 'octaveScale', label: 'Octave scale', min: 1.1, max: 2, step: 0.05 },
  { path: 'stepsPerOctave', label: 'Steps per octave', min: 5, max: 100, step: 5, integer: true },
  { path: 'stepSize', label: 'Step size', min: 0.005, max: 0.1, step: 0.005 },
  { path: 'patternScale', label: 'Pattern scale', min: 1, max: 10, step: 0.25 },
  { path: 'tileSize', label: 'Tile size', min: 224, max: MAX_TILE_SIZE, step: 32, integer: true },
  { path: 'colorPreservation', label: 'Color preservation', min: 0, max: 1, step: 0.05 },
  { path: 'tvWeight', label: 'Smoothing (TV weight)', min: 0, max: 1, step: 0.05 },
  { path: 'lapLevels', label: 'Frequency bands', min: 1, max: 6, step: 1, integer: true },
  { path: 'regularizers.blurSigma', label: 'Blur strength', min: 0, max: 2, step: 0.1 },
  { path: 'regularizers.blurEvery', label: 'Blur every N steps', min: 0, max: 20, step: 1, integer: true },
  { path: 'regularizers.l2Decay', label: 'L2 decay', min: 0, max: 0.1, step: 0.005 },
];

export const STYLE_ANIMATABLE: AnimatableParam[] = [
  { path: 'contentWeight', label: 'Content weight', min: 1, max: 50, step: 1 },
  { path: 'styleWeight', label: 'Style weight', min: 1, max: 2000, step: 10 },
  { path: 'learningRate', label: 'Learning rate', min: 0.002, max: 0.05, step: 0.002 },
  { path: 'octaves', label: 'Octaves', min: 1, max: 7, step: 1, integer: true },
  { path: 'octaveScale', label: 'Octave scale', min: 1.1, max: 2, step: 0.05 },
  { path: 'stepsPerOctave', label: 'Steps per octave', min: 10, max: 150, step: 5, integer: true },
  { path: 'patternScale', label: 'Pattern scale', min: 1, max: 10, step: 0.25 },
  { path: 'tileSize', label: 'Tile size', min: 224, max: MAX_TILE_SIZE, step: 32, integer: true },
  { path: 'colorPreservation', label: 'Color preservation', min: 0, max: 1, step: 0.05 },
  { path: 'totalVariationWeight', label: 'Smoothing (TV weight)', min: 0, max: 5, step: 0.1 },
  { path: 'regularizers.blurSigma', label: 'Blur strength', min: 0, max: 2, step: 0.1 },
  { path: 'regularizers.blurEvery', label: 'Blur every N steps', min: 0, max: 20, step: 1, integer: true },
  { path: 'regularizers.l2Decay', label: 'L2 decay', min: 0, max: 0.1, step: 0.005 },
];

export function animatableFor(mode: Mode): AnimatableParam[] {
  return mode === 'deepdream' ? DREAM_ANIMATABLE : STYLE_ANIMATABLE;
}

/**
 * How a track gets from one end of its range to the other.
 *
 * Linear divides the range evenly, which is what a sweep wants when the numbers are evenly meaningful —
 * octaves 1 to 7, pattern scale 1 to 10. It is the wrong shape for a setting whose effect is a matter of
 * orders of magnitude: style weight 1 to 2000 spends its first frame going from 1 to 21, a change that
 * rewrites the picture, and its last going from 1980 to 2000, a change nobody can see. Logarithmic
 * multiplies by a constant ratio each frame instead, so every frame is the same proportional step and the
 * interesting end of the range gets the frames it deserves.
 */
export type TrackCurve = 'linear' | 'log';

/** One parameter's journey across the animation. */
export interface AnimationTrack {
  path: string;
  from: number;
  to: number;
  /** Absent means linear, so a track from before this existed reads as the one it was rendered with. */
  curve?: TrackCurve;
}

/**
 * Whether a track's ends admit a logarithmic sweep at all.
 *
 * A constant ratio cannot cross or start at zero — there is no number you can multiply 0 by to reach 5,
 * and no finite count of steps from 5 down to 0. Several of these parameters have ranges that start at
 * zero, so the curve is offered everywhere but only honoured where it means something.
 */
export function canSweepLogarithmically(track: AnimationTrack): boolean {
  return track.from > 0 && track.to > 0;
}

function isLogarithmic(track: AnimationTrack): boolean {
  return track.curve === 'log' && canSweepLogarithmically(track);
}

export interface AnimationSettings {
  tracks: AnimationTrack[];
  frames: number;
  fps: number;
}

export const DEFAULT_ANIMATION: AnimationSettings = {
  tracks: [],
  frames: 24,
  fps: 8,
};

type ParamsLike = DreamParams | StyleParams;

export function readParam(params: ParamsLike, path: string): number {
  const [head, tail] = path.split('.');
  const source = params as unknown as Record<string, unknown>;
  const value = tail ? (source[head] as Record<string, unknown> | undefined)?.[tail] : source[head];
  return typeof value === 'number' ? value : 0;
}

/** A copy with one value replaced. Nested paths rebuild only the branch they touch. */
export function withParam<T extends ParamsLike>(params: T, path: string, value: number): T {
  const [head, tail] = path.split('.');
  if (!tail) {
    return { ...params, [head]: value };
  }

  const branch = (params as unknown as Record<string, unknown>)[head] as Record<string, unknown>;
  return { ...params, [head]: { ...branch, [tail]: value } };
}

/**
 * Where a track sits on a given frame, along whichever curve the track was given. The first frame is
 * exactly `from` and the last exactly `to`, so a sweep covers the range it was asked for rather than
 * stopping a step short of the end.
 *
 * A single-frame animation sits at `from`: with nowhere to travel, the start is the only defensible answer.
 */
export function valueAtFrame(track: AnimationTrack, frameIndex: number, frameCount: number, integer: boolean): number {
  const progress = frameCount <= 1 ? 0 : frameIndex / (frameCount - 1);

  // Both formulas land a hair off their own endpoint at the extremes — 0.005 + (0.1 - 0.005) is not
  // exactly 0.1 — and the promise above is worth more than the arithmetic. Returning the ends verbatim
  // also spares the frame description a readout of 1999.9999999999998.
  if (progress <= 0) return integer ? Math.round(track.from) : track.from;
  if (progress >= 1) return integer ? Math.round(track.to) : track.to;

  const value = isLogarithmic(track)
    ? track.from * Math.pow(track.to / track.from, progress)
    : track.from + (track.to - track.from) * progress;
  return integer ? Math.round(value) : value;
}

/**
 * The parameters for one frame: the current settings with every animated track moved to where it should be.
 * Values are not clamped to the sliders' ranges — a typed-in value beyond a track's end is allowed
 * everywhere else in the app, and an animation is no place to start second-guessing it.
 */
export function paramsAtFrame<T extends ParamsLike>(
  base: T,
  settings: AnimationSettings,
  frameIndex: number,
  descriptors: AnimatableParam[],
): T {
  let result = base;

  for (const track of settings.tracks) {
    const descriptor = descriptors.find((entry) => entry.path === track.path);
    if (!descriptor) continue;
    result = withParam(result, track.path, valueAtFrame(track, frameIndex, settings.frames, !!descriptor.integer));
  }

  return result;
}

/**
 * Where a single frame sits, in the terms the sweep was set up in — "Pattern scale 2.5, Blur strength 0.8".
 * A frame number alone says nothing about which value is on screen, which is the reason for watching.
 */
export function describeFrame(
  params: DreamParams | StyleParams,
  settings: AnimationSettings,
  descriptors: AnimatableParam[],
): string {
  return settings.tracks
    .map((track) => {
      const label = descriptors.find((entry) => entry.path === track.path)?.label ?? track.path;
      // Three significant figures: interpolation lands on values like 2.7500000000000004, and a status
      // line wants "1.43" rather than "1.42857". Number() also undoes toPrecision's exponent notation,
      // so a large weight reads as 1250 rather than 1.25e+3.
      const value = Number(readParam(params, track.path).toPrecision(3));
      return `${label} ${value}`;
    })
    .join(', ');
}

/** A short line naming what is being swept, for the status text and the saved file. */
export function describeTracks(settings: AnimationSettings, descriptors: AnimatableParam[]): string {
  if (settings.tracks.length === 0) return 'nothing selected';

  return settings.tracks
    .map((track) => {
      const label = descriptors.find((entry) => entry.path === track.path)?.label ?? track.path;
      const curve = isLogarithmic(track) ? ' (log)' : '';
      return `${label} ${track.from} to ${track.to}${curve}`;
    })
    .join(', ');
}
