import { readPngText, writePngText } from './pngText';
import type { DreamParams, MixIngredient, Mode, PresetBlendEntry, StyleParams } from '../types';
import type { FeatureNetworkId } from './featureModel';

/**
 * The settings a picture was made with, carried inside the picture.
 *
 * Every finished image gets these written into a PNG text chunk, so a result found months later still says
 * how it was made and can put those settings back. Other software ignores the chunk, and the file stays an
 * ordinary PNG.
 */

/** The chunk keyword. Namespaced so it cannot collide with anything another tool writes. */
const KEYWORD = 'dream-parameters';

/**
 * Bumped only when old files would otherwise be read wrongly. Fields appearing and disappearing needs no
 * bump: reading validates every field against the current defaults and drops anything it does not know.
 */
const FORMAT_VERSION = 1;

export interface SavedParameters {
  mode: Mode;
  featureNetworkId: FeatureNetworkId;
  presetId: string;
  dream: DreamParams;
  style: StyleParams;
}

export function encodeParameters(parameters: SavedParameters): string {
  return JSON.stringify({ version: FORMAT_VERSION, app: 'dream-by-joyographic', ...parameters });
}

/**
 * These read a value out of a file, so nothing is trusted: every field is checked for type and range and
 * falls back to the current setting when it is missing, malformed, or out of bounds. A file with one bad
 * number still restores everything else rather than being rejected whole.
 */
function readNumber(value: unknown, fallback: number, min: number, max: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
}

function readBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function readOneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === 'string' && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

function readRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};
}

function readDreamParams(raw: unknown, fallback: DreamParams, styleDefaults: StyleEntryDefaults): DreamParams {
  const source = readRecord(raw);
  const regularizers = readRecord(source.regularizers);

  return {
    octaves: Math.round(readNumber(source.octaves, fallback.octaves, 1, 12)),
    octaveScale: readNumber(source.octaveScale, fallback.octaveScale, 1.01, 4),
    stepsPerOctave: Math.round(readNumber(source.stepsPerOctave, fallback.stepsPerOctave, 1, 2000)),
    stepSize: readNumber(source.stepSize, fallback.stepSize, 0, 10),
    tileSize: Math.round(readNumber(source.tileSize, fallback.tileSize, 32, 4096)),
    tvWeight: readNumber(source.tvWeight, fallback.tvWeight, 0, 100),
    lapLevels: Math.round(readNumber(source.lapLevels, fallback.lapLevels, 1, 12)),
    colorSpace: readOneOf(source.colorSpace, ['rgb', 'hsv'] as const, fallback.colorSpace),
    colorPreservation: readNumber(source.colorPreservation, fallback.colorPreservation, 0, 1),
    normalizeSaturation: readBoolean(source.normalizeSaturation, fallback.normalizeSaturation),
    normalizeBrightness: readBoolean(source.normalizeBrightness, fallback.normalizeBrightness),
    patternScale: readNumber(source.patternScale, fallback.patternScale, 1, 64),
    regularizers: {
      l2Decay: readNumber(regularizers.l2Decay, fallback.regularizers.l2Decay, 0, 1),
      blurSigma: readNumber(regularizers.blurSigma, fallback.regularizers.blurSigma, 0, 32),
      blurEvery: Math.round(readNumber(regularizers.blurEvery, fallback.regularizers.blurEvery, 0, 1000)),
    },
    // A picture saved before these existed was made from the whole layer with the plain mean, so a missing
    // value means that rather than whatever happens to be set now — otherwise it would not reproduce.
    focusSeed: Math.round(readNumber(source.focusSeed, 0, 0, 2 ** 31 - 1)),
    focusChannels: readNumber(source.focusChannels, fallback.focusChannels, 1, 4096),
    morphSeed: Math.round(readNumber(source.morphSeed, 0, 0, 2 ** 31 - 1)),
    morph: readNumber(source.morph, 0, 0, 1),
    objective: readOneOf(source.objective, ['mean', 'squared'] as const, 'mean'),
    presetWeight: readNumber(source.presetWeight, 1, 0, 1000),
    presetBlend: readPresetBlend(source.presetBlend),
    // A picture from before DeepDream had a mix was the main dream alone.
    mainTurns: Math.round(readNumber(source.mainTurns, 1, 0, 100)),
    mix: readMix(source.mix, styleDefaults),
  };
}

/** What a saved style entry is missing when it predates carrying these itself. */
type StyleEntryDefaults = Pick<StyleParams, 'contentWeight' | 'learningRate' | 'totalVariationWeight'>;

function readString(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value.slice(0, 200) : fallback;
}

/** Missing means the preset alone, which is what every picture saved before blending was made with. */
function readPresetBlend(raw: unknown): PresetBlendEntry[] {
  if (!Array.isArray(raw)) return [];

  const blend: PresetBlendEntry[] = [];
  for (const entry of raw.slice(0, 16)) {
    const source = readRecord(entry);
    const id = readString(source.id, '');
    if (!id || typeof source.presetId !== 'string' || blend.some((existing) => existing.id === id)) continue;
    blend.push({ id, presetId: source.presetId.slice(0, 200), weight: readNumber(source.weight, 1, 0, 1000) });
  }
  return blend;
}

/**
 * A saved mix, entry by entry, dropping any entry that is not recognisably one. A style entry's image is
 * not in the file — only which built-in it was, if it was one — so the app reloads that, and asks for an
 * image for any that was the user's own.
 */
function readMix(raw: unknown, styleDefaults: StyleEntryDefaults): MixIngredient[] {
  if (!Array.isArray(raw)) return [];

  const mix: MixIngredient[] = [];
  for (const entry of raw.slice(0, 32)) {
    const source = readRecord(entry);
    const id = readString(source.id, '');
    if (!id || mix.some((existing) => existing.id === id)) continue;
    const turns = Math.round(readNumber(source.turns, 1, 0, 100));

    if (source.kind === 'style') {
      mix.push({
        kind: 'style',
        id,
        turns,
        templateId: typeof source.templateId === 'string' ? source.templateId.slice(0, 200) : null,
        name: readString(source.name, 'Style'),
        styleWeight: readNumber(source.styleWeight, 400, 0, 1000000),
        contentWeight: readNumber(source.contentWeight, styleDefaults.contentWeight, 0, 100000),
        learningRate: readNumber(source.learningRate, styleDefaults.learningRate, 0.00001, 10),
        totalVariationWeight: readNumber(source.totalVariationWeight, styleDefaults.totalVariationWeight, 0, 1000),
      });
    } else if (source.kind === 'dream' && typeof source.presetId === 'string') {
      mix.push({
        kind: 'dream',
        id,
        turns,
        presetId: source.presetId.slice(0, 200),
        name: readString(source.name, 'Dream'),
        presetWeight: readNumber(source.presetWeight, 1, 0, 1000),
        presetBlend: readPresetBlend(source.presetBlend),
        stepSize: readNumber(source.stepSize, 0.02, 0, 10),
        focusSeed: Math.round(readNumber(source.focusSeed, 0, 0, 2 ** 31 - 1)),
        focusChannels: readNumber(source.focusChannels, 8, 1, 4096),
        morphSeed: Math.round(readNumber(source.morphSeed, 0, 0, 2 ** 31 - 1)),
        morph: readNumber(source.morph, 0, 0, 1),
        objective: readOneOf(source.objective, ['mean', 'squared'] as const, 'mean'),
        lapLevels: Math.round(readNumber(source.lapLevels, 1, 1, 12)),
        tvWeight: readNumber(source.tvWeight, 0, 0, 100),
      });
    }
  }
  return mix;
}

function readStyleParams(raw: unknown, fallback: StyleParams): StyleParams {
  const source = readRecord(raw);
  const regularizers = readRecord(source.regularizers);

  const contentWeight = readNumber(source.contentWeight, fallback.contentWeight, 0, 100000);
  const totalVariationWeight = readNumber(source.totalVariationWeight, fallback.totalVariationWeight, 0, 1000);
  const learningRate = readNumber(source.learningRate, fallback.learningRate, 0.00001, 10);

  return {
    contentWeight,
    styleWeight: readNumber(source.styleWeight, fallback.styleWeight, 0, 1000000),
    totalVariationWeight,
    learningRate,
    octaves: Math.round(readNumber(source.octaves, fallback.octaves, 1, 12)),
    octaveScale: readNumber(source.octaveScale, fallback.octaveScale, 1.01, 4),
    stepsPerOctave: Math.round(readNumber(source.stepsPerOctave, fallback.stepsPerOctave, 1, 2000)),
    tileSize: Math.round(readNumber(source.tileSize, fallback.tileSize, 32, 4096)),
    colorSpace: readOneOf(source.colorSpace, ['rgb', 'hsv'] as const, fallback.colorSpace),
    colorPreservation: readNumber(source.colorPreservation, fallback.colorPreservation, 0, 1),
    normalizeSaturation: readBoolean(source.normalizeSaturation, fallback.normalizeSaturation),
    normalizeBrightness: readBoolean(source.normalizeBrightness, fallback.normalizeBrightness),
    patternScale: readNumber(source.patternScale, fallback.patternScale, 1, 64),
    regularizers: {
      l2Decay: readNumber(regularizers.l2Decay, fallback.regularizers.l2Decay, 0, 1),
      blurSigma: readNumber(regularizers.blurSigma, fallback.regularizers.blurSigma, 0, 32),
      blurEvery: Math.round(readNumber(regularizers.blurEvery, fallback.regularizers.blurEvery, 0, 1000)),
    },
    // As with the dream focus settings: a picture from before the mix existed was the template alone.
    templateTurns: Math.round(readNumber(source.templateTurns, 1, 0, 100)),
    // A style entry saved before it carried these ran with the template's.
    mix: readMix(source.mix, { contentWeight, learningRate, totalVariationWeight }),
  };
}

/**
 * Parses stored parameters, filling anything missing or invalid from `fallback`. Null means the text is not
 * ours at all — absent, unparseable, or from a format version this build does not understand.
 */
export function decodeParameters(text: string | null, fallback: SavedParameters): SavedParameters | null {
  if (!text) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }

  const source = readRecord(parsed);
  if (source.version !== FORMAT_VERSION) return null;

  return {
    mode: readOneOf(source.mode, ['deepdream', 'style'] as const, fallback.mode),
    featureNetworkId: readOneOf(source.featureNetworkId, ['mobilenet', 'vgg19'] as const, fallback.featureNetworkId),
    // Not checked against the known presets: which exist depends on the network, which may still be
    // loading. A preset that turns out not to exist is dropped when the list is rebuilt.
    presetId: typeof source.presetId === 'string' ? source.presetId.slice(0, 200) : fallback.presetId,
    dream: readDreamParams(source.dream, fallback.dream, fallback.style),
    style: readStyleParams(source.style, fallback.style),
  };
}

/** The image with its parameters written in. Synchronous, so a save can stay inside its click. */
export function embedParameters(bytes: Uint8Array, parameters: SavedParameters): Uint8Array {
  return writePngText(bytes, KEYWORD, encodeParameters(parameters));
}

export function readEmbeddedParameters(bytes: Uint8Array, fallback: SavedParameters): SavedParameters | null {
  return decodeParameters(readPngText(bytes, KEYWORD), fallback);
}

/** Whether a file is worth opening to look for parameters. Only PNGs can carry them. */
export function couldCarryParameters(file: File): boolean {
  return file.type === 'image/png' || /\.png$/i.test(file.name);
}
