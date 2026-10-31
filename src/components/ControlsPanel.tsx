import { useState, type ReactNode } from 'react';
import { TOOLS, modifierHint, toolDefinition } from './tools';
import type { SavedParameters } from '../ml/imageMetadata';
import {
  animatableFor,
  canSweepLogarithmically,
  readParam,
  type AnimationSettings,
  type TrackCurve,
} from '../ml/animation';
import type {
  BrushSettings,
  ColorSpace,
  SelectionSettings,
  ToolId,
  DreamObjective,
  DreamParams,
  DreamPreset,
  EngineStatus,
  ImageRegularizers,
  MixDreamIngredient,
  MixIngredient,
  Mode,
  PresetBlendEntry,
  StyleParams,
} from '../types';
import { getDeviceLimits } from '../ml/deviceLimits';
import { TemplatePicker } from './TemplatePicker';
import { BUILT_IN_TEMPLATES } from '../templates/builtInTemplates';

// Matches the cap `computeTiledGradient` enforces, so the slider can't offer a size that is silently
// clamped on a phone.
const MAX_TILE_SIZE = getDeviceLimits().maxTileSize;

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3v12" />
      <path d="M7 10l5 5 5-5" />
      <path d="M5 21h14" />
    </svg>
  );
}

function SnapshotIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" />
      <circle cx="12" cy="14" r="3.3" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
      <rect x="6" y="5" width="4" height="14" rx="1" />
      <rect x="14" y="5" width="4" height="14" rx="1" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
      <path d="M7 5v14l12-7z" />
    </svg>
  );
}

function CancelIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function RecordIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
      <circle cx="12" cy="12" r="7" />
    </svg>
  );
}

function statusText(status: EngineStatus, isPaused: boolean): string {
  switch (status.phase) {
    case 'idle':
      return 'Upload images and click Generate.';
    case 'loading-model':
      return 'Loading MobileNet feature model…';
    case 'ready':
      return 'Model ready.';
    case 'running':
      return isPaused
        ? `Paused at step ${status.step + 1} / ${status.totalSteps}`
        : `Generating… step ${status.step + 1} / ${status.totalSteps}`;
    case 'done':
      return 'Done.';
    case 'error':
      return `Error: ${status.message}`;
  }
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  tooltip: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

/**
 * Binary floating point leaves artifacts like 0.012000000000000001 once a value has been through
 * arithmetic. Rounding to 12 significant figures drops those without truncating a precision the user
 * actually typed, and String() then renders it without trailing zeros.
 */
function formatReadout(value: number): string {
  if (!Number.isFinite(value)) return '';
  return String(Number(value.toPrecision(12)));
}

function Slider({ label, value, min, max, step, tooltip, onChange, disabled }: SliderProps) {
  // While the field is being edited its raw text is authoritative, so partial input a user is midway
  // through typing — "", "-", "0." — survives instead of being rewritten on every keystroke. Clearing it
  // on blur hands display back to the committed value, which also normalizes whatever they typed.
  const [draft, setDraft] = useState<string | null>(null);

  // Typed values are deliberately not clamped, so a slider can be pushed past the range its track offers.
  // The range input itself cannot represent that, so it is fed a clamped value and pins to the end of its
  // track; the readout keeps showing the real number and marks itself as out of range.
  const clamped = Math.min(max, Math.max(min, value));
  const isOutOfRange = value !== clamped;

  const commit = (text: string) => {
    const parsed = Number(text.trim());
    if (text.trim() !== '' && Number.isFinite(parsed)) onChange(parsed);
    setDraft(null);
  };

  return (
    <label className="slider-row" title={tooltip}>
      <span className="slider-label">
        {label}
        <input
          className={`slider-value${isOutOfRange ? ' slider-value--out-of-range' : ''}`}
          type="text"
          inputMode="decimal"
          aria-label={`${label} value`}
          value={draft ?? formatReadout(value)}
          disabled={disabled}
          onChange={(e) => {
            const text = e.target.value;
            setDraft(text);
            const parsed = Number(text.trim());
            if (text.trim() !== '' && Number.isFinite(parsed)) onChange(parsed);
          }}
          onFocus={(e) => e.target.select()}
          onBlur={(e) => commit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              commit(e.currentTarget.value);
              e.currentTarget.blur();
            } else if (e.key === 'Escape') {
              setDraft(null);
              e.currentTarget.blur();
            }
          }}
        />
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={clamped}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

interface ToggleProps {
  label: string;
  checked: boolean;
  tooltip: string;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

function Toggle({ label, checked, tooltip, onChange, disabled }: ToggleProps) {
  return (
    <label className="toggle-row" title={tooltip}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

interface PresetBlendEditorProps {
  presets: DreamPreset[];
  /** The preset everything else is blended with. */
  presetId: string;
  presetWeight: number;
  blend: PresetBlendEntry[];
  onChange: (change: { presetId?: string; presetWeight?: number; presetBlend?: PresetBlendEntry[] }) => void;
  /** Whether to show a picker for `presetId` itself, for places that do not already have one. */
  pickPrimary?: boolean;
  disabled: boolean;
}

function PresetSelect({
  presets,
  value,
  onChange,
  disabled,
}: {
  presets: DreamPreset[];
  value: string;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      {presets.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </select>
  );
}

/**
 * Any number of presets amplified at once, each with its share of the push.
 *
 * Shares rather than raw weights: each preset's pull is scaled to the same size before its weight applies,
 * so 0.5 and 0.5 is an even split however differently the layers behind them respond. A single preset is
 * untouched by any of this and runs exactly as it always did.
 */
function PresetBlendEditor({ presets, presetId, presetWeight, blend, onChange, pickPrimary, disabled }: PresetBlendEditorProps) {
  const setEntry = (id: string, patch: Partial<PresetBlendEntry>) =>
    onChange({ presetBlend: blend.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)) });
  // The next preset down the list that is not already in play, as the likeliest one to want next.
  const used = new Set([presetId, ...blend.map((entry) => entry.presetId)]);
  const next = presets.find((p) => !used.has(p.id)) ?? presets[0];

  return (
    <>
      {pickPrimary && (
        <label className="field-row" title="The preset this dream amplifies.">
          <span>Preset</span>
          <PresetSelect presets={presets} value={presetId} onChange={(id) => onChange({ presetId: id })} disabled={disabled} />
        </label>
      )}
      {blend.length > 0 && (
        <Slider
          label={`${presets.find((p) => p.id === presetId)?.name ?? 'Preset'} weight`}
          value={presetWeight}
          min={0}
          max={1}
          step={0.05}
          disabled={disabled}
          tooltip="This preset's share of the push, against the others blended in. Every preset's pull is scaled to the same strength before its weight applies, so equal weights are an even split."
          onChange={(v) => onChange({ presetWeight: Math.max(0, v) })}
        />
      )}
      {blend.map((entry) => (
        <div key={entry.id} className="preset-blend-entry">
          <div className="field-row">
            <span>+</span>
            <PresetSelect
              presets={presets}
              value={entry.presetId}
              onChange={(id) => setEntry(entry.id, { presetId: id })}
              disabled={disabled}
            />
            <button
              className="btn btn--secondary"
              onClick={() => onChange({ presetBlend: blend.filter((other) => other.id !== entry.id) })}
              disabled={disabled}
              title="Stops blending this preset in."
            >
              Remove
            </button>
          </div>
          <Slider
            label="Weight"
            value={entry.weight}
            min={0}
            max={1}
            step={0.05}
            disabled={disabled}
            tooltip="This preset's share of the push, against the others. 0 sits it out without removing it."
            onChange={(v) => setEntry(entry.id, { weight: Math.max(0, v) })}
          />
        </div>
      ))}
      <div className="controls-actions">
        <button
          className="btn btn--secondary"
          onClick={() =>
            next && onChange({ presetBlend: [...blend, { id: newIngredientId(), presetId: next.id, weight: 1 }] })
          }
          disabled={disabled || !next}
          title="Amplifies another preset at the same time — fine textures under abstract objects, say — with its own weight."
        >
          Blend in a preset
        </button>
      </div>
    </>
  );
}

interface PresetPanelProps {
  mode: Mode;
  presets: DreamPreset[];
  selectedPresetId: string;
  onPresetChange: (id: string) => void;
  dreamParams: DreamParams;
  onDreamParamsChange: (params: DreamParams) => void;
  isRunning: boolean;
}

/** A fresh variation that is never 0, since 0 means the whole layer rather than a variation of it. */
function rollVariation(current: number): number {
  let next = current;
  while (next === current) next = 1 + Math.floor(Math.random() * 999);
  return next;
}

type VariationFields = Pick<DreamParams, 'focusSeed' | 'focusChannels' | 'morphSeed' | 'morph'>;

/**
 * Variation, and a continuous morph from it toward a second one. Variations themselves are unrelated sets of
 * channels, so stepping the number jumps between families; the morph is the control that moves smoothly, and
 * the one to animate. Shared by the DeepDream tab and every DeepDream entry in a style mix.
 */
function VariationControls({
  values,
  onChange,
  disabled,
  compact = false,
}: {
  values: VariationFields;
  onChange: (patch: Partial<VariationFields>) => void;
  disabled: boolean;
  /** Fewer explanations, for the mix's nested entries. */
  compact?: boolean;
}) {
  const morphing = values.morphSeed !== values.focusSeed;
  const focused = values.focusSeed > 0 || (morphing && values.morphSeed > 0);

  return (
    <>
      <Slider
        label="Variation"
        value={values.focusSeed}
        min={0}
        max={999}
        step={1}
        disabled={disabled}
        tooltip={
          compact
            ? 'Which set of feature detectors this dream amplifies, as on the DeepDream tab. 0 is the whole layer.'
            : "Which of the layer's feature detectors to amplify. 0 amplifies all of them at once, which averages their ideas into the familiar blend. Any other number picks a small, fixed set of them, and each set has its own family of creatures and objects — the same number always gives the same set, so a good one can be typed back in later. Neighbouring numbers are unrelated sets: to move smoothly between two, use Morph. Strongest on the late-layer presets."
        }
        onChange={(v) => onChange({ focusSeed: Math.max(0, Math.round(v)) })}
      />
      <div className="controls-actions">
        <button
          className="btn btn--secondary"
          onClick={() => onChange({ focusSeed: rollVariation(values.focusSeed) })}
          disabled={disabled}
          title="Picks a different variation at random. Generate again to see what it draws."
        >
          Roll variation
        </button>
        <button
          className="btn btn--secondary"
          onClick={() => onChange({ focusSeed: 0 })}
          disabled={disabled || values.focusSeed === 0}
          title="Goes back to amplifying the whole layer."
        >
          Whole layer
        </button>
      </div>
      <Slider
        label="Morph toward"
        value={values.morphSeed}
        min={0}
        max={999}
        step={1}
        disabled={disabled}
        tooltip="A second variation to blend toward, with Morph setting how far. 0 is the whole layer. Set it to the same number as Variation to switch morphing off."
        onChange={(v) => onChange({ morphSeed: Math.max(0, Math.round(v)) })}
      />
      <div className="controls-actions">
        <button
          className="btn btn--secondary"
          onClick={() => onChange({ morphSeed: rollVariation(values.morphSeed) })}
          disabled={disabled}
          title="Picks a different variation to morph toward at random."
        >
          Roll target
        </button>
        <button
          className="btn btn--secondary"
          onClick={() =>
            onChange({ focusSeed: values.morphSeed, morphSeed: values.focusSeed, morph: 1 - values.morph })
          }
          disabled={disabled || !morphing}
          title="Swaps the two variations and flips Morph to match, so the image comes out the same — handy for rolling a new target from wherever you've got to."
        >
          Swap
        </button>
      </div>
      {morphing && (
        <Slider
          label="Morph"
          value={values.morph}
          min={0}
          max={1}
          step={0.01}
          disabled={disabled}
          tooltip="How far from Variation toward Morph toward: 0 is the first alone, 1 is the second alone, and every point between is a continuous mix of the two. Each side is balanced to the same strength, so 0.5 really is half and half. Animate this for one family turning into another."
          onChange={(v) => onChange({ morph: v })}
        />
      )}
      {focused && (
        <Slider
          label="Focus channels"
          value={values.focusChannels}
          min={1}
          max={64}
          step={compact ? 1 : 0.25}
          disabled={disabled}
          tooltip="How many feature detectors a variation amplifies. One or two give the purest and strangest forms; more mix them toward the whole-layer look. Raising this keeps the detectors already chosen and adds to them, so the variation fills out rather than changing — and a fraction fades the next one in, so this moves smoothly too."
          onChange={(v) => onChange({ focusChannels: Math.max(1, v) })}
        />
      )}
    </>
  );
}

export function PresetPanel({
  mode,
  presets,
  selectedPresetId,
  onPresetChange,
  dreamParams,
  onDreamParamsChange,
  isRunning,
}: PresetPanelProps) {
  if (mode !== 'deepdream') return null;

  return (
    <div className="preset-panel">
      <label
        className="field-row"
        title="Chooses which layer(s) of the MobileNet network to amplify. Early layers pick out fine edges, grain, and texture; later layers pick out increasingly abstract, object-like forms."
      >
        <span>Preset</span>
        <select value={selectedPresetId} onChange={(e) => onPresetChange(e.target.value)} disabled={isRunning}>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <p className="field-hint">{presets.find((p) => p.id === selectedPresetId)?.description}</p>
      <PresetBlendEditor
        presets={presets}
        presetId={selectedPresetId}
        presetWeight={dreamParams.presetWeight}
        blend={dreamParams.presetBlend}
        onChange={(change) => {
          if (change.presetId) onPresetChange(change.presetId);
          onDreamParamsChange({ ...dreamParams, ...change });
        }}
        disabled={isRunning}
      />

      <VariationControls
        values={dreamParams}
        onChange={(patch) => onDreamParamsChange({ ...dreamParams, ...patch })}
        disabled={isRunning}
      />
      <label
        className="field-row"
        title="How the network's response is scored. Average rewards every response equally and spreads detail evenly. Sharpened rewards the strongest responses most, so whatever the network half-sees is pushed hardest: fewer, bolder, more fully formed shapes."
      >
        <span>Objective</span>
        <select
          value={dreamParams.objective}
          onChange={(e) => onDreamParamsChange({ ...dreamParams, objective: e.target.value as DreamObjective })}
          disabled={isRunning}
        >
          <option value="mean">Average</option>
          <option value="squared">Sharpened</option>
        </select>
      </label>
    </div>
  );
}

interface VideoOptionsPanelProps {
  fps: number;
  onFpsChange: (fps: number) => void;
  isRunning: boolean;
}

export function VideoOptionsPanel({ fps, onFpsChange, isRunning }: VideoOptionsPanelProps) {
  return (
    <div className="video-options-panel">
      <Slider
        label="Video frame rate"
        value={fps}
        min={1}
        max={24}
        step={1}
        disabled={isRunning}
        tooltip="How many frames per second to sample from the input video and re-encode the output at. Lower values process far fewer frames (much faster, choppier); higher values are smoother but take proportionally longer since every sampled frame runs the full pipeline below."
        onChange={onFpsChange}
      />
      <p className="field-hint">
        Each sampled frame runs the full DeepDream / Style Transfer pipeline, so processing a video takes roughly
        (frame count) × (time for one image).
      </p>
    </div>
  );
}

interface AnimationPanelProps {
  mode: Mode;
  settings: AnimationSettings;
  onSettingsChange: (settings: AnimationSettings) => void;
  dreamParams: DreamParams;
  styleParams: StyleParams;
  /** Names the blend's and the mix's presets in track labels. */
  presets: DreamPreset[];
  isRunning: boolean;
  /** False when there is nothing to render from — no photo, a video, or no style template. */
  canRender: boolean;
  summary: string;
  onRender: () => void;
}

/**
 * Picking what to sweep and over what range.
 *
 * A parameter starts at its current value on both ends, so switching one on and pressing render changes
 * nothing until a range is actually set — the settings on screen are the obvious place to begin, and the
 * alternative of guessing a range would quietly discard what the user had already dialled in.
 */
export function AnimationPanel({
  mode,
  settings,
  onSettingsChange,
  dreamParams,
  styleParams,
  presets,
  isRunning,
  canRender,
  summary,
  onRender,
}: AnimationPanelProps) {
  const descriptors = animatableFor(mode, { dreamParams, styleParams, presets });
  const params = mode === 'deepdream' ? dreamParams : styleParams;

  const trackFor = (path: string) => settings.tracks.find((track) => track.path === path);

  const toggle = (path: string, on: boolean) => {
    if (!on) {
      onSettingsChange({ ...settings, tracks: settings.tracks.filter((track) => track.path !== path) });
      return;
    }
    const current = readParam(params, path);
    onSettingsChange({ ...settings, tracks: [...settings.tracks, { path, from: current, to: current }] });
  };

  const setEnd = (path: string, end: 'from' | 'to', value: number) => {
    onSettingsChange({
      ...settings,
      tracks: settings.tracks.map((track) => (track.path === path ? { ...track, [end]: value } : track)),
    });
  };

  const setCurve = (path: string, curve: TrackCurve) => {
    onSettingsChange({
      ...settings,
      tracks: settings.tracks.map((track) => (track.path === path ? { ...track, curve } : track)),
    });
  };

  return (
    <div className="slider-panel">
      <p className="field-hint">
        Sweeps the chosen settings from one value to another, running the whole pipeline once per frame and
        saving the result as a video. Several can move at once, each over its own range.
      </p>

      <div className="animation-tracks">
        {descriptors.map((descriptor) => {
          const track = trackFor(descriptor.path);
          return (
            <div key={descriptor.path} className="animation-track">
              <Toggle
                label={descriptor.label}
                checked={!!track}
                disabled={isRunning}
                tooltip={`Sweep ${descriptor.label} across the animation. Its slider range is ${descriptor.min} to ${descriptor.max}, but the ends below are not limited to it.`}
                onChange={(on) => toggle(descriptor.path, on)}
              />
              {track && (
                <div className="animation-range">
                  <label>
                    <span>from</span>
                    <input
                      type="text"
                      inputMode="decimal"
                      className="slider-value"
                      value={formatReadout(track.from)}
                      disabled={isRunning}
                      aria-label={`${descriptor.label} start value`}
                      onChange={(e) => {
                        const parsed = Number(e.target.value.trim());
                        if (e.target.value.trim() !== '' && Number.isFinite(parsed)) {
                          setEnd(descriptor.path, 'from', parsed);
                        }
                      }}
                    />
                  </label>
                  <label>
                    <span>to</span>
                    <input
                      type="text"
                      inputMode="decimal"
                      className="slider-value"
                      value={formatReadout(track.to)}
                      disabled={isRunning}
                      aria-label={`${descriptor.label} end value`}
                      onChange={(e) => {
                        const parsed = Number(e.target.value.trim());
                        if (e.target.value.trim() !== '' && Number.isFinite(parsed)) {
                          setEnd(descriptor.path, 'to', parsed);
                        }
                      }}
                    />
                  </label>
                  <label
                    className={`animation-curve${canSweepLogarithmically(track) ? '' : ' animation-curve--unavailable'}`}
                    title={
                      canSweepLogarithmically(track)
                        ? `Step ${descriptor.label} by a constant ratio rather than a constant amount, so each frame is the same proportional change. Worth it when the range covers orders of magnitude — the low end gets the frames where the difference is visible.`
                        : 'A logarithmic sweep multiplies by a fixed ratio each frame, which cannot start at or cross zero. Give both ends a value above zero to use it.'
                    }
                  >
                    <input
                      type="checkbox"
                      checked={track.curve === 'log' && canSweepLogarithmically(track)}
                      disabled={isRunning || !canSweepLogarithmically(track)}
                      aria-label={`Sweep ${descriptor.label} logarithmically`}
                      onChange={(e) => setCurve(descriptor.path, e.target.checked ? 'log' : 'linear')}
                    />
                    <span>log</span>
                  </label>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Toggle
        label="Frame feedback"
        checked={settings.feedback}
        disabled={isRunning}
        tooltip="Starts each frame from the one before instead of from the photo. A full run turns even a tiny change of setting into a different arrangement of forms, so separate frames flicker; with feedback the forms already drawn stay put and drift as the settings move, the way DeepDream videos are made. The first frame is a full run; the rest are a short pass each."
        onChange={(on) => onSettingsChange({ ...settings, feedback: on })}
      />
      {settings.feedback && (
        <>
          <Slider
            label="Steps per frame"
            value={settings.feedbackSteps}
            min={1}
            max={40}
            step={1}
            disabled={isRunning}
            tooltip="How many steps each frame after the first gets, at one octave. Fewer keeps frames closer to each other and renders faster; more lets each frame follow the settings more closely, and lets the dream grow deeper over the sweep."
            onChange={(v) => onSettingsChange({ ...settings, feedbackSteps: Math.max(1, Math.round(v)) })}
          />
          <Slider
            label="Pull toward photo"
            value={settings.feedbackAnchor}
            min={0}
            max={0.5}
            step={0.01}
            disabled={isRunning}
            tooltip="How much of the original photo is mixed back into each frame before it is dreamed on. 0 lets the dream keep compounding, which over a long sweep can wash out into pure texture; a little holds the photo's shapes underneath."
            onChange={(v) => onSettingsChange({ ...settings, feedbackAnchor: v })}
          />
        </>
      )}
      <Slider
        label="Frames"
        value={settings.frames}
        min={2}
        max={120}
        step={1}
        disabled={isRunning}
        tooltip="How many frames the sweep is divided into. Every frame is a full run, so this multiplies the time a single Generate takes. Frames are held compressed while the sweep runs, so the count asked for is the count rendered; if a sweep ever did outgrow the memory it is allowed, it would step through its range more coarsely rather than stop partway along it."
        onChange={(v) => onSettingsChange({ ...settings, frames: Math.max(2, Math.round(v)) })}
      />
      <Slider
        label="Playback frame rate"
        value={settings.fps}
        min={1}
        max={30}
        step={1}
        disabled={isRunning}
        tooltip="How fast the finished video plays. It costs nothing to change — the frames are already rendered — so it only decides whether the sweep reads as a slow reveal or a quick flick."
        onChange={(v) => onSettingsChange({ ...settings, fps: Math.max(1, Math.round(v)) })}
      />

      <div className="controls-actions">
        <button
          className="btn btn--primary"
          onClick={onRender}
          disabled={isRunning || !canRender || settings.tracks.length === 0}
          title="Renders every frame of the sweep and downloads the result as a video."
        >
          Render animation
        </button>
      </div>
      <p className="field-hint">
        {settings.tracks.length === 0
          ? 'Choose at least one setting to sweep.'
          : `Sweeping ${summary} over ${settings.frames} frames.`}
      </p>
    </div>
  );
}

interface ParametersFromImageProps {
  parameters: SavedParameters | null;
  applied: boolean;
  disabled: boolean;
  onApply: () => void;
  onDismiss: () => void;
}

/**
 * Offered when the picked image turns out to have been made here and still carries its settings. Applying
 * is a choice rather than something that happens on load: rewriting every slider because of what a file
 * contained would be a surprise, and the picture may have been opened just to look at.
 */
export function ParametersFromImage({ parameters, applied, disabled, onApply, onDismiss }: ParametersFromImageProps) {
  if (!parameters) return null;

  const summary =
    parameters.mode === 'deepdream'
      ? `DeepDream, ${parameters.dream.octaves} octaves, pattern scale ${parameters.dream.patternScale}`
      : `Style transfer, ${parameters.style.octaves} octaves, pattern scale ${parameters.style.patternScale}`;

  return (
    <div className="found-parameters">
      <span className="found-parameters-title">This image carries the settings it was made with</span>
      <span className="found-parameters-summary">{summary}</span>
      <div className="controls-actions">
        <button
          className="btn btn--primary"
          onClick={onApply}
          disabled={disabled || applied}
          title="Restores the mode, network, preset and every slider to what produced this image."
        >
          {applied ? 'Settings restored' : 'Use these settings'}
        </button>
        <button className="btn btn--secondary" onClick={onDismiss} title="Keep the settings you have now.">
          Dismiss
        </button>
      </div>
    </div>
  );
}

interface BrushPanelProps {
  tool: ToolId;
  onToolChange: (tool: ToolId) => void;
  settings: BrushSettings;
  onSettingsChange: (settings: BrushSettings) => void;
  /** False when there is nothing to work on — no photo yet, a video, or a run in flight. */
  available: boolean;
  isPainting: boolean;
  isRunning: boolean;
}

export function BrushPanel({
  tool,
  onToolChange,
  settings,
  onSettingsChange,
  available,
  isPainting,
  isRunning,
}: BrushPanelProps) {
  const set = (next: Partial<BrushSettings>) => onSettingsChange({ ...settings, ...next });
  const active = toolDefinition(tool);

  return (
    <div className="slider-panel">
      <div
        className="tool-row"
        role="radiogroup"
        aria-label="Tool"
        title="What the pointer does on the image. Selections made with the wand, bucket, lasso or selection brush confine everything else — Generate, Paint and Apply all stay inside them."
      >
        {TOOLS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            role="radio"
            aria-checked={tool === entry.id}
            aria-label={entry.label}
            title={`${entry.label} — ${entry.hint}`}
            className={`tool-button${tool === entry.id ? ' tool-button--active' : ''}`}
            disabled={isRunning}
            onClick={() => onToolChange(entry.id)}
          >
            {entry.icon}
          </button>
        ))}
      </div>
      {tool !== 'none' && <p className="field-hint">{active.hint}</p>}
      {modifierHint(tool) && <p className="field-hint field-hint--keys">{modifierHint(tool)}</p>}
      {tool !== 'none' && !available && (
        <p className="field-hint field-hint--warn">
          Load a photo first — the brush needs a still image to paint on, and cannot work on a video.
        </p>
      )}
      {tool === 'paint' && (
        <>
          <Slider
            label="Brush size"
            value={settings.radius}
            min={8}
            max={400}
            step={4}
            disabled={isRunning}
            tooltip="Radius of the dab, in pixels of the image being worked on. Bigger dabs cover ground faster but take proportionally longer to compute, so a very large brush stops feeling responsive — the whole point of a small one is that it can finish inside a frame."
            onChange={(v) => set({ radius: v })}
          />
          <Slider
            label="Feathering"
            value={settings.feather}
            min={0}
            max={1}
            step={0.05}
            disabled={isRunning}
            tooltip="How much of the radius is spent fading out. 0 gives a hard edge that shows the outline of every dab; 1 fades from the center out, so the effect only ever tints and blends invisibly into the image behind. Around 0.5 keeps a definite mark while hiding the seam."
            onChange={(v) => set({ feather: v })}
          />
          <Slider
            label="Steps per dab"
            value={settings.stepsPerDab}
            min={1}
            max={20}
            step={1}
            disabled={isRunning}
            tooltip="How many iterations run each time the brush is applied. More builds the effect up faster while you hold, but each dab takes longer, so the brush responds more coarsely to being moved. Low values give fine control over how far it goes; high values are quicker to reach a strong effect."
            onChange={(v) => set({ stepsPerDab: v })}
          />
          <p className="field-hint">
            {isPainting ? 'Painting…' : 'Hold in place to keep iterating; release to stop. Download saves the painted image.'}
          </p>
        </>
      )}
    </div>
  );
}

interface SelectionPanelProps {
  tool: ToolId;
  settings: SelectionSettings;
  onSettingsChange: (settings: SelectionSettings) => void;
  /** Share of the image currently selected, for the summary line. */
  selectedFraction: number;
  isBusy: boolean;
  isRunning: boolean;
  onApply: () => void;
  onInvert: () => void;
  onClear: () => void;
}

export function SelectionPanel({
  tool,
  settings,
  onSettingsChange,
  selectedFraction,
  isBusy,
  isRunning,
  onApply,
  onInvert,
  onClear,
}: SelectionPanelProps) {
  if (tool === 'none') return null;

  const set = (next: Partial<SelectionSettings>) => onSettingsChange({ ...settings, ...next });
  const hasSelection = selectedFraction > 0;

  return (
    <div className="slider-panel selection-panel">
      {(tool === 'wand' || tool === 'bucket') && (
        <>
          <Slider
            label="Tolerance"
            value={settings.tolerance}
            min={0}
            max={1}
            step={0.01}
            disabled={isRunning}
            tooltip="How different a pixel may be from the one you click and still be taken. 0 takes only an exact color match; higher values reach across shading and gradients, and near 1 the whole image goes in one click. Measured as distance in RGB, scaled so the number means the same thing on any image."
            onChange={(v) => set({ tolerance: v })}
          />
          <Toggle
            label="Contiguous"
            checked={settings.contiguous}
            disabled={isRunning}
            tooltip="On, the selection spreads outward from the pixel you clicked and stops at anything too different, so it takes one connected region. Off, it takes every pixel in the image of a similar color no matter where it is — useful for something like every patch of sky through a row of trees."
            onChange={(v) => set({ contiguous: v })}
          />
        </>
      )}
      {tool === 'select-brush' && (
        <Slider
          label="Selection brush size"
          value={settings.brushRadius}
          min={4}
          max={300}
          step={4}
          disabled={isRunning}
          tooltip="Radius of the selection brush, in pixels of the image being worked on."
          onChange={(v) => set({ brushRadius: v })}
        />
      )}
      <Slider
        label="Selection feather"
        value={settings.feather}
        min={0}
        max={80}
        step={1}
        disabled={isRunning}
        tooltip="How far the selection's edge fades, in pixels either side of it. 0 applies the effect right up to a hard boundary, which shows as a visible cut; a wider fade blends what is applied smoothly into the image around it. The wash on the image shows the softened edge, so what you see is what will be applied."
        onChange={(v) => set({ feather: v })}
      />
      <div className="controls-actions">
        <button
          className="btn btn--primary"
          onClick={onApply}
          disabled={!hasSelection || isBusy || isRunning}
          title="Runs the current mode over the selected area only, blended in through the feathered edge."
        >
          Apply to selection
        </button>
        <button
          className="btn btn--secondary"
          onClick={onInvert}
          disabled={isBusy || isRunning}
          title="Selects everything that is not selected, and vice versa."
        >
          Invert
        </button>
        <button
          className="btn btn--secondary"
          onClick={onClear}
          disabled={!hasSelection || isBusy || isRunning}
          title="Drops the selection, so tools act on the whole image again."
        >
          Clear
        </button>
      </div>
      <p className="field-hint">
        {isBusy
          ? 'Working…'
          : hasSelection
            ? `${(selectedFraction * 100).toFixed(1)}% of the image selected. Generate, Paint and Apply all stay inside it.`
            : 'Nothing selected — tools act on the whole image.'}
      </p>
    </div>
  );
}

interface SliderPanelProps {
  mode: Mode;
  dreamParams: DreamParams;
  onDreamParamsChange: (params: DreamParams) => void;
  styleParams: StyleParams;
  onStyleParamsChange: (params: StyleParams) => void;
  isRunning: boolean;
}

export function SliderPanel({
  mode,
  dreamParams,
  onDreamParamsChange,
  styleParams,
  onStyleParamsChange,
  isRunning,
}: SliderPanelProps) {
  return (
    <div className="slider-panel">
      {mode === 'deepdream' ? (
        <>
          <Slider
            label="Octaves"
            value={dreamParams.octaves}
            min={1}
            max={7}
            step={1}
            disabled={isRunning}
            tooltip="How many times the image is progressively scaled up during processing. More octaves build the pattern at multiple sizes at once, giving richer, more elaborate detail — but each extra octave takes longer to run."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, octaves: v })}
          />
          <Slider
            label="Octave scale"
            value={dreamParams.octaveScale}
            min={1.1}
            max={2}
            step={0.05}
            disabled={isRunning}
            tooltip="How much larger each successive octave is than the one before it. A higher scale makes bigger jumps in pattern size between octaves, spreading detail across more dramatically different scales."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, octaveScale: v })}
          />
          <Slider
            label="Steps per octave"
            value={dreamParams.stepsPerOctave}
            min={5}
            max={100}
            step={5}
            disabled={isRunning}
            tooltip="How many gradient-ascent steps run at each octave. More steps intensify and refine the effect at each scale, but increase processing time proportionally."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, stepsPerOctave: v })}
          />
          <Slider
            label="Step size"
            value={dreamParams.stepSize}
            min={0.005}
            max={0.1}
            step={0.005}
            disabled={isRunning}
            tooltip="How strongly each step nudges the image toward the target pattern. Higher values build the effect faster and more dramatically, but can quickly turn noisy or overcooked."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, stepSize: v })}
          />
          <Slider
            label="Pattern scale"
            value={dreamParams.patternScale}
            min={1}
            max={10}
            step={0.25}
            disabled={isRunning}
            tooltip="How large the drawn patterns come out. The network draws its shapes at one fixed size, so the only way to make them bigger in the finished image is to give it fewer pixels to draw on — that is what this does, working at a coarser resolution and enlarging the result. 1 is the finest detail the network can produce; higher values give bigger, softer motifs and run faster. There is a ceiling: once the working image is smaller than the network's own view, raising this further stops making patterns bigger. Octaves spread detail across a range of scales; this sets where that range sits, and it is the brush's scale control too."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, patternScale: v })}
          />
          <Slider
            label="Tile size"
            value={dreamParams.tileSize}
            min={224}
            max={MAX_TILE_SIZE}
            step={32}
            disabled={isRunning}
            tooltip="The size of the tiles the image is split into while processing. Smaller tiles capture more native detail on large images but take longer per step — 224 is the network's native resolution and gives maximum fidelity."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, tileSize: v })}
          />
        </>
      ) : (
        <>
          <Slider
            label="Content weight"
            value={styleParams.contentWeight}
            min={1}
            max={50}
            step={1}
            disabled={isRunning}
            tooltip="How strongly the result is pulled to preserve the original photo's content and layout. Higher values keep the underlying scene more recognizable."
            onChange={(v) => onStyleParamsChange({ ...styleParams, contentWeight: v })}
          />
          <Slider
            label="Style weight"
            value={styleParams.styleWeight}
            min={1}
            max={2000}
            step={10}
            disabled={isRunning}
            tooltip="How strongly the result is pulled to match the template's colors, textures, and patterns. Higher values make the style more dominant over the original content."
            onChange={(v) => onStyleParamsChange({ ...styleParams, styleWeight: v })}
          />
          <Slider
            label="Learning rate"
            value={styleParams.learningRate}
            min={0.002}
            max={0.05}
            step={0.002}
            disabled={isRunning}
            tooltip="How large a step the optimizer takes on each iteration. Higher values converge faster but can overshoot, producing unstable or noisy results."
            onChange={(v) => onStyleParamsChange({ ...styleParams, learningRate: v })}
          />
          <Slider
            label="Octaves"
            value={styleParams.octaves}
            min={1}
            max={7}
            step={1}
            disabled={isRunning}
            tooltip="How many times the image is progressively scaled up during processing, letting style patterns form at multiple sizes. More octaves add detail but take longer to run."
            onChange={(v) => onStyleParamsChange({ ...styleParams, octaves: v })}
          />
          <Slider
            label="Octave scale"
            value={styleParams.octaveScale}
            min={1.1}
            max={2}
            step={0.05}
            disabled={isRunning}
            tooltip="How much larger each successive octave is than the one before it. A higher scale makes bigger jumps in pattern size between octaves."
            onChange={(v) => onStyleParamsChange({ ...styleParams, octaveScale: v })}
          />
          <Slider
            label="Steps per octave"
            value={styleParams.stepsPerOctave}
            min={10}
            max={150}
            step={5}
            disabled={isRunning}
            tooltip="How many optimization steps run at each octave. More steps refine the result further, but increase processing time proportionally."
            onChange={(v) => onStyleParamsChange({ ...styleParams, stepsPerOctave: v })}
          />
          <Slider
            label="Pattern scale"
            value={styleParams.patternScale}
            min={1}
            max={10}
            step={0.25}
            disabled={isRunning}
            tooltip="How large the template's motifs come out. The optimizer draws through a fixed-size view of the network, so the only lever on how big a motif lands is how many pixels it is given to work on — this runs at a coarser resolution and enlarges the result. 1 is the finest the network can draw; higher values give bigger, softer motifs and run faster. There is a ceiling: once the working image is smaller than the network's own view, raising this further stops making motifs bigger. This is the brush's scale control too."
            onChange={(v) => onStyleParamsChange({ ...styleParams, patternScale: v })}
          />
          <Slider
            label="Tile size"
            value={styleParams.tileSize}
            min={224}
            max={MAX_TILE_SIZE}
            step={32}
            disabled={isRunning}
            tooltip="The size of the tiles the image is split into while processing. Smaller tiles capture more native detail on large images but take longer per step — 224 is the network's native resolution and gives maximum fidelity."
            onChange={(v) => onStyleParamsChange({ ...styleParams, tileSize: v })}
          />
        </>
      )}
      <p className="field-hint">
        Smaller tiles capture more native detail on large images but take longer per step — 224 is the
        network&apos;s native resolution and gives maximum fidelity.
      </p>
    </div>
  );
}

interface RegularizerPanelProps {
  mode: Mode;
  dreamParams: DreamParams;
  onDreamParamsChange: (params: DreamParams) => void;
  styleParams: StyleParams;
  onStyleParamsChange: (params: StyleParams) => void;
  isRunning: boolean;
}

/**
 * Controls for the priors that keep gradient ascent producing something image-like. Every one defaults to
 * off, so this panel starts as a no-op and each slider is a deliberate opt-in.
 */
export function RegularizerPanel({
  mode,
  dreamParams,
  onDreamParamsChange,
  styleParams,
  onStyleParamsChange,
  isRunning,
}: RegularizerPanelProps) {
  const isDream = mode === 'deepdream';
  const regularizers = isDream ? dreamParams.regularizers : styleParams.regularizers;
  const colorSpace = isDream ? dreamParams.colorSpace : styleParams.colorSpace;

  const setRegularizers = (next: Partial<ImageRegularizers>) => {
    const merged = { ...regularizers, ...next };
    if (isDream) {
      onDreamParamsChange({ ...dreamParams, regularizers: merged });
    } else {
      onStyleParamsChange({ ...styleParams, regularizers: merged });
    }
  };

  const colorPreservation = isDream ? dreamParams.colorPreservation : styleParams.colorPreservation;

  const setColorPreservation = (next: number) => {
    if (isDream) {
      onDreamParamsChange({ ...dreamParams, colorPreservation: next });
    } else {
      onStyleParamsChange({ ...styleParams, colorPreservation: next });
    }
  };

  const normalizesSaturation = isDream ? dreamParams.normalizeSaturation : styleParams.normalizeSaturation;
  const normalizesBrightness = isDream ? dreamParams.normalizeBrightness : styleParams.normalizeBrightness;

  const setNormalizeSaturation = (next: boolean) => {
    if (isDream) {
      onDreamParamsChange({ ...dreamParams, normalizeSaturation: next });
    } else {
      onStyleParamsChange({ ...styleParams, normalizeSaturation: next });
    }
  };

  const setNormalizeBrightness = (next: boolean) => {
    if (isDream) {
      onDreamParamsChange({ ...dreamParams, normalizeBrightness: next });
    } else {
      onStyleParamsChange({ ...styleParams, normalizeBrightness: next });
    }
  };

  const setColorSpace = (next: ColorSpace) => {
    if (isDream) {
      onDreamParamsChange({ ...dreamParams, colorSpace: next });
    } else {
      onStyleParamsChange({ ...styleParams, colorSpace: next });
    }
  };

  return (
    <div className="slider-panel">
      <label
        className="field-row"
        title="Which coordinates the optimizer steps in. RGB moves the three color channels independently. HSV moves hue, saturation and value instead, so one step is a rotation around the color wheel, a change in vividness, and a change in brightness — the same size of step reaches very different images. Note that HSV does not keep colors truer: raising value and lowering saturation both brighten a pixel, so an ascent that likes brightness drains saturation. Use Color preservation below for that. The network is always shown RGB either way."
      >
        <span>Color space</span>
        <select
          value={colorSpace}
          onChange={(e) => setColorSpace(e.target.value as ColorSpace)}
          disabled={isRunning}
        >
          <option value="rgb">RGB (default)</option>
          <option value="hsv">HSV (hue / saturation / value)</option>
        </select>
      </label>
      <Slider
        label="Color preservation"
        value={colorPreservation}
        min={0}
        max={1}
        step={0.05}
        disabled={isRunning}
        tooltip="Restores the original image's hue and saturation after every step, keeping only the brightness the run drew. Because it applies every step its effect compounds, so the useful range is the low end: 0.1 cuts color drift to about a fifth without touching how much structure appears, 0.2 to a twelfth, and 1 locks color to the original exactly so the effect shows up purely as light and shade. Works in either color space, so an RGB run can keep its colors too. Hue is blended the short way around the wheel, so reds either side of the wrap stay red."
        onChange={setColorPreservation}
      />
      <Toggle
        label="Hold average saturation"
        checked={normalizesSaturation}
        disabled={isRunning}
        tooltip="Rescales saturation after every step so the image's average matches the one it started with. Unlike Color preservation, which pins each pixel's color where it was, this fixes only the average — the run stays free to make one area more vivid and another less, and only the drift of the whole image is taken away. That drift is what shows up as washing out over a long run, and as saturation wandering between the frames of a video, where every frame is a separate run that would otherwise land somewhere slightly different."
        onChange={setNormalizeSaturation}
      />
      <Toggle
        label="Hold average brightness"
        checked={normalizesBrightness}
        disabled={isRunning}
        tooltip="Rescales brightness after every step so the image's average matches the one it started with. Maximizing activations likes brightness and climbs toward it given the chance, so a long run drifts lighter and a video's frames drift apart. Independent of the saturation switch: this scales all three channels together, which leaves hue and saturation exactly as they were, so both can be on at once without fighting."
        onChange={setNormalizeBrightness}
      />
      {mode === 'style' && (
        <Slider
          label="Smoothing (TV weight)"
          value={styleParams.totalVariationWeight}
          min={0}
          max={5}
          step={0.1}
          disabled={isRunning}
          tooltip="Total variation: penalizes the difference between neighboring pixels, the classic prior against high-frequency noise (Mahendran & Vedaldi 2015). Unlike the others here it is a term in the loss the optimizer minimizes, not something applied afterward — and it is the one regularizer that defaults to on, since style transfer speckles badly without it."
          onChange={(v) => onStyleParamsChange({ ...styleParams, totalVariationWeight: v })}
        />
      )}
      {mode === 'deepdream' && (
        <>
          <Slider
            label="Smoothing (TV weight)"
            value={dreamParams.tvWeight}
            min={0}
            max={1}
            step={0.05}
            disabled={isRunning}
            tooltip="Total variation: penalizes the difference between neighboring pixels, which is the classic prior against the high-frequency noise that unconstrained gradient ascent produces (Mahendran & Vedaldi 2015). Read it as the share of each step spent smoothing rather than amplifying — 0 is off, and past about 0.5 smoothing wins and detail dissolves."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, tvWeight: v })}
          />
          <Slider
            label="Frequency bands"
            value={dreamParams.lapLevels}
            min={1}
            max={6}
            step={1}
            disabled={isRunning}
            tooltip="Laplacian pyramid gradient normalization (Mordvintsev et al. 2015, the 'lapnorm' technique). A raw gradient is dominated by its lowest frequencies, so the pattern grows in broad smears; splitting it into this many frequency bands and normalizing each separately gives fine detail an equal say. 1 is a plain whole-image normalization — the original behavior. 4 is the classic setting."
            onChange={(v) => onDreamParamsChange({ ...dreamParams, lapLevels: v })}
          />
        </>
      )}
      <Slider
        label="Blur strength (σ)"
        value={regularizers.blurSigma}
        min={0}
        max={2}
        step={0.1}
        disabled={isRunning}
        tooltip="Standard deviation of a Gaussian blur applied periodically to the image itself (Yosinski et al. 2015). Blurring between steps suppresses the high-frequency structure the network keeps reaching for, and the pattern that survives repeated blurring is the one that is genuinely there at a larger scale. 0 is off; it only takes effect when the interval below is set."
        onChange={(v) => setRegularizers({ blurSigma: v })}
      />
      <Slider
        label="Blur every N steps"
        value={regularizers.blurEvery}
        min={0}
        max={20}
        step={1}
        disabled={isRunning}
        tooltip="How often the blur above runs. 0 disables it entirely; 1 blurs after every step, which is heavy-handed; 4 to 8 is the usual range — often enough to hold noise down, rarely enough that detail still accumulates in between."
        onChange={(v) => setRegularizers({ blurEvery: v })}
      />
      <Slider
        label="L2 decay"
        value={regularizers.l2Decay}
        min={0}
        max={0.1}
        step={0.005}
        disabled={isRunning}
        tooltip="Shrinks every pixel a little toward the image's average color after each step (Simonyan et al. 2014; Yosinski et al. 2015). Ascent pushes pixels toward the extremes and they stick there once clipped; a small decay bleeds that off continuously. Costs contrast in exchange, so keep it low — 0.01 is already noticeable."
        onChange={(v) => setRegularizers({ l2Decay: v })}
      />
      <p className="field-hint">
        The color space chooses what a step means; the rest are priors on what a natural image looks like —
        without one, the true optimum of gradient ascent is high-frequency noise rather than a picture. All
        start off, except style transfer&apos;s smoothing.
      </p>
    </div>
  );
}

interface ActionsBarProps {
  status: EngineStatus;
  isPaused: boolean;
  isRunning: boolean;
  canGenerate: boolean;
  hasResult: boolean;
  recordMovie: boolean;
  isRecordingMovie: boolean;
  recordingSupported: boolean;
  recordUnavailableForVideo: boolean;
  frameProgressLabel?: string | null;
  /** The DeepDream / Style Transfer picker, shown above Generate — it decides what Generate will run. */
  modeTabs?: ReactNode;
  onGenerate: () => void;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  onDownload: () => void;
  onSaveCurrentStep: () => void;
  onToggleRecordMovie: () => void;
}

export function ActionsBar({
  status,
  isPaused,
  isRunning,
  canGenerate,
  hasResult,
  recordMovie,
  isRecordingMovie,
  recordingSupported,
  recordUnavailableForVideo,
  frameProgressLabel,
  modeTabs,
  onGenerate,
  onCancel,
  onPause,
  onResume,
  onDownload,
  onSaveCurrentStep,
  onToggleRecordMovie,
}: ActionsBarProps) {
  const progress = status.phase === 'running' ? (status.step + 1) / status.totalSteps : status.phase === 'done' ? 1 : 0;

  return (
    <div className="actions-panel">
      {modeTabs}
      <div className="controls-actions">
        {!isRunning && (
          <button
            className="btn btn--primary"
            onClick={onGenerate}
            disabled={!canGenerate}
            title="Runs DeepDream or Style Transfer on the uploaded image(s) using the current preset and slider settings. With a selection active it runs inside the selection only, blended in through its feathered edge."
          >
            Generate
          </button>
        )}
        <button
          className={`btn btn--secondary btn--icon${recordMovie ? ' btn--icon-armed' : ''}`}
          onClick={onToggleRecordMovie}
          disabled={isRunning || !recordingSupported || recordUnavailableForVideo}
          aria-pressed={recordMovie}
          aria-label="Record movie"
          title={
            !recordingSupported
              ? "This browser doesn't support recording canvas video (MediaRecorder / captureStream)."
              : recordUnavailableForVideo
                ? "Not available when processing a video — the processed video downloads automatically when it's done."
                : 'When armed, Generate also records a movie of the run: two seconds on the starting image, the full render at a steady frame rate, then two seconds on the final result. Downloads automatically when done, as .mp4 (or .webm in browsers without MP4 recording).'
          }
        >
          <RecordIcon />
        </button>
        {isRunning && (
          <button
            className="btn btn--secondary btn--icon"
            onClick={isPaused ? onResume : onPause}
            aria-label={isPaused ? 'Resume' : 'Pause'}
            title={
              isPaused
                ? 'Continues the run from exactly the step where it was paused.'
                : 'Pauses the run after the current step finishes, so you can resume later from exactly where it left off.'
            }
          >
            {isPaused ? <PlayIcon /> : <PauseIcon />}
          </button>
        )}
        {isRunning && (
          <button
            className="btn btn--secondary btn--icon"
            onClick={onCancel}
            aria-label="Cancel"
            title="Stops the current run immediately. Progress made on this run will be lost."
          >
            <CancelIcon />
          </button>
        )}
        {isRunning && (
          <button
            className="btn btn--secondary btn--icon"
            onClick={onSaveCurrentStep}
            aria-label="Save current step"
            title="Downloads the current in-progress frame as a PNG without stopping the run."
          >
            <SnapshotIcon />
          </button>
        )}
        <button
          className="btn btn--secondary btn--icon"
          onClick={onDownload}
          disabled={!hasResult}
          aria-label="Download PNG"
          title="Downloads the finished result as a PNG file."
        >
          <DownloadIcon />
        </button>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>
      <p className={`actions-status-text${status.phase === 'error' ? ' actions-status-text--error' : ''}`}>
        {isRecordingMovie && <span className="recording-indicator">● REC</span>}
        {frameProgressLabel && <span className="frame-progress-indicator">{frameProgressLabel}</span>}
        {statusText(status, isPaused)}
      </p>
    </div>
  );
}

interface MixPanelProps {
  mode: Mode;
  mix: MixIngredient[];
  /** Turns the tab's own main template or main dream takes each time round. */
  mainTurns: number;
  onChange: (change: { mix?: MixIngredient[]; mainTurns?: number }) => void;
  /** Preview URL of the image behind each style ingredient, by ingredient id. */
  mixPreviews: Record<string, string>;
  onMixImageChange: (id: string, file: File | null) => void;
  /** Where a new dream's settings start: the DeepDream tab as it stands. */
  dreamParams: DreamParams;
  /** Where a new style's settings start: the Style Transfer tab as it stands. */
  styleParams: StyleParams;
  presets: DreamPreset[];
  selectedPresetId: string;
  isRunning: boolean;
}

function newIngredientId(): string {
  return Math.random().toString(36).slice(2, 10);
}

type DreamNameFields = Pick<
  MixDreamIngredient,
  'presetId' | 'presetBlend' | 'focusSeed' | 'focusChannels' | 'morphSeed' | 'morph' | 'objective'
>;

/**
 * "Organic Forms + Abstract Objects #7 · 4 ch · sharpened" — enough to tell two tunings apart in a list and
 * in a track label. Rebuilt whenever one of these changes, so it never describes a dream it no longer is.
 */
function dreamIngredientName(presets: DreamPreset[], fields: DreamNameFields): string {
  const nameOf = (id: string) => presets.find((p) => p.id === id)?.name ?? id;
  const names = [fields.presetId, ...fields.presetBlend.map((entry) => entry.presetId)].map(nameOf).join(' + ');
  // The morph's endpoints, not its amount: the name labels animation tracks, and a track label that changed
  // whenever the morph slider moved would be no use for finding the track.
  const morphing = fields.morph > 0 && fields.morphSeed !== fields.focusSeed;
  const variation = (seed: number) => (seed > 0 ? `#${seed}` : 'whole');
  const parts = [
    morphing
      ? `${names} ${variation(fields.focusSeed)}→${variation(fields.morphSeed)}`
      : fields.focusSeed > 0
        ? `${names} #${fields.focusSeed}`
        : names,
  ];
  if (fields.focusSeed > 0 || (morphing && fields.morphSeed > 0)) parts.push(`${+fields.focusChannels.toFixed(2)} ch`);
  if (fields.objective === 'squared') parts.push('sharpened');
  return parts.join(' · ');
}

/**
 * Further styles and tuned DeepDream presets, taking turns with the main template on one image.
 *
 * A run walks a cycle — the template's turns, then each ingredient's, then round again — one step per turn,
 * so the image is pulled toward each in rotation and settles where they balance. Turns set that balance;
 * a style's weight and a dream's step size set how hard each one pulls while it has the image.
 */
export function MixPanel({
  mode,
  mix,
  mainTurns,
  onChange,
  mixPreviews,
  onMixImageChange,
  dreamParams,
  styleParams,
  presets,
  selectedPresetId,
  isRunning,
}: MixPanelProps) {
  const update = (id: string, patch: Partial<MixIngredient>) =>
    onChange({
      mix: mix.map((entry) => {
        if (entry.id !== id) return entry;
        const next = { ...entry, ...patch } as MixIngredient;
        return next.kind === 'dream' ? { ...next, name: dreamIngredientName(presets, next) } : next;
      }),
    });
  const remove = (id: string) => {
    onMixImageChange(id, null);
    onChange({ mix: mix.filter((entry) => entry.id !== id) });
  };

  const addStyle = () =>
    onChange({
      mix: [
        ...mix,
        {
          kind: 'style',
          id: newIngredientId(),
          turns: 1,
          templateId: null,
          name: `Style ${mix.filter((entry) => entry.kind === 'style').length + 1}`,
          styleWeight: styleParams.styleWeight,
          contentWeight: styleParams.contentWeight,
          learningRate: styleParams.learningRate,
          totalVariationWeight: styleParams.totalVariationWeight,
        },
      ],
    });

  // A new dream starts as a copy of the DeepDream tab's, as the likeliest thing to want more of — then
  // everything about it can be changed in place.
  const presetId = presets.some((p) => p.id === selectedPresetId) ? selectedPresetId : presets[0]?.id;
  const addDream = () => {
    if (!presetId) return;
    const fields = {
      presetId,
      presetWeight: dreamParams.presetWeight,
      presetBlend: dreamParams.presetBlend,
      focusSeed: dreamParams.focusSeed,
      focusChannels: dreamParams.focusChannels,
      morphSeed: dreamParams.morphSeed,
      morph: dreamParams.morph,
      objective: dreamParams.objective,
    };
    onChange({
      mix: [
        ...mix,
        {
          kind: 'dream',
          id: newIngredientId(),
          turns: 1,
          ...fields,
          name: dreamIngredientName(presets, fields),
          stepSize: dreamParams.stepSize,
          lapLevels: dreamParams.lapLevels,
          tvWeight: dreamParams.tvWeight,
        },
      ],
    });
  };

  const mainName = mode === 'style' ? 'template' : 'main dream';
  const cycle = Math.round(mainTurns) + mix.reduce((sum, entry) => sum + Math.round(entry.turns), 0);

  return (
    <div className="slider-panel">
      <Slider
        label={mode === 'style' ? 'Template turns' : 'Main dream turns'}
        value={mainTurns}
        min={0}
        max={10}
        step={1}
        disabled={isRunning}
        tooltip={`How many steps in a row the ${mainName} — this tab's own settings — takes each time round the mix. 0 leaves it out of the rotation, so only the mix's own dreams and styles act on the image.`}
        onChange={(v) => onChange({ mainTurns: Math.max(0, Math.round(v)) })}
      />

      {mix.map((entry) => (
        <div key={entry.id} className="mix-ingredient">
          <div className="mix-ingredient-header">
            <span className="mix-ingredient-kind">{entry.kind === 'style' ? 'Style' : 'Dream'}</span>
            <span className="mix-ingredient-name">{entry.name}</span>
          </div>
          {entry.kind === 'style' ? (
            <>
              <TemplatePicker
                label="Style image"
                selectedId={entry.templateId}
                previewUrl={mixPreviews[entry.id]}
                disabled={isRunning}
                onSelectBuiltIn={(id, file) => {
                  const name = BUILT_IN_TEMPLATES.find((t) => t.id === id)?.name ?? entry.name;
                  update(entry.id, { templateId: id, name });
                  onMixImageChange(entry.id, file);
                }}
                onSelectFile={(file) => {
                  update(entry.id, { templateId: null, name: file.name.replace(/\.[^.]+$/, '') || entry.name });
                  onMixImageChange(entry.id, file);
                }}
              />
              <Slider
                label="Style weight"
                value={entry.styleWeight}
                min={1}
                max={2000}
                step={10}
                disabled={isRunning}
                tooltip="How strongly this style pulls the image toward its own colors and textures during its turns."
                onChange={(v) => update(entry.id, { styleWeight: v })}
              />
              <Slider
                label="Content weight"
                value={entry.contentWeight}
                min={1}
                max={50}
                step={1}
                disabled={isRunning}
                tooltip="How hard this style holds on to the photo's structure while it paints. Low lets it repaint freely, over whatever the dreams have drawn too."
                onChange={(v) => update(entry.id, { contentWeight: v })}
              />
              <Slider
                label="Learning rate"
                value={entry.learningRate}
                min={0.002}
                max={0.05}
                step={0.002}
                disabled={isRunning}
                tooltip="How far each of this style's turns moves the image. It is this style's step size, set against the dreams' own."
                onChange={(v) => update(entry.id, { learningRate: v })}
              />
              <Slider
                label="Smoothing (TV weight)"
                value={entry.totalVariationWeight}
                min={0}
                max={5}
                step={0.1}
                disabled={isRunning}
                tooltip="How much this style's turns smooth away fine, noisy detail."
                onChange={(v) => update(entry.id, { totalVariationWeight: v })}
              />
            </>
          ) : (
            <>
              <PresetBlendEditor
                presets={presets}
                presetId={entry.presetId}
                presetWeight={entry.presetWeight}
                blend={entry.presetBlend}
                onChange={(change) => update(entry.id, change)}
                pickPrimary
                disabled={isRunning}
              />
              <Slider
                label="Step size"
                value={entry.stepSize}
                min={0.005}
                max={0.1}
                step={0.005}
                disabled={isRunning}
                tooltip="How far each of this dream's turns pushes the image toward what its preset sees, set against what everything else in the mix pushes toward on its own turns."
                onChange={(v) => update(entry.id, { stepSize: v })}
              />
              <VariationControls values={entry} onChange={(patch) => update(entry.id, patch)} disabled={isRunning} compact />
              <label
                className="field-row"
                title="Average spreads detail evenly; Sharpened pushes hardest on what the network half-sees, for fewer, bolder forms."
              >
                <span>Objective</span>
                <select
                  value={entry.objective}
                  onChange={(e) => update(entry.id, { objective: e.target.value as DreamObjective })}
                  disabled={isRunning}
                >
                  <option value="mean">Average</option>
                  <option value="squared">Sharpened</option>
                </select>
              </label>
              <Slider
                label="Frequency bands"
                value={entry.lapLevels}
                min={1}
                max={6}
                step={1}
                disabled={isRunning}
                tooltip="As on the DeepDream tab: more bands balance this dream's push across coarse and fine detail instead of letting fine grain dominate."
                onChange={(v) => update(entry.id, { lapLevels: Math.max(1, Math.round(v)) })}
              />
              <Slider
                label="Smoothing (TV weight)"
                value={entry.tvWeight}
                min={0}
                max={1}
                step={0.05}
                disabled={isRunning}
                tooltip="The share of this dream's step spent smoothing rather than amplifying."
                onChange={(v) => update(entry.id, { tvWeight: v })}
              />
            </>
          )}
          <Slider
            label="Turns"
            value={entry.turns}
            min={0}
            max={10}
            step={1}
            disabled={isRunning}
            tooltip="How many steps in a row this takes each time round the mix. 0 sits it out without removing it."
            onChange={(v) => update(entry.id, { turns: Math.max(0, Math.round(v)) })}
          />
          <div className="controls-actions">
            <button className="btn btn--secondary" onClick={() => remove(entry.id)} disabled={isRunning}>
              Remove
            </button>
          </div>
        </div>
      ))}

      <div className="controls-actions">
        <button
          className="btn btn--secondary"
          onClick={addDream}
          disabled={isRunning || !presetId}
          title="Adds a dream to the rotation, starting as a copy of the DeepDream tab's settings. Change its preset, blend, variation and the rest right here."
        >
          Add dream
        </button>
        <button
          className="btn btn--secondary"
          onClick={addStyle}
          disabled={isRunning}
          title="Adds a style image to the rotation, with its own weights and learning rate, starting from the Style Transfer tab's."
        >
          Add style
        </button>
      </div>
      <p className="field-hint">
        {mix.length === 0
          ? `Plain ${mode === 'style' ? 'style transfer' : 'DeepDream'}. Add dreams and styles — any number, in any combination — to take turns with the ${mainName}.`
          : `Each cycle is ${cycle} step${cycle === 1 ? '' : 's'}; Steps per octave counts them all. Octaves, tiling and color settings come from this tab. The brush uses the ${mainName} and the dreams, not the extra styles.`}
      </p>
    </div>
  );
}
