import type { DreamParams, DreamPreset, EngineStatus, Mode, StyleParams } from '../types';

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

function Slider({ label, value, min, max, step, tooltip, onChange, disabled }: SliderProps) {
  return (
    <label className="slider-row" title={tooltip}>
      <span className="slider-label">
        {label} <span className="slider-value">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

interface PresetPanelProps {
  mode: Mode;
  presets: DreamPreset[];
  selectedPresetId: string;
  onPresetChange: (id: string) => void;
  isRunning: boolean;
}

export function PresetPanel({ mode, presets, selectedPresetId, onPresetChange, isRunning }: PresetPanelProps) {
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
            max={6}
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
            label="Tile size"
            value={dreamParams.tileSize}
            min={224}
            max={512}
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
            label="Smoothing (TV weight)"
            value={styleParams.totalVariationWeight}
            min={0}
            max={5}
            step={0.1}
            disabled={isRunning}
            tooltip="Penalizes noisy, high-frequency detail to keep the result smooth. Higher values reduce speckling and graininess, at the cost of some fine detail."
            onChange={(v) => onStyleParamsChange({ ...styleParams, totalVariationWeight: v })}
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
            max={5}
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
            label="Tile size"
            value={styleParams.tileSize}
            min={224}
            max={512}
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

interface ActionsBarProps {
  status: EngineStatus;
  isPaused: boolean;
  isRunning: boolean;
  canGenerate: boolean;
  hasResult: boolean;
  onGenerate: () => void;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  onDownload: () => void;
  onSaveCurrentStep: () => void;
}

export function ActionsBar({
  status,
  isPaused,
  isRunning,
  canGenerate,
  hasResult,
  onGenerate,
  onCancel,
  onPause,
  onResume,
  onDownload,
  onSaveCurrentStep,
}: ActionsBarProps) {
  const progress = status.phase === 'running' ? (status.step + 1) / status.totalSteps : status.phase === 'done' ? 1 : 0;

  return (
    <div className="actions-panel">
      <div className="controls-actions">
        <button
          className="btn btn--primary"
          onClick={onGenerate}
          disabled={!canGenerate || isRunning}
          title="Runs DeepDream or Style Transfer on the uploaded image(s) using the current preset and slider settings."
        >
          {isRunning ? (isPaused ? 'Paused' : 'Generating…') : 'Generate'}
        </button>
        {isRunning && (
          <button
            className="btn btn--secondary"
            onClick={isPaused ? onResume : onPause}
            title={
              isPaused
                ? 'Continues the run from exactly the step where it was paused.'
                : 'Pauses the run after the current step finishes, so you can resume later from exactly where it left off.'
            }
          >
            {isPaused ? 'Resume' : 'Pause'}
          </button>
        )}
        {isRunning && (
          <button
            className="btn btn--secondary"
            onClick={onCancel}
            title="Stops the current run immediately. Progress made on this run will be lost."
          >
            Cancel
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
        {statusText(status, isPaused)}
      </p>
    </div>
  );
}
