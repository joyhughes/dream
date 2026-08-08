import type { DreamParams, DreamPreset, Mode, StyleParams } from '../types';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

function Slider({ label, value, min, max, step, onChange, disabled }: SliderProps) {
  return (
    <label className="slider-row">
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

interface ControlsPanelProps {
  mode: Mode;
  presets: DreamPreset[];
  selectedPresetId: string;
  onPresetChange: (id: string) => void;
  dreamParams: DreamParams;
  onDreamParamsChange: (params: DreamParams) => void;
  styleParams: StyleParams;
  onStyleParamsChange: (params: StyleParams) => void;
  onGenerate: () => void;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  isRunning: boolean;
  isPaused: boolean;
  canGenerate: boolean;
}

export function ControlsPanel({
  mode,
  presets,
  selectedPresetId,
  onPresetChange,
  dreamParams,
  onDreamParamsChange,
  styleParams,
  onStyleParamsChange,
  onGenerate,
  onCancel,
  onPause,
  onResume,
  isRunning,
  isPaused,
  canGenerate,
}: ControlsPanelProps) {
  return (
    <div className="controls-panel">
      {mode === 'deepdream' ? (
        <>
          <label className="field-row">
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

          <Slider
            label="Octaves"
            value={dreamParams.octaves}
            min={1}
            max={6}
            step={1}
            disabled={isRunning}
            onChange={(v) => onDreamParamsChange({ ...dreamParams, octaves: v })}
          />
          <Slider
            label="Octave scale"
            value={dreamParams.octaveScale}
            min={1.1}
            max={2}
            step={0.05}
            disabled={isRunning}
            onChange={(v) => onDreamParamsChange({ ...dreamParams, octaveScale: v })}
          />
          <Slider
            label="Steps per octave"
            value={dreamParams.stepsPerOctave}
            min={5}
            max={100}
            step={5}
            disabled={isRunning}
            onChange={(v) => onDreamParamsChange({ ...dreamParams, stepsPerOctave: v })}
          />
          <Slider
            label="Step size"
            value={dreamParams.stepSize}
            min={0.005}
            max={0.1}
            step={0.005}
            disabled={isRunning}
            onChange={(v) => onDreamParamsChange({ ...dreamParams, stepSize: v })}
          />
          <Slider
            label="Tile size"
            value={dreamParams.tileSize}
            min={224}
            max={512}
            step={32}
            disabled={isRunning}
            onChange={(v) => onDreamParamsChange({ ...dreamParams, tileSize: v })}
          />
          <p className="field-hint">
            Smaller tiles capture more native detail on large images but take longer per step — 224 is the
            network&apos;s native resolution and gives maximum fidelity.
          </p>
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
            onChange={(v) => onStyleParamsChange({ ...styleParams, contentWeight: v })}
          />
          <Slider
            label="Style weight"
            value={styleParams.styleWeight}
            min={1}
            max={2000}
            step={10}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, styleWeight: v })}
          />
          <Slider
            label="Smoothing (TV weight)"
            value={styleParams.totalVariationWeight}
            min={0}
            max={5}
            step={0.1}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, totalVariationWeight: v })}
          />
          <Slider
            label="Learning rate"
            value={styleParams.learningRate}
            min={0.002}
            max={0.05}
            step={0.002}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, learningRate: v })}
          />
          <Slider
            label="Octaves"
            value={styleParams.octaves}
            min={1}
            max={5}
            step={1}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, octaves: v })}
          />
          <Slider
            label="Octave scale"
            value={styleParams.octaveScale}
            min={1.1}
            max={2}
            step={0.05}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, octaveScale: v })}
          />
          <Slider
            label="Steps per octave"
            value={styleParams.stepsPerOctave}
            min={10}
            max={150}
            step={5}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, stepsPerOctave: v })}
          />
          <Slider
            label="Tile size"
            value={styleParams.tileSize}
            min={224}
            max={512}
            step={32}
            disabled={isRunning}
            onChange={(v) => onStyleParamsChange({ ...styleParams, tileSize: v })}
          />
          <p className="field-hint">
            Smaller tiles capture more native detail on large images but take longer per step — 224 is the
            network&apos;s native resolution and gives maximum fidelity.
          </p>
        </>
      )}

      <div className="controls-actions">
        <button className="btn btn--primary" onClick={onGenerate} disabled={!canGenerate || isRunning}>
          {isRunning ? (isPaused ? 'Paused' : 'Generating…') : 'Generate'}
        </button>
        {isRunning && (
          <button className="btn btn--secondary" onClick={isPaused ? onResume : onPause}>
            {isPaused ? 'Resume' : 'Pause'}
          </button>
        )}
        {isRunning && (
          <button className="btn btn--secondary" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
