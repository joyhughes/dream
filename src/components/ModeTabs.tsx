import type { Mode } from '../types';

interface ModeTabsProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}

export function ModeTabs({ mode, onChange, disabled }: ModeTabsProps) {
  return (
    <div className="mode-tabs" role="tablist">
      <button
        role="tab"
        aria-selected={mode === 'deepdream'}
        className={`mode-tab${mode === 'deepdream' ? ' mode-tab--active' : ''}`}
        onClick={() => onChange('deepdream')}
        disabled={disabled}
      >
        DeepDream
      </button>
      <button
        role="tab"
        aria-selected={mode === 'style'}
        className={`mode-tab${mode === 'style' ? ' mode-tab--active' : ''}`}
        onClick={() => onChange('style')}
        disabled={disabled}
      >
        Style Transfer
      </button>
    </div>
  );
}
