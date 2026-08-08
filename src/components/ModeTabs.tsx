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
        title="DeepDream amplifies patterns the MobileNet network already 'sees' in your photo, enhancing whatever textures and shapes activate its layers most strongly — the classic dream-like effect."
      >
        DeepDream
      </button>
      <button
        role="tab"
        aria-selected={mode === 'style'}
        className={`mode-tab${mode === 'style' ? ' mode-tab--active' : ''}`}
        onClick={() => onChange('style')}
        disabled={disabled}
        title="Style Transfer repaints your photo using the colors, textures, and brushwork of a second template image, while keeping your photo's original content and layout."
      >
        Style Transfer
      </button>
    </div>
  );
}
