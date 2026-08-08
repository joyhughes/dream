import type { ReactNode } from 'react';

interface OverlayPanelProps {
  title: string;
  side: 'left' | 'right';
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export function OverlayPanel({ title, side, open, onToggle, children }: OverlayPanelProps) {
  return (
    <div className={`overlay-panel overlay-panel--${side} ${open ? 'overlay-panel--open' : 'overlay-panel--collapsed'}`}>
      <div className="overlay-panel-header">
        {open && <span className="overlay-panel-title">{title}</span>}
        <button
          className="overlay-toggle"
          onClick={onToggle}
          aria-expanded={open}
          title={open ? `Collapse the ${title.toLowerCase()} panel` : `Expand the ${title.toLowerCase()} panel`}
        >
          {open ? '−' : '≡'}
        </button>
      </div>
      {open && <div className="overlay-panel-body">{children}</div>}
    </div>
  );
}
