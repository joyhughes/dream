import { useCallback, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface HoverPopupProps {
  trigger: ReactNode;
  children: ReactNode;
}

/**
 * Shows `children` in a portal positioned just to the right of `trigger`, while either is
 * hovered/focused. Rendered via a portal (not CSS `:hover` nesting) because the trigger lives
 * inside a scrollable overlay panel (`overflow-y: auto`), which would otherwise clip the popup.
 */
export function HoverPopup({ trigger, children }: HoverPopupProps) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number | null>(null);

  const clearCloseTimer = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };

  const handleOpen = useCallback(() => {
    clearCloseTimer();
    const rect = wrapperRef.current?.getBoundingClientRect();
    if (rect) {
      setCoords({ top: rect.top, left: rect.right + 14 });
    }
    setOpen(true);
  }, []);

  const handleClose = useCallback(() => {
    clearCloseTimer();
    closeTimer.current = window.setTimeout(() => setOpen(false), 150);
  }, []);

  return (
    <div ref={wrapperRef} onMouseEnter={handleOpen} onMouseLeave={handleClose} onFocus={handleOpen} onBlur={handleClose}>
      {trigger}
      {open &&
        coords &&
        createPortal(
          <div
            className="template-popup"
            style={{ top: coords.top, left: coords.left }}
            onMouseEnter={handleOpen}
            onMouseLeave={handleClose}
          >
            {children}
          </div>,
          document.body,
        )}
    </div>
  );
}
