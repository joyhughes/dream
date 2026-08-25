import { useCallback, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface HoverPopupProps {
  trigger: ReactNode;
  children: ReactNode;
}

interface PopupPosition {
  top: number;
  left: number;
  maxHeight: number;
}

/** Gap between the popup and the trigger, and the minimum it keeps from any viewport edge. */
const MARGIN = 14;
/** Popup width, kept in step with `.template-popup`'s max-width so the left edge can be clamped. */
const POPUP_WIDTH = 320;
/** Below this much room, the popup is moved up the screen rather than squeezed against the bottom. */
const MIN_USEFUL_HEIGHT = 240;

/**
 * Shows `children` in a portal positioned just to the right of `trigger`, while either is
 * hovered/focused. Rendered via a portal (not CSS `:hover` nesting) because the trigger lives
 * inside a scrollable overlay panel (`overflow-y: auto`), which would otherwise clip the popup.
 *
 * The popup is fixed-positioned, so nothing else keeps it on screen — it is placed against the
 * viewport here, and handed the height it has room for so its contents can scroll within it.
 */
export function HoverPopup({ trigger, children }: HoverPopupProps) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<PopupPosition | null>(null);
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
      const viewportHeight = window.innerHeight;

      // Line up with the trigger, unless that leaves too little room underneath to be worth showing.
      let top = rect.top;
      if (viewportHeight - top - MARGIN < MIN_USEFUL_HEIGHT) {
        top = Math.max(MARGIN, viewportHeight - MIN_USEFUL_HEIGHT - MARGIN);
      }

      const left = Math.max(MARGIN, Math.min(rect.right + MARGIN, window.innerWidth - POPUP_WIDTH - MARGIN));

      setPosition({ top, left, maxHeight: viewportHeight - top - MARGIN });
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
        position &&
        createPortal(
          <div
            className="template-popup"
            style={{ top: position.top, left: position.left, maxHeight: position.maxHeight }}
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
