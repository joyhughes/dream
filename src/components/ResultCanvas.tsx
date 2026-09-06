import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from 'react';
import type { EngineStatus, ToolId } from '../types';
import { selectionModeFor, toolDefinition, type SelectionMode } from './tools';

/** What the pointer does on the canvas right now. Supplied only when a tool is active. */
export interface CanvasTool {
  id: ToolId;
  /** Draws a circle at the pointer for tools that act over a radius; omitted for click and path tools. */
  cursorRadius?: number;
  onStart: (x: number, y: number, mode: SelectionMode) => void;
  onMove: (x: number, y: number, mode: SelectionMode) => void;
  onEnd: () => void;
}

/** The badge riding beside the pointer: which tool is in hand, and what the held keys will make it do. */
function ToolCursorBadge({ id, mode }: { id: ToolId; mode: SelectionMode }) {
  const definition = toolDefinition(id);
  const showsMode = definition.usesSelectionModes && mode !== 'replace';

  return (
    <span className={`tool-cursor tool-cursor--${mode}`}>
      {definition.icon}
      {showsMode && <span className="tool-cursor-mode">{mode === 'add' ? '+' : '\u2212'}</span>}
    </span>
  );
}

interface ResultCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement>;
  /** Overlay the selection is drawn into, kept exactly on top of the image. */
  overlayRef: RefObject<HTMLCanvasElement>;
  status: EngineStatus;
  resultImageUrl: string | null;
  /** Shown between picking an image and having a result for it. Omitted for a video base file. */
  basePreviewUrl?: string;
  tool?: CanvasTool;
}

interface DisplayGeometry {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/**
 * The canvas is sized to the working image but drawn with `object-fit: contain`, so it is letterboxed
 * inside its element whenever the aspect ratios differ. Pointer coordinates have to be un-letterboxed
 * before they mean anything in image space — without this a tool lands somewhere other than the cursor
 * on every image that is not exactly the shape of its container.
 */
function displayGeometry(canvas: HTMLCanvasElement): DisplayGeometry {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  return {
    scale,
    offsetX: (rect.width - canvas.width * scale) / 2,
    offsetY: (rect.height - canvas.height * scale) / 2,
  };
}

export function ResultCanvas({
  canvasRef,
  overlayRef,
  status,
  resultImageUrl,
  basePreviewUrl,
  tool,
}: ResultCanvasProps) {
  const stageRef = useRef<HTMLDivElement>(null);
  // Tracked on hover as well as during a stroke, so the cursor says which tool is in hand and which mode a
  // click would be in before it is committed to.
  const [cursor, setCursor] = useState<{ left: number; top: number; size: number; mode: SelectionMode } | null>(
    null,
  );

  // Once a run finishes, the completed result is shown as a plain <img> from a Blob URL instead of the live
  // canvas — a GPU process reset (common after the computer sleeps) can silently wipe a GPU-composited canvas
  // and invalidate the WebGPU device, but a plain image resource doesn't depend on either. With no result to
  // show, the image the user picked stands in, so the stage always shows whatever is about to be worked on
  // rather than going blank. While a tool is active the canvas is what is being worked on, so it wins.
  const stillImageUrl = tool || status.phase === 'running' ? null : resultImageUrl ?? basePreviewUrl ?? null;

  const toImagePoint = useCallback(
    (event: ReactPointerEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      const rect = canvas.getBoundingClientRect();
      const { scale, offsetX, offsetY } = displayGeometry(canvas);
      return {
        x: (event.clientX - rect.left - offsetX) / scale,
        y: (event.clientY - rect.top - offsetY) / scale,
      };
    },
    [canvasRef],
  );

  const updateCursor = useCallback(
    (event: ReactPointerEvent) => {
      const canvas = canvasRef.current;
      if (!tool || !stageRef.current || !canvas) {
        setCursor(null);
        return;
      }

      const { scale } = displayGeometry(canvas);
      const stageRect = stageRef.current.getBoundingClientRect();
      // Radius tools draw their real footprint; the rest get a zero-size anchor the badge hangs off.
      const size = (tool.cursorRadius ?? 0) * 2 * scale;
      setCursor({
        left: event.clientX - stageRect.left - size / 2,
        top: event.clientY - stageRect.top - size / 2,
        size,
        mode: selectionModeFor(event.shiftKey, event.altKey),
      });
    },
    [tool, canvasRef],
  );

  const handlePointerDown = (event: ReactPointerEvent) => {
    if (!tool) return;
    const point = toImagePoint(event);
    if (!point) return;

    // Capture so a stroke that wanders off the canvas keeps reporting, and releases cleanly.
    event.currentTarget.setPointerCapture(event.pointerId);
    updateCursor(event);
    tool.onStart(point.x, point.y, selectionModeFor(event.shiftKey, event.altKey));
  };

  const handlePointerMove = (event: ReactPointerEvent) => {
    if (!tool) return;
    updateCursor(event);
    const point = toImagePoint(event);
    if (point) tool.onMove(point.x, point.y, selectionModeFor(event.shiftKey, event.altKey));
  };

  const handlePointerUp = (event: ReactPointerEvent) => {
    if (!tool) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    tool.onEnd();
  };

  const stageClass = [
    'result-stage',
    tool ? 'result-stage--tool' : null,
    tool ? 'result-stage--hide-pointer' : null,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      ref={stageRef}
      className={stageClass}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onPointerLeave={() => setCursor(null)}
    >
      <canvas ref={canvasRef} className="result-canvas" style={stillImageUrl ? { display: 'none' } : undefined} />
      {stillImageUrl && (
        <img src={stillImageUrl} alt={resultImageUrl ? 'Result' : 'Image to alter'} className="result-canvas" />
      )}
      {/* Sized and positioned exactly like the image below it, so mask pixels land on image pixels. */}
      <canvas ref={overlayRef} className="result-canvas selection-overlay" />
      {tool && cursor && (
        <div
          className={`tool-cursor-anchor tool-cursor-anchor--${cursor.mode}`}
          style={{ left: cursor.left, top: cursor.top, width: cursor.size, height: cursor.size }}
        >
          {cursor.size > 0 && <span className="brush-cursor" />}
          <ToolCursorBadge id={tool.id} mode={cursor.mode} />
        </div>
      )}
    </div>
  );
}
