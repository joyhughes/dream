import type { RefObject } from 'react';
import type { EngineStatus } from '../types';

interface ResultCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement>;
  status: EngineStatus;
  resultImageUrl: string | null;
  /** Shown between picking an image and having a result for it. Omitted for a video base file. */
  basePreviewUrl?: string;
}

export function ResultCanvas({ canvasRef, status, resultImageUrl, basePreviewUrl }: ResultCanvasProps) {
  // Once a run finishes, the completed result is shown as a plain <img> from a Blob URL instead of the live
  // canvas — a GPU process reset (common after the computer sleeps) can silently wipe a GPU-composited canvas
  // and invalidate the WebGPU device, but a plain image resource doesn't depend on either. With no result to
  // show, the image the user picked stands in, so the stage always shows whatever is about to be worked on
  // rather than going blank.
  const stillImageUrl = status.phase === 'running' ? null : resultImageUrl ?? basePreviewUrl ?? null;

  return (
    <div className="result-stage">
      <canvas ref={canvasRef} className="result-canvas" style={stillImageUrl ? { display: 'none' } : undefined} />
      {stillImageUrl && (
        <img src={stillImageUrl} alt={resultImageUrl ? 'Result' : 'Image to alter'} className="result-canvas" />
      )}
    </div>
  );
}
