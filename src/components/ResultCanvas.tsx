import type { RefObject } from 'react';
import type { EngineStatus } from '../types';

interface ResultCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement>;
  status: EngineStatus;
  resultImageUrl: string | null;
}

export function ResultCanvas({ canvasRef, status, resultImageUrl }: ResultCanvasProps) {
  // Once a run finishes, the completed result is shown as a plain <img> from a Blob URL instead of the live
  // canvas — a GPU process reset (common after the computer sleeps) can silently wipe a GPU-composited canvas
  // and invalidate the WebGPU device, but a plain image resource doesn't depend on either.
  const showPersistedImage = status.phase !== 'running' && !!resultImageUrl;

  return (
    <div className="result-stage">
      <canvas ref={canvasRef} className="result-canvas" style={showPersistedImage ? { display: 'none' } : undefined} />
      {showPersistedImage && <img src={resultImageUrl} alt="Result" className="result-canvas" />}
    </div>
  );
}
