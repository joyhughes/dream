import type { RefObject } from 'react';
import type { EngineStatus } from '../types';

interface ResultCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement>;
  status: EngineStatus;
  onDownload: () => void;
  hasResult: boolean;
  resultImageUrl: string | null;
}

function statusText(status: EngineStatus): string {
  switch (status.phase) {
    case 'idle':
      return 'Upload images and click Generate.';
    case 'loading-model':
      return 'Loading MobileNet feature model…';
    case 'ready':
      return 'Model ready.';
    case 'running':
      return `Generating… step ${status.step + 1} / ${status.totalSteps}`;
    case 'done':
      return 'Done.';
    case 'error':
      return `Error: ${status.message}`;
  }
}

export function ResultCanvas({ canvasRef, status, onDownload, hasResult, resultImageUrl }: ResultCanvasProps) {
  const progress = status.phase === 'running' ? (status.step + 1) / status.totalSteps : status.phase === 'done' ? 1 : 0;

  // Once a run finishes, the completed result is shown as a plain <img> from a Blob URL instead of the live
  // canvas — a GPU process reset (common after the computer sleeps) can silently wipe a GPU-composited canvas
  // and invalidate the WebGPU device, but a plain image resource doesn't depend on either.
  const showPersistedImage = status.phase !== 'running' && !!resultImageUrl;

  return (
    <div className="result-panel">
      <div className="result-canvas-wrapper">
        <canvas ref={canvasRef} className="result-canvas" style={showPersistedImage ? { display: 'none' } : undefined} />
        {showPersistedImage && <img src={resultImageUrl} alt="Result" className="result-canvas" />}
      </div>
      <div className="result-progress-track">
        <div className="result-progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="result-status-row">
        <span className={`result-status${status.phase === 'error' ? ' result-status--error' : ''}`}>
          {statusText(status)}
        </span>
        <button className="btn btn--secondary" onClick={onDownload} disabled={!hasResult}>
          Download PNG
        </button>
      </div>
    </div>
  );
}
