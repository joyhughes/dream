import type { EngineStatus } from '../types';

interface StatusBarProps {
  status: EngineStatus;
  isPaused: boolean;
  hasResult: boolean;
  onDownload: () => void;
  onSaveCurrentStep: () => void;
}

function statusText(status: EngineStatus, isPaused: boolean): string {
  switch (status.phase) {
    case 'idle':
      return 'Upload images and click Generate.';
    case 'loading-model':
      return 'Loading MobileNet feature model…';
    case 'ready':
      return 'Model ready.';
    case 'running':
      return isPaused
        ? `Paused at step ${status.step + 1} / ${status.totalSteps}`
        : `Generating… step ${status.step + 1} / ${status.totalSteps}`;
    case 'done':
      return 'Done.';
    case 'error':
      return `Error: ${status.message}`;
  }
}

export function StatusBar({ status, isPaused, hasResult, onDownload, onSaveCurrentStep }: StatusBarProps) {
  const progress = status.phase === 'running' ? (status.step + 1) / status.totalSteps : status.phase === 'done' ? 1 : 0;
  const canSaveCurrentStep = status.phase === 'running';

  return (
    <div className="status-bar">
      <div className="status-bar-progress-track">
        <div className="status-bar-progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="status-bar-row">
        <span className={`status-bar-text${status.phase === 'error' ? ' status-bar-text--error' : ''}`}>
          {statusText(status, isPaused)}
        </span>
        <div className="status-bar-actions">
          {canSaveCurrentStep && (
            <button
              className="btn btn--secondary"
              onClick={onSaveCurrentStep}
              title="Downloads the current in-progress frame as a PNG without stopping the run."
            >
              Save current step
            </button>
          )}
          <button
            className="btn btn--secondary"
            onClick={onDownload}
            disabled={!hasResult}
            title="Downloads the finished result as a PNG file."
          >
            Download PNG
          </button>
        </div>
      </div>
    </div>
  );
}
