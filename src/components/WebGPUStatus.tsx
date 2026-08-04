import type { BackendInfo } from '../ml/tfSetup';

interface WebGPUStatusProps {
  info: BackendInfo | null;
  error: string | null;
}

export function WebGPUStatus({ info, error }: WebGPUStatusProps) {
  if (error) {
    return <div className="status-pill status-pill--error">ML init failed: {error}</div>;
  }

  if (!info) {
    return <div className="status-pill status-pill--pending">Initializing ML backend…</div>;
  }

  const isWebgpu = info.backend === 'webgpu';

  return (
    <div className={`status-pill${isWebgpu ? ' status-pill--ok' : ' status-pill--warn'}`}>
      Backend: <strong>{info.backend}</strong>
      {!isWebgpu && ' (WebGPU unavailable, using WebGL fallback)'}
      {info.adapterInfo && <span className="status-detail"> · {info.adapterInfo}</span>}
    </div>
  );
}
