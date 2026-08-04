import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

export interface BackendInfo {
  backend: string;
  webgpuAvailable: boolean;
  adapterInfo?: string;
}

let readyPromise: Promise<BackendInfo> | null = null;

async function pickBackend(): Promise<BackendInfo> {
  const webgpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;

  if (webgpuAvailable) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();

      let adapterInfo: string | undefined;
      try {
        const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter();
        const info = adapter ? await adapter.requestAdapterInfo?.() : undefined;
        adapterInfo = info ? `${info.vendor ?? ''} ${info.architecture ?? ''}`.trim() : undefined;
      } catch {
        // adapter info is best-effort only
      }

      return { backend: tf.getBackend(), webgpuAvailable: true, adapterInfo };
    } catch (err) {
      console.warn('WebGPU backend failed to initialize, falling back.', err);
    }
  }

  await tf.setBackend('webgl');
  await tf.ready();
  return { backend: tf.getBackend(), webgpuAvailable };
}

export function initializeML(): Promise<BackendInfo> {
  if (!readyPromise) {
    readyPromise = pickBackend();
  }
  return readyPromise;
}

export { tf };
