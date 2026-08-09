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

  try {
    await tf.setBackend('webgl');
    await tf.ready();
    return { backend: tf.getBackend(), webgpuAvailable };
  } catch (err) {
    console.warn('WebGL backend failed to initialize, falling back to CPU.', err);
  }

  await tf.setBackend('cpu');
  await tf.ready();
  return { backend: tf.getBackend(), webgpuAvailable };
}

export function initializeML(): Promise<BackendInfo> {
  if (!readyPromise) {
    readyPromise = pickBackend();
  }
  return readyPromise;
}

/** Cheap sanity computation to confirm the active backend is actually producing correct results. */
async function isBackendHealthy(): Promise<boolean> {
  try {
    const value = tf.tidy(() => tf.scalar(2).add(tf.scalar(3)).dataSync()[0]);
    return value === 5;
  } catch {
    return false;
  }
}

/**
 * Re-validates the active backend before a run and transparently recreates it if it's gone bad.
 * A GPU process reset (common after the computer sleeps, or when the browser reclaims GPU
 * resources from a backgrounded tab) can silently invalidate the WebGPU/WebGL device tfjs is
 * holding onto — after that, ops don't throw, they just quietly return zeroed/garbage tensors,
 * which render as a black frame. `tf.setBackend()` alone won't fix this: if the backend name
 * hasn't changed, tfjs reuses the existing (broken) instance rather than recreating it, so the
 * broken backend has to be explicitly removed first to force its factory to run again.
 */
export async function ensureBackendHealthy(): Promise<BackendInfo> {
  const info = await initializeML();

  if (await isBackendHealthy()) {
    return info;
  }

  const brokenBackend = tf.getBackend();

  // The CPU backend is plain deterministic JS with no external device to lose, so a failed
  // sanity check there points to something else being wrong. Unlike webgpu/webgl, tf.removeBackend()
  // also deregisters the backend's factory, permanently disabling it for the rest of the session —
  // not worth that risk for a backend that should never actually fail this check.
  if (brokenBackend === 'cpu') {
    console.warn('CPU backend failed a sanity check; not attempting to reinitialize it.');
    return info;
  }

  console.warn(`ML backend "${brokenBackend}" failed a sanity check (likely a lost GPU device) — reinitializing.`);
  try {
    tf.removeBackend(brokenBackend);
  } catch (err) {
    console.warn('Failed to remove broken backend before reinitializing.', err);
  }
  readyPromise = null;
  return initializeML();
}

export { tf };
