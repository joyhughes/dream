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

interface WebGpuBufferPool {
  freeBuffers: Map<string, GPUBuffer[]>;
  numFreeBuffers: number;
  numBytesAllocated: number;
}

interface WebGlTexturePool {
  freeTextures: Record<string, { texture: WebGLTexture }[]> | null;
  numFreeTextures: number;
  _numBytesFree: number;
  _numBytesAllocated: number;
}

/**
 * Hands the GPU memory tf.js is holding for reuse back to the driver.
 *
 * Disposing a tensor doesn't free its GPU buffer: both GPU backends park it in a pool keyed by its exact
 * byte size (WebGPU) or shape (WebGL), to be handed out again the next time a tensor of that size is made,
 * and nothing in either backend ever empties that pool. Within one run that is exactly right — every step
 * of an octave makes the same shapes. Across a sweep it is a leak: moving pattern scale, octave scale,
 * octaves or tile size changes every full-image tensor's size on every frame, so each frame leaves its
 * whole working set in the pool under sizes no later frame asks for, and a long animation climbs by
 * hundreds of megabytes a frame until the GPU process gives out.
 *
 * Call it between runs, never during one — the next step would only have to allocate everything again.
 * This reaches into backend internals, so every access is guarded: a tf.js that has rearranged them makes
 * this a no-op rather than a crash.
 */
export function releasePooledGpuMemory(): void {
  const backend = tf.backend() as unknown as {
    bufferManager?: WebGpuBufferPool;
    textureManager?: WebGlTexturePool;
    gpgpu?: { deleteMatrixTexture(texture: WebGLTexture): void };
  };

  const buffers = backend.bufferManager;
  if (buffers?.freeBuffers instanceof Map) {
    // The keys stay, emptied: `releaseBuffer` treats a pool with no keys at all as a disposed backend and
    // silently stops returning buffers to it.
    for (const pooled of buffers.freeBuffers.values()) {
      for (const buffer of pooled) {
        buffers.numBytesAllocated -= buffer.size;
        // Safe with work still in flight: WebGPU defers the actual free until submitted commands finish.
        buffer.destroy();
      }
      buffers.numFreeBuffers -= pooled.length;
      pooled.length = 0;
    }
  }

  const textures = backend.textureManager;
  if (backend.gpgpu && textures?.freeTextures && typeof textures.freeTextures === 'object') {
    for (const pooled of Object.values(textures.freeTextures)) {
      for (const entry of pooled) backend.gpgpu.deleteMatrixTexture(entry.texture);
      textures.numFreeTextures -= pooled.length;
      pooled.length = 0;
    }
    textures._numBytesAllocated -= textures._numBytesFree;
    textures._numBytesFree = 0;
  }
}

export { tf };
