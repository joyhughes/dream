/**
 * Per-device ceilings for the things in this app whose cost scales with image size or frame count.
 *
 * Desktop browsers will hand a tab several gigabytes without complaining. Mobile Safari will not:
 * once a page's footprint crosses a threshold that is well under 1 GB on most iPhones, iOS kills the
 * whole WebContent process. There is no catchable error and no warning beforehand — the tab simply
 * reloads itself or shows "A problem repeatedly occurred" — so the only defense is to not allocate
 * that much in the first place. Every limit here exists to keep a peak allocation off that cliff.
 */

import { isMobileBrowser } from './platform';

interface DeviceLimits {
  /** True for phones/tablets, where the per-tab memory ceiling is low and enforced by process death. */
  memoryConstrained: boolean;
  /** Longest side, in pixels, that an uploaded image or video frame is worked on at. */
  workingMaxDimension: number;
  /**
   * The same, for style transfer. Lower, because style transfer holds far more per pixel than DeepDream:
   * a content target per tile, Gram targets, Adam's two moment tensors, and a tape whose tile pass runs a
   * deeper slice of the network. DeepDream runs fine on a phone at the full working size; style transfer
   * was reaching the end of a run and then losing the tab to the final full-resolution GPU readback.
   */
  styleWorkingMaxDimension: number;
  /** Upper bound on the tile size used for tiled gradients, regardless of what the slider allows. */
  maxTileSize: number;
  /** Byte budget for the compressed frames held in memory while recording a movie or processing a video. */
  frameStoreBudgetBytes: number;
}

function detectMemoryConstrained(): boolean {
  if (isMobileBrowser()) {
    return true;
  }

  if (typeof navigator === 'undefined') {
    return false;
  }

  // Not implemented in Safari or Firefox, so this only ever adds detections — never the mobile one above.
  const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof deviceMemory === 'number' && deviceMemory <= 4;
}

let cached: DeviceLimits | null = null;

export function getDeviceLimits(): DeviceLimits {
  if (cached) {
    return cached;
  }

  const memoryConstrained = detectMemoryConstrained();

  cached = memoryConstrained
    ? {
        memoryConstrained,
        // A 1024x768 float32 RGB tensor is ~9 MB, and the tiled-gradient inner loop keeps a handful of
        // full-image tensors alive at once. At 2048 those same tensors are ~38 MB each, which on top of
        // the loaded network is already past what an iPhone tab survives.
        workingMaxDimension: 1024,
        styleWorkingMaxDimension: 768,
        maxTileSize: 256,
        frameStoreBudgetBytes: 64 * 1024 * 1024,
      }
    : {
        memoryConstrained,
        workingMaxDimension: 2048,
        styleWorkingMaxDimension: 2048,
        maxTileSize: 512,
        frameStoreBudgetBytes: 256 * 1024 * 1024,
      };

  return cached;
}

/**
 * How many frames of `width`x`height` the in-memory frame store may hold before exceeding its budget.
 *
 * Frames are kept as ImageBitmaps, which cost roughly their pixel count in RGBA bytes, and both the
 * movie recorder and the video pipeline hold every frame until the whole sequence is encoded at the
 * end. Never returns less than 2, so a sequence is always at least something rather than nothing.
 */
export function maxFramesInStore(width: number, height: number): number {
  const bytesPerFrame = Math.max(1, width * height * 4);
  return Math.max(2, Math.floor(getDeviceLimits().frameStoreBudgetBytes / bytesPerFrame));
}
