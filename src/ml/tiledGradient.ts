import { tf } from './tfSetup';
import { getDeviceLimits } from './deviceLimits';

export interface TileSpec {
  y: number;
  x: number;
  h: number;
  w: number;
}

/** Circularly shifts a HWC tensor by (shiftY, shiftX), wrapping content around each edge. */
export function rollImage(img: tf.Tensor3D, shiftY: number, shiftX: number): tf.Tensor3D {
  return tf.tidy(() => {
    const [h, w] = img.shape;
    const sy = ((shiftY % h) + h) % h;
    const sx = ((shiftX % w) + w) % w;

    const rolledY =
      sy === 0
        ? img
        : (tf.concat([img.slice([h - sy, 0, 0], [sy, w, 3]), img.slice([0, 0, 0], [h - sy, w, 3])], 0) as tf.Tensor3D);

    const rolledXY =
      sx === 0
        ? rolledY
        : (tf.concat([rolledY.slice([0, w - sx, 0], [h, sx, 3]), rolledY.slice([0, 0, 0], [h, w - sx, 3])], 1) as tf.Tensor3D);

    return rolledXY;
  });
}

function tileStartPositions(dim: number, tileSize: number): number[] {
  if (dim <= tileSize) {
    return [0];
  }
  const starts: number[] = [];
  let pos = 0;
  while (pos < dim) {
    starts.push(Math.min(pos, dim - tileSize));
    pos += tileSize;
  }
  return Array.from(new Set(starts));
}

/**
 * The tile size a run will actually use: what was asked for, capped by the device's ceiling and by the
 * image itself. Every tile pass holds a full-image gradient plus the autodiff tape for one tile, and that
 * tape scales with the tile's area — capping here rather than at each caller covers both the dream and the
 * style path, and also catches a too-large tile size restored from a previously saved result.
 */
export function effectiveTileSize(requestedTileSize: number, height: number, width: number): number {
  return Math.min(requestedTileSize, getDeviceLimits().maxTileSize, height, width);
}

export function computeTileGrid(height: number, width: number, tileSize: number): TileSpec[] {
  // Tiles are always kept square (clamped by whichever of height/width is smaller), even when the
  // image itself isn't. tf.js's CPU backend has a real bug in its gradient for a non-square
  // resizeBilinear input (the H/W dims come back transposed) — since every tile gets resized to
  // the network's square 224x224 input downstream, a non-square tile would trigger it. Square
  // tiles sidestep that entirely, and as a bonus don't distort the tile's aspect ratio on resize.
  const tileDim = Math.min(tileSize, height, width);
  const ys = tileStartPositions(height, tileDim);
  const xs = tileStartPositions(width, tileDim);

  const specs: TileSpec[] = [];
  for (const y of ys) {
    for (const x of xs) {
      specs.push({ y, x, h: tileDim, w: tileDim });
    }
  }
  return specs;
}

/**
 * Computes the gradient of a per-tile loss with respect to the full image, by splitting the image into a
 * grid of `tileSize`-native-pixel tiles (each fed to the network without further downsampling), running
 * `tf.grad` per tile, and accumulating the results into one full-resolution gradient.
 *
 * This is what lets the network respond to detail at native resolution instead of only ever seeing a single
 * global downsample to its fixed 224x224 input — the classic "tiled gradients" DeepDream technique. A random
 * jitter (circular shift) is applied each call so the tile grid lands in a different place every time,
 * otherwise the fixed tile boundaries show up as a visible grid pattern in the result.
 *
 * Yields to the event loop periodically since a large image can mean dozens of sequential tile passes.
 */
export async function computeTiledGradient(
  image: tf.Tensor3D,
  requestedTileSize: number,
  lossFn: (tile: tf.Tensor3D, tileSpec: TileSpec) => tf.Scalar,
): Promise<tf.Tensor3D> {
  const [h, w] = image.shape;

  const tileSize = effectiveTileSize(requestedTileSize, h, w);

  const shiftY = Math.floor(Math.random() * tileSize);
  const shiftX = Math.floor(Math.random() * tileSize);
  const rolled = tf.tidy(() => tf.keep(rollImage(image, shiftY, shiftX)) as tf.Tensor3D);

  const tiles = computeTileGrid(h, w, tileSize);
  let gradAccum = tf.tidy(() => tf.keep(tf.zeros([h, w, 3])) as tf.Tensor3D);

  for (let i = 0; i < tiles.length; i++) {
    const spec = tiles[i];

    const next = tf.tidy(() => {
      const gradFn = tf.grad((imgArg: tf.Tensor) => {
        const tile = (imgArg as tf.Tensor3D).slice([spec.y, spec.x, 0], [spec.h, spec.w, 3]);
        return lossFn(tile, spec);
      });
      const g = gradFn(rolled) as tf.Tensor3D;
      return tf.keep(gradAccum.add(g)) as tf.Tensor3D;
    });

    gradAccum.dispose();
    gradAccum = next;

    if (i % 4 === 3) {
      await tf.nextFrame();
    }
  }

  rolled.dispose();

  const unrolled = tf.tidy(() => tf.keep(rollImage(gradAccum, -shiftY, -shiftX)) as tf.Tensor3D);
  gradAccum.dispose();

  return unrolled;
}
