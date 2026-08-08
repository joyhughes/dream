import { tf } from './tfSetup';

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

export function computeTileGrid(height: number, width: number, tileSize: number): TileSpec[] {
  const tileH = Math.min(tileSize, height);
  const tileW = Math.min(tileSize, width);
  const ys = tileStartPositions(height, tileSize);
  const xs = tileStartPositions(width, tileSize);

  const specs: TileSpec[] = [];
  for (const y of ys) {
    for (const x of xs) {
      specs.push({ y, x, h: tileH, w: tileW });
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
  tileSize: number,
  lossFn: (tile: tf.Tensor3D, tileSpec: TileSpec) => tf.Scalar,
): Promise<tf.Tensor3D> {
  const [h, w] = image.shape;
  const tileH = Math.min(tileSize, h);
  const tileW = Math.min(tileSize, w);

  const shiftY = Math.floor(Math.random() * tileH);
  const shiftX = Math.floor(Math.random() * tileW);
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
