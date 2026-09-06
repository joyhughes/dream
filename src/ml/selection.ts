import { tf } from './tfSetup';

/**
 * A selection: one weight per pixel saying how much of the effect that pixel receives.
 *
 * Held as a plain array on the CPU rather than a tensor, because every tool that writes to it is
 * inherently sequential or geometric — a flood fill walks neighbors, a lasso rasterizes scanlines — and
 * none of that is work a GPU does well. It only becomes a tensor at the moment the effect is applied.
 *
 * The stored mask is hard-edged. Feathering is applied on the way out, so moving the feather slider
 * re-softens the original edge instead of blurring an already-blurred one into mush.
 */
export class SelectionMask {
  readonly width: number;
  readonly height: number;
  private readonly weights: Float32Array;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.weights = new Float32Array(width * height);
  }

  get data(): Float32Array {
    return this.weights;
  }

  isEmpty(): boolean {
    return !this.weights.some((weight) => weight > 0);
  }

  clear() {
    this.weights.fill(0);
  }

  invert() {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = 1 - this.weights[i];
    }
  }

  selectAll() {
    this.weights.fill(1);
  }

  clone(): SelectionMask {
    const copy = new SelectionMask(this.width, this.height);
    copy.weights.set(this.weights);
    return copy;
  }

  /**
   * Magic wand. Grows out from the seed pixel through everything within `tolerance` of its color, or —
   * when not `contiguous` — takes every matching pixel in the image regardless of whether it connects.
   *
   * Tolerance is a distance in RGB, normalized so 1 spans the full diagonal of the color cube and the
   * slider behaves the same whatever the image.
   *
   * `subtract` takes the region back out of the selection instead of putting it in. Which pixels the region
   * covers is decided by color either way — what is already selected has no bearing on where it reaches.
   */
  wand(
    pixels: Float32Array,
    seedX: number,
    seedY: number,
    tolerance: number,
    contiguous: boolean,
    subtract = false,
  ) {
    const { width, height } = this;
    const seedIndex = (Math.floor(seedY) * width + Math.floor(seedX)) * 3;
    if (seedIndex < 0 || seedIndex >= pixels.length) return;

    const [seedR, seedG, seedB] = [pixels[seedIndex], pixels[seedIndex + 1], pixels[seedIndex + 2]];
    const limit = tolerance * Math.sqrt(3);
    const weight = subtract ? 0 : 1;

    const matches = (index: number): boolean => {
      const dr = pixels[index * 3] - seedR;
      const dg = pixels[index * 3 + 1] - seedG;
      const db = pixels[index * 3 + 2] - seedB;
      return Math.sqrt(dr * dr + dg * dg + db * db) <= limit;
    };

    if (!contiguous) {
      for (let i = 0; i < width * height; i++) {
        if (matches(i)) this.weights[i] = weight;
      }
      return;
    }

    // An explicit stack rather than recursion: a large flat region is hundreds of thousands of pixels
    // deep and would overflow the call stack.
    const visited = new Uint8Array(width * height);
    const stack: number[] = [Math.floor(seedY) * width + Math.floor(seedX)];

    while (stack.length > 0) {
      const index = stack.pop()!;
      if (visited[index]) continue;
      visited[index] = 1;
      if (!matches(index)) continue;

      this.weights[index] = weight;

      const x = index % width;
      const y = (index - x) / width;
      if (x > 0) stack.push(index - 1);
      if (x < width - 1) stack.push(index + 1);
      if (y > 0) stack.push(index - width);
      if (y < height - 1) stack.push(index + width);
    }
  }

  /**
   * Lasso. Fills the interior of a freehand outline by the even-odd rule: a pixel is inside when a ray
   * cast from it crosses the outline an odd number of times, which handles a path that crosses itself
   * without needing to know where it did.
   *
   * `subtract` clears the enclosed area rather than selecting it.
   */
  fillPolygon(points: Array<{ x: number; y: number }>, subtract = false) {
    if (points.length < 3) return;

    const weight = subtract ? 0 : 1;
    const minY = Math.max(0, Math.floor(Math.min(...points.map((p) => p.y))));
    const maxY = Math.min(this.height - 1, Math.ceil(Math.max(...points.map((p) => p.y))));

    for (let y = minY; y <= maxY; y++) {
      const center = y + 0.5;
      const crossings: number[] = [];

      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        // A segment counts when the scanline passes between its endpoints. The half-open comparison is
        // what stops a vertex exactly on the scanline being counted twice.
        if (a.y <= center !== b.y <= center) {
          crossings.push(a.x + ((center - a.y) / (b.y - a.y)) * (b.x - a.x));
        }
      }

      crossings.sort((p, q) => p - q);
      for (let i = 0; i + 1 < crossings.length; i += 2) {
        const from = Math.max(0, Math.ceil(crossings[i] - 0.5));
        const to = Math.min(this.width - 1, Math.floor(crossings[i + 1] - 0.5));
        for (let x = from; x <= to; x++) {
          this.weights[y * this.width + x] = weight;
        }
      }
    }
  }

  /** Selection brush. Paints a round dab into the mask, or takes it back out when `subtract`. */
  stamp(centerX: number, centerY: number, radius: number, subtract: boolean) {
    const left = Math.max(0, Math.floor(centerX - radius));
    const right = Math.min(this.width - 1, Math.ceil(centerX + radius));
    const top = Math.max(0, Math.floor(centerY - radius));
    const bottom = Math.min(this.height - 1, Math.ceil(centerY + radius));

    for (let y = top; y <= bottom; y++) {
      for (let x = left; x <= right; x++) {
        const dx = x + 0.5 - centerX;
        const dy = y + 0.5 - centerY;
        if (dx * dx + dy * dy > radius * radius) continue;

        const index = y * this.width + x;
        this.weights[index] = subtract ? 0 : 1;
      }
    }
  }

  /**
   * The smallest box containing the selection, grown by `pad` so a feathered edge is not clipped by the
   * bound of the hard mask it came from. Null when nothing is selected.
   */
  bounds(pad = 0): { x: number; y: number; width: number; height: number } | null {
    let minX = this.width;
    let minY = this.height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        if (this.weights[y * this.width + x] > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (maxX < 0) return null;

    const x = Math.max(0, minX - pad);
    const y = Math.max(0, minY - pad);
    return {
      x,
      y,
      width: Math.min(this.width, maxX + 1 + pad) - x,
      height: Math.min(this.height, maxY + 1 + pad) - y,
    };
  }

  /**
   * The mask with its edge softened, as a flat array. `radius` is how far the fade actually reaches, in
   * pixels either side of the original edge.
   *
   * Three box blurs in a row rather than a true Gaussian: repeated box blur converges on a Gaussian
   * quickly — three passes are already visually indistinguishable — and each pass is a running sum, so
   * the cost does not grow with the feather radius the way a real kernel's would.
   *
   * Those three passes stack, though, reaching three times the radius of any one of them. Dividing here
   * is what makes the number on the slider the distance the user actually sees the edge fade over.
   */
  feathered(radius: number): Float32Array {
    if (radius <= 0) return this.weights;

    const buffer = Float32Array.from(this.weights);
    const scratch = new Float32Array(buffer.length);
    const r = Math.max(1, Math.round(radius / 3));

    for (let pass = 0; pass < 3; pass++) {
      boxBlurPass(buffer, scratch, this.width, this.height, r, true);
      boxBlurPass(scratch, buffer, this.width, this.height, r, false);
    }

    return buffer;
  }

  /** The feathered mask over one box, as an [h, w, 1] tensor ready to blend with. */
  toTensor(bounds: { x: number; y: number; width: number; height: number }, featherRadius: number): tf.Tensor3D {
    const source = this.feathered(featherRadius);
    const crop = new Float32Array(bounds.width * bounds.height);

    for (let y = 0; y < bounds.height; y++) {
      const from = (bounds.y + y) * this.width + bounds.x;
      crop.set(source.subarray(from, from + bounds.width), y * bounds.width);
    }

    return tf.tensor3d(crop, [bounds.height, bounds.width, 1]);
  }
}

/** One axis of a box blur, as a running sum so its cost does not depend on the radius. */
function boxBlurPass(
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
  horizontal: boolean,
) {
  const lineCount = horizontal ? height : width;
  const lineLength = horizontal ? width : height;
  const step = horizontal ? 1 : width;
  const window = radius * 2 + 1;

  for (let line = 0; line < lineCount; line++) {
    const start = horizontal ? line * width : line;

    // Seed the window, clamping at the edge so the border does not fade toward zero on its own.
    let sum = source[start] * (radius + 1);
    for (let i = 1; i <= radius; i++) {
      sum += source[start + Math.min(i, lineLength - 1) * step];
    }

    for (let i = 0; i < lineLength; i++) {
      target[start + i * step] = sum / window;
      const leaving = source[start + Math.max(0, i - radius) * step];
      const entering = source[start + Math.min(lineLength - 1, i + radius + 1) * step];
      sum += entering - leaving;
    }
  }
}
