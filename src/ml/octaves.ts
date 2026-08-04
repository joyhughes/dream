/**
 * Builds a size pyramid from smallest to full resolution. The last entry is always exactly [h, w],
 * so callers can optimize coarse-to-fine and finish at native resolution.
 */
export function computeOctaveShapes(h: number, w: number, octaves: number, octaveScale: number): [number, number][] {
  const shapes: [number, number][] = [];
  for (let k = 0; k < octaves; k++) {
    const exponent = octaves - 1 - k;
    const factor = Math.pow(octaveScale, -exponent);
    shapes.push([Math.max(8, Math.round(h * factor)), Math.max(8, Math.round(w * factor))]);
  }
  return shapes;
}
