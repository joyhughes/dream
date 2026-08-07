type Point = [number, number];

interface Palette {
  background: string;
  motifs: string[];
}

const PALETTES: Palette[] = [
  { background: '#7a1f2b', motifs: ['#e8b84b', '#1c3f5f', '#e8dcc8', '#3d7a5c'] },
  { background: '#1c3f5f', motifs: ['#e8b84b', '#7a1f2b', '#e8dcc8', '#3d7a5c'] },
  { background: '#3d2645', motifs: ['#d98e4a', '#e8dcc8', '#7a1f2b', '#c9a15a'] },
];

/**
 * Outline of a "capsule" formed by two circles (a large bulb + a smaller offset head) joined by their
 * external tangent lines — the classic construction for a paisley/boteh comma silhouette. Traced as an
 * explicit point list (rather than canvas arc() with a cw/ccw flag) so the sweep direction is unambiguous.
 */
function buildTeardropOutline(
  bulbRadius: number,
  tipRadius: number,
  tipOffsetX: number,
  tipOffsetY: number,
  segments: number,
): Point[] {
  const d = Math.hypot(tipOffsetX, tipOffsetY);
  const ratio = Math.max(-1, Math.min(1, (bulbRadius - tipRadius) / d));
  const alpha = Math.asin(ratio);
  const phi = Math.atan2(tipOffsetY, tipOffsetX);
  const betaHigh = phi + Math.PI / 2 - alpha;
  const betaLow = phi - Math.PI / 2 + alpha;

  const points: Point[] = [];

  const bulbSpan = betaLow + Math.PI * 2 - betaHigh;
  const bulbSteps = Math.max(8, Math.round((segments * bulbSpan) / (Math.PI * 2)));
  for (let i = 0; i <= bulbSteps; i++) {
    const a = betaHigh + (bulbSpan * i) / bulbSteps;
    points.push([Math.cos(a) * bulbRadius, Math.sin(a) * bulbRadius]);
  }

  const tipSpan = betaHigh - betaLow;
  const tipSteps = Math.max(6, Math.round((segments * tipSpan) / (Math.PI * 2)));
  for (let i = 0; i <= tipSteps; i++) {
    const a = betaLow + (tipSpan * i) / tipSteps;
    points.push([tipOffsetX + Math.cos(a) * tipRadius, tipOffsetY + Math.sin(a) * tipRadius]);
  }

  return points;
}

function fillOutline(ctx: CanvasRenderingContext2D, outline: Point[], color: string) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(outline[0][0], outline[0][1]);
  for (let i = 1; i < outline.length; i++) {
    ctx.lineTo(outline[i][0], outline[i][1]);
  }
  ctx.closePath();
  ctx.fill();
}

function drawPaisleyMotif(ctx: CanvasRenderingContext2D, palette: string[]) {
  const outer = buildTeardropOutline(1, 0.32, -0.15, -1.05, 48);

  ctx.save();
  fillOutline(ctx, outer, palette[0]);

  ctx.scale(0.72, 0.72);
  fillOutline(ctx, outer, palette[1 % palette.length]);
  ctx.restore();

  ctx.save();
  ctx.fillStyle = palette[2 % palette.length];
  const dotCount = 5 + Math.floor(Math.random() * 3);
  for (let i = 0; i < dotCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 0.55;
    const dotRadius = 0.03 + Math.random() * 0.05;
    ctx.beginPath();
    ctx.arc(Math.cos(angle) * radius, 0.15 + Math.sin(angle) * radius * 0.6, dotRadius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/** Procedurally draws a tileable paisley/boteh pattern — original artwork, no licensing concerns. */
export function generatePaisleyPattern(size = 512): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable.');

  const palette = PALETTES[Math.floor(Math.random() * PALETTES.length)];
  ctx.fillStyle = palette.background;
  ctx.fillRect(0, 0, size, size);

  const cellSize = size / 3.4;
  const motifScale = cellSize * 0.42;

  let row = 0;
  for (let y = -cellSize; y < size + cellSize; y += cellSize * 0.75) {
    const offsetX = row % 2 === 0 ? 0 : cellSize / 2;
    for (let x = -cellSize; x < size + cellSize; x += cellSize) {
      ctx.save();
      ctx.translate(x + offsetX, y);
      ctx.rotate((Math.random() - 0.5) * 0.6 + (row % 2 === 0 ? 0 : Math.PI));
      const scale = motifScale * (0.85 + Math.random() * 0.3);
      ctx.scale(scale, scale);
      drawPaisleyMotif(ctx, palette.motifs);
      ctx.restore();
    }
    row++;
  }

  return canvas;
}
