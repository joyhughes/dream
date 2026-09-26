export function pickSupportedVideoMimeType(): string | undefined {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface TimedFrame {
  bitmap: ImageBitmap;
  holdMs: number;
}

/**
 * Encodes a sequence of already-captured frames into a video, each held for exactly its own
 * `holdMs` — independent of how long it actually took to produce those frames in the first place.
 *
 * Captures via `captureStream(0)`'s manual mode (advanced only by explicit `requestFrame()`
 * calls) onto a dedicated off-screen canvas, rather than sampling continuously at a fixed rate:
 * this ties each frame's on-screen duration in the output to the `holdMs` asked for, not to real
 * capture timing.
 */
export async function encodeFrameSequence(
  frames: Iterable<TimedFrame> | AsyncIterable<TimedFrame>,
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  // Kept off-screen (not display:none) so browsers don't throttle a canvas they consider hidden.
  canvas.style.position = 'fixed';
  canvas.style.top = '-99999px';
  canvas.style.left = '-99999px';
  document.body.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not create a 2D context for video encoding.');

  const stream = canvas.captureStream(0);
  const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
  const mimeType = pickSupportedVideoMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const draw = (bitmap: ImageBitmap) => {
    ctx.drawImage(bitmap, 0, 0, width, height);
    track.requestFrame();
  };

  recorder.start();

  // Each frame's hold is measured against a running schedule rather than slept for after the draw,
  // so whatever the source spends producing the next frame — decoding a stored one, most of all —
  // comes out of that frame's own hold instead of being added to it.
  let drawn = 0;
  let dueAt = performance.now();

  for await (const frame of frames) {
    draw(frame.bitmap);
    drawn += 1;
    dueAt += frame.holdMs;
    await sleep(Math.max(0, dueAt - performance.now()));
  }

  if (drawn > 0) {
    // The canvas still holds the last frame drawn, so this needs no bitmap of its own — and the
    // source may well have closed that one already. requestFrame() only queues the capture:
    // stopping immediately after can drop it before the browser processes it, truncating the
    // recording right before its intended final duration.
    track.requestFrame();
  }
  await sleep(150);

  const blob = await new Promise<Blob>((resolve) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || 'video/webm' }));
    recorder.stop();
  });

  stream.getTracks().forEach((t) => t.stop());
  canvas.remove();
  return blob;
}
