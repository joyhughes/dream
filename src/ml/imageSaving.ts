/**
 * Getting a finished image off the page and into the user's own storage.
 *
 * On a desktop browser an `<a download>` click is the whole story. On iOS it is not: Safari treats a
 * blob download as a file bound for Files, chimes as though it succeeded, and the image never appears
 * in Photos. The only route into the photo library is the share sheet's "Save Image", so prefer
 * `navigator.share` with a file wherever the browser can share one.
 */

import { isMobileBrowser } from './platform';

/**
 * Saves a blob as a file the user can find later. Must be called from a click handler: the share sheet
 * needs the user activation that a gesture provides, and it is spent by the first `await`.
 */
export async function saveImage(blob: Blob, filename: string): Promise<void> {
  const type = blob.type || 'image/png';
  const file = new File([blob], filename, { type });

  // Only on a phone or tablet. On a desktop the share sheet is a detour around what a download already
  // does well, and the file lands somewhere the user knows how to find.
  if (isMobileBrowser() && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
      return;
    } catch (err) {
      // A dismissed share sheet is a deliberate "no" — don't answer it with a download the user
      // didn't ask for. Anything else means the sheet failed, and a download beats saving nothing.
      if (err instanceof DOMException && err.name === 'AbortError') return;
      console.warn('Share sheet failed, falling back to a download:', err);
    }
  }

  downloadBlob(file, filename);
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.rel = 'noopener';
  document.body.append(link);
  link.click();
  link.remove();
  // Safari cancels a download whose blob URL is revoked while it is still reading from it, so hold the
  // URL well past the click rather than freeing it in the same task.
  setTimeout(() => URL.revokeObjectURL(url), 40_000);
}

/**
 * PNG bytes from a canvas with no `await` on the way. `canvas.toBlob` is asynchronous, and by the time
 * its callback runs the user activation `saveImage` needs is gone, so the share sheet never opens.
 * `toDataURL` is synchronous, which keeps the save in the same task as the click that asked for it.
 */
export function canvasToPngBlob(canvas: HTMLCanvasElement): Blob {
  const dataUrl = canvas.toDataURL('image/png');
  const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: 'image/png' });
}
