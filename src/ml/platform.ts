/**
 * Which kind of device the app is running on, where that changes what the app should do.
 * Two callers so far: the per-tab memory ceilings in `deviceLimits`, and how a finished image is saved.
 */
export function isMobileBrowser(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  const ua = navigator.userAgent;

  // iPadOS 13+ reports a desktop Macintosh user agent; a touch-capable "Mac" is the giveaway.
  const isIOS = /iP(hone|od|ad)/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  return isIOS || /Android/.test(ua);
}
