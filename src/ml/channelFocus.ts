/**
 * Picks which of a layer's channels a DreamParams `focusSeed` asks DeepDream to amplify.
 *
 * Amplifying a whole late layer pushes on hundreds of feature detectors at once, and what comes out is
 * their average — the same blend of eyes, snouts and fur on every run. Each channel on its own is a
 * single detector with its own idea of a creature, so pushing on a handful of them instead gives a
 * distinct family of forms, and a different handful gives a different family. The seed names a handful;
 * the same seed always names the same one, so a variation worth keeping can be typed back in.
 */

/** mulberry32: small, fast, and good enough to shuffle channel indices reproducibly. */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a, so two layers in one preset draw different channels from the same seed. */
function hashName(name: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    hash ^= name.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

const cache = new Map<string, number[]>();

/**
 * The channel indices to amplify in a layer of `channelCount` channels, or null for all of them.
 * A seed of 0 means the whole layer, which is the classic effect and what every run did before this.
 */
export function focusedChannels(nodeName: string, channelCount: number, seed: number, count: number): number[] | null {
  const take = Math.round(count);
  if (seed <= 0 || take <= 0 || take >= channelCount) return null;

  const key = `${nodeName}:${channelCount}:${seed}:${take}`;
  const cached = cache.get(key);
  if (cached) return cached;

  // A partial Fisher-Yates shuffle: the first `take` entries are a uniform draw without repeats. Growing
  // `count` with the seed held keeps the channels already drawn and adds to them, so a sweep of the count
  // reads as one family of forms filling out rather than a new one every frame.
  const random = seededRandom(Math.round(seed) ^ hashName(nodeName));
  const indices = Array.from({ length: channelCount }, (_, i) => i);
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(random() * (channelCount - i));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const picked = indices.slice(0, take);
  cache.set(key, picked);
  return picked;
}
