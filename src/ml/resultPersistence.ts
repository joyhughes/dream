const DB_NAME = 'dream-app';
const STORE_NAME = 'last-result';
const RESULT_KEY = 'latest';
const BASE_KEY = 'base-image';

/**
 * A phone's photo is a few megabytes; a phone's video is not. Persisting a large base file would spend
 * seconds of IndexedDB write time on every pick, so files past this size are simply not kept — losing
 * the restore for a video is a much smaller cost than stalling the picker.
 */
const MAX_PERSISTED_BASE_BYTES = 32 * 1024 * 1024;

interface PersistedBaseImage {
  blob: Blob;
  name: string;
  type: string;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function put(key: string, value: unknown): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

async function del(key: string): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

async function get<T>(key: string): Promise<T | null> {
  const db = await openDb();
  try {
    return await new Promise<T | null>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve((req.result as T | undefined) ?? null);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

/**
 * Persists the last completed result as a plain PNG blob, independent of any WebGPU/tensor state.
 * A system sleep can reset the browser's GPU process, which wipes any GPU-composited canvas and can
 * invalidate the WebGPU device outright — this survives that (and a full tab reload/discard) since it's
 * just bytes in IndexedDB, not something the GPU owns.
 */
export async function saveLastResultBlob(blob: Blob): Promise<void> {
  try {
    await put(RESULT_KEY, blob);
  } catch (err) {
    console.warn('Failed to persist last result:', err);
  }
}

export async function loadLastResultBlob(): Promise<Blob | null> {
  try {
    return await get<Blob>(RESULT_KEY);
  } catch (err) {
    console.warn('Failed to load last persisted result:', err);
    return null;
  }
}

/**
 * Forgets the stored result. Called whenever a new base image is picked: a result only means anything
 * next to the image it was made from, and a result restored beside a different image reads as the app
 * having thrown away the picture the user was working on.
 */
export async function clearLastResult(): Promise<void> {
  try {
    await del(RESULT_KEY);
  } catch (err) {
    console.warn('Failed to clear last persisted result:', err);
  }
}

/**
 * Persists the image being worked on. iOS discards and reloads a backgrounded tab far more readily than
 * a desktop browser does, and a `File` handed over by the picker does not survive that reload — without
 * this the tab comes back with an empty dropzone.
 */
export async function saveBaseImage(file: File): Promise<void> {
  if (file.size > MAX_PERSISTED_BASE_BYTES) {
    await clearBaseImage();
    return;
  }
  try {
    const record: PersistedBaseImage = { blob: file.slice(), name: file.name, type: file.type };
    await put(BASE_KEY, record);
  } catch (err) {
    console.warn('Failed to persist base image:', err);
  }
}

export async function loadBaseImage(): Promise<File | null> {
  try {
    const record = await get<PersistedBaseImage>(BASE_KEY);
    if (!record?.blob) return null;
    return new File([record.blob], record.name, { type: record.type || record.blob.type });
  } catch (err) {
    console.warn('Failed to load persisted base image:', err);
    return null;
  }
}

export async function clearBaseImage(): Promise<void> {
  try {
    await del(BASE_KEY);
  } catch (err) {
    console.warn('Failed to clear persisted base image:', err);
  }
}
