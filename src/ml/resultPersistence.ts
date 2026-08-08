const DB_NAME = 'dream-app';
const STORE_NAME = 'last-result';
const KEY = 'latest';

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

/**
 * Persists the last completed result as a plain PNG blob, independent of any WebGPU/tensor state.
 * A system sleep can reset the browser's GPU process, which wipes any GPU-composited canvas and can
 * invalidate the WebGPU device outright — this survives that (and a full tab reload/discard) since it's
 * just bytes in IndexedDB, not something the GPU owns.
 */
export async function saveLastResultBlob(blob: Blob): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(blob, KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (err) {
    console.warn('Failed to persist last result:', err);
  }
}

export async function loadLastResultBlob(): Promise<Blob | null> {
  try {
    const db = await openDb();
    const blob = await new Promise<Blob | null>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(KEY);
      req.onsuccess = () => resolve((req.result as Blob | undefined) ?? null);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return blob;
  } catch (err) {
    console.warn('Failed to load last persisted result:', err);
    return null;
  }
}
