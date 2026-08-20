import type { FeatureModel, FeatureNetworkId } from './featureModel';
import { loadMobilenetFeatureModel } from './mobilenetFeatures';
import { loadVgg19FeatureModel } from './vgg19Features';

const loadPromises = new Map<FeatureNetworkId, Promise<FeatureModel>>();

/**
 * Loads (and thereafter reuses) a feature network. VGG-19 in particular is an ~80 MB download, so it is
 * only fetched if the user actually selects it — and a failed load drops the cached promise so selecting
 * it again retries rather than replaying the error forever.
 */
export function loadFeatureModel(id: FeatureNetworkId): Promise<FeatureModel> {
  let promise = loadPromises.get(id);
  if (!promise) {
    promise = (id === 'vgg19' ? loadVgg19FeatureModel() : loadMobilenetFeatureModel()).catch((err) => {
      loadPromises.delete(id);
      throw err;
    });
    loadPromises.set(id, promise);
  }
  return promise;
}

export function isFeatureModelLoaded(id: FeatureNetworkId): boolean {
  return loadPromises.has(id);
}
