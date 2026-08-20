import { tf } from './tfSetup';

export type FeatureNetworkId = 'mobilenet' | 'vgg19';

export interface DiscoveredLayer {
  nodeName: string;
  depthIndex: number;
}

/**
 * A frozen classifier used purely as a feature extractor, behind an interface that hides how it wants
 * its input and how its intermediate activations are addressed. MobileNet is a fixed-224 graph model
 * addressed by graph node name; VGG-19 is a fully-convolutional layers model addressed by Keras layer
 * name and fed near-native-resolution tiles. Everything downstream (DeepDream, style transfer) only
 * needs "preprocess this tile" and "give me these activations".
 */
export interface FeatureModel {
  id: FeatureNetworkId;
  /** Usable activation layers, ordered shallow to deep. */
  layers: DiscoveredLayer[];
  /**
   * Layers this network is specifically known to give good Gram-matrix style statistics at. When absent,
   * style transfer falls back to sampling the layer list by depth fraction.
   */
  styleLayerNames?: string[];
  /** Layer to match content at, when the network has a well-established choice. */
  contentLayerName?: string;
  /** Turns a [0,1] HWC tile into the batched input this network expects. Must stay differentiable. */
  preprocess(image01: tf.Tensor3D): tf.Tensor4D;
  /** Runs the network and returns the requested intermediate activations. Differentiable via tf.grad. */
  activations(input: tf.Tensor4D, nodeNames: string[]): tf.Tensor[];
}

export interface FeatureNetworkOption {
  id: FeatureNetworkId;
  label: string;
  description: string;
  /** Rough download size, shown in the UI so the cost of switching is not a surprise. */
  downloadLabel: string;
}

export const FEATURE_NETWORKS: FeatureNetworkOption[] = [
  {
    id: 'mobilenet',
    label: 'MobileNet V2',
    description: 'Fast. Good for DeepDream; style transfer comes out softer and blotchier.',
    downloadLabel: '~14 MB',
  },
  {
    id: 'vgg19',
    label: 'VGG-19',
    description: 'The network classic style transfer was built on. Much stronger style, several times slower.',
    downloadLabel: '~80 MB',
  },
];
