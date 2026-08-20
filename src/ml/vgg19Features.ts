import { tf } from './tfSetup';
import type { DiscoveredLayer, FeatureModel } from './featureModel';

/**
 * Keras' `VGG19(include_top=False)` converted to a TF.js layers model — the convolutional trunk only
 * (~80 MB), with the 400 MB of fully-connected classifier weights left off since feature extraction
 * never touches them. Pinned to a commit rather than a branch so the weights can't change underneath us,
 * and served through jsDelivr, which fronts GitHub with CORS headers and a CDN cache (raw.githubusercontent
 * is neither cached nor intended for this).
 */
const VGG19_MODEL_URL =
  'https://cdn.jsdelivr.net/gh/paulsp94/tfjs_vgg19_imagenet@ea6bea2ac90e492592b552346e02abcb0eafa443/model/model.json';

/**
 * Unlike MobileNet, VGG-19 is fully convolutional here (declared input shape [null, null, null, 3]), so
 * tiles can be fed at close to native resolution instead of being squashed to a fixed 224. That is a real
 * quality gain — no resampling between the tile and what the network scores — but activation memory grows
 * with the square of the input, and VGG's early blocks are enormous (a 320x320 input already holds ~37 MB in
 * block1_conv1 alone, before the gradient tape doubles it). This cap keeps a tile's forward+backward pass
 * inside a few hundred MB; larger tiles get downsampled to it, exactly as they would for MobileNet.
 */
const VGG19_MAX_INPUT_SIZE = 320;

/** Keras' VGG preprocessing ("caffe" mode): [0,1] RGB to [0,255] BGR, minus the ImageNet channel means. */
const VGG_MEAN_BGR = [103.939, 116.779, 123.68];

/**
 * The layer set from Gatys et al., which is what the classic style-transfer tools (Dreamscope among them)
 * used. One convolution from each block: block1 carries colour and fine grain, block5 carries large
 * compositional structure, and the Gram matrices of all five together are what make the style read as a
 * coherent medium rather than a texture swatch.
 */
const VGG19_STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'];

/** Also from Gatys et al.: deep enough to pin down layout without dictating local texture. */
const VGG19_CONTENT_LAYER = 'block4_conv2';

const VGG19_CONV_LAYERS = [
  'block1_conv1', 'block1_conv2',
  'block2_conv1', 'block2_conv2',
  'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
  'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
  'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4',
];

function preprocessForVgg19(image01: tf.Tensor3D): tf.Tensor4D {
  return tf.tidy(() => {
    const [h, w] = image01.shape;
    const longest = Math.max(h, w);
    const scaled =
      longest > VGG19_MAX_INPUT_SIZE
        ? (tf.image.resizeBilinear(image01, [
            Math.max(1, Math.round((h * VGG19_MAX_INPUT_SIZE) / longest)),
            Math.max(1, Math.round((w * VGG19_MAX_INPUT_SIZE) / longest)),
          ]) as tf.Tensor3D)
        : image01;

    const bgr = tf.reverse(scaled.mul(255), -1).sub(VGG_MEAN_BGR);
    return bgr.expandDims(0) as tf.Tensor4D;
  });
}

export async function loadVgg19FeatureModel(): Promise<FeatureModel> {
  const model = await tf.loadLayersModel(VGG19_MODEL_URL);

  // One sub-model per distinct set of requested layers. VGG is a straight chain, so a sub-model that stops
  // at the deepest layer asked for skips everything past it; caching by name set means that sub-model is
  // built once instead of on every tile of every step.
  const subModels = new Map<string, tf.LayersModel>();
  const subModelFor = (nodeNames: string[]): tf.LayersModel => {
    const key = nodeNames.join('|');
    let sub = subModels.get(key);
    if (!sub) {
      const outputs = nodeNames.map((name) => model.getLayer(name).output as tf.SymbolicTensor);
      sub = tf.model({ inputs: model.inputs, outputs });
      subModels.set(key, sub);
    }
    return sub;
  };

  const layers: DiscoveredLayer[] = VGG19_CONV_LAYERS.map((nodeName, depthIndex) => ({ nodeName, depthIndex }));

  return {
    id: 'vgg19',
    layers,
    styleLayerNames: VGG19_STYLE_LAYERS,
    contentLayerName: VGG19_CONTENT_LAYER,
    preprocess: preprocessForVgg19,
    activations(input, nodeNames) {
      const result = subModelFor(nodeNames).predict(input);
      return Array.isArray(result) ? result : [result];
    },
  };
}
