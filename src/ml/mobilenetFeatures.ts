import * as mobilenetLib from '@tensorflow-models/mobilenet';
import { tf } from './tfSetup';
import type { DiscoveredLayer, FeatureModel } from './featureModel';

/**
 * This MobileNetV2 build (TF-Hub classification signature, loaded via @tensorflow-models/mobilenet) declares
 * a fixed graph input shape of [-1, 224, 224, 3]. Unlike a raw fully-convolutional backbone, GraphModel.execute()
 * rejects any other spatial size outright — this is not a quality/speed tradeoff knob, it cannot be changed
 * without swapping in a different model.
 */
export const MOBILENET_INPUT_SIZE = 224;

interface GraphNode {
  name: string;
  op: string;
}

interface IntrospectableGraphModel {
  executor?: { graph?: { nodes?: Record<string, GraphNode> } };
}

/**
 * Resizes a [0,1] HWC tensor into the NHWC batch MobileNet expects.
 * This build of MobileNetV2 (TF-Hub, alpha 1.0) takes its input already in [0,1], so no rescale is needed.
 * Kept as a distinct differentiable step (not a no-op passthrough) so the gradient path is explicit.
 */
export function preprocessForMobilenet(image01: tf.Tensor3D): tf.Tensor4D {
  return tf.tidy(() => {
    const resized = tf.image.resizeBilinear(image01, [MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE]);
    return resized.expandDims(0) as tf.Tensor4D;
  });
}

export async function loadMobilenetFeatureModel(): Promise<FeatureModel> {
  const net = await mobilenetLib.load({ version: 2, alpha: 1.0 });
  const graphModel = (net as unknown as { model: tf.GraphModel }).model;

  const nodes = (graphModel as unknown as IntrospectableGraphModel).executor?.graph?.nodes ?? {};
  const names = Object.keys(nodes);

  const activationNames = names.filter((name) => {
    const op = nodes[name]?.op ?? '';
    return /relu/i.test(name) || /relu/i.test(op);
  });

  if (activationNames.length === 0) {
    throw new Error('Could not discover intermediate activation layers in the MobileNet graph.');
  }

  const layers: DiscoveredLayer[] = activationNames.map((nodeName, depthIndex) => ({ nodeName, depthIndex }));

  return {
    id: 'mobilenet',
    layers,
    preprocess: preprocessForMobilenet,
    activations(input, nodeNames) {
      const result = graphModel.execute(input, nodeNames);
      return Array.isArray(result) ? result : [result];
    },
  };
}
