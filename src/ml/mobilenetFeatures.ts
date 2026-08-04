import * as mobilenetLib from '@tensorflow-models/mobilenet';
import { tf } from './tfSetup';

export interface DiscoveredLayer {
  nodeName: string;
  depthIndex: number;
}

export interface FeatureModel {
  graphModel: tf.GraphModel;
  layers: DiscoveredLayer[];
}

interface GraphNode {
  name: string;
  op: string;
}

interface IntrospectableGraphModel {
  executor?: { graph?: { nodes?: Record<string, GraphNode> } };
}

let loadPromise: Promise<FeatureModel> | null = null;

async function loadInternal(): Promise<FeatureModel> {
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

  return { graphModel, layers };
}

export function loadFeatureModel(): Promise<FeatureModel> {
  if (!loadPromise) {
    loadPromise = loadInternal();
  }
  return loadPromise;
}

/** Runs the frozen graph and returns the requested intermediate activations. Differentiable via tf.grad/variableGrads. */
export function getActivations(graphModel: tf.GraphModel, input: tf.Tensor4D, nodeNames: string[]): tf.Tensor[] {
  const result = graphModel.execute(input, nodeNames);
  return Array.isArray(result) ? result : [result];
}
