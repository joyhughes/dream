export type Mode = 'deepdream' | 'style';

export interface DreamLayerTarget {
  nodeName: string;
  label: string;
  weight: number;
}

export interface DreamPreset {
  id: string;
  name: string;
  description: string;
  layers: DreamLayerTarget[];
}

export interface DreamParams {
  octaves: number;
  octaveScale: number;
  stepsPerOctave: number;
  stepSize: number;
}

export interface StyleParams {
  imageSize: number;
  contentWeight: number;
  styleWeight: number;
  totalVariationWeight: number;
  learningRate: number;
  octaves: number;
  octaveScale: number;
  stepsPerOctave: number;
}

export interface ProgressUpdate {
  step: number;
  totalSteps: number;
  canvas: HTMLCanvasElement;
}

export type EngineStatus =
  | { phase: 'idle' }
  | { phase: 'loading-model' }
  | { phase: 'ready' }
  | { phase: 'running'; step: number; totalSteps: number }
  | { phase: 'done' }
  | { phase: 'error'; message: string };
