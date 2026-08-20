import type { DreamPreset } from '../types';
import type { DiscoveredLayer } from './featureModel';

interface PresetSpec {
  id: string;
  name: string;
  description: string;
  /** Fraction (0-1) into the sorted activation-layer list this preset targets. */
  depthFraction: number;
  /** How many consecutive discovered layers around that point to blend together. */
  span: number;
}

const PRESET_SPECS: PresetSpec[] = [
  { id: 'fine-textures', name: 'Fine Textures', description: 'Early layers: edges, grain, and color texture.', depthFraction: 0.12, span: 2 },
  { id: 'patterns', name: 'Patterns & Ripples', description: 'Low-mid layers: repeating patterns and local structure.', depthFraction: 0.32, span: 2 },
  { id: 'organic-forms', name: 'Organic Forms', description: 'Mid layers: swirling, organic, wave-like forms.', depthFraction: 0.52, span: 2 },
  { id: 'structures', name: 'Complex Structures', description: 'Mid-high layers: architectural and compound shapes.', depthFraction: 0.72, span: 2 },
  { id: 'abstract', name: 'Abstract Objects', description: 'Late layers: high-level, abstract, object-like forms.', depthFraction: 0.9, span: 2 },
];

/** Builds the DeepDream preset list from whatever activation layers were actually discovered in the loaded model. */
export function buildPresets(layers: DiscoveredLayer[]): DreamPreset[] {
  const sorted = [...layers].sort((a, b) => a.depthIndex - b.depthIndex);
  const n = sorted.length;

  return PRESET_SPECS.map((spec) => {
    const center = Math.min(n - 1, Math.max(0, Math.round(spec.depthFraction * (n - 1))));
    const start = Math.max(0, Math.min(n - spec.span, center - Math.floor(spec.span / 2)));
    const picked = sorted.slice(start, start + spec.span);

    return {
      id: spec.id,
      name: spec.name,
      description: spec.description,
      layers: picked.map((layer) => ({ nodeName: layer.nodeName, label: layer.nodeName, weight: 1 })),
    };
  });
}
