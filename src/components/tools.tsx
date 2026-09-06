/* eslint-disable react-refresh/only-export-components -- the tool table and its icons belong together;
   splitting them to satisfy fast refresh would put a component's identity in one file and its icon in
   another for no benefit at this size. */
import type { ReactNode } from 'react';
import type { ToolId } from '../types';

/**
 * What a click does to the existing selection. Held here rather than read off the keyboard at each call
 * site so the panel, the cursor and the tool handlers cannot drift apart on what a modifier means.
 */
export type SelectionMode = 'replace' | 'add' | 'subtract';

/**
 * Option subtracts, shift adds, nothing replaces — the convention every image editor uses, so the keys
 * already in someone's fingers do what they expect here. Option wins when both are down, on the grounds
 * that taking something out is the more deliberate of the two.
 */
export function selectionModeFor(shiftKey: boolean, altKey: boolean): SelectionMode {
  if (altKey) return 'subtract';
  if (shiftKey) return 'add';
  return 'replace';
}

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

function OffIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <path d="M5 12h14" />
    </svg>
  );
}

function PaintIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <path d="M15.5 4.5l4 4L10 18l-5 1 1-5z" />
      <path d="M13.5 6.5l4 4" />
    </svg>
  );
}

function WandIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <path d="M5 19L15 9" />
      <path d="M14 4.5l1 2.5 2.5 1-2.5 1-1 2.5-1-2.5L10.5 8l2.5-1z" />
      <path d="M19.5 13.5l.6 1.4 1.4.6-1.4.6-.6 1.4-.6-1.4-1.4-.6 1.4-.6z" />
    </svg>
  );
}

function BucketIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <path d="M11 3l8 8-7 7-8-8z" />
      <path d="M4 12h14" />
      <path d="M20.5 15.5c1 1.4 1.5 2.3 1.5 3a1.5 1.5 0 0 1-3 0c0-.7.5-1.6 1.5-3z" />
    </svg>
  );
}

function LassoIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <path d="M12 4c4.4 0 8 2.5 8 5.5S16.4 15 12 15 4 12.5 4 9.5 7.6 4 12 4z" />
      <path d="M8 14.5c-.7 1.6-.4 3 .8 3.6 1 .5 1.9.1 2.2-.7.3-.8-.2-1.5-.9-1.5-.9 0-1.4.9-1.1 2 .3 1.2 1.4 2.1 2.9 2.1" />
    </svg>
  );
}

function SelectBrushIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke} aria-hidden="true">
      <circle cx="12" cy="12" r="6.5" strokeDasharray="3 2.5" />
      <path d="M12 9v6M9 12h6" />
    </svg>
  );
}

export interface ToolDefinition {
  id: ToolId;
  label: string;
  hint: string;
  icon: ReactNode;
  /** Whether the selection modifiers mean anything for this tool. Paint draws the effect, not a selection. */
  usesSelectionModes: boolean;
}

export const TOOLS: ToolDefinition[] = [
  {
    id: 'none',
    label: 'Off',
    hint: 'The image is just an image; the pointer does nothing to it.',
    icon: <OffIcon />,
    usesSelectionModes: false,
  },
  {
    id: 'paint',
    label: 'Paint',
    hint: 'Press and hold to build the effect up under the cursor — it keeps iterating for as long as you hold — and drag to paint a stroke. Confined to the selection when there is one.',
    icon: <PaintIcon />,
    usesSelectionModes: false,
  },
  {
    id: 'wand',
    label: 'Magic wand',
    hint: 'Click a pixel to select everything of a similar color, spreading out from where you clicked.',
    icon: <WandIcon />,
    usesSelectionModes: true,
  },
  {
    id: 'bucket',
    label: 'Paint bucket',
    hint: 'The wand and Apply in one click: finds the region under the cursor and floods the effect into it. With a modifier held it only changes the selection, and applies nothing.',
    icon: <BucketIcon />,
    usesSelectionModes: true,
  },
  {
    id: 'lasso',
    label: 'Lasso',
    hint: 'Drag to draw a freehand outline; releasing closes it and takes what is inside.',
    icon: <LassoIcon />,
    usesSelectionModes: true,
  },
  {
    id: 'select-brush',
    label: 'Selection brush',
    hint: 'Paint the selection on by hand.',
    icon: <SelectBrushIcon />,
    usesSelectionModes: true,
  },
];

export function toolDefinition(id: ToolId): ToolDefinition {
  return TOOLS.find((tool) => tool.id === id) ?? TOOLS[0];
}

/** The line under the tool buttons that says what the modifiers will do with the tool now chosen. */
export function modifierHint(id: ToolId): string | null {
  if (!toolDefinition(id).usesSelectionModes) return null;
  return 'No key replaces the selection · shift adds · option subtracts';
}
