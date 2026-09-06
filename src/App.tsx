import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ImageDropzone } from './components/ImageDropzone';
import { BuiltInTemplatePicker } from './components/BuiltInTemplatePicker';
import { ModeTabs } from './components/ModeTabs';
import {
  PresetPanel,
  SliderPanel,
  VideoOptionsPanel,
  ActionsBar,
  RegularizerPanel,
  BrushPanel,
  SelectionPanel,
  ParametersFromImage,
} from './components/ControlsPanel';
import { ResultCanvas } from './components/ResultCanvas';
import { HoverPopup } from './components/HoverPopup';
import { initializeML, ensureBackendHealthy } from './ml/tfSetup';
import { loadFeatureModel } from './ml/featureModels';
import type { FeatureModel, FeatureNetworkId } from './ml/featureModel';
import { buildPresets } from './ml/presets';
import { runDeepDream } from './ml/deepdream';
import { runStyleTransfer } from './ml/styleTransfer';
import { imageToWorkingTensor, loadImageFromFile, renderTensorToCanvas, workingDimensions } from './ml/imageUtils';
import { getDeviceLimits, maxFramesInStore } from './ml/deviceLimits';
import {
  clearLastResult,
  loadBaseImage,
  loadLastResultBlob,
  saveBaseImage,
  saveLastResultBlob,
} from './ml/resultPersistence';
import { canvasToPngBytes, downloadBlob, saveImage } from './ml/imageSaving';
import { isPng } from './ml/pngText';
import { PauseController } from './ml/pauseController';
import { MovieRecorder, isMovieRecordingSupported } from './ml/movieRecorder';
import { encodeFrameSequence } from './ml/frameEncoding';
import { VideoFrameSource } from './ml/videoFrames';
import { DreamBrush } from './ml/brush';
import {
  couldCarryParameters,
  embedParameters,
  readEmbeddedParameters,
  type SavedParameters,
} from './ml/imageMetadata';
import { SelectionMask } from './ml/selection';
import { BUILT_IN_TEMPLATES } from './templates/builtInTemplates';
import type { CanvasTool } from './components/ResultCanvas';
import type { SelectionMode } from './components/tools';
import type {
  BrushSettings,
  SelectionSettings,
  ToolId,
  DreamParams,
  DreamPreset,
  EngineStatus,
  ImageRegularizers,
  Mode,
  StyleParams,
} from './types';
import { AppFrame, ControlGroup } from './simui';
import { FeatureNetworkPicker } from './components/FeatureNetworkPicker';
import { tf } from './ml/tfSetup';

const DEFAULT_TEMPLATE_ID = 'paisley-color';

/**
 * How often an in-progress run stores what's on the canvas. A run that dies in its last moments — an iOS
 * tab kill leaves nothing to catch — otherwise loses everything, and a frame a few steps short of the end
 * is worth far more than an empty viewport after the reload.
 */
const PROGRESS_PERSIST_INTERVAL_MS = 30_000;

/**
 * A GPU that runs out of memory or gets reset mid-run surfaces as whatever low-level call happened to be
 * in flight — on iOS that's Safari's "map async not successful" from a failed buffer readback, which tells
 * the person looking at it nothing about what to do next. The backend is rebuilt on the next run by
 * `ensureBackendHealthy`, so this only needs to explain what happened and which knobs lower the cost.
 */
function describeRunError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);

  if (/map async|device (is )?lost|out of memory|GPUDevice|createBuffer/i.test(message)) {
    return 'The GPU ran out of memory or was reset partway through. Try a smaller tile size or fewer octaves, then Generate again.';
  }

  return message;
}

// 320 where there is room for it, less on a phone — see `deviceLimits`.
const DEFAULT_TILE_SIZE = Math.min(320, getDeviceLimits().maxTileSize);

/**
 * Every regularizer defaults to off, so a fresh session reproduces exactly what this app produced before
 * they existed. They are opt-in tools, not a new house style.
 */
const NO_REGULARIZERS: ImageRegularizers = {
  l2Decay: 0,
  blurSigma: 0,
  blurEvery: 0,
};

const DEFAULT_DREAM_PARAMS: DreamParams = {
  octaves: 3,
  octaveScale: 1.4,
  stepsPerOctave: 20,
  stepSize: 0.02,
  tileSize: DEFAULT_TILE_SIZE,
  tvWeight: 0,
  lapLevels: 1,
  colorSpace: 'rgb',
  colorPreservation: 0,
  normalizeSaturation: false,
  normalizeBrightness: false,
  patternScale: 1,
  regularizers: NO_REGULARIZERS,
};

const DEFAULT_STYLE_PARAMS: StyleParams = {
  contentWeight: 8,
  styleWeight: 400,
  totalVariationWeight: 1,
  learningRate: 0.015,
  octaves: 3,
  octaveScale: 1.4,
  stepsPerOctave: 40,
  tileSize: DEFAULT_TILE_SIZE,
  colorSpace: 'rgb',
  colorPreservation: 0,
  normalizeSaturation: false,
  normalizeBrightness: false,
  patternScale: 1,
  regularizers: NO_REGULARIZERS,
};

const DEFAULT_BRUSH: BrushSettings = {
  radius: 64,
  feather: 0.5,
  stepsPerDab: 4,
};

const DEFAULT_SELECTION: SelectionSettings = {
  tolerance: 0.15,
  contiguous: true,
  feather: 12,
  brushRadius: 40,
};

function App() {
  const [mode, setMode] = useState<Mode>('deepdream');

  const [initError, setInitError] = useState<string | null>(null);
  const [featureModel, setFeatureModel] = useState<FeatureModel | null>(null);
  const [featureNetworkId, setFeatureNetworkId] = useState<FeatureNetworkId>('mobilenet');
  const [presets, setPresets] = useState<DreamPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');

  const [baseFile, setBaseFile] = useState<File | null>(null);
  const [basePreviewUrl, setBasePreviewUrl] = useState<string>();
  const [templateFile, setTemplateFile] = useState<File | null>(null);
  const [templatePreviewUrl, setTemplatePreviewUrl] = useState<string>();

  const [dreamParams, setDreamParams] = useState<DreamParams>(DEFAULT_DREAM_PARAMS);
  const [styleParams, setStyleParams] = useState<StyleParams>(DEFAULT_STYLE_PARAMS);

  const [engineStatus, setEngineStatus] = useState<EngineStatus>({ phase: 'loading-model' });
  const [hasResult, setHasResult] = useState(false);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [recordMovie, setRecordMovie] = useState(false);
  const [isRecordingMovie, setIsRecordingMovie] = useState(false);
  const [videoFps, setVideoFps] = useState(8);
  const [frameProgress, setFrameProgress] = useState<{ index: number; total: number } | null>(null);

  const [tool, setTool] = useState<ToolId>('none');
  const [brushSettings, setBrushSettings] = useState<BrushSettings>(DEFAULT_BRUSH);
  const [selectionSettings, setSelectionSettings] = useState<SelectionSettings>(DEFAULT_SELECTION);
  const [isPainting, setIsPainting] = useState(false);
  // Parameters found inside the picked image, waiting for the user to say whether to use them.
  const [offeredParameters, setOfferedParameters] = useState<SavedParameters | null>(null);
  const [parametersApplied, setParametersApplied] = useState(false);
  // Bumped whenever the mask changes, purely to drive a redraw of the overlay and the panel's summary —
  // the mask itself lives in a ref, since it is written pixel by pixel and must not clone on every stroke.
  const [selectionVersion, setSelectionVersion] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pauseControllerRef = useRef<PauseController | null>(null);
  const movieRecorderRef = useRef<MovieRecorder | null>(null);
  const lastProgressPersistRef = useRef(0);

  // The image being painted on, and the pointer state driving it. All refs: the paint loop reads these
  // every tick and must see the current values, not the ones captured when the stroke began.
  const brushRef = useRef<DreamBrush | null>(null);
  const brushTemplateRef = useRef<tf.Tensor3D | null>(null);
  const paintingRef = useRef(false);
  const pointerRef = useRef<{ x: number; y: number } | null>(null);
  const brushSettingsRef = useRef(brushSettings);
  brushSettingsRef.current = brushSettings;
  const selectionSettingsRef = useRef(selectionSettings);
  selectionSettingsRef.current = selectionSettings;
  const isRunningRef = useRef(false);

  const overlayRef = useRef<HTMLCanvasElement>(null);
  // Scratch canvas the in-progress patch is rendered to before being drawn into the main one.
  const patchPreviewRef = useRef<HTMLCanvasElement | null>(null);
  const selectionRef = useRef<SelectionMask | null>(null);
  const lassoPointsRef = useRef<Array<{ x: number; y: number }>>([]);
  // The lasso only acts when the stroke ends, so whether it adds or subtracts is decided when the stroke
  // begins and held — releasing shift midway through drawing should not change what the outline does.
  const lassoModeRef = useRef<SelectionMode>('replace');
  // A stroke's mode is fixed when it starts. Re-reading the keys mid-stroke would let a plain stroke
  // clear itself on every move, and would change what a lasso means after most of it had been drawn.
  const strokeModeRef = useRef<SelectionMode>('replace');
  const toolRef = useRef<ToolId>('none');
  toolRef.current = tool;
  const paintContextRef = useRef({ mode, dreamParams, styleParams, presets, selectedPresetId, featureModel });
  paintContextRef.current = { mode, dreamParams, styleParams, presets, selectedPresetId, featureModel };
  // Whether the user has picked their own image yet, which decides who wins if the restore below
  // finishes after they've already moved on.
  const hasOwnBaseImageRef = useRef(false);

  // Re-runs whenever the selected feature network changes: the presets are derived from whichever
  // network's layers are loaded, so they have to be rebuilt alongside it. Preset ids are stable across
  // networks, so an existing selection survives the switch.
  useEffect(() => {
    let cancelled = false;

    setFeatureModel(null);
    setInitError(null);
    setEngineStatus({ phase: 'loading-model' });

    (async () => {
      try {
        await initializeML();
        if (cancelled) return;

        const model = await loadFeatureModel(featureNetworkId);
        if (cancelled) return;
        setFeatureModel(model);

        const builtPresets = buildPresets(model.layers);
        setPresets(builtPresets);
        setSelectedPresetId((current) =>
          builtPresets.some((preset) => preset.id === current) ? current : builtPresets[0]?.id ?? '',
        );
        setEngineStatus({ phase: 'idle' });
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setInitError(message);
        setEngineStatus({ phase: 'error', message });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [featureNetworkId]);

  // iOS discards and reloads a backgrounded tab far more readily than a desktop browser does, and the
  // picked `File` doesn't survive that. Both halves are restored together or not at all: a result on its
  // own, with no image beside it, reads as the app having lost the picture and gone back to an older one.
  useEffect(() => {
    (async () => {
      const base = await loadBaseImage();
      if (!base) return;
      const blob = await loadLastResultBlob();

      // These reads take long enough on a phone that the user can pick an image first. Theirs wins.
      if (hasOwnBaseImageRef.current) return;

      setBaseFile(base);
      setBasePreviewUrl(URL.createObjectURL(base));
      if (blob) {
        setResultBlob(blob);
        setHasResult(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (!resultBlob) {
      setResultImageUrl(null);
      return;
    }
    const url = URL.createObjectURL(resultBlob);
    setResultImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [resultBlob]);

  useEffect(() => {
    return () => {
      if (basePreviewUrl) URL.revokeObjectURL(basePreviewUrl);
    };
  }, [basePreviewUrl]);

  useEffect(() => {
    return () => {
      if (templatePreviewUrl) URL.revokeObjectURL(templatePreviewUrl);
    };
  }, [templatePreviewUrl]);

  const handleBaseFile = useCallback((file: File) => {
    hasOwnBaseImageRef.current = true;
    // A new photo means the painted image, and any selection over it, are of the wrong thing entirely.
    brushRef.current?.dispose();
    brushRef.current = null;
    selectionRef.current = null;
    setBaseFile(file);
    setBasePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });

    // A result only means anything next to the image it came from. Dropping the previous one here is
    // what stops an older run's output from sitting in the viewport on top of a freshly picked image.
    setResultBlob(null);
    setHasResult(false);
    setEngineStatus((status) => (status.phase === 'done' ? { phase: 'idle' } : status));
    void clearLastResult();
    void saveBaseImage(file);

    // An image saved from here carries the settings it was made with. Read them, but do not apply them —
    // silently rewriting every slider because of what a file contained would be its own kind of surprise.
    setOfferedParameters(null);
    setParametersApplied(false);
    if (couldCarryParameters(file)) {
      void (async () => {
        try {
          const bytes = new Uint8Array(await file.arrayBuffer());
          const found = readEmbeddedParameters(bytes, parametersRef.current);
          if (found) setOfferedParameters(found);
        } catch (err) {
          console.warn('Could not read parameters from the image:', err);
        }
      })();
    }
  }, []);

  const applyOfferedParameters = useCallback(() => {
    const found = offeredParameters;
    if (!found) return;

    setMode(found.mode);
    setFeatureNetworkId(found.featureNetworkId);
    setDreamParams(found.dream);
    setStyleParams(found.style);
    // Set last, and left as-is if the network's preset list turns out not to contain it: switching networks
    // rebuilds that list asynchronously, and it drops an unknown id on its own.
    setSelectedPresetId(found.presetId);
    setParametersApplied(true);
  }, [offeredParameters]);

  const handleTemplateFile = useCallback((file: File) => {
    brushTemplateRef.current?.dispose();
    brushTemplateRef.current = null;
    setTemplateFile(file);
    setTemplatePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const defaultTemplate = BUILT_IN_TEMPLATES.find((t) => t.id === DEFAULT_TEMPLATE_ID);
      if (!defaultTemplate) return;
      const file = await defaultTemplate.getFile();
      if (cancelled) return;
      handleTemplateFile(file);
    })();

    return () => {
      cancelled = true;
    };
  }, [handleTemplateFile]);

  const persistProgressSnapshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const now = Date.now();
    if (now - lastProgressPersistRef.current < PROGRESS_PERSIST_INTERVAL_MS) return;
    lastProgressPersistRef.current = now;

    canvas.toBlob((blob) => {
      if (blob) void saveLastResultBlob(blob);
    }, 'image/png');
  }, []);

  const isBaseVideo = !!baseFile && baseFile.type.startsWith('video/');

  /** Everything worth carrying inside a saved picture. */
  const currentParameters = useMemo<SavedParameters>(
    () => ({
      mode,
      featureNetworkId,
      presetId: selectedPresetId,
      dream: dreamParams,
      style: styleParams,
    }),
    [mode, featureNetworkId, selectedPresetId, dreamParams, styleParams],
  );
  const parametersRef = useRef(currentParameters);
  parametersRef.current = currentParameters;

  /**
   * Writes the settings into a PNG. Synchronous on purpose: a save has to reach iOS's share sheet inside
   * the click that asked for it, and awaiting anything spends the activation that allows it.
   */
  const stampParameters = useCallback((bytes: Uint8Array): Blob => {
    const stamped = isPng(bytes) ? embedParameters(bytes, parametersRef.current) : bytes;
    return new Blob([stamped], { type: 'image/png' });
  }, []);

  /** The same, for a blob whose bytes we do not already have. Used off the save path, where waiting is fine. */
  const stampParametersAsync = useCallback(
    async (blob: Blob): Promise<Blob> => {
      try {
        return stampParameters(new Uint8Array(await blob.arrayBuffer()));
      } catch (err) {
        console.warn('Could not write parameters into the image:', err);
        return blob;
      }
    },
    [stampParameters],
  );

  /** Throws the painted image away, so the next stroke starts from whatever the app is showing now. */
  const discardBrush = useCallback(() => {
    brushRef.current?.dispose();
    brushRef.current = null;
    brushTemplateRef.current?.dispose();
    brushTemplateRef.current = null;
  }, []);

  /**
   * The image the brush paints on: the last result if there is one, otherwise the picked photo. Built once
   * per painting session and kept until something replaces the image underneath it.
   */
  const ensureBrush = useCallback(async (): Promise<DreamBrush | null> => {
    if (brushRef.current) return brushRef.current;
    if (!baseFile || isBaseVideo) return null;

    const source = resultBlob ? new File([resultBlob], 'result.png', { type: 'image/png' }) : baseFile;
    const img = await loadImageFromFile(source);
    const tensor = imageToWorkingTensor(img, getDeviceLimits().workingMaxDimension);
    try {
      brushRef.current = new DreamBrush(tensor);
    } finally {
      tensor.dispose();
    }

    if (canvasRef.current) {
      await renderTensorToCanvas(brushRef.current.current, canvasRef.current);
    }
    return brushRef.current;
  }, [baseFile, isBaseVideo, resultBlob]);

  /** Runs the current mode over one brush patch. Whatever Generate would do, at the size of a dab. */
  const processBrushPatch = useCallback(async (patch: tf.Tensor3D): Promise<tf.Tensor3D> => {
    const context = paintContextRef.current;
    const steps = Math.max(1, Math.round(brushSettingsRef.current.stepsPerDab));
    if (!context.featureModel) throw new Error('Feature model is not loaded.');

    // One octave, always: the brush's scale control is pattern scale, which sets the size of what gets
    // drawn directly. An octave pyramid inside a dab would spread detail across scales the user did not ask
    // for, and cost several passes per tick where the budget is one frame.
    if (context.mode === 'deepdream') {
      const preset = context.presets.find((p) => p.id === context.selectedPresetId);
      if (!preset) throw new Error('No preset selected.');
      return runDeepDream(patch, {
        featureModel: context.featureModel,
        preset,
        params: { ...context.dreamParams, octaves: 1, stepsPerOctave: steps },
      });
    }

    if (!brushTemplateRef.current) throw new Error('No style template loaded.');
    return runStyleTransfer(patch, brushTemplateRef.current, {
      featureModel: context.featureModel,
      params: { ...context.styleParams, octaves: 1, stepsPerOctave: steps },
    });
  }, []);

  /**
   * Keeps dabbing at wherever the pointer is until it lifts, so holding still builds the effect up in one
   * place while moving paints a stroke. One dab is in flight at a time — a dab takes as long as it takes,
   * and queueing them on pointer events would run further and further behind the cursor.
   */
  const runPaintLoop = useCallback(async () => {
    const brush = await ensureBrush();
    if (!brush) {
      paintingRef.current = false;
      setIsPainting(false);
      return;
    }

    try {
      while (paintingRef.current) {
        const point = pointerRef.current;
        if (!point) break;

        const selection = selectionRef.current;
        const restrictTo =
          selection && !selection.isEmpty()
            ? selection.feathered(selectionSettingsRef.current.feather)
            : undefined;
        await brush.dab(point.x, point.y, brushSettingsRef.current, processBrushPatch, restrictTo);
        if (canvasRef.current) {
          await renderTensorToCanvas(brush.current, canvasRef.current);
        }
        await tf.nextFrame();
      }
    } catch (err) {
      console.error('Painting failed:', err);
      setEngineStatus({ phase: 'error', message: describeRunError(err) });
    } finally {
      paintingRef.current = false;
      setIsPainting(false);
    }

    // The stroke is the unit of work worth keeping: this is what Download saves and what a killed tab
    // comes back to.
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.toBlob((blob) => {
        if (!blob) return;
        void stampParametersAsync(blob).then((stamped) => {
          setResultBlob(stamped);
          setHasResult(true);
          void saveLastResultBlob(stamped);
        });
      }, 'image/png');
    }
  }, [ensureBrush, processBrushPatch, stampParametersAsync]);

  const handleBrushStart = useCallback(
    (x: number, y: number) => {
      if (paintingRef.current || isRunningRef.current) return;
      pointerRef.current = { x, y };
      paintingRef.current = true;
      setIsPainting(true);
      void runPaintLoop();
    },
    [runPaintLoop],
  );

  const handleBrushMove = useCallback((x: number, y: number) => {
    if (paintingRef.current) pointerRef.current = { x, y };
  }, []);

  const handleBrushEnd = useCallback(() => {
    paintingRef.current = false;
  }, []);

  /** The selection, sized to the image being worked on. Created on first use. */
  const ensureSelection = useCallback(async (): Promise<SelectionMask | null> => {
    const brush = await ensureBrush();
    if (!brush) return null;

    const [height, width] = brush.shape;
    if (!selectionRef.current || selectionRef.current.width !== width || selectionRef.current.height !== height) {
      selectionRef.current = new SelectionMask(width, height);
    }
    return selectionRef.current;
  }, [ensureBrush]);

  const touchSelection = useCallback(() => setSelectionVersion((version) => version + 1), []);

  /**
   * The wand and the bucket both need the image's pixels on the CPU to trace a region. That is a readback,
   * but a click's worth — not something inside a loop.
   */
  const readPixels = useCallback(async (): Promise<Float32Array | null> => {
    const brush = await ensureBrush();
    return brush ? (brush.current.dataSync() as Float32Array) : null;
  }, [ensureBrush]);

  /** Runs the current mode over the selection's box, blended through its feathered edge. */
  const applyToSelection = useCallback(async () => {
    const selection = selectionRef.current;
    const brush = brushRef.current;
    if (!selection || !brush || selection.isEmpty() || paintingRef.current || isRunningRef.current) return;

    const feather = selectionSettingsRef.current.feather;
    const bounds = selection.bounds(Math.ceil(feather) + 1);
    if (!bounds) return;

    paintingRef.current = true;
    setIsPainting(true);
    try {
      const mask = selection.toTensor(bounds, feather);
      try {
        await brush.applyPatch(
          { y: bounds.y, x: bounds.x, height: bounds.height, width: bounds.width },
          mask,
          processBrushPatch,
        );
      } finally {
        mask.dispose();
      }

      if (canvasRef.current) {
        await renderTensorToCanvas(brush.current, canvasRef.current);
        canvasRef.current.toBlob((blob) => {
          if (!blob) return;
          void stampParametersAsync(blob).then((stamped) => {
            setResultBlob(stamped);
            setHasResult(true);
            void saveLastResultBlob(stamped);
          });
        }, 'image/png');
      }
    } catch (err) {
      console.error('Applying to the selection failed:', err);
      setEngineStatus({ phase: 'error', message: describeRunError(err) });
    } finally {
      paintingRef.current = false;
      setIsPainting(false);
    }
  }, [processBrushPatch, stampParametersAsync]);

  const handleToolStart = useCallback(
    (x: number, y: number, mode: SelectionMode) => {
      const active = toolRef.current;
      strokeModeRef.current = mode;

      if (active === 'paint') {
        handleBrushStart(x, y);
        return;
      }

      void (async () => {
        const selection = await ensureSelection();
        if (!selection) return;

        // Replace means exactly that: whatever was selected goes, and this gesture starts over.
        if (mode === 'replace') selection.clear();
        const subtract = mode === 'subtract';

        if (active === 'wand' || active === 'bucket') {
          const pixels = await readPixels();
          if (!pixels) return;
          const { tolerance, contiguous } = selectionSettingsRef.current;
          selection.wand(pixels, x, y, tolerance, contiguous, subtract);
          touchSelection();
          // The bucket is the wand plus the thing you were going to do next. With a modifier held the
          // gesture is about the selection, so it stops there rather than flooding anything.
          if (active === 'bucket' && mode === 'replace') await applyToSelection();
          return;
        }

        if (active === 'lasso') {
          lassoPointsRef.current = [{ x, y }];
          lassoModeRef.current = mode;
          return;
        }

        if (active === 'select-brush') {
          selection.stamp(x, y, selectionSettingsRef.current.brushRadius, subtract);
          touchSelection();
        }
      })();
    },
    [applyToSelection, ensureSelection, handleBrushStart, readPixels, touchSelection],
  );

  const handleToolMove = useCallback(
    (x: number, y: number) => {
      const active = toolRef.current;
      if (active === 'paint') {
        handleBrushMove(x, y);
        return;
      }

      if (active === 'lasso' && lassoPointsRef.current.length > 0) {
        const points = lassoPointsRef.current;
        const last = points[points.length - 1];
        // Thin the path: a point per pixel of mouse movement is thousands of segments to rasterize, and
        // two pixels of resolution is far below what the outline shows.
        if (Math.hypot(x - last.x, y - last.y) >= 2) {
          points.push({ x, y });
          touchSelection();
        }
        return;
      }

      if (active === 'select-brush' && selectionRef.current) {
        // The stroke's own mode, not whatever is held now. A replace stroke cleared once when it began
        // and adds from then on; re-reading the keys would have it wipe itself with every move.
        selectionRef.current.stamp(x, y, selectionSettingsRef.current.brushRadius, strokeModeRef.current === 'subtract');
        touchSelection();
      }
    },
    [handleBrushMove, touchSelection],
  );

  const handleToolEnd = useCallback(() => {
    const active = toolRef.current;
    if (active === 'paint') {
      handleBrushEnd();
      return;
    }

    if (active === 'lasso') {
      const points = lassoPointsRef.current;
      lassoPointsRef.current = [];
      // Releasing closes the outline back to where it started, which is what makes it a region.
      if (points.length >= 3 && selectionRef.current) {
        selectionRef.current.fillPolygon(points, lassoModeRef.current === 'subtract');
      }
      touchSelection();
    }
  }, [handleBrushEnd, touchSelection]);

  const clearSelection = useCallback(() => {
    selectionRef.current?.clear();
    touchSelection();
  }, [touchSelection]);

  const invertSelection = useCallback(() => {
    void (async () => {
      const selection = await ensureSelection();
      if (!selection) return;
      selection.invert();
      touchSelection();
    })();
  }, [ensureSelection, touchSelection]);

  const canGenerate =
    engineStatus.phase !== 'loading-model' &&
    engineStatus.phase !== 'error' &&
    !!baseFile &&
    (mode === 'deepdream' || !!templateFile) &&
    !!featureModel;

  const handleGenerate = useCallback(async () => {
    if (!baseFile || !featureModel) return;
    if (mode === 'style' && !templateFile) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const pauseController = new PauseController();
    pauseControllerRef.current = pauseController;

    let baseTensor: tf.Tensor3D | null = null;
    let templateTensor: tf.Tensor3D | null = null;
    let movieRecorder: MovieRecorder | null = null;
    let videoSource: VideoFrameSource | null = null;
    const processedVideoFrames: ImageBitmap[] = [];

    const stepsPerRun =
      mode === 'deepdream' ? dreamParams.octaves * dreamParams.stepsPerOctave : styleParams.octaves * styleParams.stepsPerOctave;

    // Runs the currently-selected algorithm (DeepDream or Style Transfer) on one image tensor,
    // reporting progress as a step count offset against totalSteps — shared by the single-image
    // path and the per-frame video-processing loop below.
    const drawFullPreview = async (image: tf.Tensor3D) => {
      if (canvasRef.current) await renderTensorToCanvas(image, canvasRef.current);
    };

    const runOnce = async (
      inputTensor: tf.Tensor3D,
      stepOffset: number,
      totalSteps: number,
      // How an in-progress image reaches the screen. A whole-image run replaces the canvas; a run confined
      // to a selection draws its patch back into place instead.
      renderPreview: (image: tf.Tensor3D) => Promise<void> = drawFullPreview,
    ): Promise<tf.Tensor3D> => {
      if (mode === 'deepdream') {
        const preset = presets.find((p) => p.id === selectedPresetId);
        if (!preset) throw new Error('No preset selected.');

        return runDeepDream(inputTensor, {
          featureModel,
          preset,
          params: dreamParams,
          signal: controller.signal,
          pauseController,
          onProgress: async ({ octave, step, image }) => {
            setEngineStatus({ phase: 'running', step: stepOffset + octave * dreamParams.stepsPerOctave + step, totalSteps });
            await renderPreview(image);
            persistProgressSnapshot();
            await movieRecorder?.captureStep();
          },
        });
      }

      return runStyleTransfer(inputTensor, templateTensor!, {
        featureModel,
        params: styleParams,
        signal: controller.signal,
        pauseController,
        onProgress: async ({ octave, step, image }) => {
          setEngineStatus({ phase: 'running', step: stepOffset + octave * styleParams.stepsPerOctave + step, totalSteps });
          await renderPreview(image);
          persistProgressSnapshot();
          await movieRecorder?.captureStep();
        },
      });
    };

    try {
      setHasResult(false);
      lastProgressPersistRef.current = Date.now();
      // Cleared for the whole run, not just on success: if this one is cancelled or dies, the viewport
      // should keep the work in progress rather than fall back to the previous result.
      setResultBlob(null);
      setIsPaused(false);
      setFrameProgress(null);
      setEngineStatus({ phase: 'running', step: 0, totalSteps: 1 });

      // A GPU process reset (e.g. after the computer sleeps) can silently invalidate the
      // WebGPU/WebGL device tfjs is holding, after which ops quietly return zeroed tensors
      // instead of throwing — recreate the backend up front if that's happened.
      await ensureBackendHealthy();

      // Style transfer holds far more per pixel than DeepDream, so on a phone it works at a smaller size —
      // a finished 768px result beats a 1024px run that takes the tab down with it. See `deviceLimits`.
      const limits = getDeviceLimits();
      const workingMax = mode === 'style' ? limits.styleWorkingMaxDimension : limits.workingMaxDimension;

      if (mode === 'style') {
        const templateImg = await loadImageFromFile(templateFile!);
        templateTensor = imageToWorkingTensor(templateImg, workingMax);
      }

      const selection = selectionRef.current;
      const bounds =
        selection && !selection.isEmpty() && !isBaseVideo
          ? selection.bounds(Math.ceil(selectionSettings.feather) + 1)
          : null;

      if (bounds && selection) {
        // Generate stays inside the selection, running the full pipeline — every octave, every step — over
        // its box rather than the cut-down pass the brush uses. The result is blended back through the same
        // feathered edge, so what Generate leaves behind matches what the wash on screen promised.
        const brush = await ensureBrush();
        if (!brush) throw new Error('Nothing to work on.');

        if (canvasRef.current) {
          await renderTensorToCanvas(brush.current, canvasRef.current);
        }

        const mask = selection.toTensor(bounds, selectionSettings.feather);
        const basePatch = tf.tidy(
          () =>
            tf.keep(
              brush.current.slice([bounds.y, bounds.x, 0], [bounds.height, bounds.width, 3]),
            ) as tf.Tensor3D,
        );

        // Previews are composited through the mask and drawn into place, so the run is watched where it is
        // happening and the soft edge is visible throughout rather than appearing at the end.
        //
        // A preview arrives at whatever resolution its octave is working at, not the patch's — the first
        // octave of a three-octave run is the patch divided by octaveScale squared. The whole-image path
        // never had to care, since it hands previews straight to a canvas that resizes to fit; here they
        // are blended against a fixed-size base, so they have to be brought up to it first.
        const drawPatchPreview = async (image: tf.Tensor3D) => {
          const canvas = canvasRef.current;
          if (!canvas) return;

          const blended = tf.tidy(() => {
            const sized =
              image.shape[0] === bounds.height && image.shape[1] === bounds.width
                ? image
                : (tf.image.resizeBilinear(image, [bounds.height, bounds.width]) as tf.Tensor3D);
            return tf.keep(basePatch.add(sized.sub(basePatch).mul(mask))) as tf.Tensor3D;
          });
          try {
            const scratch = patchPreviewRef.current ?? (patchPreviewRef.current = document.createElement('canvas'));
            await renderTensorToCanvas(blended, scratch);
            canvas.getContext('2d')?.drawImage(scratch, bounds.x, bounds.y);
          } finally {
            blended.dispose();
          }
        };

        try {
          await brush.applyPatch({ y: bounds.y, x: bounds.x, height: bounds.height, width: bounds.width }, mask, (patch) =>
            runOnce(patch, 0, stepsPerRun, drawPatchPreview),
          );
        } finally {
          mask.dispose();
          basePatch.dispose();
        }

        if (canvasRef.current) {
          const canvas = canvasRef.current;
          await renderTensorToCanvas(brush.current, canvas);
          canvas.toBlob((blob) => {
            if (!blob) return;
            void stampParametersAsync(blob).then((stamped) => {
              setResultBlob(stamped);
              void saveLastResultBlob(stamped);
            });
          }, 'image/png');
        }

        setHasResult(true);
        setEngineStatus({ phase: 'done' });
      } else if (isBaseVideo) {
        videoSource = await VideoFrameSource.load(baseFile, videoFps);

        // Every processed frame is held as an ImageBitmap until the whole clip is encoded at the end,
        // so it is the output store — not decoding — that sets how long a clip fits in memory. Process
        // as much of the clip as that budget holds; the progress label reports the capped count, and a
        // shortened video is a far better outcome than the tab being killed partway through.
        const [workingW, workingH] = workingDimensions(videoSource.info.width, videoSource.info.height, workingMax);
        const frameCount = Math.min(videoSource.info.frameCount, maxFramesInStore(workingW, workingH));
        const totalSteps = stepsPerRun * frameCount;

        for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
          if (controller.signal.aborted) break;
          setFrameProgress({ index: frameIndex, total: frameCount });

          const frameImage = await videoSource.seekToFrame(frameIndex);
          const frameTensor = imageToWorkingTensor(frameImage, workingMax);

          const frameResult = await runOnce(frameTensor, frameIndex * stepsPerRun, totalSteps);
          frameTensor.dispose();

          if (canvasRef.current) {
            await renderTensorToCanvas(frameResult, canvasRef.current);
            processedVideoFrames.push(await createImageBitmap(canvasRef.current));
          }
          frameResult.dispose();
        }

        setFrameProgress(null);

        if (processedVideoFrames.length > 0 && canvasRef.current) {
          const holdMs = 1000 / videoFps;
          const outFrames = processedVideoFrames.map((bitmap) => ({ bitmap, holdMs }));
          const videoBlob = await encodeFrameSequence(outFrames, canvasRef.current.width, canvasRef.current.height);
          downloadBlob(videoBlob, `dream-${mode}-video-${Date.now()}.webm`);
        }

        setEngineStatus({ phase: 'done' });
      } else {
        const baseImg = await loadImageFromFile(baseFile);
        baseTensor = imageToWorkingTensor(baseImg, workingMax);

        if (canvasRef.current) {
          await renderTensorToCanvas(baseTensor, canvasRef.current);
        }

        if (recordMovie && canvasRef.current) {
          movieRecorder = new MovieRecorder(canvasRef.current);
          movieRecorderRef.current = movieRecorder;
          setIsRecordingMovie(true);
          await movieRecorder.start();
        }

        const result = await runOnce(baseTensor, 0, stepsPerRun);

        if (canvasRef.current) {
          const canvas = canvasRef.current;
          try {
            await renderTensorToCanvas(result, canvas);
          } catch (err) {
            // The largest single GPU->CPU transfer of the run, and so the likeliest thing to fail on a
            // phone that is nearly out of memory. The canvas still holds the last progress frame, a few
            // steps short of the finished image — worth keeping rather than failing the whole run.
            console.warn('Final render failed; keeping the last frame already on the canvas.', err);
          }
          canvas.toBlob((blob) => {
            if (!blob) return;
            void stampParametersAsync(blob).then((stamped) => {
              setResultBlob(stamped);
              void saveLastResultBlob(stamped);
            });
          }, 'image/png');
        }
        // Nothing reads the finished tensor once it's on the canvas and encoded to a PNG — and it is a
        // full-resolution float32 buffer, so holding it kept a run's worth of GPU memory alive right
        // through the next run.
        // Hand the finished image straight to the brush, rather than dropping it and letting it reload
        // from the saved PNG. That reload raced the canvas's asynchronous encode: with a tool active the
        // brush is re-primed the moment the run ends, which is before `toBlob` has produced anything, so
        // it reloaded the *previous* image and painted it over the result that had just finished.
        brushRef.current?.dispose();
        brushRef.current = new DreamBrush(result);
        result.dispose();
        setHasResult(true);
        setEngineStatus({ phase: 'done' });

        if (movieRecorder) {
          const videoBlob = await movieRecorder.finish();
          movieRecorderRef.current = null;
          setIsRecordingMovie(false);
          downloadBlob(videoBlob, `dream-${mode}-movie-${Date.now()}.webm`);
        }
      }
    } catch (err) {
      console.error('Run failed:', err);
      setEngineStatus({ phase: 'error', message: describeRunError(err) });
    } finally {
      baseTensor?.dispose();
      templateTensor?.dispose();
      videoSource?.dispose();
      processedVideoFrames.forEach((bitmap) => bitmap.close());
      abortControllerRef.current = null;
      pauseControllerRef.current = null;
      setIsPaused(false);
      setFrameProgress(null);
      if (movieRecorderRef.current) {
        movieRecorderRef.current.abort();
        movieRecorderRef.current = null;
      }
      setIsRecordingMovie(false);
    }
  }, [
    baseFile,
    templateFile,
    featureModel,
    mode,
    presets,
    selectedPresetId,
    dreamParams,
    styleParams,
    recordMovie,
    isBaseVideo,
    videoFps,
    persistProgressSnapshot,
    ensureBrush,
    selectionSettings.feather,
    stampParametersAsync,
  ]);

  const handleCancel = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const handlePause = useCallback(() => {
    pauseControllerRef.current?.pause();
    setIsPaused(true);
  }, []);

  const handleResume = useCallback(() => {
    pauseControllerRef.current?.resume();
    setIsPaused(false);
  }, []);

  // Both save handlers hand off synchronously, keeping the click's user activation alive so `saveImage`
  // can open the share sheet — the only way onto an iPhone's camera roll.
  const handleSaveCurrentStep = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    void saveImage(stampParameters(canvasToPngBytes(canvas)), `dream-${mode}-step-${Date.now()}.png`);
  }, [mode, stampParameters]);

  const handleDownload = useCallback(() => {
    if (!resultBlob) return;
    void saveImage(resultBlob, `dream-${mode}-${Date.now()}.png`);
  }, [resultBlob, mode]);

  const isRunning = engineStatus.phase === 'running';
  isRunningRef.current = isRunning;

  // Recomputed when the mask changes, so the panel can say how much is selected and enable Apply.
  const selectedFraction = useMemo(() => {
    void selectionVersion;
    const selection = selectionRef.current;
    if (!selection) return 0;
    let total = 0;
    for (const weight of selection.data) total += weight;
    return total / (selection.width * selection.height);
  }, [selectionVersion]);

  const activeTool: CanvasTool = useMemo(
    () => ({
      id: tool,
      cursorRadius:
        tool === 'paint' ? brushSettings.radius : tool === 'select-brush' ? selectionSettings.brushRadius : undefined,
      onStart: handleToolStart,
      onMove: handleToolMove,
      onEnd: handleToolEnd,
    }),
    [tool, brushSettings.radius, selectionSettings.brushRadius, handleToolStart, handleToolMove, handleToolEnd],
  );
  const recordingSupported = isMovieRecordingSupported();

  // The brush needs a still photo to paint on and a network to paint with; style transfer also needs its
  // template, loaded here so the first dab of a stroke isn't the one that pays for it.
  const toolsAvailable = !!baseFile && !isBaseVideo && !!featureModel && !isRunning;
  const canBrush = toolsAvailable && tool !== 'none';

  useEffect(() => {
    if (tool === 'none' || mode !== 'style' || !templateFile) return;

    let cancelled = false;
    (async () => {
      const img = await loadImageFromFile(templateFile);
      const tensor = imageToWorkingTensor(img, getDeviceLimits().styleWorkingMaxDimension);
      if (cancelled) {
        tensor.dispose();
        return;
      }
      brushTemplateRef.current?.dispose();
      brushTemplateRef.current = tensor;
    })();

    return () => {
      cancelled = true;
    };
  }, [tool, mode, templateFile]);

  // Switching the brush on shows the canvas instead of the still image, so the image has to be on the
  // canvas by then — otherwise the viewport goes blank until the first dab lands.
  useEffect(() => {
    if (!canBrush) return;
    void ensureBrush();
  }, [canBrush, ensureBrush]);

  /**
   * Draws the selection over the image: a wash across what is selected, plus the outline of a lasso still
   * being drawn. The wash uses the feathered mask, so the softness of the edge is visible before it is
   * committed to rather than being a number that has to be imagined.
   */
  useEffect(() => {
    const overlay = overlayRef.current;
    const canvas = canvasRef.current;
    if (!overlay || !canvas) return;

    const selection = selectionRef.current;
    const width = selection?.width ?? canvas.width;
    const height = selection?.height ?? canvas.height;
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }

    const context = overlay.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, width, height);

    if (tool === 'none') return;

    if (selection && !selection.isEmpty()) {
      const weights = selection.feathered(selectionSettings.feather);
      const image = context.createImageData(width, height);
      for (let i = 0; i < weights.length; i++) {
        const weight = weights[i];
        if (weight <= 0.002) continue;
        image.data[i * 4] = 110;
        image.data[i * 4 + 1] = 130;
        image.data[i * 4 + 2] = 255;
        image.data[i * 4 + 3] = Math.round(weight * 90);
      }
      context.putImageData(image, 0, 0);
    }

    const points = lassoPointsRef.current;
    if (points.length > 1) {
      context.strokeStyle = 'rgba(255, 255, 255, 0.95)';
      context.lineWidth = Math.max(1, width / 400);
      context.setLineDash([6, 4]);
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (const point of points.slice(1)) context.lineTo(point.x, point.y);
      context.closePath();
      context.stroke();
    }
  }, [selectionVersion, selectionSettings.feather, tool]);

  // Painting owns tensors that outlive any one render, so they have to be released when the app is.
  useEffect(() => discardBrush, [discardBrush]);

  return (
    <AppFrame
      className="dream-frame"
      title="Dream by Joyographic"
      subtitle={initError ?? undefined}
      viewportFill
      controlsLabel="Dream controls"
      viewportLabel="Result"
      viewport={
        <ResultCanvas
          canvasRef={canvasRef}
          status={engineStatus}
          resultImageUrl={resultImageUrl}
          basePreviewUrl={isBaseVideo ? undefined : basePreviewUrl}
          overlayRef={overlayRef}
          tool={canBrush ? activeTool : undefined}
        />
      }
      controls={
        <>
          <ActionsBar
            status={engineStatus}
            isPaused={isPaused}
            isRunning={isRunning}
            canGenerate={canGenerate}
            hasResult={hasResult}
            recordMovie={recordMovie}
            isRecordingMovie={isRecordingMovie}
            recordingSupported={recordingSupported}
            recordUnavailableForVideo={isBaseVideo}
            frameProgressLabel={frameProgress ? `Frame ${frameProgress.index + 1} / ${frameProgress.total}` : null}
            modeTabs={<ModeTabs mode={mode} onChange={setMode} disabled={isRunning} />}
            onGenerate={handleGenerate}
            onCancel={handleCancel}
            onPause={handlePause}
            onResume={handleResume}
            onDownload={handleDownload}
            onSaveCurrentStep={handleSaveCurrentStep}
            onToggleRecordMovie={() => setRecordMovie((r) => !r)}
          />

          <ControlGroup title="Setup">
            <div className="dropzones-row">
              <ImageDropzone
                label="Image to alter"
                hint={recordingSupported ? 'The photo or video DeepDream / style transfer will transform' : 'The photo DeepDream / style transfer will transform'}
                tooltip={
                  recordingSupported
                    ? "The photo or video that DeepDream or Style Transfer will transform. Drop a file here or click to browse. For a video, every sampled frame runs through the full pipeline and the result downloads as a new video."
                    : 'The photo that DeepDream or Style Transfer will transform. Drop an image here or click to browse your files.'
                }
                onFileSelected={handleBaseFile}
                previewUrl={basePreviewUrl}
                previewIsVideo={isBaseVideo}
                acceptVideo={recordingSupported}
              />
              {isBaseVideo && <VideoOptionsPanel fps={videoFps} onFpsChange={setVideoFps} isRunning={isRunning} />}
              {mode === 'style' && (
                <HoverPopup
                  trigger={
                    <ImageDropzone
                      label="Dream template (style)"
                      hint="The image whose style/patterns get imprinted onto the first image"
                      tooltip="The style image whose colors, textures, and patterns get imprinted onto your photo. Images with strong, distinctive visual patterns tend to work best. Hover to pick a built-in template."
                      onFileSelected={handleTemplateFile}
                      previewUrl={templatePreviewUrl}
                    />
                  }
                >
                  <BuiltInTemplatePicker onSelect={handleTemplateFile} disabled={isRunning} />
                </HoverPopup>
              )}
            </div>

            <ParametersFromImage
              parameters={offeredParameters}
              applied={parametersApplied}
              disabled={isRunning}
              onApply={applyOfferedParameters}
              onDismiss={() => setOfferedParameters(null)}
            />

            <FeatureNetworkPicker
              value={featureNetworkId}
              onChange={setFeatureNetworkId}
              isLoading={!featureModel && !initError}
              disabled={isRunning}
            />

            <PresetPanel
              mode={mode}
              presets={presets}
              selectedPresetId={selectedPresetId}
              onPresetChange={setSelectedPresetId}
              isRunning={isRunning}
            />
          </ControlGroup>

          <ControlGroup title="Parameters">
            <SliderPanel
              mode={mode}
              dreamParams={dreamParams}
              onDreamParamsChange={setDreamParams}
              styleParams={styleParams}
              onStyleParamsChange={setStyleParams}
              isRunning={isRunning}
            />
          </ControlGroup>

          <ControlGroup title="Tools" defaultOpen={false}>
            <BrushPanel
              tool={tool}
              onToolChange={setTool}
              settings={brushSettings}
              onSettingsChange={setBrushSettings}
              available={toolsAvailable}
              isPainting={isPainting}
              isRunning={isRunning}
            />
            <SelectionPanel
              tool={tool}
              settings={selectionSettings}
              onSettingsChange={setSelectionSettings}
              selectedFraction={selectedFraction}
              isBusy={isPainting}
              isRunning={isRunning}
              onApply={() => void applyToSelection()}
              onInvert={invertSelection}
              onClear={clearSelection}
            />
          </ControlGroup>

          <ControlGroup title="Regularizers" defaultOpen={false}>
            <RegularizerPanel
              mode={mode}
              dreamParams={dreamParams}
              onDreamParamsChange={setDreamParams}
              styleParams={styleParams}
              onStyleParamsChange={setStyleParams}
              isRunning={isRunning}
            />
          </ControlGroup>
        </>
      }
    />
  );
}

export default App;
