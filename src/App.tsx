import { useCallback, useEffect, useRef, useState } from 'react';
import { ImageDropzone } from './components/ImageDropzone';
import { BuiltInTemplatePicker } from './components/BuiltInTemplatePicker';
import { ModeTabs } from './components/ModeTabs';
import { PresetPanel, SliderPanel, VideoOptionsPanel, ActionsBar, RegularizerPanel } from './components/ControlsPanel';
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
import { canvasToPngBlob, downloadBlob, saveImage } from './ml/imageSaving';
import { PauseController } from './ml/pauseController';
import { MovieRecorder, isMovieRecordingSupported } from './ml/movieRecorder';
import { encodeFrameSequence } from './ml/frameEncoding';
import { VideoFrameSource } from './ml/videoFrames';
import { BUILT_IN_TEMPLATES } from './templates/builtInTemplates';
import type { DreamParams, DreamPreset, EngineStatus, ImageRegularizers, Mode, StyleParams } from './types';
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
  regularizers: NO_REGULARIZERS,
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

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pauseControllerRef = useRef<PauseController | null>(null);
  const movieRecorderRef = useRef<MovieRecorder | null>(null);
  const lastProgressPersistRef = useRef(0);
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
  }, []);

  const handleTemplateFile = useCallback((file: File) => {
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
    const runOnce = async (inputTensor: tf.Tensor3D, stepOffset: number, totalSteps: number): Promise<tf.Tensor3D> => {
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
            if (canvasRef.current) await renderTensorToCanvas(image, canvasRef.current);
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
          if (canvasRef.current) await renderTensorToCanvas(image, canvasRef.current);
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

      if (isBaseVideo) {
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
            setResultBlob(blob);
            void saveLastResultBlob(blob);
          }, 'image/png');
        }
        // Nothing reads the finished tensor once it's on the canvas and encoded to a PNG — and it is a
        // full-resolution float32 buffer, so holding it kept a run's worth of GPU memory alive right
        // through the next run.
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
    void saveImage(canvasToPngBlob(canvas), `dream-${mode}-step-${Date.now()}.png`);
  }, [mode]);

  const handleDownload = useCallback(() => {
    if (!resultBlob) return;
    void saveImage(resultBlob, `dream-${mode}-${Date.now()}.png`);
  }, [resultBlob, mode]);

  const isRunning = engineStatus.phase === 'running';
  const recordingSupported = isMovieRecordingSupported();

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
