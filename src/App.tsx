import { useCallback, useEffect, useRef, useState } from 'react';
import { ImageDropzone } from './components/ImageDropzone';
import { BuiltInTemplatePicker } from './components/BuiltInTemplatePicker';
import { ModeTabs } from './components/ModeTabs';
import { WebGPUStatus } from './components/WebGPUStatus';
import { PresetPanel, SliderPanel, ActionsBar } from './components/ControlsPanel';
import { ResultCanvas } from './components/ResultCanvas';
import { OverlayPanel } from './components/OverlayPanel';
import { HoverPopup } from './components/HoverPopup';
import { initializeML, ensureBackendHealthy, type BackendInfo } from './ml/tfSetup';
import { loadFeatureModel, type FeatureModel } from './ml/mobilenetFeatures';
import { buildPresets } from './ml/presets';
import { runDeepDream } from './ml/deepdream';
import { runStyleTransfer } from './ml/styleTransfer';
import { imageToWorkingTensor, loadImageFromFile, renderTensorToCanvas } from './ml/imageUtils';
import { loadLastResultBlob, saveLastResultBlob } from './ml/resultPersistence';
import { PauseController } from './ml/pauseController';
import { MovieRecorder, isMovieRecordingSupported } from './ml/movieRecorder';
import { BUILT_IN_TEMPLATES } from './templates/builtInTemplates';
import type { DreamParams, DreamPreset, EngineStatus, Mode, StyleParams } from './types';
import { tf } from './ml/tfSetup';

const DEFAULT_TEMPLATE_ID = 'paisley-color';

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const DEFAULT_DREAM_PARAMS: DreamParams = {
  octaves: 3,
  octaveScale: 1.4,
  stepsPerOctave: 20,
  stepSize: 0.02,
  tileSize: 320,
};

const DEFAULT_STYLE_PARAMS: StyleParams = {
  contentWeight: 8,
  styleWeight: 400,
  totalVariationWeight: 1,
  learningRate: 0.015,
  octaves: 3,
  octaveScale: 1.4,
  stepsPerOctave: 40,
  tileSize: 320,
};

function App() {
  const [mode, setMode] = useState<Mode>('deepdream');

  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [featureModel, setFeatureModel] = useState<FeatureModel | null>(null);
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
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [recordMovie, setRecordMovie] = useState(false);
  const [isRecordingMovie, setIsRecordingMovie] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pauseControllerRef = useRef<PauseController | null>(null);
  const resultTensorRef = useRef<tf.Tensor3D | null>(null);
  const movieRecorderRef = useRef<MovieRecorder | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const info = await initializeML();
        if (cancelled) return;
        setBackendInfo(info);

        const model = await loadFeatureModel();
        if (cancelled) return;
        setFeatureModel(model);

        const builtPresets = buildPresets(model.layers);
        setPresets(builtPresets);
        setSelectedPresetId(builtPresets[0]?.id ?? '');
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
  }, []);

  useEffect(() => {
    (async () => {
      const blob = await loadLastResultBlob();
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
    setBaseFile(file);
    setBasePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
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

    try {
      setHasResult(false);
      setIsPaused(false);
      setEngineStatus({ phase: 'running', step: 0, totalSteps: 1 });

      // A GPU process reset (e.g. after the computer sleeps) can silently invalidate the
      // WebGPU/WebGL device tfjs is holding, after which ops quietly return zeroed tensors
      // instead of throwing — recreate the backend up front if that's happened.
      setBackendInfo(await ensureBackendHealthy());

      const baseImg = await loadImageFromFile(baseFile);
      baseTensor = imageToWorkingTensor(baseImg);

      if (canvasRef.current) {
        await renderTensorToCanvas(baseTensor, canvasRef.current);
      }

      if (recordMovie && canvasRef.current) {
        movieRecorder = new MovieRecorder(canvasRef.current);
        movieRecorderRef.current = movieRecorder;
        setIsRecordingMovie(true);
        await movieRecorder.start();
      }

      let result: tf.Tensor3D;

      if (mode === 'deepdream') {
        const preset = presets.find((p) => p.id === selectedPresetId);
        if (!preset) throw new Error('No preset selected.');

        const totalSteps = dreamParams.octaves * dreamParams.stepsPerOctave;

        result = await runDeepDream(baseTensor, {
          featureModel,
          preset,
          params: dreamParams,
          signal: controller.signal,
          pauseController,
          onProgress: async ({ octave, step, image }) => {
            const doneSteps = octave * dreamParams.stepsPerOctave + step;
            setEngineStatus({ phase: 'running', step: doneSteps, totalSteps });
            if (canvasRef.current) await renderTensorToCanvas(image, canvasRef.current);
            await movieRecorder?.captureStep();
          },
        });
      } else {
        const templateImg = await loadImageFromFile(templateFile!);
        templateTensor = imageToWorkingTensor(templateImg);

        const totalSteps = styleParams.octaves * styleParams.stepsPerOctave;

        result = await runStyleTransfer(baseTensor, templateTensor, {
          featureModel,
          params: styleParams,
          signal: controller.signal,
          pauseController,
          onProgress: async ({ octave, step, image }) => {
            const doneSteps = octave * styleParams.stepsPerOctave + step;
            setEngineStatus({ phase: 'running', step: doneSteps, totalSteps });
            if (canvasRef.current) await renderTensorToCanvas(image, canvasRef.current);
            await movieRecorder?.captureStep();
          },
        });
      }

      if (canvasRef.current) {
        await renderTensorToCanvas(result, canvasRef.current);
        const canvas = canvasRef.current;
        canvas.toBlob((blob) => {
          if (!blob) return;
          setResultBlob(blob);
          void saveLastResultBlob(blob);
        }, 'image/png');
      }
      resultTensorRef.current?.dispose();
      resultTensorRef.current = result;
      setHasResult(true);
      setEngineStatus({ phase: 'done' });

      if (movieRecorder) {
        const videoBlob = await movieRecorder.finish();
        movieRecorderRef.current = null;
        setIsRecordingMovie(false);
        downloadBlob(videoBlob, `dream-${mode}-movie-${Date.now()}.webm`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setEngineStatus({ phase: 'error', message });
    } finally {
      baseTensor?.dispose();
      templateTensor?.dispose();
      abortControllerRef.current = null;
      pauseControllerRef.current = null;
      setIsPaused(false);
      if (movieRecorderRef.current) {
        movieRecorderRef.current.abort();
        movieRecorderRef.current = null;
      }
      setIsRecordingMovie(false);
    }
  }, [baseFile, templateFile, featureModel, mode, presets, selectedPresetId, dreamParams, styleParams, recordMovie]);

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

  const handleSaveCurrentStep = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      downloadBlob(blob, `dream-${mode}-step-${Date.now()}.png`);
    }, 'image/png');
  }, [mode]);

  const handleDownload = useCallback(() => {
    if (!resultBlob) return;
    downloadBlob(resultBlob, `dream-${mode}-${Date.now()}.png`);
  }, [resultBlob, mode]);

  const isRunning = engineStatus.phase === 'running';
  const recordingSupported = isMovieRecordingSupported();

  return (
    <div className="stage">
      <ResultCanvas canvasRef={canvasRef} status={engineStatus} resultImageUrl={resultImageUrl} />

      <div className="left-stack">
        <header
          className="app-header"
          title="Browser-based DeepDream and neural style transfer, running on your own GPU via WebGPU."
        >
          <h1>Dream</h1>
          <WebGPUStatus info={backendInfo} error={initError} />
        </header>

        <OverlayPanel title="Setup" side="left" open={leftPanelOpen} onToggle={() => setLeftPanelOpen((o) => !o)}>
          <ModeTabs mode={mode} onChange={setMode} disabled={isRunning} />

          <div className="dropzones-row">
            <ImageDropzone
              label="Image to alter"
              hint="The photo DeepDream / style transfer will transform"
              tooltip="The photo that DeepDream or Style Transfer will transform. Drop an image here or click to browse your files."
              onFileSelected={handleBaseFile}
              previewUrl={basePreviewUrl}
            />
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

          <ActionsBar
            status={engineStatus}
            isPaused={isPaused}
            isRunning={isRunning}
            canGenerate={canGenerate}
            hasResult={hasResult}
            recordMovie={recordMovie}
            isRecordingMovie={isRecordingMovie}
            recordingSupported={recordingSupported}
            onGenerate={handleGenerate}
            onCancel={handleCancel}
            onPause={handlePause}
            onResume={handleResume}
            onDownload={handleDownload}
            onSaveCurrentStep={handleSaveCurrentStep}
            onToggleRecordMovie={() => setRecordMovie((r) => !r)}
          />

          <PresetPanel
            mode={mode}
            presets={presets}
            selectedPresetId={selectedPresetId}
            onPresetChange={setSelectedPresetId}
            isRunning={isRunning}
          />
        </OverlayPanel>
      </div>

      <OverlayPanel title="Parameters" side="right" open={rightPanelOpen} onToggle={() => setRightPanelOpen((o) => !o)}>
        <SliderPanel
          mode={mode}
          dreamParams={dreamParams}
          onDreamParamsChange={setDreamParams}
          styleParams={styleParams}
          onStyleParamsChange={setStyleParams}
          isRunning={isRunning}
        />
      </OverlayPanel>
    </div>
  );
}

export default App;
