import { useCallback, useRef, useState, type DragEvent } from 'react';
import { ensureBrowserDecodableImage, isHeicFile } from '../ml/imageUtils';

interface ImageDropzoneProps {
  label: string;
  hint?: string;
  tooltip?: string;
  onFileSelected: (file: File) => void;
  previewUrl?: string;
  previewIsVideo?: boolean;
  acceptVideo?: boolean;
}

export function ImageDropzone({
  label,
  hint,
  tooltip,
  onFileSelected,
  previewUrl,
  previewIsVideo,
  acceptVideo,
}: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [conversionError, setConversionError] = useState<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0];
      const isVideo = !!acceptVideo && !!file && file.type.startsWith('video/');
      if (!file || !(file.type.startsWith('image/') || isHeicFile(file) || isVideo)) {
        return;
      }

      setConversionError(null);

      if (isVideo || !isHeicFile(file)) {
        onFileSelected(file);
        return;
      }

      setIsConverting(true);
      try {
        const decodable = await ensureBrowserDecodableImage(file);
        onFileSelected(decodable);
      } catch (err) {
        console.error('HEIC conversion failed:', err);
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'object' && err !== null && 'message' in err
              ? String((err as { message: unknown }).message)
              : 'Could not convert this HEIC file (see console for details).';
        setConversionError(message);
      } finally {
        setIsConverting(false);
      }
    },
    [onFileSelected, acceptVideo],
  );

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    void handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="dropzone-wrapper">
      <span className="dropzone-label">{label}</span>
      <div
        className={`dropzone${isDragOver ? ' dropzone--active' : ''}${previewUrl ? ' dropzone--filled' : ''}`}
        onClick={() => !isConverting && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        title={tooltip}
      >
        {isConverting ? (
          <div className="dropzone-placeholder">
            <span>Converting HEIC photo…</span>
          </div>
        ) : previewUrl ? (
          previewIsVideo ? (
            <video src={previewUrl} className="dropzone-preview" muted loop autoPlay playsInline />
          ) : (
            <img src={previewUrl} alt={label} className="dropzone-preview" />
          )
        ) : (
          <div className="dropzone-placeholder">
            <span>Drop {acceptVideo ? 'image or video' : 'image'} or click to browse</span>
            {hint && <small>{hint}</small>}
            {conversionError && <small className="dropzone-error">{conversionError}</small>}
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={acceptVideo ? 'image/*,video/*,.heic,.heif,image/heic,image/heif' : 'image/*,.heic,.heif,image/heic,image/heif'}
          hidden
          onChange={(e) => void handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
