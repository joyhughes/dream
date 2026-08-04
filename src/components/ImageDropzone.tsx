import { useCallback, useRef, useState, type DragEvent } from 'react';
import { ensureBrowserDecodableImage, isHeicFile } from '../ml/imageUtils';

interface ImageDropzoneProps {
  label: string;
  hint?: string;
  onFileSelected: (file: File) => void;
  previewUrl?: string;
}

export function ImageDropzone({ label, hint, onFileSelected, previewUrl }: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [conversionError, setConversionError] = useState<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0];
      if (!file || !(file.type.startsWith('image/') || isHeicFile(file))) {
        return;
      }

      setConversionError(null);

      if (!isHeicFile(file)) {
        onFileSelected(file);
        return;
      }

      setIsConverting(true);
      try {
        const decodable = await ensureBrowserDecodableImage(file);
        onFileSelected(decodable);
      } catch (err) {
        setConversionError(err instanceof Error ? err.message : 'Could not convert this HEIC file.');
      } finally {
        setIsConverting(false);
      }
    },
    [onFileSelected],
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
      >
        {isConverting ? (
          <div className="dropzone-placeholder">
            <span>Converting HEIC photo…</span>
          </div>
        ) : previewUrl ? (
          <img src={previewUrl} alt={label} className="dropzone-preview" />
        ) : (
          <div className="dropzone-placeholder">
            <span>Drop image or click to browse</span>
            {hint && <small>{hint}</small>}
            {conversionError && <small className="dropzone-error">{conversionError}</small>}
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.heic,.heif,image/heic,image/heif"
          hidden
          onChange={(e) => void handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
