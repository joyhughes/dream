import { useCallback, useRef, useState, type DragEvent } from 'react';

interface ImageDropzoneProps {
  label: string;
  hint?: string;
  onFileSelected: (file: File) => void;
  previewUrl?: string;
}

export function ImageDropzone({ label, hint, onFileSelected, previewUrl }: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && file.type.startsWith('image/')) {
        onFileSelected(file);
      }
    },
    [onFileSelected],
  );

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="dropzone-wrapper">
      <span className="dropzone-label">{label}</span>
      <div
        className={`dropzone${isDragOver ? ' dropzone--active' : ''}${previewUrl ? ' dropzone--filled' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
      >
        {previewUrl ? (
          <img src={previewUrl} alt={label} className="dropzone-preview" />
        ) : (
          <div className="dropzone-placeholder">
            <span>Drop image or click to browse</span>
            {hint && <small>{hint}</small>}
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
