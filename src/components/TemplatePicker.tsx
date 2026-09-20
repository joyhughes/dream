import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { BUILT_IN_TEMPLATES } from '../templates/builtInTemplates';
import { ensureBrowserDecodableImage, isHeicFile } from '../ml/imageUtils';

interface TemplatePickerProps {
  /** Id of the built-in currently in use, or null when the template is the user's own file. */
  selectedId: string | null;
  previewUrl?: string;
  disabled?: boolean;
  onSelectBuiltIn: (id: string, file: File) => void;
  onSelectFile: (file: File) => void;
}

/** Gap the popup keeps from the trigger and from any viewport edge. */
const MARGIN = 14;
/** Kept in step with `.template-popup`'s max-width so the left edge can be clamped against the viewport. */
const POPUP_WIDTH = 320;
/** Below this much room the popup is moved up the screen rather than squeezed against the bottom. */
const MIN_USEFUL_HEIGHT = 240;

interface Position {
  top: number;
  left: number;
  maxHeight: number;
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 16V5" />
      <path d="M7.5 9.5L12 5l4.5 4.5" />
      <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
    </svg>
  );
}

/**
 * Choosing the style image.
 *
 * This is the only way to change it. There used to also be a drop target behind the popup, which meant two
 * controls doing one job and a tap on a phone landing on whichever the browser decided — the file picker,
 * usually, so the built-ins were unreachable. The trigger now only opens this, and everything that can set
 * the template lives inside it, the user's own file included.
 */
export function TemplatePicker({
  selectedId,
  previewUrl,
  disabled,
  onSelectBuiltIn,
  onSelectFile,
}: TemplatePickerProps) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<Position | null>(null);
  const [thumbnails, setThumbnails] = useState<Record<string, string>>({});
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const triggerRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const closeTimer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        BUILT_IN_TEMPLATES.map(async (template) => [template.id, await template.getThumbnailUrl()] as const),
      );
      if (!cancelled) setThumbnails(Object.fromEntries(entries));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const clearCloseTimer = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };

  const close = useCallback(() => {
    clearCloseTimer();
    setOpen(false);
  }, []);

  const place = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const viewportHeight = window.innerHeight;
    let top = rect.top;
    if (viewportHeight - top - MARGIN < MIN_USEFUL_HEIGHT) {
      top = Math.max(MARGIN, viewportHeight - MIN_USEFUL_HEIGHT - MARGIN);
    }

    // To the right where there is room, otherwise clamped inside the viewport — on a phone that puts it
    // over the controls rather than off the edge of the screen.
    const left = Math.max(MARGIN, Math.min(rect.right + MARGIN, window.innerWidth - POPUP_WIDTH - MARGIN));
    setPosition({ top, left, maxHeight: viewportHeight - top - MARGIN });
  }, []);

  const openPopup = useCallback(() => {
    if (disabled) return;
    clearCloseTimer();
    place();
    setOpen(true);
  }, [disabled, place]);

  // Dismissal, for the ways out that are not picking something: tapping elsewhere, or Escape.
  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!popupRef.current?.contains(target) && !triggerRef.current?.contains(target)) close();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };

    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open, close, place]);

  useEffect(() => clearCloseTimer, []);

  const pickBuiltIn = async (id: string) => {
    const template = BUILT_IN_TEMPLATES.find((t) => t.id === id);
    if (!template) return;

    setError(null);
    setPendingId(id);
    try {
      const file = await template.getFile();
      onSelectBuiltIn(id, file);
      // Closing here rather than on the tap that started it: a phone's tap would otherwise leave the popup
      // covering the image the moment it became worth looking at.
      close();
    } catch (err) {
      console.error('Could not load the template:', err);
      setError('That template could not be loaded.');
    } finally {
      setPendingId(null);
    }
  };

  const handleUpload = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file || !(file.type.startsWith('image/') || isHeicFile(file))) return;

    setError(null);
    setPendingId('upload');
    try {
      // iPhone photos are HEIC, which only Safari can decode; converting here means the rest of the app
      // never has to know.
      onSelectFile(isHeicFile(file) ? await ensureBrowserDecodableImage(file) : file);
      close();
    } catch (err) {
      console.error('Could not read that image:', err);
      setError('That image could not be read.');
    } finally {
      setPendingId(null);
    }
  };

  const selectedName = selectedId
    ? (BUILT_IN_TEMPLATES.find((t) => t.id === selectedId)?.name ?? 'Template')
    : previewUrl
      ? 'Your image'
      : 'None chosen';

  return (
    <div className="dropzone-wrapper">
      <span className="dropzone-label">Dream template (style)</span>
      <div
        ref={triggerRef}
        // Hover only opens it for a real mouse. iOS synthesizes hover from a tap, which used to open the
        // popup on the way to a click that then did something else entirely.
        onPointerEnter={(event) => {
          if (event.pointerType === 'mouse') openPopup();
        }}
        onPointerLeave={(event) => {
          if (event.pointerType !== 'mouse') return;
          clearCloseTimer();
          closeTimer.current = window.setTimeout(close, 200);
        }}
      >
        <button
          type="button"
          className={`template-trigger${open ? ' template-trigger--open' : ''}`}
          disabled={disabled}
          aria-haspopup="dialog"
          aria-expanded={open}
          onClick={() => (open ? close() : openPopup())}
          title="The style image whose colors, textures and patterns get imprinted onto your photo. Opens the picker, where you can choose a built-in template or upload your own."
        >
          {previewUrl ? (
            <img src={previewUrl} alt="" className="template-trigger-preview" />
          ) : (
            <span className="template-trigger-preview template-trigger-preview--empty" />
          )}
          <span className="template-trigger-text">
            <span className="template-trigger-name">{selectedName}</span>
            <small>Tap to change</small>
          </span>
        </button>
      </div>

      {open &&
        position &&
        createPortal(
          <div
            ref={popupRef}
            className="template-popup"
            role="dialog"
            aria-label="Choose a style template"
            style={{ top: position.top, left: position.left, maxHeight: position.maxHeight }}
            onPointerEnter={clearCloseTimer}
            onPointerLeave={(event) => {
              if (event.pointerType !== 'mouse') return;
              clearCloseTimer();
              closeTimer.current = window.setTimeout(close, 200);
            }}
          >
            <div className="builtin-templates">
              <span className="builtin-templates-label">Choose a template, or use your own image</span>
              {error && <span className="field-hint field-hint--warn">{error}</span>}
              <div className="builtin-templates-row">
                {/* First, so the way to bring in your own picture is the first thing read rather than the
                    last thing found after scrolling past fifteen thumbnails. */}
                <button
                  type="button"
                  className="builtin-template-thumb builtin-template-thumb--upload"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={pendingId !== null}
                  title="Use one of your own images as the style template."
                >
                  <span className="template-upload-icon">
                    <UploadIcon />
                  </span>
                  <span>{pendingId === 'upload' ? 'Reading…' : 'Upload'}</span>
                </button>

                {BUILT_IN_TEMPLATES.map((template) => (
                  <button
                    key={template.id}
                    type="button"
                    className={`builtin-template-thumb${selectedId === template.id ? ' builtin-template-thumb--active' : ''}`}
                    onClick={() => void pickBuiltIn(template.id)}
                    disabled={pendingId !== null}
                    aria-pressed={selectedId === template.id}
                    title={`Use the "${template.name}" built-in image as the style template.`}
                  >
                    {thumbnails[template.id] && (
                      <img src={thumbnails[template.id]} alt={template.name} loading="lazy" />
                    )}
                    <span>{pendingId === template.id ? 'Loading…' : template.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>,
          document.body,
        )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.heic,.heif"
        hidden
        onChange={(event) => {
          void handleUpload(event.target.files);
          // Cleared so choosing the same file twice still fires a change.
          event.target.value = '';
        }}
      />
    </div>
  );
}
