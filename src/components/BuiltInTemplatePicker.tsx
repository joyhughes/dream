import { useEffect, useState } from 'react';
import { BUILT_IN_TEMPLATES } from '../templates/builtInTemplates';

interface BuiltInTemplatePickerProps {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

export function BuiltInTemplatePicker({ onSelect, disabled }: BuiltInTemplatePickerProps) {
  const [thumbnails, setThumbnails] = useState<Record<string, string>>({});
  const [pendingId, setPendingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const entries = await Promise.all(
        BUILT_IN_TEMPLATES.map(async (template) => [template.id, await template.getThumbnailUrl()] as const),
      );
      if (!cancelled) {
        setThumbnails(Object.fromEntries(entries));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const handlePick = async (id: string) => {
    const template = BUILT_IN_TEMPLATES.find((t) => t.id === id);
    if (!template) return;

    setPendingId(id);
    try {
      const file = await template.getFile();
      onSelect(file);
    } finally {
      setPendingId(null);
    }
  };

  return (
    <div className="builtin-templates">
      <span
        className="builtin-templates-label"
        title="Skip uploading your own style image and use one of these ready-made templates instead."
      >
        Or use a built-in template
      </span>
      <div className="builtin-templates-row">
        {BUILT_IN_TEMPLATES.map((template) => (
          <button
            key={template.id}
            type="button"
            className="builtin-template-thumb"
            onClick={() => void handlePick(template.id)}
            disabled={disabled || pendingId !== null}
            title={`Use the "${template.name}" built-in image as the style template.`}
          >
            {thumbnails[template.id] && <img src={thumbnails[template.id]} alt={template.name} />}
            <span>{pendingId === template.id ? 'Loading…' : template.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
