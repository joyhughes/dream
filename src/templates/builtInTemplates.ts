import { generatePaisleyPattern } from './paisleyPattern';
import paisleyColorUrl from '../assets/templates/paisley-color.jpg';
import paisleyBwUrl from '../assets/templates/paisley-bw.jpg';

export interface BuiltInTemplate {
  id: string;
  name: string;
  getThumbnailUrl: () => Promise<string>;
  getFile: () => Promise<File>;
}

async function fileFromUrl(url: string, filename: string, type: string): Promise<File> {
  const response = await fetch(url);
  const blob = await response.blob();
  return new File([blob], filename, { type });
}

function staticTemplate(id: string, name: string, url: string, filename: string): BuiltInTemplate {
  return {
    id,
    name,
    getThumbnailUrl: () => Promise.resolve(url),
    getFile: () => fileFromUrl(url, filename, 'image/jpeg'),
  };
}

function canvasToFile(canvas: HTMLCanvasElement, filename: string): Promise<File> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Failed to generate pattern.'));
        return;
      }
      resolve(new File([blob], filename, { type: 'image/png' }));
    }, 'image/png');
  });
}

const generatedPaisley: BuiltInTemplate = {
  id: 'paisley-generated',
  name: 'Paisley (Generated)',
  getThumbnailUrl: async () => generatePaisleyPattern(160).toDataURL('image/png'),
  getFile: async () => canvasToFile(generatePaisleyPattern(512), 'paisley-generated.png'),
};

export const BUILT_IN_TEMPLATES: BuiltInTemplate[] = [
  staticTemplate('paisley-color', 'Paisley (Color)', paisleyColorUrl, 'paisley-color.jpg'),
  staticTemplate('paisley-bw', 'Paisley (B&W)', paisleyBwUrl, 'paisley-bw.jpg'),
  generatedPaisley,
];
