import { generatePaisleyPattern } from './paisleyPattern';
import paisleyColorUrl from '../assets/templates/paisley-color.jpg';
import paisleyBwUrl from '../assets/templates/paisley-bw.jpg';
import hongKongUrl from '../assets/templates/hong-kong.jpg';
import dotPaintingUrl from '../assets/templates/dot-painting.jpg';
import greatWaveUrl from '../assets/templates/great-wave.jpg';
import monetBridgeUrl from '../assets/templates/monet-bridge.jpg';
import facesUrl from '../assets/templates/faces.jpg';
import flagsUrl from '../assets/templates/flags.jpg';
import flowerBedsUrl from '../assets/templates/flower-beds.jpg';
import lightningUrl from '../assets/templates/lightning.jpg';
import crackedEarthUrl from '../assets/templates/cracked-earth.jpg';
import pandasUrl from '../assets/templates/pandas.jpg';

import paisleyColorThumb from '../assets/templates/thumbs/paisley-color.jpg';
import paisleyBwThumb from '../assets/templates/thumbs/paisley-bw.jpg';
import hongKongThumb from '../assets/templates/thumbs/hong-kong.jpg';
import dotPaintingThumb from '../assets/templates/thumbs/dot-painting.jpg';
import greatWaveThumb from '../assets/templates/thumbs/great-wave.jpg';
import monetBridgeThumb from '../assets/templates/thumbs/monet-bridge.jpg';
import facesThumb from '../assets/templates/thumbs/faces.jpg';
import flagsThumb from '../assets/templates/thumbs/flags.jpg';
import flowerBedsThumb from '../assets/templates/thumbs/flower-beds.jpg';
import lightningThumb from '../assets/templates/thumbs/lightning.jpg';
import crackedEarthThumb from '../assets/templates/thumbs/cracked-earth.jpg';
import pandasThumb from '../assets/templates/thumbs/pandas.jpg';

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

/**
 * Every template is stored twice: a ~1280px JPEG that becomes the style image, and a 240px JPEG for the
 * picker. With a dozen of them, showing the picker used to mean fetching several megabytes of full-size
 * images to draw them at 72px; the thumbnails are a few hundred kilobytes for the whole set, and the
 * full image is fetched only for the one actually chosen.
 */
function staticTemplate(id: string, name: string, url: string, thumbnailUrl: string): BuiltInTemplate {
  return {
    id,
    name,
    getThumbnailUrl: () => Promise.resolve(thumbnailUrl),
    getFile: () => fileFromUrl(url, `${id}.jpg`, 'image/jpeg'),
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
  staticTemplate('paisley-color', 'Paisley (Color)', paisleyColorUrl, paisleyColorThumb),
  staticTemplate('paisley-bw', 'Paisley (B&W)', paisleyBwUrl, paisleyBwThumb),
  generatedPaisley,
  staticTemplate('hong-kong', 'Hong Kong', hongKongUrl, hongKongThumb),
  staticTemplate('dot-painting', 'Dot Painting', dotPaintingUrl, dotPaintingThumb),
  staticTemplate('great-wave', 'Great Wave', greatWaveUrl, greatWaveThumb),
  staticTemplate('monet-bridge', 'Monet Bridge', monetBridgeUrl, monetBridgeThumb),
  staticTemplate('flower-beds', 'Flower Beds', flowerBedsUrl, flowerBedsThumb),
  staticTemplate('lightning', 'Lightning', lightningUrl, lightningThumb),
  staticTemplate('cracked-earth', 'Cracked Earth', crackedEarthUrl, crackedEarthThumb),
  staticTemplate('flags', 'Flags', flagsUrl, flagsThumb),
  staticTemplate('faces', 'Faces', facesUrl, facesThumb),
  staticTemplate('pandas', 'Pandas', pandasUrl, pandasThumb),
];
