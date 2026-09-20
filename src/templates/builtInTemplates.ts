import paisleyColorUrl from '../assets/templates/paisley-color.jpg';
import dotPaintingUrl from '../assets/templates/dot-painting.jpg';
import eyeballsUrl from '../assets/templates/eyeballs.jpg';
import stainedGlassUrl from '../assets/templates/stained-glass.jpg';
import hongKong2Url from '../assets/templates/hong-kong-2.jpg';
import starryNightUrl from '../assets/templates/starry-night.jpg';
import discoBallUrl from '../assets/templates/disco-ball.jpg';
import greatWaveUrl from '../assets/templates/great-wave.jpg';
import monetBridgeUrl from '../assets/templates/monet-bridge.jpg';
import facesUrl from '../assets/templates/faces.jpg';
import flagsUrl from '../assets/templates/flags.jpg';
import flowerBedsUrl from '../assets/templates/flower-beds.jpg';
import lightningUrl from '../assets/templates/lightning.jpg';
import crackedEarthUrl from '../assets/templates/cracked-earth.jpg';
import pandasUrl from '../assets/templates/pandas.jpg';

import paisleyColorThumb from '../assets/templates/thumbs/paisley-color.jpg';
import dotPaintingThumb from '../assets/templates/thumbs/dot-painting.jpg';
import eyeballsThumb from '../assets/templates/thumbs/eyeballs.jpg';
import stainedGlassThumb from '../assets/templates/thumbs/stained-glass.jpg';
import hongKong2Thumb from '../assets/templates/thumbs/hong-kong-2.jpg';
import starryNightThumb from '../assets/templates/thumbs/starry-night.jpg';
import discoBallThumb from '../assets/templates/thumbs/disco-ball.jpg';
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

export const BUILT_IN_TEMPLATES: BuiltInTemplate[] = [
  staticTemplate('paisley-color', 'Paisley', paisleyColorUrl, paisleyColorThumb),
  staticTemplate('hong-kong-2', 'Hong Kong', hongKong2Url, hongKong2Thumb),
  staticTemplate('dot-painting', 'Dot Painting', dotPaintingUrl, dotPaintingThumb),
  staticTemplate('stained-glass', 'Stained Glass', stainedGlassUrl, stainedGlassThumb),
  staticTemplate('starry-night', 'Starry Night', starryNightUrl, starryNightThumb),
  staticTemplate('great-wave', 'Great Wave', greatWaveUrl, greatWaveThumb),
  staticTemplate('monet-bridge', 'Monet Bridge', monetBridgeUrl, monetBridgeThumb),
  staticTemplate('flower-beds', 'Flower Beds', flowerBedsUrl, flowerBedsThumb),
  staticTemplate('disco-ball', 'Disco Ball', discoBallUrl, discoBallThumb),
  staticTemplate('lightning', 'Lightning', lightningUrl, lightningThumb),
  staticTemplate('cracked-earth', 'Cracked Earth', crackedEarthUrl, crackedEarthThumb),
  staticTemplate('flags', 'Flags', flagsUrl, flagsThumb),
  staticTemplate('faces', 'Faces', facesUrl, facesThumb),
  staticTemplate('pandas', 'Pandas', pandasUrl, pandasThumb),
  staticTemplate('eyeballs', 'Eyeballs', eyeballsUrl, eyeballsThumb),
];
