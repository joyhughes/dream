/**
 * Reading and writing PNG text chunks.
 *
 * A PNG is a signature followed by a chain of chunks, each one a length, a four-letter type, its data, and
 * a CRC. Decoders skip chunk types they do not recognize, so a `tEXt` chunk carrying our own data rides
 * along in a file every other program still reads as an ordinary image.
 *
 * Everything here works on bytes rather than Blobs, and synchronously. Saving an image on iOS has to reach
 * the share sheet inside the click that asked for it, and the user activation that permits it is spent by
 * the first `await` — so metadata cannot be something the save path waits for.
 */

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** The highest code point a Latin-1 byte can hold; anything above it has to be escaped. */
const MAX_LATIN1 = 0xff;

let crcTable: Uint32Array | null = null;

/** The CRC-32 PNG specifies for every chunk. Table built once, on first use. */
function crc32(bytes: Uint8Array, start: number, end: number): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let value = i;
      for (let bit = 0; bit < 8; bit++) {
        value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
      }
      crcTable[i] = value;
    }
  }

  let crc = 0xffffffff;
  for (let i = start; i < end; i++) {
    crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0;
}

function writeUint32(bytes: Uint8Array, offset: number, value: number) {
  bytes[offset] = (value >>> 24) & 0xff;
  bytes[offset + 1] = (value >>> 16) & 0xff;
  bytes[offset + 2] = (value >>> 8) & 0xff;
  bytes[offset + 3] = value & 0xff;
}

export function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && SIGNATURE.every((byte, index) => bytes[index] === byte);
}

interface Chunk {
  type: string;
  dataStart: number;
  dataEnd: number;
  /** Offset of this chunk's length field, i.e. where the chunk begins. */
  start: number;
}

/**
 * Walks the chunk chain. Stops at the first malformed length rather than reading past the buffer, so a
 * truncated or non-PNG file yields whatever was readable instead of throwing.
 */
function* chunks(bytes: Uint8Array): Generator<Chunk> {
  let offset = 8;

  while (offset + 12 <= bytes.length) {
    const length = readUint32(bytes, offset);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) return;

    yield {
      type: String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]),
      dataStart,
      dataEnd,
      start: offset,
    };

    offset = dataEnd + 4;
  }
}

/**
 * PNG's `tEXt` is Latin-1, so anything above it is escaped to a `\uXXXX` sequence on the way in. JSON reads
 * those escapes back as the original characters, which keeps the chunk spec-legal without the caller having
 * to think about what is in its strings.
 */
function toLatin1(text: string): Uint8Array {
  let escaped = '';

  // Escaped per UTF-16 code unit, not per code point. A character outside the basic plane is a surrogate
  // pair, and treating it as one unit escapes only its first half and drops the second — the emoji goes in
  // and a replacement character comes back. JSON reassembles the pair from the two escapes.
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    escaped += code > MAX_LATIN1 ? '\\u' + code.toString(16).padStart(4, '0') : text[i];
  }

  const bytes = new Uint8Array(escaped.length);
  for (let i = 0; i < escaped.length; i++) bytes[i] = escaped.charCodeAt(i) & 0xff;
  return bytes;
}

function fromLatin1(bytes: Uint8Array, start: number, end: number): string {
  let text = '';
  for (let i = start; i < end; i++) text += String.fromCharCode(bytes[i]);
  return text;
}

/** Offset of the null byte separating a tEXt chunk's keyword from its text, or -1 when there is none. */
function separatorIn(bytes: Uint8Array, chunk: Chunk): number {
  for (let i = chunk.dataStart; i < chunk.dataEnd; i++) {
    if (bytes[i] === 0) return i;
  }
  return -1;
}

/** The text stored under `keyword`, or null when this is not a PNG or carries no such chunk. */
export function readPngText(bytes: Uint8Array, keyword: string): string | null {
  if (!isPng(bytes)) return null;

  for (const chunk of chunks(bytes)) {
    if (chunk.type !== 'tEXt') continue;

    const separator = separatorIn(bytes, chunk);
    if (separator < 0) continue;

    if (fromLatin1(bytes, chunk.dataStart, separator) === keyword) {
      return fromLatin1(bytes, separator + 1, chunk.dataEnd);
    }
  }

  return null;
}

/**
 * A copy of the image with `text` stored under `keyword`, replacing any chunk already using that keyword so
 * re-saving does not accumulate copies. Returns the bytes unchanged if this is not a PNG.
 *
 * The chunk goes immediately before IEND, which must stay last.
 */
export function writePngText(bytes: Uint8Array, keyword: string, text: string): Uint8Array {
  if (!isPng(bytes)) return bytes;

  let endStart = -1;
  const replaced: Chunk[] = [];

  for (const chunk of chunks(bytes)) {
    if (chunk.type === 'IEND') {
      endStart = chunk.start;
      break;
    }
    if (chunk.type !== 'tEXt') continue;

    const separator = separatorIn(bytes, chunk);
    if (separator >= 0 && fromLatin1(bytes, chunk.dataStart, separator) === keyword) {
      replaced.push(chunk);
    }
  }

  if (endStart < 0) return bytes;

  const keywordBytes = toLatin1(keyword);
  const textBytes = toLatin1(text);
  const dataLength = keywordBytes.length + 1 + textBytes.length;
  const chunkLength = dataLength + 12;

  const removedLength = replaced.reduce((total, chunk) => total + (chunk.dataEnd + 4 - chunk.start), 0);
  const output = new Uint8Array(bytes.length - removedLength + chunkLength);

  // Copy everything ahead of IEND, skipping any earlier chunk under this keyword.
  let written = 0;
  let read = 0;
  for (const chunk of replaced) {
    output.set(bytes.subarray(read, chunk.start), written);
    written += chunk.start - read;
    read = chunk.dataEnd + 4;
  }
  output.set(bytes.subarray(read, endStart), written);
  written += endStart - read;

  const chunkStart = written;
  writeUint32(output, chunkStart, dataLength);
  output.set([0x74, 0x45, 0x58, 0x74], chunkStart + 4); // "tEXt"
  output.set(keywordBytes, chunkStart + 8);
  output[chunkStart + 8 + keywordBytes.length] = 0;
  output.set(textBytes, chunkStart + 9 + keywordBytes.length);
  writeUint32(output, chunkStart + 8 + dataLength, crc32(output, chunkStart + 4, chunkStart + 8 + dataLength));
  written += chunkLength;

  // Then IEND, still last.
  output.set(bytes.subarray(endStart), written);
  return output;
}
