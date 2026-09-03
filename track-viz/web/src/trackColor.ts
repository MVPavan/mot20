const FNV_OFFSET_BASIS = 2_166_136_261;
const FNV_PRIME = 16_777_619;
const CHROMA = 172;
const MINIMUM_CHANNEL = 58;

export function trackColor(sequence: string, trackId: number): string {
  let hash = FNV_OFFSET_BASIS;
  const bytes = new TextEncoder().encode(`${sequence}\u001f${trackId}`);
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  const hue = hash % 360;
  const distance = 60 - Math.abs((hue % 120) - 60);
  const intermediate = Math.floor((CHROMA * distance + 30) / 60);
  const sectors = [
    [CHROMA, intermediate, 0],
    [intermediate, CHROMA, 0],
    [0, CHROMA, intermediate],
    [0, intermediate, CHROMA],
    [intermediate, 0, CHROMA],
    [CHROMA, 0, intermediate],
  ];
  const rgb = sectors[Math.floor(hue / 60)].map((channel) => channel + MINIMUM_CHANNEL);
  return `#${rgb.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}