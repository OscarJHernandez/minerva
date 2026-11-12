/** Preparing a picture of a formula for upload: check it, shrink it, re-encode it in the browser. */

export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const MAX_SOURCE_BYTES = 20_000_000;
export const MAX_SIDE = 1600;
/** A PNG larger than this is sent as JPEG instead (the server accepts up to 4 MB). */
const PNG_LIMIT_BYTES = 3_000_000;

export interface PreparedImage {
  /** base64 without the `data:` prefix */
  base64: string;
  /** a blob URL for the thumbnail; the owner must revoke it */
  previewUrl: string;
}

export class ImageError extends Error {}

/** Scale `width` x `height` down so the longer side is at most `max`; never scales up. */
export function fitWithin(
  width: number,
  height: number,
  max: number = MAX_SIDE,
): { width: number; height: number } {
  const longest = Math.max(width, height);
  if (longest <= max) return { width, height };
  const scale = max / longest;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/** Why a file cannot be used, or null. */
export function checkImageFile(file: Pick<File, "type" | "size">): string | null {
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) return "Use a PNG, JPEG or WebP image.";
  if (file.size > MAX_SOURCE_BYTES) return "That image is too large (20 MB at most).";
  return null;
}

function toBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new ImageError("Could not process the image."))),
      type,
      quality,
    ),
  );
}

async function toBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/**
 * Downscales and re-encodes through a canvas. Re-encoding also drops EXIF data (such as the
 * location a phone photo carries), so only the pixels leave the browser.
 */
export async function prepareImage(file: File): Promise<PreparedImage> {
  const problem = checkImageFile(file);
  if (problem) throw new ImageError(problem);
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new ImageError("That file could not be read as an image.");
  }
  try {
    const size = fitWithin(bitmap.width, bitmap.height);
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext("2d");
    if (!context) throw new ImageError("Could not process the image.");
    context.fillStyle = "#fff"; // transparent screenshots would otherwise turn black in JPEG
    context.fillRect(0, 0, size.width, size.height);
    context.drawImage(bitmap, 0, 0, size.width, size.height);
    let blob = await toBlob(canvas, "image/png");
    if (blob.size > PNG_LIMIT_BYTES) blob = await toBlob(canvas, "image/jpeg", 0.9);
    return { base64: await toBase64(blob), previewUrl: URL.createObjectURL(blob) };
  } finally {
    bitmap.close();
  }
}
