/**
 * Image preprocessing logic extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

export const MODEL_INPUT_SIZE = 640;

/**
 * Compute letterbox parameters for resizing an image to fit within a square
 * target canvas while preserving aspect ratio.
 *
 * @param {number} imgWidth   — original image width in pixels (>= 1)
 * @param {number} imgHeight  — original image height in pixels (>= 1)
 * @param {number} [targetSize=640]
 * @returns {{ scaledW: number, scaledH: number, scaleX: number, scaleY: number, padX: number, padY: number }}
 */
export function computeLetterboxParams(imgWidth, imgHeight, targetSize = MODEL_INPUT_SIZE) {
  const scale = Math.min(targetSize / imgWidth, targetSize / imgHeight);
  const scaledW = Math.min(Math.max(1, Math.round(imgWidth * scale)), targetSize);
  const scaledH = Math.min(Math.max(1, Math.round(imgHeight * scale)), targetSize);
  const padX = Math.floor((targetSize - scaledW) / 2);
  const padY = Math.floor((targetSize - scaledH) / 2);
  return {
    scaledW,
    scaledH,
    scaleX: scaledW / imgWidth,
    scaleY: scaledH / imgHeight,
    padX,
    padY,
  };
}

/**
 * Resize and letterbox-pad an image to 640×640, returning a Float32Array
 * tensor in CHW format (shape [1, 3, 640, 640]) with values normalized to
 * [0, 1], plus the scale and padding info needed to map detections back to
 * the original image space.
 *
 * @param {HTMLImageElement | HTMLCanvasElement} imageElement
 * @returns {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }}
 */
export function preprocessImage(imageElement) {
  const origW = imageElement.naturalWidth ?? imageElement.width;
  const origH = imageElement.naturalHeight ?? imageElement.height;

  const { scaledW, scaledH, scaleX, scaleY, padX, padY } =
    computeLetterboxParams(origW, origH, MODEL_INPUT_SIZE);

  // Draw onto an offscreen canvas
  const canvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const ctx = canvas.getContext('2d');

  // Fill with gray padding (128, 128, 128)
  ctx.fillStyle = 'rgb(128, 128, 128)';
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  // Draw the scaled image centered
  ctx.drawImage(imageElement, padX, padY, scaledW, scaledH);

  // Read pixel data (RGBA, HWC layout)
  const imageData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const pixels = imageData.data; // Uint8ClampedArray, length = 640*640*4

  // Build CHW Float32Array: [R flat, G flat, B flat]
  const numPixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const tensor = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    tensor[i]               = pixels[i * 4]     / 255.0; // R
    tensor[numPixels + i]   = pixels[i * 4 + 1] / 255.0; // G
    tensor[2 * numPixels + i] = pixels[i * 4 + 2] / 255.0; // B
  }

  return { tensor, scaleX, scaleY, padX, padY };
}
