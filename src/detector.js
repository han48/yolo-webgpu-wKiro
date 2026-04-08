/**
 * Detector logic extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

export const CONFIDENCE_THRESHOLD = 0.25;

/**
 * Filter detections by confidence threshold.
 * Only detections with confidence >= threshold are kept.
 *
 * @param {Array<{ classIndex: number, className: string, confidence: number, box: { x: number, y: number, width: number, height: number } }>} detections
 * @param {number} [threshold=0.25]
 * @returns {Array} filtered detections
 */
export function filterByConfidence(detections, threshold = CONFIDENCE_THRESHOLD) {
  return detections.filter(d => d.confidence >= threshold);
}

/**
 * Scale a single bounding box from 640×640 model space back to original image space,
 * accounting for letterbox padding.
 *
 * Formula:
 *   x_orig = (cx - w/2 - padX) / scaleX
 *   y_orig = (cy - h/2 - padY) / scaleY
 *   w_orig = w / scaleX
 *   h_orig = h / scaleY
 *
 * @param {number} cx     — center x in 640×640 space
 * @param {number} cy     — center y in 640×640 space
 * @param {number} w      — width in 640×640 space
 * @param {number} h      — height in 640×640 space
 * @param {number} scaleX — scaledW / origW
 * @param {number} scaleY — scaledH / origH
 * @param {number} padX   — horizontal padding (px in 640 space)
 * @param {number} padY   — vertical padding (px in 640 space)
 * @returns {{ x: number, y: number, width: number, height: number }} box in original image space
 */
export function scaleBox(cx, cy, w, h, scaleX, scaleY, padX, padY) {
  return {
    x:      (cx - w / 2 - padX) / scaleX,
    y:      (cy - h / 2 - padY) / scaleY,
    width:  w / scaleX,
    height: h / scaleY,
  };
}
