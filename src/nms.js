/**
 * NMS (Non-Maximum Suppression) logic extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

/**
 * Compute Intersection over Union (IoU) between two bounding boxes.
 * Boxes are in { x, y, width, height } format where (x, y) is top-left.
 *
 * @param {{ x: number, y: number, width: number, height: number }} boxA
 * @param {{ x: number, y: number, width: number, height: number }} boxB
 * @returns {number} IoU value in [0, 1]
 */
export function computeIoU(boxA, boxB) {
  const xA1 = boxA.x, yA1 = boxA.y, xA2 = boxA.x + boxA.width,  yA2 = boxA.y + boxA.height;
  const xB1 = boxB.x, yB1 = boxB.y, xB2 = boxB.x + boxB.width,  yB2 = boxB.y + boxB.height;

  const interX1 = Math.max(xA1, xB1);
  const interY1 = Math.max(yA1, yB1);
  const interX2 = Math.min(xA2, xB2);
  const interY2 = Math.min(yA2, yB2);

  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const intersection = interW * interH;

  if (intersection === 0) return 0;

  const areaA = boxA.width * boxA.height;
  const areaB = boxB.width * boxB.height;
  const union = areaA + areaB - intersection;

  return union <= 0 ? 0 : intersection / union;
}

/**
 * Apply Non-Maximum Suppression to a list of detections.
 * Detections are sorted by confidence descending; boxes of the same class
 * with IoU > iouThreshold are suppressed, keeping the highest-confidence box.
 *
 * @param {Array<{ classIndex: number, className: string, confidence: number, box: { x: number, y: number, width: number, height: number } }>} detections
 * @param {number} iouThreshold  — typically 0.45
 * @returns {Array} filtered detections
 */
export function applyNMS(detections, iouThreshold) {
  // Sort by confidence descending
  const sorted = detections.slice().sort((a, b) => b.confidence - a.confidence);

  const kept = [];
  const suppressed = new Uint8Array(sorted.length);

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (sorted[i].classIndex !== sorted[j].classIndex) continue;
      if (computeIoU(sorted[i].box, sorted[j].box) > iouThreshold) {
        suppressed[j] = 1;
      }
    }
  }

  return kept;
}
