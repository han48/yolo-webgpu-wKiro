/**
 * Renderer utilities extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

/**
 * Fixed HSL colors for each of the 10 detection classes, indexed by class index.
 * Hue steps of 36 degrees starting from 0.
 */
const CLASS_COLORS_BY_INDEX = [
  'hsl(0, 80%, 55%)',    // 0: Candy
  'hsl(36, 80%, 55%)',   // 1: Piece
  'hsl(72, 80%, 55%)',   // 2: 1 Yen
  'hsl(108, 80%, 55%)',  // 3: 5 Yen
  'hsl(144, 80%, 55%)',  // 4: 10 Yen
  'hsl(180, 80%, 55%)',  // 5: 50 Yen
  'hsl(216, 80%, 55%)',  // 6: 100 Yen
  'hsl(252, 80%, 55%)',  // 7: 500 Yen
  'hsl(288, 80%, 55%)',  // 8: Container
  'hsl(324, 80%, 55%)',  // 9: Packaging
];

/**
 * Return the HSL color string for a given class index (0–9).
 * Returns a fallback gray for out-of-range indices.
 *
 * @param {number} classIndex — integer in [0, 9]
 * @returns {string} CSS color string, e.g. "hsl(0, 80%, 55%)"
 */
export function getClassColor(classIndex) {
  return CLASS_COLORS_BY_INDEX[classIndex] ?? 'hsl(0, 0%, 50%)';
}

/**
 * Format a detection label as "ClassName: 0.XX".
 * Confidence is rounded to exactly 2 decimal places using toFixed(2).
 *
 * @param {string} className   — e.g. "100 Yen"
 * @param {number} confidence  — value in [0, 1]
 * @returns {string} e.g. "100 Yen: 0.88"
 */
export function formatLabel(className, confidence) {
  return `${className}: ${confidence.toFixed(2)}`;
}

/**
 * Aggregate a list of detections into per-class stats, sorted by count descending.
 *
 * @param {Array<{ className: string, confidence: number }>} detections
 * @returns {Array<{ className: string, count: number, avgConfidence: number }>}
 */
export function aggregateDetections(detections) {
  /** @type {Map<string, { count: number, sumConfidence: number }>} */
  const statsMap = new Map();
  for (const det of detections) {
    const existing = statsMap.get(det.className);
    if (existing) {
      existing.count += 1;
      existing.sumConfidence += det.confidence;
    } else {
      statsMap.set(det.className, { count: 1, sumConfidence: det.confidence });
    }
  }

  const stats = [];
  for (const [className, { count, sumConfidence }] of statsMap) {
    stats.push({ className, count, avgConfidence: sumConfidence / count });
  }

  stats.sort((a, b) => b.count - a.count);
  return stats;
}

/**
 * Format an average confidence value (0–1) as a percentage string with 1 decimal place.
 * e.g. 0.875 → "87.5%"
 *
 * @param {number} confidence — value in [0, 1]
 * @returns {string} e.g. "87.5%"
 */
export function formatConfidencePercent(confidence) {
  return `${(confidence * 100).toFixed(1)}%`;
}
