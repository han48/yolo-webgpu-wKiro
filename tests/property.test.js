/**
 * Property-Based Tests — YOLO Image Detection
 * Feature: yolo-image-detection
 *
 * Validates: Requirements 2.2, 2.3
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { isAcceptedFileType, ACCEPTED_TYPES } from '../src/validation.js';

/**
 * Property 1: File format validation
 * Tag: Feature: yolo-image-detection, Property 1: File format validation
 *
 * For any file MIME type, the app SHALL accept the file if and only if its
 * type is one of `image/png`, `image/jpeg`, or `image/webp`.
 * Any other MIME type must be rejected.
 *
 * Validates: Requirements 2.2, 2.3
 */
describe('Property 1: File format validation', () => {
  it('accepts image/png, image/jpeg, image/webp and rejects everything else', () => {
    // Arbitrary MIME type strings: combine type/subtype patterns plus random strings
    const mimeTypeArb = fc.oneof(
      // Valid accepted types — must all be accepted
      fc.constantFrom(...ACCEPTED_TYPES),
      // Common image types that are NOT accepted
      fc.constantFrom(
        'image/gif',
        'image/bmp',
        'image/tiff',
        'image/svg+xml',
        'image/avif',
        'image/heic',
      ),
      // Completely random strings (non-image types, empty, etc.)
      fc.string({ minLength: 0, maxLength: 50 }),
      // Well-formed type/subtype patterns
      fc.tuple(
        fc.constantFrom('image', 'video', 'audio', 'application', 'text'),
        fc.stringMatching(/^[a-z0-9+\-]{1,20}$/),
      ).map(([type, subtype]) => `${type}/${subtype}`),
    );

    fc.assert(
      fc.property(mimeTypeArb, (mimeType) => {
        const accepted = isAcceptedFileType(mimeType);
        const shouldAccept = ACCEPTED_TYPES.includes(mimeType);
        return accepted === shouldAccept;
      }),
      { numRuns: 100 },
    );
  });

  it('always accepts the three valid MIME types', () => {
    for (const validType of ACCEPTED_TYPES) {
      expect(isAcceptedFileType(validType)).toBe(true);
    }
  });

  it('rejects any MIME type not in the accepted list', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 50 }).filter(
          (s) => !ACCEPTED_TYPES.includes(s),
        ),
        (mimeType) => {
          return isAcceptedFileType(mimeType) === false;
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 2: Preprocessing output shape and normalization
 * Tag: Feature: yolo-image-detection, Property 2: Preprocessing output shape and normalization
 *
 * For any input image with arbitrary width and height (>= 1px), the
 * `preprocessImage` function SHALL return a Float32Array tensor of exactly
 * 1 x 3 x 640 x 640 = 1,228,800 elements, with all values in [0.0, 1.0].
 *
 * Validates: Requirements 3.2
 */
import { beforeAll } from 'vitest';
import { computeLetterboxParams, preprocessImage, MODEL_INPUT_SIZE } from '../src/preprocessor.js';

const EXPECTED_TENSOR_LENGTH = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3; // 1,228,800

describe('Property 2: Preprocessing output shape and normalization', () => {
  /**
   * Pure math property: letterbox params keep the scaled image within the
   * 640×640 canvas and produce non-negative padding.
   */
  it('letterbox params always fit within 640x640 for any image dimensions', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        (width, height) => {
          const { scaledW, scaledH, padX, padY } = computeLetterboxParams(width, height);

          // Scaled dimensions must be positive and fit within the target canvas
          return (
            scaledW >= 1 &&
            scaledH >= 1 &&
            scaledW <= MODEL_INPUT_SIZE &&
            scaledH <= MODEL_INPUT_SIZE &&
            padX >= 0 &&
            padY >= 0 &&
            // The image placed at padX must not exceed the canvas width
            padX + scaledW <= MODEL_INPUT_SIZE &&
            padY + scaledH <= MODEL_INPUT_SIZE
          );
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * Pure math property: scale factors are always positive.
   * (Images smaller than 640px are upscaled, so scaleX/scaleY can be > 1.)
   */
  it('scaleX and scaleY are always positive for any image dimensions', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        (width, height) => {
          const { scaleX, scaleY } = computeLetterboxParams(width, height);
          return scaleX > 0 && scaleY > 0;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * Full preprocessImage property: tensor length = 1,228,800 and all values in [0, 1].
   *
   * jsdom does not implement HTMLCanvasElement.getContext(), so we mock the
   * OffscreenCanvas and its 2D context to return synthetic pixel data.
   * The mock fills the 640×640 canvas with a known RGBA value so we can
   * verify normalization independently of the browser rendering engine.
   */
  it('tensor has exactly 1,228,800 elements and all values in [0.0, 1.0]', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        // Pixel values 0–255 for R, G, B channels
        fc.integer({ min: 0, max: 255 }),
        fc.integer({ min: 0, max: 255 }),
        fc.integer({ min: 0, max: 255 }),
        (width, height, r, g, b) => {
          const numPixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

          // Build synthetic RGBA pixel data (solid color)
          const pixels = new Uint8ClampedArray(numPixels * 4);
          for (let i = 0; i < numPixels; i++) {
            pixels[i * 4]     = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255;
          }

          // Mock OffscreenCanvas so preprocessImage can run without a real browser
          globalThis.OffscreenCanvas = class MockOffscreenCanvas {
            constructor() {}
            getContext() {
              return {
                fillStyle: '',
                fillRect() {},
                drawImage() {},
                getImageData() {
                  return { data: pixels };
                },
              };
            }
          };

          // Mock image element with the given dimensions
          const mockImage = { naturalWidth: width, naturalHeight: height };

          const result = preprocessImage(mockImage);

          // 1. Tensor length must be exactly 1,228,800
          if (result.tensor.length !== EXPECTED_TENSOR_LENGTH) return false;

          // 2. All values must be in [0.0, 1.0]
          for (let i = 0; i < result.tensor.length; i++) {
            if (result.tensor[i] < 0.0 || result.tensor[i] > 1.0) return false;
          }

          // 3. Verify normalization: solid color r,g,b → expected normalized values
          const expectedR = r / 255;
          const expectedG = g / 255;
          const expectedB = b / 255;
          const tolerance = 1e-6;

          if (Math.abs(result.tensor[0] - expectedR) > tolerance) return false;
          if (Math.abs(result.tensor[numPixels] - expectedG) > tolerance) return false;
          if (Math.abs(result.tensor[2 * numPixels] - expectedB) > tolerance) return false;

          return true;
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 4: NMS removes high-overlap boxes
 * Tag: Feature: yolo-image-detection, Property 4: NMS removes high-overlap boxes
 *
 * For any list of detections after NMS with IoU threshold 0.45, no two
 * remaining boxes of the same class SHALL have an IoU greater than 0.45.
 * The box with the highest confidence in each overlapping group SHALL be retained.
 *
 * Validates: Requirements 3.4
 */
import { computeIoU, applyNMS } from '../src/nms.js';

const NMS_THRESHOLD = 0.45;

// Arbitrary for a single bounding box
const boundingBoxArb = fc.record({
  x:      fc.integer({ min: 0, max: 500 }),
  y:      fc.integer({ min: 0, max: 500 }),
  width:  fc.integer({ min: 1, max: 200 }),
  height: fc.integer({ min: 1, max: 200 }),
});

// Arbitrary for a single detection
const detectionArb = fc.record({
  classIndex: fc.integer({ min: 0, max: 9 }),
  className:  fc.constantFrom('Candy','Piece','1 Yen','5 Yen','10 Yen','50 Yen','100 Yen','500 Yen','Container','Packaging'),
  confidence: fc.float({ min: 0, max: 1, noNaN: true }),
  box:        boundingBoxArb,
});

describe('Property 4: NMS removes high-overlap boxes', () => {
  it('no two remaining boxes of the same class have IoU > 0.45 after NMS', () => {
    fc.assert(
      fc.property(
        fc.array(detectionArb, { minLength: 0, maxLength: 30 }),
        (detections) => {
          const result = applyNMS(detections, NMS_THRESHOLD);

          // For every pair of kept detections with the same classIndex,
          // their IoU must be <= NMS_THRESHOLD
          for (let i = 0; i < result.length; i++) {
            for (let j = i + 1; j < result.length; j++) {
              if (result[i].classIndex !== result[j].classIndex) continue;
              const iou = computeIoU(result[i].box, result[j].box);
              if (iou > NMS_THRESHOLD) return false;
            }
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  it('the highest-confidence box in each overlapping group is retained', () => {
    fc.assert(
      fc.property(
        fc.array(detectionArb, { minLength: 1, maxLength: 30 }),
        (detections) => {
          const result = applyNMS(detections, NMS_THRESHOLD);
          const resultSet = new Set(result);

          // For every kept detection, no suppressed detection of the same class
          // that overlaps it (IoU > threshold) should have a higher confidence
          for (const kept of result) {
            for (const candidate of detections) {
              if (resultSet.has(candidate)) continue;
              if (candidate.classIndex !== kept.classIndex) continue;
              const iou = computeIoU(kept.box, candidate.box);
              if (iou > NMS_THRESHOLD) {
                // The suppressed candidate must NOT have higher confidence than the kept box
                if (candidate.confidence > kept.confidence) return false;
              }
            }
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 3: Confidence threshold filtering
 * Tag: Feature: yolo-image-detection, Property 3: Confidence threshold filtering
 *
 * For any list of raw detections with arbitrary confidence scores, after
 * applying the confidence filter with threshold 0.25, every detection in the
 * output SHALL have `confidence >= 0.25`, and no detection with
 * `confidence < 0.25` SHALL appear in the output.
 *
 * Validates: Requirements 3.5
 */
import { filterByConfidence, CONFIDENCE_THRESHOLD } from '../src/detector.js';

describe('Property 3: Confidence threshold filtering', () => {
  it('every detection in output has confidence >= 0.25', () => {
    fc.assert(
      fc.property(
        fc.array(detectionArb, { minLength: 0, maxLength: 50 }),
        (detections) => {
          const result = filterByConfidence(detections, CONFIDENCE_THRESHOLD);
          return result.every(d => d.confidence >= CONFIDENCE_THRESHOLD);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('no detection with confidence < 0.25 appears in output', () => {
    fc.assert(
      fc.property(
        fc.array(detectionArb, { minLength: 0, maxLength: 50 }),
        (detections) => {
          const result = filterByConfidence(detections, CONFIDENCE_THRESHOLD);
          return !result.some(d => d.confidence < CONFIDENCE_THRESHOLD);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('all detections with confidence >= 0.25 from input are preserved (no false negatives)', () => {
    fc.assert(
      fc.property(
        fc.array(detectionArb, { minLength: 0, maxLength: 50 }),
        (detections) => {
          const result = filterByConfidence(detections, CONFIDENCE_THRESHOLD);
          const resultSet = new Set(result);
          // Every input detection with confidence >= threshold must appear in output
          return detections
            .filter(d => d.confidence >= CONFIDENCE_THRESHOLD)
            .every(d => resultSet.has(d));
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 5: Bounding box coordinate scaling
 * Tag: Feature: yolo-image-detection, Property 5: Bounding box coordinate scaling
 *
 * For any original image dimensions (W, H) and any bounding box in 640x640 space
 * with valid scaleX, scaleY, padX, padY, the scaled coordinates SHALL map the box
 * back to the correct position in the original image space. Specifically, a box at
 * the center of the 640x640 space SHALL map to the center of the original image.
 *
 * Validates: Requirements 4.5
 */
import { scaleBox } from '../src/detector.js';

describe('Property 5: Bounding box coordinate scaling', () => {
  const TOLERANCE = 1e-9;

  it('scaled coordinates map correctly back to original image space', () => {
    /**
     * **Validates: Requirements 4.5**
     *
     * For any image dimensions and any box in 640x640 space, applying scaleBox
     * with the letterbox params must produce coordinates that satisfy the inverse
     * of the letterbox transform.
     */
    fc.assert(
      fc.property(
        // Original image dimensions
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        // Box in 640x640 space: cx, cy in [0, 640], w/h in [1, 640]
        fc.float({ min: 0, max: 640, noNaN: true }),
        fc.float({ min: 0, max: 640, noNaN: true }),
        fc.float({ min: 1, max: 640, noNaN: true }),
        fc.float({ min: 1, max: 640, noNaN: true }),
        (imgW, imgH, cx, cy, w, h) => {
          const { scaleX, scaleY, padX, padY } = computeLetterboxParams(imgW, imgH);
          const result = scaleBox(cx, cy, w, h, scaleX, scaleY, padX, padY);

          // Verify the inverse formula holds exactly:
          //   x_orig = (cx - w/2 - padX) / scaleX
          //   y_orig = (cy - h/2 - padY) / scaleY
          //   w_orig = w / scaleX
          //   h_orig = h / scaleY
          const expectedX = (cx - w / 2 - padX) / scaleX;
          const expectedY = (cy - h / 2 - padY) / scaleY;
          const expectedW = w / scaleX;
          const expectedH = h / scaleY;

          return (
            Math.abs(result.x - expectedX) <= TOLERANCE &&
            Math.abs(result.y - expectedY) <= TOLERANCE &&
            Math.abs(result.width  - expectedW) <= TOLERANCE &&
            Math.abs(result.height - expectedH) <= TOLERANCE
          );
        },
      ),
      { numRuns: 100 },
    );
  });

  it('a box at the center of 640x640 maps to the center of the original image', () => {
    /**
     * **Validates: Requirements 4.5**
     *
     * A box whose center is exactly at (320, 320) in 640x640 space (the center
     * of the model input) must map to the center of the original image.
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        // Box size in 640x640 space (small box centered at 320,320)
        fc.float({ min: 1, max: 100, noNaN: true }),
        fc.float({ min: 1, max: 100, noNaN: true }),
        (imgW, imgH, boxW, boxH) => {
          const { scaleX, scaleY, padX, padY, scaledW, scaledH } =
            computeLetterboxParams(imgW, imgH);

          // Center of the 640x640 canvas
          const cx640 = padX + scaledW / 2;
          const cy640 = padY + scaledH / 2;

          const result = scaleBox(cx640, cy640, boxW, boxH, scaleX, scaleY, padX, padY);

          // The center of the scaled box should be at the center of the original image
          const resultCx = result.x + result.width / 2;
          const resultCy = result.y + result.height / 2;

          const expectedCx = imgW / 2;
          const expectedCy = imgH / 2;

          // Allow small floating-point tolerance (rounding in computeLetterboxParams)
          const tol = 1.0;
          return (
            Math.abs(resultCx - expectedCx) <= tol &&
            Math.abs(resultCy - expectedCy) <= tol
          );
        },
      ),
      { numRuns: 100 },
    );
  });

  it('scaled box dimensions are proportional to original (w_orig = w / scaleX)', () => {
    /**
     * **Validates: Requirements 4.5**
     *
     * The width and height of the scaled box must be exactly w/scaleX and h/scaleY.
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4000 }),
        fc.integer({ min: 1, max: 4000 }),
        fc.float({ min: 0, max: 640, noNaN: true }),
        fc.float({ min: 0, max: 640, noNaN: true }),
        fc.float({ min: 1, max: 640, noNaN: true }),
        fc.float({ min: 1, max: 640, noNaN: true }),
        (imgW, imgH, cx, cy, w, h) => {
          const { scaleX, scaleY, padX, padY } = computeLetterboxParams(imgW, imgH);
          const result = scaleBox(cx, cy, w, h, scaleX, scaleY, padX, padY);

          const expectedW = w / scaleX;
          const expectedH = h / scaleY;

          return (
            Math.abs(result.width  - expectedW) <= TOLERANCE &&
            Math.abs(result.height - expectedH) <= TOLERANCE
          );
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 6: Detection rendering completeness
 * Tag: Feature: yolo-image-detection, Property 6: Detection rendering completeness
 *
 * For any list of N detections, each detection SHALL be rendered with:
 * (a) a bounding box in the color corresponding to its class index, and
 * (b) a label string containing the class name and confidence score rounded
 *     to 2 decimal places.
 *
 * Validates: Requirements 4.1, 4.2, 4.3
 */
import { formatLabel, getClassColor } from '../src/renderer.js';

/** Expected HSL colors indexed by class index 0–9 */
const EXPECTED_COLORS = [
  'hsl(0, 80%, 55%)',
  'hsl(36, 80%, 55%)',
  'hsl(72, 80%, 55%)',
  'hsl(108, 80%, 55%)',
  'hsl(144, 80%, 55%)',
  'hsl(180, 80%, 55%)',
  'hsl(216, 80%, 55%)',
  'hsl(252, 80%, 55%)',
  'hsl(288, 80%, 55%)',
  'hsl(324, 80%, 55%)',
];

describe('Property 6: Detection rendering completeness', () => {
  /**
   * **Validates: Requirements 4.1, 4.2, 4.3**
   *
   * For any detection, formatLabel(className, confidence) must contain the
   * class name and the confidence rounded to exactly 2 decimal places.
   */
  it('label contains class name for any detection', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 50 }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (className, confidence) => {
          const label = formatLabel(className, confidence);
          return label.includes(className);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('label contains confidence rounded to exactly 2 decimal places', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 50 }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (className, confidence) => {
          const label = formatLabel(className, confidence);
          const expected = confidence.toFixed(2);
          return label.includes(expected);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('label format is "ClassName: 0.XX" for any detection list', () => {
    /**
     * **Validates: Requirements 4.1, 4.2, 4.3**
     *
     * For any list of detections, each label must match the pattern
     * "ClassName: 0.XX" where 0.XX is confidence rounded to 2 decimal places.
     */
    const detectionArb2 = fc.record({
      classIndex: fc.integer({ min: 0, max: 9 }),
      className: fc.string({ minLength: 1, maxLength: 50 }),
      confidence: fc.float({ min: 0, max: 1, noNaN: true }),
    });

    fc.assert(
      fc.property(
        fc.array(detectionArb2, { minLength: 0, maxLength: 20 }),
        (detections) => {
          for (const det of detections) {
            const label = formatLabel(det.className, det.confidence);
            // Must contain class name
            if (!label.includes(det.className)) return false;
            // Must contain confidence rounded to 2 decimal places
            if (!label.includes(det.confidence.toFixed(2))) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  it('getClassColor returns correct HSL color for each class index 0–9', () => {
    /**
     * **Validates: Requirements 4.2**
     *
     * For each class index 0–9, getClassColor must return the exact HSL string
     * defined in the design (hue = classIndex * 36, saturation 80%, lightness 55%).
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 9 }),
        (classIndex) => {
          const color = getClassColor(classIndex);
          return color === EXPECTED_COLORS[classIndex];
        },
      ),
      { numRuns: 100 },
    );
  });

  it('getClassColor hue follows 36-degree step pattern for all valid indices', () => {
    /**
     * **Validates: Requirements 4.2**
     *
     * The hue for class index i must be i * 36 degrees.
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 9 }),
        (classIndex) => {
          const color = getClassColor(classIndex);
          const expectedHue = classIndex * 36;
          return color === `hsl(${expectedHue}, 80%, 55%)`;
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 8: Detection table correctness
 * Tag: Feature: yolo-image-detection, Property 8: Detection table correctness
 *
 * For any non-empty list of detections, the rendered table SHALL:
 * (a) contain columns for class name, count, and average confidence;
 * (b) display average confidence as a percentage string (e.g. "87.5%"); and
 * (c) have rows sorted by count in descending order.
 *
 * Validates: Requirements 5.2, 5.3, 5.4, 5.5
 */
import { aggregateDetections, formatConfidencePercent } from '../src/renderer.js';

// Arbitrary for a single detection used in table tests
const tableDetectionArb = fc.record({
  classIndex: fc.integer({ min: 0, max: 9 }),
  className: fc.constantFrom(
    'Candy', 'Piece', '1 Yen', '5 Yen', '10 Yen',
    '50 Yen', '100 Yen', '500 Yen', 'Container', 'Packaging',
  ),
  confidence: fc.float({ min: 0, max: 1, noNaN: true }),
  box: fc.record({
    x: fc.integer({ min: 0, max: 500 }),
    y: fc.integer({ min: 0, max: 500 }),
    width: fc.integer({ min: 1, max: 200 }),
    height: fc.integer({ min: 1, max: 200 }),
  }),
});

describe('Property 8: Detection table correctness', () => {
  /**
   * **Validates: Requirements 5.5**
   *
   * For any non-empty detection list, aggregateDetections must return rows
   * sorted by count in descending order.
   */
  it('rows are sorted by count in descending order', () => {
    fc.assert(
      fc.property(
        fc.array(tableDetectionArb, { minLength: 1, maxLength: 50 }),
        (detections) => {
          const stats = aggregateDetections(detections);
          for (let i = 0; i < stats.length - 1; i++) {
            if (stats[i].count < stats[i + 1].count) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 5.3**
   *
   * For any non-empty detection list, avgConfidence for each class must equal
   * the sum of confidences divided by the count.
   */
  it('avgConfidence is calculated correctly (sum / count) for each class', () => {
    fc.assert(
      fc.property(
        fc.array(tableDetectionArb, { minLength: 1, maxLength: 50 }),
        (detections) => {
          const stats = aggregateDetections(detections);

          // Recompute expected avg per class from raw detections
          const expected = new Map();
          for (const det of detections) {
            const e = expected.get(det.className) ?? { sum: 0, count: 0 };
            e.sum += det.confidence;
            e.count += 1;
            expected.set(det.className, e);
          }

          const TOLERANCE = 1e-9;
          for (const row of stats) {
            const { sum, count } = expected.get(row.className);
            const expectedAvg = sum / count;
            if (Math.abs(row.avgConfidence - expectedAvg) > TOLERANCE) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 5.4**
   *
   * formatConfidencePercent(0.875) must equal "87.5%".
   */
  it('formatConfidencePercent(0.875) === "87.5%"', () => {
    expect(formatConfidencePercent(0.875)).toBe('87.5%');
  });

  /**
   * **Validates: Requirements 5.4**
   *
   * For any confidence value in [0, 1], formatConfidencePercent must return
   * a string ending with "%".
   */
  it('formatConfidencePercent always produces a string ending with "%"', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1, noNaN: true }),
        (confidence) => {
          const result = formatConfidencePercent(confidence);
          return typeof result === 'string' && result.endsWith('%');
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 5.4**
   *
   * The numeric part of formatConfidencePercent must equal
   * (confidence * 100) rounded to 1 decimal place.
   */
  it('percentage value equals avgConfidence * 100 rounded to 1 decimal', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1, noNaN: true }),
        (confidence) => {
          const result = formatConfidencePercent(confidence);
          const numericPart = parseFloat(result.slice(0, -1)); // strip "%"
          const expected = parseFloat((confidence * 100).toFixed(1));
          return Math.abs(numericPart - expected) < 1e-9;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
   *
   * Combined: for any non-empty detection list, aggregateDetections + formatConfidencePercent
   * together produce rows sorted by count descending with correct percentage strings.
   */
  it('combined: sorted rows with correct percentage display for any detection list', () => {
    fc.assert(
      fc.property(
        fc.array(tableDetectionArb, { minLength: 1, maxLength: 50 }),
        (detections) => {
          const stats = aggregateDetections(detections);

          // (a) Must have className, count, avgConfidence fields
          for (const row of stats) {
            if (typeof row.className !== 'string') return false;
            if (typeof row.count !== 'number' || row.count < 1) return false;
            if (typeof row.avgConfidence !== 'number') return false;
          }

          // (b) Percentage string format is correct
          for (const row of stats) {
            const pct = formatConfidencePercent(row.avgConfidence);
            if (!pct.endsWith('%')) return false;
            const expected = (row.avgConfidence * 100).toFixed(1) + '%';
            if (pct !== expected) return false;
          }

          // (c) Sorted by count descending
          for (let i = 0; i < stats.length - 1; i++) {
            if (stats[i].count < stats[i + 1].count) return false;
          }

          return true;
        },
      ),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 7: Results cleared on new image selection
 * Tag: Feature: yolo-image-detection, Property 7: Results cleared on new image selection
 *
 * For any sequence of image selections, after selecting a new image, no
 * detection results (bounding boxes, table rows) from the previous inference
 * SHALL remain visible on the UI.
 *
 * Validates: Requirements 2.5
 */
import { clearResults } from '../src/ui.js';

/**
 * Create a mock 2D canvas context that tracks clearRect calls.
 * jsdom does not implement HTMLCanvasElement.getContext(), so we use a mock
 * that records whether clearRect was called with the full canvas dimensions.
 */
function createMockContext(canvas) {
  const calls = { clearRect: [] };
  return {
    calls,
    fillStyle: '',
    fillRect() {},
    clearRect(x, y, w, h) {
      calls.clearRect.push({ x, y, w, h });
    },
    getImageData() {
      return { data: new Uint8ClampedArray(canvas.width * canvas.height * 4) };
    },
  };
}

/**
 * Create a minimal mock canvas element with a controllable 2D context.
 * Returns the canvas and its mock context so tests can inspect calls.
 */
function createMockCanvas(width = 640, height = 480) {
  const canvas = { width, height, _ctx: null };
  const ctx = createMockContext(canvas);
  canvas._ctx = ctx;
  canvas.getContext = () => ctx;
  return canvas;
}

/**
 * Create a minimal jsdom DOM structure for the table parts.
 */
function createMockTableDOM() {
  const tableSection = document.createElement('div');
  tableSection.style.display = 'block';

  const tableBody = document.createElement('tbody');
  tableSection.appendChild(tableBody);

  return { tableSection, tableBody };
}

/**
 * Simulate a previous detection result by marking the canvas as "dirty"
 * and adding table rows.
 */
function simulateDetectionResult(mockCanvas, tableSection, tableBody, numRows) {
  // Mark canvas as having content (simulate bounding boxes drawn)
  mockCanvas._ctx.calls.clearRect = []; // reset tracking

  // Show the table section
  tableSection.style.display = 'block';

  // Add table rows to simulate detection results
  for (let i = 0; i < numRows; i++) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>Class ${i}</td><td>${i + 1}</td><td>75.0%</td>`;
    tableBody.appendChild(tr);
  }
}

describe('Property 7: Results cleared on new image selection', () => {
  /**
   * **Validates: Requirements 2.5**
   *
   * For any sequence of image selections (simulated by calling clearResults
   * multiple times), after each call the stats table SHALL be hidden.
   */
  it('table section is hidden after every image selection in a sequence', () => {
    fc.assert(
      fc.property(
        // Number of image selections in the sequence (1 to 10)
        fc.integer({ min: 1, max: 10 }),
        // Number of table rows from the previous detection (0 to 20)
        fc.integer({ min: 0, max: 20 }),
        (numSelections, numRows) => {
          const mockCanvas = createMockCanvas();
          const { tableSection, tableBody } = createMockTableDOM();

          for (let i = 0; i < numSelections; i++) {
            simulateDetectionResult(mockCanvas, tableSection, tableBody, numRows);
            clearResults(mockCanvas, tableSection, tableBody);

            // The table section must be hidden after each selection
            if (tableSection.style.display !== 'none') return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 2.5**
   *
   * For any sequence of image selections, after each clearResults call,
   * the table body SHALL contain no rows.
   */
  it('table body has no rows after every image selection in a sequence', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        fc.integer({ min: 0, max: 20 }),
        (numSelections, numRows) => {
          const mockCanvas = createMockCanvas();
          const { tableSection, tableBody } = createMockTableDOM();

          for (let i = 0; i < numSelections; i++) {
            simulateDetectionResult(mockCanvas, tableSection, tableBody, numRows);
            clearResults(mockCanvas, tableSection, tableBody);

            // No table rows should remain
            if (tableBody.children.length !== 0) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 2.5**
   *
   * For any sequence of image selections, after each clearResults call,
   * clearRect SHALL have been called on the canvas context, indicating
   * that the result canvas was cleared.
   */
  it('clearRect is called on the canvas context after every image selection', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        fc.integer({ min: 0, max: 20 }),
        (numSelections, numRows) => {
          const mockCanvas = createMockCanvas();
          const { tableSection, tableBody } = createMockTableDOM();

          for (let i = 0; i < numSelections; i++) {
            simulateDetectionResult(mockCanvas, tableSection, tableBody, numRows);
            clearResults(mockCanvas, tableSection, tableBody);

            // clearRect must have been called at least once
            if (mockCanvas._ctx.calls.clearRect.length === 0) return false;

            // The clearRect call must cover the full canvas dimensions
            const call = mockCanvas._ctx.calls.clearRect[0];
            if (call.x !== 0 || call.y !== 0) return false;
            if (call.w !== mockCanvas.width || call.h !== mockCanvas.height) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });

  /**
   * **Validates: Requirements 2.5**
   *
   * Combined property: for any sequence of image selections, after each
   * clearResults call, ALL of the following hold simultaneously:
   * - table section is hidden
   * - table body has no rows
   * - clearRect was called on the canvas (canvas content cleared)
   */
  it('combined: all results cleared simultaneously after each image selection', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        fc.integer({ min: 1, max: 20 }),
        (numSelections, numRows) => {
          const mockCanvas = createMockCanvas();
          const { tableSection, tableBody } = createMockTableDOM();

          for (let i = 0; i < numSelections; i++) {
            simulateDetectionResult(mockCanvas, tableSection, tableBody, numRows);
            clearResults(mockCanvas, tableSection, tableBody);

            // (a) Table section hidden
            if (tableSection.style.display !== 'none') return false;

            // (b) No table rows
            if (tableBody.children.length !== 0) return false;

            // (c) Canvas cleared via clearRect covering full dimensions
            const clearCalls = mockCanvas._ctx.calls.clearRect;
            if (clearCalls.length === 0) return false;
            const call = clearCalls[0];
            if (call.x !== 0 || call.y !== 0) return false;
            if (call.w !== mockCanvas.width || call.h !== mockCanvas.height) return false;
          }
          return true;
        },
      ),
      { numRuns: 100 },
    );
  });
});
