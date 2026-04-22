/**
 * Detector for anchor-based YOLO models (yolo12 and similar).
 * Output format: [1, 4+numClasses, 8400] — cx, cy, w, h + class scores (already sigmoid).
 * Requires NMS post-processing.
 */

/* global ort */

function filterByConfidence(detections, threshold = 0.25) {
  return detections.filter(d => d.confidence >= threshold);
}

function computeIoU(boxA, boxB) {
  const xA1 = boxA.x, yA1 = boxA.y, xA2 = boxA.x + boxA.width,  yA2 = boxA.y + boxA.height;
  const xB1 = boxB.x, yB1 = boxB.y, xB2 = boxB.x + boxB.width,  yB2 = boxB.y + boxB.height;
  const interW = Math.max(0, Math.min(xA2, xB2) - Math.max(xA1, xB1));
  const interH = Math.max(0, Math.min(yA2, yB2) - Math.max(yA1, yB1));
  const intersection = interW * interH;
  if (intersection === 0) return 0;
  const union = boxA.width * boxA.height + boxB.width * boxB.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function applyNMS(detections, iouThreshold) {
  const sorted = detections.slice().sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  const suppressed = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (sorted[i].classIndex !== sorted[j].classIndex) continue;
      if (computeIoU(sorted[i].box, sorted[j].box) > iouThreshold) suppressed[j] = 1;
    }
  }
  return kept;
}

export class DetectorYolo {
  /**
   * @param {ort.InferenceSession} session
   * @param {string[]} classNames
   */
  constructor(session, classNames) {
    this.session    = session;
    this.classNames = classNames;
  }

  /**
   * Run full detection pipeline: inference → parse → filter → NMS → scale.
   *
   * @param {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }} preprocessResult
   * @param {number} [confThreshold=0.25]
   * @param {number} [iouThreshold=0.45]
   * @returns {Promise<Array>}
   */
  async runDetectionYolo26(preprocessResult, confThreshold = 0.25, iouThreshold = 0.45) {
    const { tensor, scaleX, scaleY, padX, padY } = preprocessResult;
    const NUM_ANCHORS = 8400;

    const ortTensor  = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    const results    = await this.session.run({ images: ortTensor });
    const outputData = results[Object.keys(results)[0]].data;

    // Parse [1, 4+C, 8400] — rows: cx, cy, w, h, class0..classN
    const raw = [];
    for (let i = 0; i < NUM_ANCHORS; i++) {
      const cx = outputData[0 * NUM_ANCHORS + i];
      const cy = outputData[1 * NUM_ANCHORS + i];
      const w  = outputData[2 * NUM_ANCHORS + i];
      const h  = outputData[3 * NUM_ANCHORS + i];

      let confidence = -Infinity, classIndex = 0;
      for (let c = 0; c < this.classNames.length; c++) {
        const score = outputData[(4 + c) * NUM_ANCHORS + i];
        if (score > confidence) { confidence = score; classIndex = c; }
      }
      raw.push({
        classIndex,
        className:  this.classNames[classIndex],
        confidence,
        box: { x: cx - w / 2, y: cy - h / 2, width: w, height: h },
      });
    }

    const filtered = filterByConfidence(raw, confThreshold);
    const nmsed    = applyNMS(filtered, iouThreshold);

    return nmsed.map(det => {
      const { x, y, width, height } = det.box;
      return {
        ...det,
        box: {
          x:      Math.max(0, (x - padX) / scaleX),
          y:      Math.max(0, (y - padY) / scaleY),
          width:  width  / scaleX,
          height: height / scaleY,
        },
      };
    });
  }
}
