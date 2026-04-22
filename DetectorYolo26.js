/**
 * Detector for YOLO26 NMS-free models (ultralytics >= 8.4.41 export).
 * Output format: [1, 300, 6] — x1, y1, x2, y2, confidence, class_id.
 * Coordinates are in 640×640 letterboxed space. No NMS needed.
 */

/* global ort */
export class DetectorYolo26 {
  /**
   * @param {ort.InferenceSession} session
   * @param {string[]} classNames
   */
  constructor(session, classNames) {
    this.session    = session;
    this.classNames = classNames;
  }

  /**
   * Run NMS-free detection pipeline: inference → parse → filter → scale.
   *
   * @param {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }} preprocessResult
   * @param {number} [confThreshold=0.25]
   * @returns {Promise<Array<{ classIndex: number, className: string, confidence: number, box: { x: number, y: number, width: number, height: number } }>>}
   */
  async runDetectionYolo26(preprocessResult, confThreshold = 0.25) {
    const { tensor, scaleX, scaleY, padX, padY } = preprocessResult;
    const NUM_QUERIES = 300;

    const ortTensor = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    const results   = await this.session.run({ images: ortTensor });
    const data      = results[Object.keys(results)[0]].data; // Float32Array [300*6]

    const detections = [];

    for (let i = 0; i < NUM_QUERIES; i++) {
      const offset = i * 6;
      const x1   = data[offset];
      const y1   = data[offset + 1];
      const x2   = data[offset + 2];
      const y2   = data[offset + 3];
      const conf = data[offset + 4];
      const cls  = Math.round(data[offset + 5]);

      if (conf < confThreshold) continue;

      // Scale from 640-letterbox space back to original image space
      detections.push({
        classIndex: cls,
        className:  this.classNames[cls] ?? `class_${cls}`,
        confidence: conf,
        box: {
          x:      Math.max(0, (x1 - padX) / scaleX),
          y:      Math.max(0, (y1 - padY) / scaleY),
          width:  (x2 - x1) / scaleX,
          height: (y2 - y1) / scaleY,
        },
      });
    }

    return detections;
  }
}
