/**
 * YOLO detection pipeline — preprocessing, model loading, inference, rendering.
 * Used by index.html via ES module import.
 */

/* global ort */

// ── DetectorYolo — anchor-based (yolo12 and similar) ─────────────────────────
// Output format: [1, 4+numClasses, 8400]

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

class DetectorYolo {
  constructor(session, classNames) {
    this.session    = session;
    this.classNames = classNames;
  }

  async runDetectionYolo26(preprocessResult, confThreshold = 0.25, iouThreshold = 0.45) {
    const { tensor, scaleX, scaleY, padX, padY } = preprocessResult;
    const NUM_ANCHORS = 8400;

    const ortTensor  = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    const results    = await this.session.run({ images: ortTensor });
    const outputData = results[Object.keys(results)[0]].data;

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
      raw.push({ classIndex, className: this.classNames[classIndex], confidence,
        box: { x: cx - w / 2, y: cy - h / 2, width: w, height: h } });
    }

    const nmsed = applyNMS(filterByConfidence(raw, confThreshold), iouThreshold);
    return nmsed.map(det => {
      const { x, y, width, height } = det.box;
      return { ...det, box: {
        x:      Math.max(0, (x - padX) / scaleX),
        y:      Math.max(0, (y - padY) / scaleY),
        width:  width  / scaleX,
        height: height / scaleY,
      }};
    });
  }
}

// ── DetectorYolo26 — NMS-free (ultralytics >= 8.4.41) ────────────────────────
// Output format: [1, 300, 6] — x1, y1, x2, y2, confidence, class_id

class DetectorYolo26 {
  constructor(session, classNames) {
    this.session    = session;
    this.classNames = classNames;
  }

  async runDetectionYolo26(preprocessResult, confThreshold = 0.25) {
    const { tensor, scaleX, scaleY, padX, padY } = preprocessResult;
    const NUM_QUERIES = 300;

    const ortTensor = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    const results   = await this.session.run({ images: ortTensor });
    const data      = results[Object.keys(results)[0]].data;

    const detections = [];
    for (let i = 0; i < NUM_QUERIES; i++) {
      const offset = i * 6;
      const x1 = data[offset], y1 = data[offset + 1];
      const x2 = data[offset + 2], y2 = data[offset + 3];
      const conf = data[offset + 4];
      const cls  = Math.round(data[offset + 5]);
      if (conf < confThreshold) continue;
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

// ── Constants ─────────────────────────────────────────────────────────────────
export const MODEL_INPUT_SIZE = 640;

// ── Model Loading ─────────────────────────────────────────────────────────────

/** @returns {Promise<ort.InferenceSession>} */
export async function loadModel(modelPath) {
  return await ort.InferenceSession.create(modelPath);
}

/** @returns {Promise<string[]>} */
export async function loadClasses(classesPath) {
  const response = await fetch(classesPath);
  if (!response.ok) throw new Error(`Không thể tải classes: ${response.status} ${response.statusText}`);
  const text = await response.text();
  return text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
}

/** @returns {Promise<Array<{id: string, name: string, modelPath: string, classesPath: string, type?: string}>>} */
export async function loadRegistry() {
  const response = await fetch('models/registry.json');
  if (!response.ok) throw new Error(`Không thể tải registry: ${response.status}`);
  const data = await response.json();
  return data.models;
}

// ── Detector Factory ──────────────────────────────────────────────────────────

/**
 * Create the appropriate detector based on registry entry type.
 * @param {ort.InferenceSession} session
 * @param {string[]} classNames
 * @param {{ type?: string }} entry
 * @returns {DetectorYolo|DetectorYolo26}
 */
export function createDetector(session, classNames, entry) {
  return entry.type === 'yolo26'
    ? new DetectorYolo26(session, classNames)
    : new DetectorYolo(session, classNames);
}

/**
 * Run detection using the given detector and preprocess result.
 * @param {DetectorYolo|DetectorYolo26} detector
 * @param {{ type?: string }} modelEntry
 * @param {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }} preprocessResult
 * @param {number} [confThreshold=0.25]
 * @param {number} [iouThreshold=0.45]
 * @returns {Promise<Array>}
 */
export async function runDetection(detector, modelEntry, preprocessResult, confThreshold = 0.25, iouThreshold = 0.45) {
  if (!detector) return [];
  return modelEntry?.type === 'yolo26'
    ? detector.runDetectionYolo26(preprocessResult, confThreshold)
    : detector.runDetectionYolo26(preprocessResult, confThreshold, iouThreshold);
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

/**
 * Letterbox-resize an HTMLImageElement to 640×640 and return CHW Float32Array tensor.
 * @param {HTMLImageElement} imageElement
 * @returns {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }}
 */
export function preprocessImage(imageElement) {
  const origW = imageElement.naturalWidth;
  const origH = imageElement.naturalHeight;
  return _letterbox(imageElement, origW, origH);
}

/**
 * Letterbox-resize an HTMLCanvasElement to 640×640 and return CHW Float32Array tensor.
 * @param {HTMLCanvasElement} srcCanvas
 * @returns {{ tensor: Float32Array, scaleX: number, scaleY: number, padX: number, padY: number }}
 */
export function preprocessFromCanvas(srcCanvas) {
  return _letterbox(srcCanvas, srcCanvas.width, srcCanvas.height);
}

function _letterbox(source, origW, origH) {
  const scale  = Math.min(MODEL_INPUT_SIZE / origW, MODEL_INPUT_SIZE / origH);
  const scaledW = Math.min(Math.max(1, Math.round(origW * scale)), MODEL_INPUT_SIZE);
  const scaledH = Math.min(Math.max(1, Math.round(origH * scale)), MODEL_INPUT_SIZE);
  const padX = Math.floor((MODEL_INPUT_SIZE - scaledW) / 2);
  const padY = Math.floor((MODEL_INPUT_SIZE - scaledH) / 2);

  const canvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(128,128,128)';
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  ctx.drawImage(source, padX, padY, scaledW, scaledH);

  const pixels = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
  const numPixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const tensor = new Float32Array(3 * numPixels);
  for (let i = 0; i < numPixels; i++) {
    tensor[i]                 = pixels[i * 4]     / 255;
    tensor[numPixels + i]     = pixels[i * 4 + 1] / 255;
    tensor[2 * numPixels + i] = pixels[i * 4 + 2] / 255;
  }
  return { tensor, scaleX: scaledW / origW, scaleY: scaledH / origH, padX, padY };
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function getClassColor(classIndex, numClasses) {
  const hue = Math.round((classIndex / Math.max(numClasses, 1)) * 360);
  return `hsl(${hue}, 80%, 55%)`;
}

function _drawBoxes(ctx, detections, numClasses) {
  ctx.lineWidth = 2;
  ctx.font = 'bold 14px system-ui, sans-serif';
  for (const det of detections) {
    const { x, y, width, height } = det.box;
    const color = getClassColor(det.classIndex, numClasses);
    const label = `${det.className}: ${det.confidence.toFixed(2)}`;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, width, height);
    const tw = ctx.measureText(label).width + 6;
    const th = 18;
    const ly = y > th ? y - th : y + height;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 3, ly + 13);
  }
}

/**
 * Draw image + bounding boxes onto a canvas (used for static image detection).
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLImageElement} image
 * @param {Array} detections
 */
export function drawDetections(canvas, image, detections) {
  canvas.width  = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
  _drawBoxes(ctx, detections, detections.reduce((m, d) => Math.max(m, d.classIndex + 1), 1));
}

/**
 * Draw bounding boxes onto an existing canvas context (used for webcam).
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} detections
 * @param {number} numClasses
 */
export function drawDetectionsOnCtx(ctx, detections, numClasses) {
  _drawBoxes(ctx, detections, numClasses);
}

/**
 * Render detection stats table.
 * @param {Array<{ className: string, confidence: number }>} detections
 */
export function renderTable(detections) {
  const tableSection = document.getElementById('table-section');
  if (!detections || detections.length === 0) {
    tableSection.style.display = 'none';
    return;
  }

  const statsMap = new Map();
  for (const det of detections) {
    const existing = statsMap.get(det.className);
    if (existing) {
      existing.count++;
      existing.sumConfidence += det.confidence;
    } else {
      statsMap.set(det.className, { count: 1, sumConfidence: det.confidence });
    }
  }

  const stats = [...statsMap.entries()]
    .map(([className, { count, sumConfidence }]) => ({ className, count, avgConfidence: sumConfidence / count }))
    .sort((a, b) => b.count - a.count);

  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  for (const { className, count, avgConfidence } of stats) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${className}</td><td>${count}</td><td>${(avgConfidence * 100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  }
  tableSection.style.display = 'block';
}
