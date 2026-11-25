// All the AI models and per-frame analysis for FaceAlbums.
//
// Real faces:   face-api (SSD MobileNet detector + 68 landmarks + 128-number ResNet descriptor)
// Things:       CenterNet (MobileNetV3) trained on COCO, 80 everyday object types
// Characters:   YOLOv8n anime face detector (deepghs) + DINOv2-small image embedding
//
// Everything runs in the browser. The face and object models ship with the app;
// the two character models are downloaded from Hugging Face the first time and cached.

import * as faceapi from './lib/face-api.esm.js';

const tf = faceapi.tf;

export const DETECT_MAX_SIDE = 1600;   // big frames are scaled down to this
export const DETECT_MIN_SIDE = 640;    // small frames are scaled up to this
export const CROP_SIZE = 160;          // stored crop size for faces, characters and things
const MIN_FACE_PX = 36;

const ANIME_DETECTOR_URL = 'https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/model.onnx';
const EMBEDDER_URL = 'https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx';
const MODEL_CACHE = 'facealbums-models-v1';

export const COCO_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
  'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush'];

// ---------- loading ----------
let objectModel = null;
let ort = null, animeSession = null, embedSession = null;
export const status = { faces: false, things: false, characters: 'off' }; // characters: off | loading | ready | failed

export async function loadCoreModels() {
  try { await tf.setBackend('webgl'); } catch { /* cpu fallback */ }
  await tf.ready();
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
  ]);
  status.faces = true;
  try {
    objectModel = await tf.loadGraphModel('./models/centernet.json');
    status.things = true;
  } catch (e) { console.error('object model', e); }
}

async function fetchModel(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { /* no Cache API */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status})`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(buf)); } catch { /* storage full */ } }
  return buf;
}

let charactersLoading = null;
export function loadCharacterModels(onProgress) {
  if (charactersLoading) return charactersLoading;
  status.characters = 'loading';
  charactersLoading = (async () => {
    ort = await import('./lib/ort/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./lib/ort/', location.href).href;
    ort.env.wasm.numThreads = 1;   // GitHub Pages cannot enable multi-threading
    const sizes = [12, 24];
    let doneMB = 0;
    const prog = (i) => (got, total) => onProgress && onProgress((doneMB + got / 1048576) / (sizes[0] + sizes[1]), i);
    const a = await fetchModel(ANIME_DETECTOR_URL, prog(0));
    doneMB += sizes[0];
    const b = await fetchModel(EMBEDDER_URL, prog(1));
    animeSession = await ort.InferenceSession.create(a, { executionProviders: ['wasm'] });
    embedSession = await ort.InferenceSession.create(b, { executionProviders: ['wasm'] });
    status.characters = 'ready';
  })().catch((e) => {
    console.error('character models', e);
    status.characters = 'failed';
    charactersLoading = null;
    throw e;
  });
  return charactersLoading;
}

// ---------- helpers ----------
const toBlob = (canvas, q = 0.82) => new Promise((r) => canvas.toBlob(r, 'image/jpeg', q));

// Square crop around a box (pixel coords on `canvas`), with padding, scaled to `size`
function squareCrop(canvas, x, y, w, h, pad, size) {
  const side = Math.max(w, h) * (1 + pad * 2);
  const cx = x + w / 2, cy = y + h / 2;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#e9edf0';
  g.fillRect(0, 0, size, size);
  g.drawImage(canvas, cx - side / 2, cy - side / 2, side, side, 0, 0, size, size);
  return c;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

function nms(boxes, iouLimit) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every((k) => iou(k, b) < iouLimit)) keep.push(b);
  return keep;
}

// Draw any image/video/bitmap source onto a working canvas at detection size
export function frameCanvas(source, sw, sh) {
  const scale = Math.min(DETECT_MAX_SIDE / Math.max(sw, sh), Math.max(1, DETECT_MIN_SIDE / Math.max(sw, sh)));
  const iw = Math.round(sw * scale), ih = Math.round(sh * scale);
  const c = document.createElement('canvas');
  c.width = iw; c.height = ih;
  c.getContext('2d', { willReadFrequently: true }).drawImage(source, 0, 0, iw, ih);
  return c;
}

// ---------- real faces ----------
async function findRealFaces(plain) {
  const iw = plain.width, ih = plain.height;
  // A border lets the detector find faces that touch the edge (close-up selfies, tight crops)
  const pad = Math.round(Math.max(iw, ih) * 0.2);
  const det = document.createElement('canvas');
  det.width = iw + pad * 2; det.height = ih + pad * 2;
  const g = det.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#808080';
  g.fillRect(0, 0, det.width, det.height);
  g.drawImage(plain, pad, pad);
  const results = await faceapi
    .detectAllFaces(det, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.35, maxResults: 60 }))
    .withFaceLandmarks()
    .withFaceDescriptors();
  const out = [];
  for (const r of results) {
    const b = r.detection.box;
    if (b.width < MIN_FACE_PX || b.height < MIN_FACE_PX) continue;
    const x = Math.max(0, b.x - pad), y = Math.max(0, b.y - pad);
    const w = Math.min(iw, b.x - pad + b.width) - x, h = Math.min(ih, b.y - pad + b.height) - y;
    if (w <= 0 || h <= 0) continue;
    out.push({
      kind: 'real', px: { x, y, w, h },
      box: { x: x / iw, y: y / ih, w: w / iw, h: h / ih },
      desc: Array.from(r.descriptor), score: r.detection.score,
      cropCanvas: squareCrop(plain, x, y, w, h, 0.35, CROP_SIZE),
    });
  }
  return out;
}

// ---------- things ----------
async function findThings(plain) {
  if (!objectModel) return [];
  const iw = plain.width, ih = plain.height;
  const res = tf.tidy(() => {
    const t = tf.image.resizeBilinear(tf.browser.fromPixels(plain).toFloat().expandDims(0), [512, 512]);
    return objectModel.execute(t, ['tower_0/detections']);
  });
  const rows = (await res.array())[0];
  res.dispose();
  const boxes = [];
  for (const r of rows) {
    const score = r[4], cls = Math.round(r[5]);
    if (!(score >= 0.35) || !COCO_LABELS[cls] || cls === 0) continue; // people are handled by face grouping
    const x = Math.max(0, r[0] / 512), y = Math.max(0, r[1] / 512);
    const w = Math.min(1, r[2] / 512) - x, h = Math.min(1, r[3] / 512) - y;
    if (w * h < 0.004) continue;
    boxes.push({ label: COCO_LABELS[cls], score, x, y, w, h });
  }
  const keep = [];
  for (const label of new Set(boxes.map((b) => b.label))) keep.push(...nms(boxes.filter((b) => b.label === label), 0.45));
  return keep.map((b) => ({
    label: b.label, score: b.score,
    box: { x: b.x, y: b.y, w: b.w, h: b.h },
    cropCanvas: squareCrop(plain, b.x * iw, b.y * ih, b.w * iw, b.h * ih, 0.08, CROP_SIZE),
  }));
}

// ---------- cartoon / anime characters ----------
async function findCharacters(plain) {
  if (status.characters !== 'ready') return [];
  const iw = plain.width, ih = plain.height;
  const S = 640;
  const scale = S / Math.max(iw, ih);
  const lw = Math.round(iw * scale), lh = Math.round(ih * scale);
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.fillStyle = 'rgb(114,114,114)';
  g.fillRect(0, 0, S, S);
  g.drawImage(plain, 0, 0, lw, lh);
  const px = g.getImageData(0, 0, S, S).data;
  const input = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    input[i] = px[p] / 255; input[i + S * S] = px[p + 1] / 255; input[i + 2 * S * S] = px[p + 2] / 255;
  }
  const feeds = { [animeSession.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, S, S]) };
  const out = (await animeSession.run(feeds))[animeSession.outputNames[0]];
  // YOLOv8 output: [1, 5, N] = cx, cy, w, h, score (some exports use [1, N, 5])
  const [, a, b] = out.dims;
  const d = out.data;
  const transposed = a > b;
  const n = transposed ? a : b;
  const get = (k, i) => (transposed ? d[i * 5 + k] : d[k * n + i]);
  const boxes = [];
  for (let i = 0; i < n; i++) {
    const score = get(4, i);
    if (score < 0.35) continue;
    const w = get(2, i) / scale, h = get(3, i) / scale;
    const x = get(0, i) / scale - w / 2, y = get(1, i) / scale - h / 2;
    if (w < 24 || h < 24) continue;
    boxes.push({ x: Math.max(0, x), y: Math.max(0, y), w: Math.min(iw - Math.max(0, x), w), h: Math.min(ih - Math.max(0, y), h), score });
  }
  const kept = nms(boxes, 0.5).slice(0, 20);
  const found = [];
  for (const k of kept) {
    found.push({
      kind: 'anime', px: k,
      box: { x: k.x / iw, y: k.y / ih, w: k.w / iw, h: k.h / ih },
      desc: await embed(squareCrop(plain, k.x, k.y, k.w, k.h, 0.25, 224)),
      score: k.score,
      cropCanvas: squareCrop(plain, k.x, k.y, k.w, k.h, 0.35, CROP_SIZE),
    });
  }
  return found;
}

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
async function embed(canvas224) {
  const S = 224;
  const px = canvas224.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, S, S).data;
  const input = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    for (let ch = 0; ch < 3; ch++) input[i + ch * S * S] = (px[p + ch] / 255 - MEAN[ch]) / STD[ch];
  }
  const name = embedSession.inputNames[0];
  const res = await embedSession.run({ [name]: new ort.Tensor('float32', input, [1, 3, S, S]) });
  let v;
  if (res.pooler_output) v = Array.from(res.pooler_output.data);
  else { const t = res[embedSession.outputNames[0]]; v = Array.from(t.data.slice(0, t.dims[t.dims.length - 1])); }
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

// ---------- one frame ----------
export async function analyzeFrame(plain, opts) {
  const [real, things, chars] = [
    await findRealFaces(plain),
    opts.things ? await findThings(plain) : [],
    opts.characters ? await findCharacters(plain) : [],
  ];
  // A drawn face that face-api also picked up counts as a character, not a person
  const people = real.filter((f) => !chars.some((c) => c.score > 0.5 && iou(c.px, f.px) > 0.3));
  return { faces: [...people, ...chars], things };
}

export { toBlob };
