// FaceAlbums: groups photos by the faces in them, fully in the browser.
// Pipeline per photo: detect faces -> find 68 landmarks -> 128-number face descriptor.
// Grouping: faces whose descriptors are close (Euclidean distance under a threshold) are the same person.

import * as faceapi from './lib/face-api.esm.js';

const DETECT_MAX_SIDE = 1600;   // big photos are scaled down to this before detection
const DETECT_MIN_SIDE = 640;    // small photos are scaled up to this
const THUMB_MAX_SIDE = 1280;    // stored preview size
const CROP_SIZE = 160;          // stored face crop size
const MIN_FACE_PX = 36;         // faces smaller than this give unreliable descriptors
const DEFAULT_STRICTNESS = 50;

// ---------- state ----------
const state = {
  photos: new Map(),   // id -> {id, key, name, w, h, added, thumb: Blob}
  faces: new Map(),    // id -> {id, photoId, box:{x,y,w,h} (0..1), desc: number[], crop: Blob, score, personId, notIn: []}
  people: new Map(),   // id -> {id, name, hidden, created}
  scanning: false,
  stopRequested: false,
  modelReady: false,
};
const urlCache = new WeakMap();
const blobUrl = (blob) => {
  if (!blob) return '';
  let u = urlCache.get(blob);
  if (!u) { u = URL.createObjectURL(blob); urlCache.set(blob, u); }
  return u;
};
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
const $ = (id) => document.getElementById(id);

// ---------- IndexedDB ----------
let db;
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('facealbums', 1);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore('photos', { keyPath: 'id' });
      d.createObjectStore('faces', { keyPath: 'id' });
      d.createObjectStore('people', { keyPath: 'id' });
      d.createObjectStore('meta', { keyPath: 'k' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function tx(stores, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode);
    const out = fn(t);
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}
function getAll(store) {
  return new Promise((resolve, reject) => {
    const r = db.transaction(store).objectStore(store).getAll();
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
const putMany = (store, items) => items.length ? tx([store], 'readwrite', (t) => items.forEach((i) => t.objectStore(store).put(i))) : Promise.resolve();
const delMany = (store, ids) => ids.length ? tx([store], 'readwrite', (t) => ids.forEach((i) => t.objectStore(store).delete(i))) : Promise.resolve();
async function getMeta(k, fallback) {
  const rows = await getAll('meta');
  const row = rows.find((r) => r.k === k);
  return row ? row.v : fallback;
}
const setMeta = (k, v) => putMany('meta', [{ k, v }]);

async function loadAll() {
  const [photos, faces, people] = await Promise.all([getAll('photos'), getAll('faces'), getAll('people')]);
  photos.forEach((p) => state.photos.set(p.id, p));
  faces.forEach((f) => state.faces.set(f.id, f));
  people.forEach((p) => state.people.set(p.id, p));
}

// ---------- settings ----------
let strictness = DEFAULT_STRICTNESS;
// strictness 0..100 maps to a distance threshold 0.60 (looser) .. 0.40 (stricter)
const threshold = () => 0.60 - strictness * 0.002;

// ---------- math ----------
function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
// Bigger, more confident detections make better album covers and cluster seeds
function faceQuality(f) { return (f.score || 0.5) * Math.sqrt(f.box.w * f.box.h); }

// Cluster centroid = average descriptor of its faces
function buildCentroids() {
  const sums = new Map();
  for (const f of state.faces.values()) {
    if (!f.personId) continue;
    let s = sums.get(f.personId);
    if (!s) { s = { v: new Array(f.desc.length).fill(0), n: 0 }; sums.set(f.personId, s); }
    for (let i = 0; i < f.desc.length; i++) s.v[i] += f.desc[i];
    s.n++;
  }
  const c = new Map();
  for (const [pid, s] of sums) c.set(pid, { v: s.v.map((x) => x / s.n), n: s.n, sum: s.v });
  return c;
}
function addToCentroid(c, desc) {
  for (let i = 0; i < desc.length; i++) c.sum[i] += desc[i];
  c.n++;
  for (let i = 0; i < desc.length; i++) c.v[i] = c.sum[i] / c.n;
}

// Put each unassigned face into the closest person, or start a new person.
function assignFaces(faceList, centroids = buildCentroids()) {
  const T = threshold();
  const sorted = [...faceList].sort((a, b) => faceQuality(b) - faceQuality(a));
  const changedPeople = new Set();
  for (const f of sorted) {
    let best = null, bestD = Infinity;
    for (const [pid, c] of centroids) {
      if (f.notIn && f.notIn.includes(pid)) continue;
      const p = state.people.get(pid);
      if (!p) continue;
      const d = dist(f.desc, c.v);
      if (d < bestD) { bestD = d; best = pid; }
    }
    if (best && bestD < T) {
      f.personId = best;
      addToCentroid(centroids.get(best), f.desc);
    } else {
      const p = { id: uid(), name: '', hidden: false, created: Date.now() };
      state.people.set(p.id, p);
      f.personId = p.id;
      centroids.set(p.id, { v: [...f.desc], sum: [...f.desc], n: 1 });
    }
    changedPeople.add(f.personId);
  }
  return changedPeople;
}

// One refinement round: move unlocked faces to their nearest centroid, then merge
// unnamed groups whose centroids are very close.
function refine() {
  const T = threshold();
  let centroids = buildCentroids();
  for (const f of state.faces.values()) {
    const cur = state.people.get(f.personId);
    if (!cur || cur.name) continue;
    let best = f.personId, bestD = dist(f.desc, centroids.get(f.personId).v);
    for (const [pid, c] of centroids) {
      if (pid === f.personId || (f.notIn && f.notIn.includes(pid))) continue;
      const d = dist(f.desc, c.v);
      if (d < bestD - 0.02 && d < T) { bestD = d; best = pid; }
    }
    f.personId = best;
  }
  centroids = buildCentroids();
  const ids = [...centroids.keys()];
  const mergedInto = new Map();
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = ids[i], b = ids[j];
      if (mergedInto.has(a) || mergedInto.has(b)) continue;
      const pa = state.people.get(a), pb = state.people.get(b);
      if (pa.name && pb.name) continue;
      if (dist(centroids.get(a).v, centroids.get(b).v) < T * 0.85) {
        const [keep, drop] = pb.name || centroids.get(b).n > centroids.get(a).n ? [b, a] : [a, b];
        mergedInto.set(drop, keep);
      }
    }
  }
  for (const f of state.faces.values()) if (mergedInto.has(f.personId)) f.personId = mergedInto.get(f.personId);
  removeEmptyPeople();
}

function removeEmptyPeople() {
  const used = new Set([...state.faces.values()].map((f) => f.personId));
  const gone = [];
  for (const id of state.people.keys()) if (!used.has(id)) { state.people.delete(id); gone.push(id); }
  return gone;
}

async function saveEverything() {
  await tx(['faces', 'people'], 'readwrite', (t) => {
    const fs = t.objectStore('faces'), ps = t.objectStore('people');
    ps.clear();
    state.people.forEach((p) => ps.put(p));
    state.faces.forEach((f) => fs.put(f));
  });
}

async function regroup() {
  const named = new Set([...state.people.values()].filter((p) => p.name).map((p) => p.id));
  const loose = [];
  for (const f of state.faces.values()) {
    if (!named.has(f.personId)) { f.personId = null; loose.push(f); }
  }
  for (const id of [...state.people.keys()]) if (!named.has(id)) state.people.delete(id);
  assignFaces(loose);
  refine();
  await saveEverything();
}

// ---------- model ----------
async function loadModels() {
  try { await faceapi.tf.setBackend('webgl'); } catch { /* falls back */ }
  await faceapi.tf.ready();
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
  ]);
  state.modelReady = true;
}

// ---------- import ----------
const isImage = (f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(f.name);
const fileKey = (f) => `${f.webkitRelativePath || f.name}|${f.size}|${f.lastModified}`;

function drawScaled(source, w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.round(w * s); c.height = Math.round(h * s);
  c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
  return c;
}
const toBlob = (canvas, q = 0.85) => new Promise((r) => canvas.toBlob(r, 'image/jpeg', q));

function cropFace(canvas, box) {
  const pad = 0.35;
  const size = Math.max(box.width, box.height) * (1 + pad * 2);
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const c = document.createElement('canvas');
  c.width = c.height = CROP_SIZE;
  const g = c.getContext('2d');
  g.fillStyle = '#e9edf0';
  g.fillRect(0, 0, CROP_SIZE, CROP_SIZE);
  g.drawImage(canvas, cx - size / 2, cy - size / 2, size, size, 0, 0, CROP_SIZE, CROP_SIZE);
  return toBlob(c, 0.8);
}

async function processFile(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch { return { skipped: true }; }
  // Small images are scaled up so faces are big enough for the detector.
  const scale = Math.min(DETECT_MAX_SIDE / Math.max(bmp.width, bmp.height), Math.max(1, DETECT_MIN_SIDE / Math.max(bmp.width, bmp.height)));
  const iw = Math.round(bmp.width * scale), ih = Math.round(bmp.height * scale);
  // A border around the photo lets the detector find faces that touch the edge
  // (close-up selfies, tightly cropped profile pictures).
  const pad = Math.round(Math.max(iw, ih) * 0.2);
  const det = document.createElement('canvas');
  det.width = iw + pad * 2; det.height = ih + pad * 2;
  const g = det.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#808080';
  g.fillRect(0, 0, det.width, det.height);
  g.drawImage(bmp, pad, pad, iw, ih);
  const thumb = await toBlob(drawScaled(bmp, bmp.width, bmp.height, THUMB_MAX_SIDE), 0.85);
  bmp.close && bmp.close();

  const results = await faceapi
    .detectAllFaces(det, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.35, maxResults: 60 }))
    .withFaceLandmarks()
    .withFaceDescriptors();

  const photo = { id: uid(), key: fileKey(file), name: file.name, w: iw, h: ih, added: Date.now(), taken: file.lastModified, thumb };
  const faces = [];
  for (const r of results) {
    const b = r.detection.box;
    if (b.width < MIN_FACE_PX || b.height < MIN_FACE_PX) continue;
    const x = Math.max(0, b.x - pad), y = Math.max(0, b.y - pad);
    const w = Math.min(iw, b.x - pad + b.width) - x, h = Math.min(ih, b.y - pad + b.height) - y;
    if (w <= 0 || h <= 0) continue;
    faces.push({
      id: uid(), photoId: photo.id,
      box: { x: x / iw, y: y / ih, w: w / iw, h: h / ih },
      desc: Array.from(r.descriptor), score: r.detection.score,
      crop: await cropFace(det, b), personId: null, notIn: [],
    });
  }
  return { photo, faces };
}

async function importFiles(fileList) {
  if (!state.modelReady) { toast('The face model is still loading. Try again in a moment.'); return; }
  if (state.scanning) { toast('Already scanning. Wait for it to finish.'); return; }
  const known = new Set([...state.photos.values()].map((p) => p.key));
  const files = [...fileList].filter((f) => isImage(f) && !known.has(fileKey(f)));
  const dupes = [...fileList].filter(isImage).length - files.length;
  if (!files.length) { toast(dupes ? 'These photos are already in your library.' : 'No photos found in that selection.'); return; }

  state.scanning = true; state.stopRequested = false;
  $('progress').hidden = false;
  let done = 0, faceCount = 0, skipped = 0;
  let pendingPhotos = [], pendingFaces = [];
  const flush = async () => {
    if (!pendingPhotos.length) return;
    assignFaces(pendingFaces);
    await putMany('photos', pendingPhotos);
    await saveEverything();
    pendingPhotos = []; pendingFaces = [];
    render();
  };

  for (const file of files) {
    if (state.stopRequested) break;
    $('progressLabel').textContent = `Scanning ${done + 1} of ${files.length}: ${file.name}`;
    try {
      const res = await processFile(file);
      if (res.skipped) skipped++;
      else {
        state.photos.set(res.photo.id, res.photo);
        res.faces.forEach((f) => state.faces.set(f.id, f));
        pendingPhotos.push(res.photo); pendingFaces.push(...res.faces);
        faceCount += res.faces.length;
      }
    } catch (e) { console.error(file.name, e); skipped++; }
    done++;
    $('progressBar').style.width = `${(done / files.length) * 100}%`;
    if (pendingPhotos.length >= 10) await flush();
    await new Promise((r) => setTimeout(r, 0));
  }
  await flush();
  refine();
  await saveEverything();

  state.scanning = false;
  $('progress').hidden = true;
  $('progressBar').style.width = '0';
  let msg = `Added ${done - skipped} photos and found ${faceCount} faces.`;
  if (skipped) msg += ` ${skipped} could not be read (HEIC photos need converting to JPG first).`;
  if (dupes) msg += ` ${dupes} were already added.`;
  toast(msg, 6000);
  render();
}

// ---------- derived data ----------
function personSummary(pid) {
  const faces = [...state.faces.values()].filter((f) => f.personId === pid);
  const photoIds = [...new Set(faces.map((f) => f.photoId))];
  const cover = faces.reduce((a, b) => (!a || faceQuality(b) > faceQuality(a) ? b : a), null);
  return { faces, photoIds, cover };
}
function allSummaries() {
  const by = new Map();
  for (const f of state.faces.values()) {
    if (!by.has(f.personId)) by.set(f.personId, []);
    by.get(f.personId).push(f);
  }
  const out = [];
  for (const [pid, faces] of by) {
    const p = state.people.get(pid);
    if (!p) continue;
    const photoIds = new Set(faces.map((f) => f.photoId));
    const cover = faces.reduce((a, b) => (!a || faceQuality(b) > faceQuality(a) ? b : a), null);
    out.push({ p, faces, count: photoIds.size, cover });
  }
  return out.sort((a, b) => (!!b.p.name - !!a.p.name) || b.count - a.count);
}
const sortPhotos = (ids) => ids.map((id) => state.photos.get(id)).filter(Boolean).sort((a, b) => (b.taken || b.added) - (a.taken || a.added));

// ---------- rendering ----------
function personCard(s, onClick) {
  const a = document.createElement('a');
  a.className = 'person-card';
  a.href = `#person/${s.p.id}`;
  if (onClick) a.addEventListener('click', onClick);
  const img = document.createElement('img');
  img.loading = 'lazy'; img.alt = s.p.name || 'Unnamed person';
  img.src = blobUrl(s.cover && s.cover.crop);
  const n = document.createElement('div');
  n.className = 'pname' + (s.p.name ? '' : ' unnamed');
  n.textContent = s.p.name || 'Add a name';
  const c = document.createElement('div');
  c.className = 'pcount';
  c.textContent = `${s.count} photo${s.count === 1 ? '' : 's'}`;
  a.append(img, n, c);
  return a;
}

function photoGrid(container, photos) {
  container.replaceChildren();
  const ids = photos.map((p) => p.id);
  photos.forEach((p, i) => {
    const img = document.createElement('img');
    img.loading = 'lazy'; img.alt = p.name; img.src = blobUrl(p.thumb);
    img.addEventListener('click', () => openViewer(ids, i));
    container.append(img);
  });
}

function showView(name) {
  for (const v of document.querySelectorAll('.view')) v.hidden = v.id !== `view-${name}`;
  for (const a of document.querySelectorAll('.tabs a')) a.classList.toggle('active', a.dataset.tab === name || (name === 'person' && a.dataset.tab === 'people'));
}

function render() {
  if (!state.modelReady && !state.photos.size) { showView('loading'); return; }
  const hash = location.hash.slice(1) || 'people';
  const [route, arg] = hash.split('/');
  if (!state.photos.size && route !== 'settings') { showView('empty'); return; }
  if (route === 'person' && state.people.has(arg)) return renderPerson(arg);
  if (route === 'photos') return renderPhotos();
  if (route === 'settings') return renderSettings();
  renderPeople();
}

function renderPeople() {
  showView('people');
  const all = allSummaries().filter((s) => !s.p.hidden);
  const main = all.filter((s) => s.count >= 2 || s.p.name);
  const once = all.filter((s) => s.count < 2 && !s.p.name);
  $('peopleCount').textContent = main.length;
  $('peopleGrid').replaceChildren(...main.map((s) => personCard(s)));
  if (!main.length) $('peopleGrid').innerHTML = '<p class="muted">No one appears in more than one photo yet. Add more photos.</p>';
  $('onceBox').hidden = !once.length;
  $('onceCount').textContent = once.length;
  $('onceGrid').replaceChildren(...once.map((s) => personCard(s)));
}

function renderPerson(pid) {
  showView('person');
  const p = state.people.get(pid);
  const { faces, photoIds, cover } = personSummary(pid);
  $('personCover').src = blobUrl(cover && cover.crop);
  const input = $('personName');
  if (document.activeElement !== input) input.value = p.name;
  input.dataset.pid = pid;
  $('personStats').textContent = `${photoIds.length} photo${photoIds.length === 1 ? '' : 's'}, ${faces.length} face${faces.length === 1 ? '' : 's'}`;
  $('hideBtn').textContent = p.hidden ? 'Unhide person' : 'Hide person';

  const others = allSummaries().filter((s) => s.p.id !== pid);
  const sel = $('mergeSelect');
  sel.replaceChildren(new Option('Merge with...', ''));
  others.forEach((s) => sel.add(new Option(`${s.p.name || 'Unnamed'} (${s.count} photo${s.count === 1 ? '' : 's'})`, s.p.id)));

  const strip = $('personFaces');
  strip.replaceChildren();
  faces.sort((a, b) => faceQuality(b) - faceQuality(a)).forEach((f) => {
    const img = document.createElement('img');
    img.src = blobUrl(f.crop); img.alt = 'Face'; img.title = 'Not this person? Click to remove';
    img.addEventListener('click', () => removeFace(f.id));
    strip.append(img);
  });
  photoGrid($('personPhotos'), sortPhotos(photoIds));
}

function renderPhotos() {
  showView('photos');
  const photos = sortPhotos([...state.photos.keys()]);
  $('photoCount').textContent = photos.length;
  photoGrid($('allPhotos'), photos);
}

async function renderSettings() {
  showView('settings');
  $('strictness').value = strictness;
  $('strictnessValue').textContent = `Current match distance: ${threshold().toFixed(2)} (default 0.50)`;
  const hidden = allSummaries().filter((s) => s.p.hidden);
  $('hiddenGrid').replaceChildren(...hidden.map((s) => personCard(s)));
  $('hiddenEmpty').hidden = !!hidden.length;
  let info = `${state.photos.size} photos, ${state.faces.size} faces, ${state.people.size} groups.`;
  if (navigator.storage && navigator.storage.estimate) {
    const e = await navigator.storage.estimate();
    info += ` Using ${(e.usage / 1048576).toFixed(1)} MB on this device.`;
  }
  $('storageInfo').textContent = info;
}

// ---------- actions ----------
async function renamePerson(pid, name) {
  name = name.trim();
  const p = state.people.get(pid);
  if (!p || p.name === name) return;
  const same = name && [...state.people.values()].find((o) => o.id !== pid && o.name.toLowerCase() === name.toLowerCase());
  if (same && confirm(`"${same.name}" already exists. Is this the same person? Click OK to merge them.`)) {
    await mergePeople(pid, same.id);
    location.hash = `#person/${same.id}`;
    return;
  }
  p.name = name;
  await putMany('people', [p]);
  toast(name ? `Named "${name}"` : 'Name removed');
}

async function mergePeople(fromId, intoId) {
  const from = state.people.get(fromId), into = state.people.get(intoId);
  if (!from || !into) return;
  if (!into.name && from.name) into.name = from.name;
  const moved = [];
  for (const f of state.faces.values()) if (f.personId === fromId) { f.personId = intoId; f.notIn = (f.notIn || []).filter((x) => x !== intoId); moved.push(f); }
  state.people.delete(fromId);
  await putMany('faces', moved);
  await putMany('people', [into]);
  await delMany('people', [fromId]);
  toast('Merged');
}

async function removeFace(faceId) {
  const f = state.faces.get(faceId);
  const oldPid = f.personId;
  f.notIn = [...new Set([...(f.notIn || []), oldPid])];
  f.personId = null;
  // see if it fits another existing person, otherwise it becomes its own group
  const centroids = buildCentroids();
  assignFaces([f], centroids);
  const gone = removeEmptyPeople();
  await putMany('faces', [f]);
  await putMany('people', [...state.people.values()]);
  await delMany('people', gone);
  if (gone.includes(oldPid)) location.hash = '#people';
  toast('Face removed from this person');
  render();
}

// ---------- viewer ----------
let viewer = { ids: [], i: 0 };
function openViewer(ids, i) { viewer = { ids, i }; $('viewer').hidden = false; drawViewer(); }
function closeViewer() { $('viewer').hidden = true; }
function drawViewer() {
  const p = state.photos.get(viewer.ids[viewer.i]);
  const stage = $('viewerStage');
  stage.replaceChildren();
  const img = document.createElement('img');
  img.src = blobUrl(p.thumb); img.alt = p.name;
  stage.append(img);
  const faces = [...state.faces.values()].filter((f) => f.photoId === p.id);
  for (const f of faces) {
    const person = state.people.get(f.personId);
    const box = document.createElement('div');
    box.className = 'facebox';
    Object.assign(box.style, { left: `${f.box.x * 100}%`, top: `${f.box.y * 100}%`, width: `${f.box.w * 100}%`, height: `${f.box.h * 100}%` });
    const label = document.createElement('span');
    label.textContent = (person && person.name) || 'Unnamed';
    box.append(label);
    box.addEventListener('click', () => { closeViewer(); location.hash = `#person/${f.personId}`; });
    stage.append(box);
  }
  $('viewerCaption').textContent = `${p.name}  |  ${viewer.i + 1} of ${viewer.ids.length}  |  ${faces.length} face${faces.length === 1 ? '' : 's'}`;
  $('viewerPrev').hidden = viewer.i === 0;
  $('viewerNext').hidden = viewer.i === viewer.ids.length - 1;
}

// ---------- ui helpers ----------
let toastTimer;
function toast(msg, ms = 3000) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

function bindEvents() {
  $('pickFiles').addEventListener('change', (e) => { importFiles(e.target.files); e.target.value = ''; });
  $('pickFolder').addEventListener('change', (e) => { importFiles(e.target.files); e.target.value = ''; });
  $('stopScan').addEventListener('click', () => { state.stopRequested = true; $('progressLabel').textContent = 'Stopping...'; });

  // drag and drop anywhere
  let depth = 0;
  window.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; document.body.classList.add('dragging'); });
  window.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; document.body.classList.remove('dragging'); } });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault(); depth = 0; document.body.classList.remove('dragging');
    if (e.dataTransfer.files.length) importFiles(e.dataTransfer.files);
  });

  const nameInput = $('personName');
  nameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') nameInput.blur(); });
  nameInput.addEventListener('change', async () => { await renamePerson(nameInput.dataset.pid, nameInput.value); render(); });

  $('mergeBtn').addEventListener('click', async () => {
    const target = $('mergeSelect').value;
    const pid = $('personName').dataset.pid;
    if (!target) { toast('Pick a person to merge with first.'); return; }
    await mergePeople(pid, target);
    location.hash = `#person/${target}`;
  });
  $('hideBtn').addEventListener('click', async () => {
    const p = state.people.get($('personName').dataset.pid);
    p.hidden = !p.hidden;
    await putMany('people', [p]);
    toast(p.hidden ? 'Hidden. Find hidden people in Settings.' : 'Unhidden');
    render();
  });

  $('strictness').addEventListener('input', (e) => {
    strictness = +e.target.value;
    $('strictnessValue').textContent = `Current match distance: ${threshold().toFixed(2)} (default 0.50)`;
  });
  $('strictness').addEventListener('change', () => setMeta('strictness', strictness));
  $('regroupBtn').addEventListener('click', async () => {
    if (state.scanning) return toast('Wait for the scan to finish.');
    $('regroupBtn').disabled = true;
    await regroup();
    $('regroupBtn').disabled = false;
    toast('Regrouped');
    location.hash = '#people';
  });
  $('wipeBtn').addEventListener('click', async () => {
    if (!confirm('Delete all photos, faces and names from FaceAlbums on this device? Your original photo files are not touched.')) return;
    await tx(['photos', 'faces', 'people'], 'readwrite', (t) => ['photos', 'faces', 'people'].forEach((s) => t.objectStore(s).clear()));
    state.photos.clear(); state.faces.clear(); state.people.clear();
    toast('All data deleted');
    location.hash = '#people';
    render();
  });

  $('viewerClose').addEventListener('click', closeViewer);
  $('viewerPrev').addEventListener('click', () => { if (viewer.i > 0) { viewer.i--; drawViewer(); } });
  $('viewerNext').addEventListener('click', () => { if (viewer.i < viewer.ids.length - 1) { viewer.i++; drawViewer(); } });
  $('viewer').addEventListener('click', (e) => { if (e.target.id === 'viewer') closeViewer(); });
  window.addEventListener('keydown', (e) => {
    if ($('viewer').hidden) return;
    if (e.key === 'Escape') closeViewer();
    if (e.key === 'ArrowLeft') $('viewerPrev').click();
    if (e.key === 'ArrowRight') $('viewerNext').click();
  });
  window.addEventListener('hashchange', () => { closeViewer(); render(); window.scrollTo(0, 0); });
}

// ---------- start ----------
async function start() {
  bindEvents();
  db = await openDB();
  await loadAll();
  strictness = await getMeta('strictness', DEFAULT_STRICTNESS);
  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  render();
  try {
    await loadModels();
  } catch (e) {
    console.error(e);
    $('view-loading').innerHTML = '<p>The face model could not load. Check your internet connection and reload the page.</p>';
    showView('loading');
    return;
  }
  render();
}

// exposed for automated tests
window.FaceAlbums = { state, importFiles, regroup, threshold, ready: start() };
