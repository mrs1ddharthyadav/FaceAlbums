// FaceAlbums: groups photos and videos by the people, cartoon characters and things in them.
// Grouping: each face becomes a list of numbers (a descriptor). Faces whose descriptors are
// close belong to the same person or character.

import * as detect from './detect.js';

const THUMB_MAX_SIDE = 1280;
const MAX_VIDEO_FRAMES = 90;        // frames sampled per video (spread over its length)
const DEFAULT_STRICTNESS = 50;

// ---------- state ----------
const state = {
  photos: new Map(),   // id -> {id, key, name, type:'image'|'video', w, h, added, taken, thumb, video?, duration?}
  faces: new Map(),    // id -> {id, photoId, kind:'real'|'anime', box, desc, crop, score, personId, notIn, t?}
  people: new Map(),   // id -> {id, kind, name, hidden, created}
  things: new Map(),   // id -> {id, photoId, label, score, box, crop, t?}
  scanning: false,
  stopRequested: false,
  modelReady: false,
};
const settings = { strictness: DEFAULT_STRICTNESS, charStrictness: DEFAULT_STRICTNESS, things: true, characters: true };

const urlCache = new WeakMap();
const blobUrl = (blob) => {
  if (!blob) return '';
  let u = urlCache.get(blob);
  if (!u) { u = URL.createObjectURL(blob); urlCache.set(blob, u); }
  return u;
};
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
const $ = (id) => document.getElementById(id);
const kindOf = (x) => x.kind || 'real';
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// ---------- IndexedDB ----------
let db;
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('facealbums', 2);
    req.onupgradeneeded = () => {
      const d = req.result;
      for (const s of ['photos', 'faces', 'people', 'things']) if (!d.objectStoreNames.contains(s)) d.createObjectStore(s, { keyPath: 'id' });
      if (!d.objectStoreNames.contains('meta')) d.createObjectStore('meta', { keyPath: 'k' });
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

async function loadAll() {
  const [photos, faces, people, things, meta] = await Promise.all(['photos', 'faces', 'people', 'things', 'meta'].map(getAll));
  photos.forEach((p) => state.photos.set(p.id, p));
  faces.forEach((f) => state.faces.set(f.id, f));
  people.forEach((p) => state.people.set(p.id, p));
  things.forEach((t) => state.things.set(t.id, t));
  for (const m of meta) if (m.k in settings) settings[m.k] = m.v;
}
const saveSetting = (k) => putMany('meta', [{ k, v: settings[k] }]);

// ---------- grouping math ----------
// Real faces: Euclidean distance between 128-number descriptors. 0.60 (loose) .. 0.40 (strict).
// Characters: cosine distance between DINOv2 embeddings. 0.40 (loose) .. 0.20 (strict).
const thresholdFor = (kind) => (kind === 'anime' ? 0.40 - settings.charStrictness * 0.002 : 0.60 - settings.strictness * 0.002);

function dist(kind, a, b) {
  if (kind === 'anime') {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return 1 - dot / (Math.sqrt(na * nb) || 1);
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
// Bigger, more confident detections make better covers and cluster seeds
const faceQuality = (f) => (f.score || 0.5) * Math.sqrt(f.box.w * f.box.h);

function buildCentroids() {
  const sums = new Map();
  for (const f of state.faces.values()) {
    if (!f.personId) continue;
    let s = sums.get(f.personId);
    if (!s) { s = { sum: new Array(f.desc.length).fill(0), n: 0, kind: kindOf(f) }; sums.set(f.personId, s); }
    for (let i = 0; i < f.desc.length; i++) s.sum[i] += f.desc[i];
    s.n++;
  }
  const c = new Map();
  for (const [pid, s] of sums) c.set(pid, { v: s.sum.map((x) => x / s.n), sum: s.sum, n: s.n, kind: s.kind });
  return c;
}
function addToCentroid(c, desc) {
  for (let i = 0; i < desc.length; i++) { c.sum[i] += desc[i]; c.v[i] = c.sum[i] / (c.n + 1); }
  c.n++;
}

// Put each unassigned face into the closest group of the same kind, or start a new group
function assignFaces(faceList, centroids = buildCentroids()) {
  const sorted = [...faceList].sort((a, b) => faceQuality(b) - faceQuality(a));
  for (const f of sorted) {
    const kind = kindOf(f);
    let best = null, bestD = Infinity;
    for (const [pid, c] of centroids) {
      if (c.kind !== kind || (f.notIn && f.notIn.includes(pid)) || !state.people.has(pid)) continue;
      const d = dist(kind, f.desc, c.v);
      if (d < bestD) { bestD = d; best = pid; }
    }
    if (best && bestD < thresholdFor(kind)) {
      f.personId = best;
      addToCentroid(centroids.get(best), f.desc);
    } else {
      const p = { id: uid(), kind, name: '', hidden: false, created: Date.now() };
      state.people.set(p.id, p);
      f.personId = p.id;
      centroids.set(p.id, { v: [...f.desc], sum: [...f.desc], n: 1, kind });
    }
  }
}

// Refinement: move faces in unnamed groups to their nearest group, then join very close unnamed groups
function refine() {
  let centroids = buildCentroids();
  for (const f of state.faces.values()) {
    const cur = state.people.get(f.personId);
    if (!cur || cur.name) continue;
    const kind = kindOf(f), T = thresholdFor(kind);
    let best = f.personId, bestD = dist(kind, f.desc, centroids.get(f.personId).v);
    for (const [pid, c] of centroids) {
      if (pid === f.personId || c.kind !== kind || (f.notIn && f.notIn.includes(pid))) continue;
      const d = dist(kind, f.desc, c.v);
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
      const ca = centroids.get(a), cb = centroids.get(b);
      if (ca.kind !== cb.kind || mergedInto.has(a) || mergedInto.has(b)) continue;
      const pa = state.people.get(a), pb = state.people.get(b);
      if (pa.name && pb.name) continue;
      if (dist(ca.kind, ca.v, cb.v) < thresholdFor(ca.kind) * 0.85) {
        const [keep, drop] = pb.name || cb.n > ca.n ? [b, a] : [a, b];
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

async function saveGroups() {
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
  for (const f of state.faces.values()) if (!named.has(f.personId)) { f.personId = null; loose.push(f); }
  for (const id of [...state.people.keys()]) if (!named.has(id)) state.people.delete(id);
  assignFaces(loose);
  refine();
  await saveGroups();
}

// ---------- import ----------
const isImage = (f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(f.name);
const isVideo = (f) => f.type.startsWith('video/') || /\.(mp4|m4v|mov|webm|mkv|3gp)$/i.test(f.name);
const fileKey = (f) => `${f.webkitRelativePath || f.name}|${f.size}|${f.lastModified}`;

function drawScaled(source, w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.round(w * s); c.height = Math.round(h * s);
  c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
  return c;
}

const opts = () => ({ things: settings.things, characters: settings.characters && detect.status.characters === 'ready' });

async function toRecords(found, photoId, t) {
  const faces = [], things = [];
  for (const f of found.faces) {
    faces.push({ id: uid(), photoId, kind: f.kind, box: f.box, desc: f.desc, score: f.score, crop: await detect.toBlob(f.cropCanvas), personId: null, notIn: [], ...(t != null ? { t } : {}) });
  }
  for (const th of found.things) {
    things.push({ id: uid(), photoId, label: th.label, score: th.score, box: th.box, crop: await detect.toBlob(th.cropCanvas), ...(t != null ? { t } : {}) });
  }
  return { faces, things };
}

async function processImage(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch { return { skipped: true }; }
  const plain = detect.frameCanvas(bmp, bmp.width, bmp.height);
  const thumb = await detect.toBlob(drawScaled(bmp, bmp.width, bmp.height, THUMB_MAX_SIDE), 0.85);
  bmp.close && bmp.close();
  const photo = { id: uid(), type: 'image', key: fileKey(file), name: file.name, w: plain.width, h: plain.height, added: Date.now(), taken: file.lastModified, thumb };
  const found = await detect.analyzeFrame(plain, opts());
  return { photo, ...(await toRecords(found, photo.id)) };
}

const once = (el, ev, ms = 15000) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => { cleanup(); reject(new Error('timeout')); }, ms);
  const ok = () => { cleanup(); resolve(); };
  const bad = () => { cleanup(); reject(new Error('media error')); };
  const cleanup = () => { clearTimeout(timer); el.removeEventListener(ev, ok); el.removeEventListener('error', bad); };
  el.addEventListener(ev, ok); el.addEventListener('error', bad);
});

async function processVideo(file, onFrame) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  try {
    v.src = url;
    await once(v, 'loadeddata');
    let duration = v.duration;
    if (!isFinite(duration)) {               // some recordings only report length after a seek to the end
      v.currentTime = 1e7; await once(v, 'seeked'); duration = v.duration;
    }
    if (!isFinite(duration) || !v.videoWidth) return { skipped: true };
    const n = Math.max(1, Math.min(MAX_VIDEO_FRAMES, Math.round(duration)));
    const step = duration / n;
    const photo = { id: uid(), type: 'video', key: fileKey(file), name: file.name, w: v.videoWidth, h: v.videoHeight, duration, added: Date.now(), taken: file.lastModified, video: file, thumb: null };

    const kept = [];          // best sighting of each face in this video
    const thingsBest = new Map();
    let thumbScore = -1;
    for (let i = 0; i < n; i++) {
      if (state.stopRequested) break;
      const t = Math.min(duration - 0.05, i * step + step / 2);
      v.currentTime = t;
      await once(v, 'seeked');
      onFrame(i + 1, n);
      const plain = detect.frameCanvas(v, v.videoWidth, v.videoHeight);
      const found = await detect.analyzeFrame(plain, opts());
      // Cover frame: the one with the most faces, else the first
      if (found.faces.length > thumbScore) {
        thumbScore = found.faces.length;
        photo.thumb = await detect.toBlob(drawScaled(plain, plain.width, plain.height, THUMB_MAX_SIDE), 0.85);
      }
      for (const f of found.faces) {
        const same = kept.find((k) => k.kind === f.kind && dist(f.kind, k.desc, f.desc) < thresholdFor(f.kind) * 0.9);
        if (!same) kept.push({ ...f, t });
        else if (faceQuality(f) > faceQuality(same)) Object.assign(same, f, { t });
      }
      for (const th of found.things) {
        const cur = thingsBest.get(th.label);
        if (!cur || th.score > cur.score) thingsBest.set(th.label, { ...th, t });
      }
      await new Promise((r) => setTimeout(r, 0));
    }
    const faces = [], things = [];
    for (const f of kept) faces.push((await toRecords({ faces: [f], things: [] }, photo.id, f.t)).faces[0]);
    for (const th of thingsBest.values()) things.push((await toRecords({ faces: [], things: [th] }, photo.id, th.t)).things[0]);
    return { photo, faces, things };
  } catch (e) {
    console.error(file.name, e);
    return { skipped: true };
  } finally {
    v.removeAttribute('src'); v.load();
    URL.revokeObjectURL(url);
  }
}

async function importFiles(fileList) {
  if (!state.modelReady) { toast('The AI models are still loading. Try again in a moment.'); return; }
  if (state.scanning) { toast('Already scanning. Wait for it to finish.'); return; }
  const media = [...fileList].filter((f) => isImage(f) || isVideo(f));
  const known = new Set([...state.photos.values()].map((p) => p.key));
  const files = media.filter((f) => !known.has(fileKey(f)));
  const dupes = media.length - files.length;
  if (!files.length) { toast(dupes ? 'These are already in your library.' : 'No photos or videos found in that selection.'); return; }

  state.scanning = true; state.stopRequested = false;
  $('progress').hidden = false;
  if (settings.characters && detect.status.characters === 'loading') {
    $('progressLabel').textContent = 'Waiting for the character model to finish downloading...';
    try { await detect.loadCharacterModels(); } catch { /* continue without characters */ }
  }
  let done = 0, faceCount = 0, thingCount = 0, skipped = 0;
  let pending = { photos: [], faces: [], things: [] };
  const flush = async () => {
    if (!pending.photos.length) return;
    assignFaces(pending.faces);
    await putMany('photos', pending.photos);
    await putMany('things', pending.things);
    await saveGroups();
    pending = { photos: [], faces: [], things: [] };
    render();
  };

  for (const file of files) {
    if (state.stopRequested) break;
    const label = `${done + 1} of ${files.length}: ${file.name}`;
    $('progressLabel').textContent = `Scanning ${label}`;
    try {
      const res = isVideo(file)
        ? await processVideo(file, (i, n) => {
          $('progressLabel').textContent = `Scanning video ${label} (frame ${i} of ${n})`;
          $('progressBar').style.width = `${((done + i / n) / files.length) * 100}%`;
        })
        : await processImage(file);
      if (res.skipped) skipped++;
      else {
        state.photos.set(res.photo.id, res.photo);
        res.faces.forEach((f) => state.faces.set(f.id, f));
        res.things.forEach((t) => state.things.set(t.id, t));
        pending.photos.push(res.photo); pending.faces.push(...res.faces); pending.things.push(...res.things);
        faceCount += res.faces.length; thingCount += res.things.length;
      }
    } catch (e) { console.error(file.name, e); skipped++; }
    done++;
    $('progressBar').style.width = `${(done / files.length) * 100}%`;
    if (pending.photos.length >= 10 || isVideo(file)) await flush();
    await new Promise((r) => setTimeout(r, 0));
  }
  await flush();
  refine();
  await saveGroups();

  state.scanning = false;
  $('progress').hidden = true;
  $('progressBar').style.width = '0';
  let msg = `Added ${done - skipped} items. Found ${plural(faceCount, 'face')} and ${plural(thingCount, 'thing')}.`;
  if (skipped) msg += ` ${skipped} could not be read (HEIC photos and some video formats need converting first).`;
  if (dupes) msg += ` ${dupes} were already added.`;
  toast(msg, 7000);
  render();
}

// ---------- derived data ----------
function groupSummaries(kind) {
  const by = new Map();
  for (const f of state.faces.values()) {
    if (kindOf(f) !== kind) continue;
    if (!by.has(f.personId)) by.set(f.personId, []);
    by.get(f.personId).push(f);
  }
  const out = [];
  for (const [pid, faces] of by) {
    const p = state.people.get(pid);
    if (!p) continue;
    const cover = faces.reduce((a, b) => (!a || faceQuality(b) > faceQuality(a) ? b : a), null);
    out.push({ p, faces, count: new Set(faces.map((f) => f.photoId)).size, cover });
  }
  return out.sort((a, b) => (!!b.p.name - !!a.p.name) || b.count - a.count);
}
function thingSummaries() {
  const by = new Map();
  for (const t of state.things.values()) {
    if (!by.has(t.label)) by.set(t.label, []);
    by.get(t.label).push(t);
  }
  return [...by].map(([label, list]) => ({
    label, list,
    count: new Set(list.map((t) => t.photoId)).size,
    cover: list.reduce((a, b) => (!a || b.score * Math.sqrt(b.box.w * b.box.h) > a.score * Math.sqrt(a.box.w * a.box.h) ? b : a), null),
  })).sort((a, b) => b.count - a.count);
}
const byDate = (a, b) => (b.taken || b.added) - (a.taken || a.added);

// ---------- rendering ----------
const SECTIONS = {
  people: { kind: 'real', title: 'People', empty: 'No one appears in more than one photo or video yet. Add more.' },
  characters: { kind: 'anime', title: 'Cartoon and anime characters', empty: 'No characters yet. Add cartoon or anime pictures or videos.' },
};

function personCard(s, route) {
  const a = document.createElement('a');
  a.className = 'person-card';
  a.href = `#${route}/${s.p.id}`;
  const img = document.createElement('img');
  img.loading = 'lazy'; img.alt = s.p.name || 'Unnamed';
  img.src = blobUrl(s.cover && s.cover.crop);
  const n = document.createElement('div');
  n.className = 'pname' + (s.p.name ? '' : ' unnamed');
  n.textContent = s.p.name || 'Add a name';
  const c = document.createElement('div');
  c.className = 'pcount';
  c.textContent = plural(s.count, 'item');
  a.append(img, n, c);
  return a;
}

function thingCard(s) {
  const a = document.createElement('a');
  a.className = 'person-card thing';
  a.href = `#thing/${encodeURIComponent(s.label)}`;
  const img = document.createElement('img');
  img.loading = 'lazy'; img.alt = s.label; img.src = blobUrl(s.cover.crop);
  const n = document.createElement('div');
  n.className = 'pname'; n.textContent = capital(s.label);
  const c = document.createElement('div');
  c.className = 'pcount'; c.textContent = plural(s.count, 'item');
  a.append(img, n, c);
  return a;
}

const PLAY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
// items: [{photo, t?}]
function mediaGrid(container, items) {
  container.replaceChildren();
  items.forEach((it, i) => {
    const tile = document.createElement('button');
    tile.className = 'tile';
    tile.setAttribute('aria-label', it.photo.name);
    const img = document.createElement('img');
    img.loading = 'lazy'; img.alt = ''; img.src = blobUrl(it.photo.thumb);
    tile.append(img);
    if (it.photo.type === 'video') {
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.innerHTML = PLAY_ICON;
      badge.append(fmtTime(it.photo.duration || 0));
      tile.append(badge);
    }
    tile.addEventListener('click', () => openViewer(items, i));
    container.append(tile);
  });
}

function showView(name) {
  for (const v of document.querySelectorAll('.view')) v.hidden = v.id !== `view-${name}`;
  const tab = { person: currentSection, group: currentSection, thing: 'things' }[name] || name;
  for (const a of document.querySelectorAll('.tabs a')) a.classList.toggle('active', a.dataset.tab === tab);
}

let currentSection = 'people';
function render() {
  if (!state.modelReady && !state.photos.size) { showView('loading'); return; }
  const hash = decodeURIComponent(location.hash.slice(1)) || 'people';
  const [route, arg] = hash.split('/');
  if (!state.photos.size && route !== 'settings') { showView('empty'); return; }
  if ((route === 'people' || route === 'characters') && arg && state.people.has(arg)) { currentSection = route; return renderPerson(arg, route); }
  if (route === 'characters') { currentSection = route; return renderGroups('characters'); }
  if (route === 'things') return renderThings();
  if (route === 'thing' && arg) return renderThing(arg);
  if (route === 'photos') return renderPhotos();
  if (route === 'settings') return renderSettings();
  currentSection = 'people';
  renderGroups('people');
}

function renderGroups(section) {
  const cfg = SECTIONS[section];
  showView('group');
  $('groupTitle').textContent = cfg.title;
  const all = groupSummaries(cfg.kind).filter((s) => !s.p.hidden);
  const main = all.filter((s) => s.count >= 2 || s.p.name);
  const single = all.filter((s) => s.count < 2 && !s.p.name);
  $('groupCount').textContent = main.length;
  $('groupGrid').replaceChildren(...main.map((s) => personCard(s, section)));
  if (!main.length) $('groupGrid').innerHTML = `<p class="muted">${cfg.empty}</p>`;
  if (section === 'characters' && !settings.characters) $('groupGrid').innerHTML = '<p class="muted">Character finding is turned off. Turn it on in Settings.</p>';
  $('onceBox').hidden = !single.length;
  $('onceCount').textContent = single.length;
  $('onceGrid').replaceChildren(...single.map((s) => personCard(s, section)));
}

function renderPerson(pid, section) {
  showView('person');
  const p = state.people.get(pid);
  const faces = [...state.faces.values()].filter((f) => f.personId === pid).sort((a, b) => faceQuality(b) - faceQuality(a));
  const items = new Map();
  for (const f of faces) if (!items.has(f.photoId) && state.photos.has(f.photoId)) items.set(f.photoId, { photo: state.photos.get(f.photoId), t: f.t });
  $('personBack').href = `#${section}`;
  $('personBackLabel').textContent = SECTIONS[section].title;
  $('personCover').src = blobUrl(faces[0] && faces[0].crop);
  const input = $('personName');
  if (document.activeElement !== input) input.value = p.name;
  input.dataset.pid = pid;
  const videos = [...items.values()].filter((i) => i.photo.type === 'video').length;
  $('personStats').textContent = `${plural(items.size - videos, 'photo')}, ${plural(videos, 'video')}`;
  $('hideBtn').textContent = p.hidden ? 'Unhide' : 'Hide';

  const sel = $('mergeSelect');
  sel.replaceChildren(new Option('Merge with...', ''));
  groupSummaries(kindOf(p)).filter((s) => s.p.id !== pid)
    .forEach((s) => sel.add(new Option(`${s.p.name || 'Unnamed'} (${plural(s.count, 'item')})`, s.p.id)));

  const strip = $('personFaces');
  strip.replaceChildren();
  faces.forEach((f) => {
    const img = document.createElement('img');
    img.src = blobUrl(f.crop); img.alt = 'Face'; img.title = 'Not this one? Click to remove';
    img.addEventListener('click', () => removeFace(f.id));
    strip.append(img);
  });
  mediaGrid($('personPhotos'), [...items.values()].sort((a, b) => byDate(a.photo, b.photo)));
}

function renderThings() {
  showView('things');
  const all = thingSummaries();
  $('thingCount').textContent = all.length;
  $('thingGrid').replaceChildren(...all.map(thingCard));
  if (!settings.things) $('thingGrid').innerHTML = '<p class="muted">Finding things is turned off. Turn it on in Settings.</p>';
  else if (!all.length) $('thingGrid').innerHTML = '<p class="muted">No things found yet. Pets, cars, food, phones and other everyday objects show up here.</p>';
}

function renderThing(label) {
  showView('thing');
  $('thingTitle').textContent = capital(label);
  const items = new Map();
  [...state.things.values()].filter((t) => t.label === label).sort((a, b) => b.score - a.score)
    .forEach((t) => { if (!items.has(t.photoId) && state.photos.has(t.photoId)) items.set(t.photoId, { photo: state.photos.get(t.photoId), t: t.t }); });
  $('thingStats').textContent = plural(items.size, 'item');
  mediaGrid($('thingPhotos'), [...items.values()].sort((a, b) => byDate(a.photo, b.photo)));
}

function renderPhotos() {
  showView('photos');
  const items = [...state.photos.values()].sort(byDate).map((photo) => ({ photo }));
  $('photoCount').textContent = items.length;
  mediaGrid($('allPhotos'), items);
}

async function renderSettings() {
  showView('settings');
  $('strictness').value = settings.strictness;
  $('charStrictness').value = settings.charStrictness;
  $('findThings').checked = settings.things;
  $('findCharacters').checked = settings.characters;
  updateStrictnessText();
  updateCharStatus();
  const hidden = [...groupSummaries('real').map((s) => [s, 'people']), ...groupSummaries('anime').map((s) => [s, 'characters'])].filter(([s]) => s.p.hidden);
  $('hiddenGrid').replaceChildren(...hidden.map(([s, r]) => personCard(s, r)));
  $('hiddenEmpty').hidden = !!hidden.length;
  const vids = [...state.photos.values()].filter((p) => p.type === 'video').length;
  let info = `${plural(state.photos.size - vids, 'photo')}, ${plural(vids, 'video')}, ${plural(state.faces.size, 'face')}, ${plural(state.things.size, 'thing')}.`;
  if (navigator.storage && navigator.storage.estimate) {
    const e = await navigator.storage.estimate();
    info += ` Using ${(e.usage / 1048576).toFixed(1)} MB on this device.`;
  }
  $('storageInfo').textContent = info;
}
function updateStrictnessText() {
  $('strictnessValue').textContent = `Match distance: ${thresholdFor('real').toFixed(2)} (default 0.50)`;
  $('charStrictnessValue').textContent = `Match distance: ${thresholdFor('anime').toFixed(2)} (default 0.30)`;
}
function updateCharStatus() {
  const s = detect.status.characters;
  $('charStatus').textContent = !settings.characters ? 'Off.'
    : s === 'ready' ? 'Character models are ready.'
      : s === 'loading' ? 'Downloading character models...'
        : s === 'failed' ? 'Character models could not be downloaded. Check your connection and reload.' : '';
}

// ---------- actions ----------
async function renamePerson(pid, name) {
  name = name.trim();
  const p = state.people.get(pid);
  if (!p || p.name === name) return;
  const same = name && [...state.people.values()].find((o) => o.id !== pid && kindOf(o) === kindOf(p) && o.name.toLowerCase() === name.toLowerCase());
  if (same && confirm(`"${same.name}" already exists. Is this the same one? Click OK to merge them.`)) {
    await mergePeople(pid, same.id);
    location.hash = `#${currentSection}/${same.id}`;
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
  assignFaces([f]);
  const gone = removeEmptyPeople();
  await putMany('faces', [f]);
  await putMany('people', [...state.people.values()]);
  await delMany('people', gone);
  if (gone.includes(oldPid)) location.hash = `#${currentSection}`;
  toast('Removed from this group');
  render();
}

// ---------- viewer ----------
let viewer = { items: [], i: 0 };
function openViewer(items, i) { viewer = { items, i }; $('viewer').hidden = false; drawViewer(); }
function closeViewer() {
  const v = $('viewerStage').querySelector('video');
  if (v) v.pause();
  $('viewer').hidden = true;
}
function nameOf(f) {
  const p = state.people.get(f.personId);
  return (p && p.name) || (kindOf(f) === 'anime' ? 'Unnamed character' : 'Unnamed');
}
function drawViewer() {
  const { photo: p, t } = viewer.items[viewer.i];
  const stage = $('viewerStage');
  const old = stage.querySelector('video');
  if (old) old.pause();
  stage.replaceChildren();
  const faces = [...state.faces.values()].filter((f) => f.photoId === p.id);
  const things = [...state.things.values()].filter((x) => x.photoId === p.id);
  const chips = $('viewerChips');
  chips.replaceChildren();

  if (p.type === 'video') {
    const v = document.createElement('video');
    v.controls = true; v.playsInline = true; v.preload = 'metadata';
    v.src = blobUrl(p.video);
    v.addEventListener('loadedmetadata', () => { if (t != null) v.currentTime = Math.max(0, t - 1); }, { once: true });
    stage.append(v);
    // Who and what is in this video, with a jump-to time
    // one chip per person or character, at their first appearance
    const seen = new Set();
    for (const f of [...faces].sort((a, b) => (a.t || 0) - (b.t || 0))) {
      if (seen.has(f.personId)) continue;
      seen.add(f.personId);
      const b = document.createElement('button');
      b.className = 'chip';
      const img = document.createElement('img'); img.src = blobUrl(f.crop); img.alt = '';
      b.append(img, `${nameOf(f)} ${fmtTime(f.t || 0)}`);
      b.addEventListener('click', () => { v.currentTime = Math.max(0, (f.t || 0) - 1); v.play(); });
      chips.append(b);
    }
  } else {
    const img = document.createElement('img');
    img.src = blobUrl(p.thumb); img.alt = p.name;
    stage.append(img);
    for (const f of faces) {
      const box = document.createElement('div');
      box.className = 'facebox' + (kindOf(f) === 'anime' ? ' anime' : '');
      Object.assign(box.style, { left: `${f.box.x * 100}%`, top: `${f.box.y * 100}%`, width: `${f.box.w * 100}%`, height: `${f.box.h * 100}%` });
      const label = document.createElement('span');
      label.textContent = nameOf(f);
      box.append(label);
      box.addEventListener('click', () => { closeViewer(); location.hash = `#${kindOf(f) === 'anime' ? 'characters' : 'people'}/${f.personId}`; });
      stage.append(box);
    }
  }
  for (const label of [...new Set(things.map((x) => x.label))]) {
    const a = document.createElement('a');
    a.className = 'chip thing-chip';
    a.href = `#thing/${encodeURIComponent(label)}`;
    a.textContent = capital(label);
    chips.append(a);
  }
  $('viewerCaption').textContent = `${p.name}  |  ${viewer.i + 1} of ${viewer.items.length}`;
  $('viewerPrev').hidden = viewer.i === 0;
  $('viewerNext').hidden = viewer.i === viewer.items.length - 1;
}

// ---------- ui helpers ----------
let toastTimer;
function toast(msg, ms = 3000) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

function startCharacterModels() {
  if (!settings.characters || detect.status.characters === 'ready' || detect.status.characters === 'loading') return;
  detect.loadCharacterModels((frac) => {
    if (!state.scanning) $('charStatus').textContent = `Downloading character models: ${Math.round(frac * 100)}%`;
  }).then(() => { updateCharStatus(); render(); })
    .catch(() => { updateCharStatus(); toast('Could not download the cartoon character models. Faces and things still work.', 6000); });
  updateCharStatus();
}

function bindEvents() {
  $('pickFiles').addEventListener('change', (e) => { importFiles(e.target.files); e.target.value = ''; });
  $('pickFolder').addEventListener('change', (e) => { importFiles(e.target.files); e.target.value = ''; });
  $('stopScan').addEventListener('click', () => { state.stopRequested = true; $('progressLabel').textContent = 'Stopping...'; });

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
    if (!target) { toast('Pick who to merge with first.'); return; }
    await mergePeople($('personName').dataset.pid, target);
    location.hash = `#${currentSection}/${target}`;
  });
  $('hideBtn').addEventListener('click', async () => {
    const p = state.people.get($('personName').dataset.pid);
    p.hidden = !p.hidden;
    await putMany('people', [p]);
    toast(p.hidden ? 'Hidden. Find hidden ones in Settings.' : 'Unhidden');
    render();
  });

  $('strictness').addEventListener('input', (e) => { settings.strictness = +e.target.value; updateStrictnessText(); });
  $('charStrictness').addEventListener('input', (e) => { settings.charStrictness = +e.target.value; updateStrictnessText(); });
  $('strictness').addEventListener('change', () => saveSetting('strictness'));
  $('charStrictness').addEventListener('change', () => saveSetting('charStrictness'));
  $('findThings').addEventListener('change', (e) => { settings.things = e.target.checked; saveSetting('things'); });
  $('findCharacters').addEventListener('change', (e) => {
    settings.characters = e.target.checked; saveSetting('characters');
    startCharacterModels(); updateCharStatus();
  });
  $('regroupBtn').addEventListener('click', async () => {
    if (state.scanning) return toast('Wait for the scan to finish.');
    $('regroupBtn').disabled = true;
    await regroup();
    $('regroupBtn').disabled = false;
    toast('Regrouped');
    location.hash = '#people';
  });
  $('wipeBtn').addEventListener('click', async () => {
    if (!confirm('Delete all photos, videos, faces, things and names from FaceAlbums on this device? Your original files are not touched.')) return;
    await tx(['photos', 'faces', 'people', 'things'], 'readwrite', (t) => ['photos', 'faces', 'people', 'things'].forEach((s) => t.objectStore(s).clear()));
    state.photos.clear(); state.faces.clear(); state.people.clear(); state.things.clear();
    toast('All data deleted');
    location.hash = '#people';
    render();
  });

  $('viewerClose').addEventListener('click', closeViewer);
  $('viewerPrev').addEventListener('click', () => { if (viewer.i > 0) { viewer.i--; drawViewer(); } });
  $('viewerNext').addEventListener('click', () => { if (viewer.i < viewer.items.length - 1) { viewer.i++; drawViewer(); } });
  $('viewer').addEventListener('click', (e) => { if (e.target.id === 'viewer') closeViewer(); });
  window.addEventListener('keydown', (e) => {
    if ($('viewer').hidden || e.target.tagName === 'VIDEO') return;
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
  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  render();
  try {
    await detect.loadCoreModels();
  } catch (e) {
    console.error(e);
    $('view-loading').innerHTML = '<p>The AI models could not load. Check your internet connection and reload the page.</p>';
    showView('loading');
    return;
  }
  state.modelReady = true;
  render();
  startCharacterModels();
}

// exposed for automated tests
window.FaceAlbums = { state, settings, detect, importFiles, regroup, ready: start() };
