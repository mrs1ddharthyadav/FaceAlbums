# FaceAlbums

Group your photos by the people in them, like the "People" section in Google Photos. Everything runs in your browser. Your photos are never uploaded.

## Features

- Add photos or a whole folder (or drag and drop them)
- Finds every face and groups photos of the same person into an album
- Name people, merge two groups that are the same person, remove a wrong face
- Photo viewer with a labelled box on every face
- Strictness slider to tune how strict grouping is, plus a "Regroup now" button
- Library is saved on your device (IndexedDB), so it is still there after a reload

## How it works

1. **Detect**: an SSD MobileNet v1 neural network finds faces in each photo.
2. **Align**: 68 facial landmarks (eyes, nose, mouth, jaw) are located.
3. **Describe**: a ResNet face recognition network turns each face into a *face descriptor*, a list of 128 numbers. Photos of the same person give descriptors that are close together.
4. **Group**: each face is compared to the average descriptor of every existing group (Euclidean distance). If it is closer than the threshold (0.50 by default), it joins that group. Otherwise it starts a new one. A refinement pass then moves faces to their nearest group and joins groups that are very close.

Named people are "locked": regrouping never moves their faces, and a face you remove from a person is never put back into that person.

Models come from [@vladmandic/face-api](https://github.com/vladmandic/face-api) (MIT License), a maintained version of face-api.js built on TensorFlow.js.

## Run locally

Any static file server works. For example:

```
python -m http.server 8000
```

Then open http://localhost:8000. Opening `index.html` directly from disk does not work, because the browser blocks loading the model files.

## Project layout

```
index.html      app page
style.css       styles
app.js          detection, grouping, storage and UI
lib/            face-api library (bundled with TensorFlow.js)
models/         neural network weights (about 12 MB)
privacy.html    Privacy Policy
terms.html      Terms and Conditions
```

## Limits

- HEIC photos (iPhone default) need converting to JPG first, except in Safari.
- Very small or blurry faces (under about 36 pixels) are skipped.
- Look-alike relatives can end up in one group. Use "remove face" or a stricter setting.
