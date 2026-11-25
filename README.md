# FaceAlbums

Group your photos and videos by the people, cartoon characters and things in them, like the "People" and "Things" sections in Google Photos. Everything runs in your browser. Nothing is uploaded.

Live app: https://mrs1ddharthyadav.github.io/FaceAlbums/

## Features

- Add photos, videos or a whole folder (or drag and drop them)
- Videos: about one frame per second is checked, and each face is saved with its time so you can jump to it
- Characters: finds cartoon and anime characters and groups the same character together
- Things: 80 kinds of everyday objects (pets, cars, food, phones and more)
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

### Characters and things

- **Characters**: a YOLOv8 anime face detector ([deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection), MIT) finds drawn faces. [DINOv2-small](https://huggingface.co/Xenova/dinov2-small) turns each one into 384 numbers, and faces with close numbers (cosine distance) are grouped. These two models (about 36 MB) are downloaded from Hugging Face the first time and cached. They run with [ONNX Runtime Web](https://onnxruntime.ai/).
- **Things**: CenterNet (MobileNetV3) trained on COCO, from [@vladmandic/human-models](https://github.com/vladmandic/human-models) (MIT).

Face models come from [@vladmandic/face-api](https://github.com/vladmandic/face-api) (MIT License), a maintained version of face-api.js built on TensorFlow.js.

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
app.js          grouping, storage, video scanning and UI
detect.js       all AI models and per-frame analysis
lib/            face-api (with TensorFlow.js) and ONNX Runtime Web
models/         face and object model weights (about 16 MB)
privacy.html    Privacy Policy
terms.html      Terms and Conditions
```

## Limits

- HEIC photos (iPhone default) need converting to JPG first, except in Safari.
- Very small or blurry faces (under about 36 pixels) are skipped.
- Look-alike relatives can end up in one group. Use "remove face" or a stricter setting.
