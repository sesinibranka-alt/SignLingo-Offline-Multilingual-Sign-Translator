# SignBridge – Offline-First Multilingual Sign Translator

SignBridge is a local Streamlit application that accepts a sign-language video, detects hand landmarks with MediaPipe, runs an ONNX ASL fingerspelling model, builds the recognized letters into text, translates common words, and provides voice output through the computer's installed text-to-speech engine.

## Current working scope

This repository is intentionally honest about the model included in the first version:

- **Input:** uploaded video (`mp4`, `mov`, `avi`, `mkv`, `webm`)
- **Computer vision:** MediaPipe Hands
- **Recognition:** pretrained ASL alphabet model (A–Z)
- **Output:** recognized English letters/text
- **Multilingual UI:** English, Tamil, Hindi
- **Offline translation:** a small built-in dictionary for common words
- **Voice:** `pyttsx3` using voices installed on the computer
- **Inference:** CPU/ONNX after the model has been downloaded once

This is **not** a claim of unrestricted sentence-level sign-language translation. The shipped recognition model is an ASL fingerspelling/alphabet model, so it should be presented as an ASL fingerspelling translator prototype.

## Run locally

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

The first run downloads the ONNX recognition model into `assets/`. Later runs use the cached model locally.

## App flow

```text
Upload sign video
       ↓
MediaPipe hand detection
       ↓
Landmark normalization
       ↓
Skeleton rendering
       ↓
ONNX ASL alphabet recognition
       ↓
Letter sequence
       ↓
Common-word translation
       ↓
Tamil / Hindi / English output
       ↓
Voice output
```

## GitHub upload

Upload these files/folders:

```text
SignBridge-App/
├── app.py
├── README.md
├── MODEL_NOTICE.md
├── requirements.txt
├── .gitignore
└── assets/
    └── .gitkeep
```

Do **not** upload `_uploaded_video.mp4`, `.venv/`, or Python cache folders.

## Important

The model file is downloaded at first launch rather than committed into this repository. This keeps the GitHub repository small and keeps the external model attribution visible.

For a competition submission, describe this version as:

> “An offline-first ASL fingerspelling translation prototype that performs local hand-landmark extraction and ONNX inference, converts recognized signs to text, provides multilingual output for supported phrases, and offers local text-to-speech.”

Do not describe it as a universal sign-language sentence translator unless a sentence-level sign-language model has been added and validated.
