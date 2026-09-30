
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyttsx3
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "assets" / "asl_cnn_model.onnx"
MODEL_URL = (
    "https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition/"
    "raw/refs/heads/main/asl_cnn_model.onnx"
)

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# A small offline phrase dictionary. The recognizer produces English letters;
# these entries make common spelled words useful in the multilingual UI.
TRANSLATIONS = {
    "hello": {"English": "Hello", "Tamil": "வணக்கம்", "Hindi": "नमस्ते"},
    "thankyou": {"English": "Thank you", "Tamil": "நன்றி", "Hindi": "धन्यवाद"},
    "thanks": {"English": "Thanks", "Tamil": "நன்றி", "Hindi": "धन्यवाद"},
    "yes": {"English": "Yes", "Tamil": "ஆம்", "Hindi": "हाँ"},
    "no": {"English": "No", "Tamil": "இல்லை", "Hindi": "नहीं"},
    "please": {"English": "Please", "Tamil": "தயவு செய்து", "Hindi": "कृपया"},
    "sorry": {"English": "Sorry", "Tamil": "மன்னிக்கவும்", "Hindi": "माफ़ कीजिए"},
    "water": {"English": "Water", "Tamil": "தண்ணீர்", "Hindi": "पानी"},
    "food": {"English": "Food", "Tamil": "உணவு", "Hindi": "खाना"},
    "help": {"English": "Help", "Tamil": "உதவி", "Hindi": "मदद"},
    "welcome": {"English": "Welcome", "Tamil": "வரவேற்கிறோம்", "Hindi": "स्वागत है"},
}

# Same preprocessing geometry used by the published ASL alphabet model.
CANVAS_SIZE = 192
REFERENCE_CANVAS = 400
HAND_FILL = 0.7
MODEL_INPUT_SIZE = 96

_WHITE_RGB = (224, 224, 224)
_RED, _PEACH, _PURPLE = (255, 48, 48), (255, 229, 180), (128, 64, 128)
_YELLOW, _GREEN, _BLUE, _GRAY = (255, 204, 0), (48, 255, 48), (21, 101, 192), (128, 128, 128)

LANDMARK_STYLE = {
    0: (_RED, 5), 1: (_RED, 5), 5: (_RED, 5), 9: (_RED, 5), 13: (_RED, 5), 17: (_RED, 5),
    2: (_PEACH, 5), 3: (_PEACH, 5), 4: (_PEACH, 5),
    6: (_PURPLE, 5), 7: (_PURPLE, 5), 8: (_PURPLE, 5),
    10: (_YELLOW, 5), 11: (_YELLOW, 5), 12: (_YELLOW, 5),
    14: (_GREEN, 5), 15: (_GREEN, 5), 16: (_GREEN, 5),
    18: (_BLUE, 5), 19: (_BLUE, 5), 20: (_BLUE, 5),
}
CONNECTION_STYLE = {
    (0, 1): (_GRAY, 3), (0, 5): (_GRAY, 3), (0, 17): (_GRAY, 3),
    (5, 9): (_GRAY, 3), (9, 13): (_GRAY, 3), (13, 17): (_GRAY, 3),
    (1, 2): (_PEACH, 2), (2, 3): (_PEACH, 2), (3, 4): (_PEACH, 2),
    (5, 6): (_PURPLE, 2), (6, 7): (_PURPLE, 2), (7, 8): (_PURPLE, 2),
    (9, 10): (_YELLOW, 2), (10, 11): (_YELLOW, 2), (11, 12): (_YELLOW, 2),
    (13, 14): (_GREEN, 2), (14, 15): (_GREEN, 2), (15, 16): (_GREEN, 2),
    (17, 18): (_BLUE, 2), (18, 19): (_BLUE, 2), (19, 20): (_BLUE, 2),
}

def scaled(value, canvas_size, minimum=1):
    return max(minimum, int(round(value * canvas_size / REFERENCE_CANVAS)))

def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        with st.spinner("Downloading the ASL recognition model (first run only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

@st.cache_resource
def load_model():
    path = ensure_model()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    outputs = [o.name for o in session.get_outputs()]
    output_name = "logits" if "logits" in outputs else outputs[0]
    return session, input_meta.name, output_name

def normalize_landmarks(points_px):
    points_px = np.asarray(points_px, dtype=np.float32)
    lo = points_px.min(axis=0)
    hi = points_px.max(axis=0)
    extent = hi - lo
    box_size = float(extent.max()) / HAND_FILL
    if box_size <= 0:
        box_size = 1.0
    centre = lo + extent / 2.0
    origin = centre - box_size / 2.0
    return ((points_px - origin) / box_size).astype(np.float32)

def render_skeleton(norm_points):
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    pts = np.rint(np.asarray(norm_points, dtype=np.float32) * CANVAS_SIZE).astype(np.int32)

    for (a, b), (color, thickness) in CONNECTION_STYLE.items():
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]),
                 color, scaled(thickness, CANVAS_SIZE), cv2.LINE_AA)

    for idx, (color, radius) in LANDMARK_STYLE.items():
        centre = tuple(pts[idx])
        r = scaled(radius, CANVAS_SIZE)
        border = max(r + 1, int(r * 1.2))
        cv2.circle(canvas, centre, border, _WHITE_RGB, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, r, color, -1, cv2.LINE_AA)

    return cv2.resize(canvas, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                      interpolation=cv2.INTER_AREA)

def predict_frame(frame_bgr, hands, session, input_name, output_name):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, 0.0, None

    hand = result.multi_hand_landmarks[0]
    h, w = frame_bgr.shape[:2]
    points = np.array([(lm.x * w, lm.y * h) for lm in hand.landmark],
                      dtype=np.float32)
    normalized = normalize_landmarks(points)
    skeleton = render_skeleton(normalized)

    tensor = np.expand_dims(skeleton.astype(np.float32), axis=0)
    logits = session.run([output_name], {input_name: tensor})[0][0]
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    idx = int(np.argmax(probs))
    letter = CLASSES[idx] if idx < len(CLASSES) else "?"
    return letter, float(probs[idx]), skeleton

def decode_video(video_bytes, confidence_threshold=0.60, sample_every=2):
    temp = APP_DIR / "_uploaded_video.mp4"
    temp.write_bytes(video_bytes)

    session, input_name, output_name = load_model()
    cap = cv2.VideoCapture(str(temp))
    if not cap.isOpened():
        temp.unlink(missing_ok=True)
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    predictions = []
    confidences = []
    preview_frame = None
    frame_index = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every == 0:
                letter, conf, skeleton = predict_frame(
                    frame, hands, session, input_name, output_name
                )
                if letter and conf >= confidence_threshold:
                    predictions.append(letter)
                    confidences.append(conf)
                if skeleton is not None:
                    preview_frame = skeleton
            frame_index += 1

    cap.release()
    temp.unlink(missing_ok=True)

    if not predictions:
        return "", 0.0, preview_frame, total, fps

    # Collapse consecutive repeated predictions so a held letter becomes one letter.
    collapsed = []
    grouped_conf = []
    last = None
    group = []
    for letter, conf in zip(predictions, confidences):
        if letter != last:
            if group:
                grouped_conf.append(float(np.mean(group)))
            collapsed.append(letter)
            group = [conf]
            last = letter
        else:
            group.append(conf)
    if group:
        grouped_conf.append(float(np.mean(group)))

    text = "".join(collapsed)
    confidence = float(np.mean(grouped_conf)) if grouped_conf else 0.0
    return text, confidence, preview_frame, total, fps

def translate_text(text, language):
    key = text.lower().replace(" ", "")
    if key in TRANSLATIONS:
        return TRANSLATIONS[key][language]

    if language == "English":
        return text

    # For words outside the small offline dictionary, keep the recognized
    # spelling rather than inventing a translation.
    return text + " (translation not in offline dictionary)"

def speak(text):
    if not text.strip():
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

st.set_page_config(
    page_title="SignBridge",
    page_icon="🤟",
    layout="wide",
)

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 18px; opacity: .75; margin-top: 0;}
.card {padding: 20px; border-radius: 16px; border: 1px solid rgba(128,128,128,.25);}
.result {font-size: 34px; font-weight: 750;}
.small {font-size: 14px; opacity: .7;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤟 SignBridge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Offline-first multilingual sign-language translator</div>',
    unsafe_allow_html=True,
)

st.write("")
language = st.selectbox("Output language", ["English", "Tamil", "Hindi"])

uploaded = st.file_uploader(
    "Upload a sign-language video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

with st.expander("What this version recognizes"):
    st.write(
        "The included pretrained model recognizes ASL fingerspelling (A–Z). "
        "The app turns the detected letters into text, translates common words "
        "from an offline dictionary, and can speak the resulting text using "
        "the computer's installed voice."
    )

if uploaded is not None:
    st.video(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        threshold = st.slider("Recognition confidence", 0.40, 0.90, 0.60, 0.05)
    with col2:
        sample_every = st.slider("Process every Nth frame", 1, 5, 2)

    if st.button("🚀 Translate Video", type="primary", use_container_width=True):
        try:
            text, confidence, skeleton, total, fps = decode_video(
                uploaded.getvalue(), threshold, sample_every
            )

            st.session_state["recognized"] = text
            st.session_state["confidence"] = confidence
            st.session_state["skeleton"] = skeleton
            st.session_state["translated"] = translate_text(text, language)

            if text:
                st.success("Video processed successfully.")
            else:
                st.warning(
                    "No confident sign was detected. Try a clearer video with the "
                    "hand fully visible and good lighting."
                )
        except Exception as exc:
            st.error(f"Processing error: {exc}")

recognized = st.session_state.get("recognized", "")
translated = st.session_state.get("translated", "")
confidence = st.session_state.get("confidence", 0.0)
skeleton = st.session_state.get("skeleton")

if recognized:
    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Detected text")
        st.markdown(f'<div class="result">{recognized}</div>', unsafe_allow_html=True)
        st.caption(f"Average accepted confidence: {confidence:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {language} output")
        st.markdown(f'<div class="result">{translated}</div>', unsafe_allow_html=True)
        if st.button("🔊 Speak", use_container_width=True):
            try:
                speak(translated)
            except Exception as exc:
                st.error(
                    "Voice output could not start. Install/configure a system "
                    f"TTS voice. Details: {exc}"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if skeleton is not None:
        st.write("")
        st.caption("Example of the hand skeleton representation used by the model")
        st.image(skeleton, width=240)

st.divider()
st.caption(
    "SignBridge prototype • Recognition runs locally after the model is cached. "
    "The pretrained recognition model is an external ASL alphabet model; see "
    "MODEL_NOTICE.md for attribution and scope."
)
