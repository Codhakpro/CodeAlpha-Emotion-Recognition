import io
import html
import random
from pathlib import Path

import joblib
import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS + CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion_cnn_advanced.pth"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder_advanced.pkl"
MEAN_PATH = BASE_DIR / "mfcc_mean.npy"
STD_PATH = BASE_DIR / "mfcc_std.npy"

SAMPLE_RATE = 16000
N_MFCC = 40
TARGET_FRAMES = 165
MAX_SECONDS = 10


# ============================================================
# STYLES
# ============================================================

st.html(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 5%, rgba(79, 93, 255, .13), transparent 25%),
            radial-gradient(circle at 92% 15%, rgba(155, 78, 255, .10), transparent 25%),
            linear-gradient(180deg, #08090e 0%, #090a10 55%, #07080c 100%);
        color: #f4f6ff;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }

    /* ---------- HERO ---------- */
    .hero {
        position: relative;
        text-align: center;
        padding: 30px 16px 30px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 13px;
        border: 1px solid rgba(130, 145, 255, .26);
        border-radius: 999px;
        background: rgba(82, 96, 255, .07);
        color: #aeb9ff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
    }

    .hero-title {
        margin: 16px 0 0;
        font-size: clamp(2.35rem, 5vw, 4.15rem);
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -2.7px;
        background: linear-gradient(100deg, #ffffff 10%, #b9c2ff 52%, #d6b8ff 92%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 690px;
        margin: 14px auto 0;
        color: #8991a5;
        line-height: 1.7;
        font-size: 14px;
    }

    /* ---------- STEPS ---------- */
    .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 0 auto 24px;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 11px 13px;
        border: 1px solid rgba(255,255,255,.055);
        border-radius: 13px;
        background: rgba(255,255,255,.025);
    }

    .step-number {
        width: 26px;
        height: 26px;
        flex: 0 0 26px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: rgba(103, 116, 255, .13);
        color: #aeb7ff;
        font-size: 11px;
        font-weight: 800;
    }

    .step-text {
        color: #9ca3b6;
        font-size: 12px;
    }

    /* ---------- PANELS ---------- */
    .panel {
        min-height: 235px;
        padding: 22px;
        border: 1px solid rgba(255,255,255,.075);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(18,20,29,.94), rgba(13,15,22,.94));
        box-shadow: 0 18px 50px rgba(0,0,0,.16);
    }

    .panel-kicker {
        color: #6f7890;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .panel-title {
        margin-top: 5px;
        color: #f0f3ff;
        font-size: 18px;
        font-weight: 800;
    }

    .panel-subtitle {
        margin-top: 5px;
        color: #747d92;
        font-size: 12px;
        line-height: 1.55;
    }

    .phrase {
        margin-top: 18px;
        padding: 18px 19px;
        border: 1px solid rgba(108, 122, 255, .22);
        border-radius: 15px;
        background: linear-gradient(135deg, rgba(76, 91, 255, .10), rgba(155, 77, 255, .055));
        color: #e9ebff;
        font-size: 15px;
        line-height: 1.7;
        font-style: italic;
    }

    .record-hint {
        margin-top: 13px;
        color: #5f687c;
        font-size: 11px;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        min-height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.08);
        background: #12151e;
        color: #e8ebf5;
        font-weight: 650;
        transition: border-color .18s ease, transform .18s ease, background .18s ease;
    }

    .stButton > button:hover {
        border-color: rgba(130,145,255,.36);
        background: #171a25;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #596aff, #7657e8);
        border: 0;
        color: white;
        box-shadow: 0 12px 28px rgba(89,106,255,.18);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6575ff, #8264ee);
    }

    /* ---------- SECTION LABELS ---------- */
    .section-kicker {
        margin-top: 30px;
        color: #707a90;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .section-title {
        margin-top: 5px;
        color: #eef1fa;
        font-size: 20px;
        font-weight: 800;
    }

    .section-subtitle {
        margin-top: 4px;
        color: #727b90;
        font-size: 12px;
    }

    /* ---------- RESULTS ---------- */
    .result-card {
        margin-top: 15px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 20px;
        background:
            radial-gradient(circle at 90% 10%, rgba(121, 93, 255, .12), transparent 35%),
            linear-gradient(145deg, rgba(23,26,39,.97), rgba(13,15,22,.97));
    }

    .result-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
    }

    .result-label {
        color: #737c90;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .emotion {
        margin-top: 7px;
        color: #f4f6ff;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 900;
        letter-spacing: -1px;
    }

    .confidence-pill {
        padding: 8px 11px;
        border: 1px solid rgba(137, 151, 255, .20);
        border-radius: 999px;
        background: rgba(94, 108, 255, .08);
        color: #b5bdff;
        font-size: 12px;
        font-weight: 750;
        white-space: nowrap;
    }

    .confidence-row,
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }

    .confidence-row {
        margin-top: 21px;
    }

    .muted {
        color: #858da1;
        font-size: 12px;
    }

    .track {
        height: 8px;
        margin-top: 7px;
        overflow: hidden;
        border-radius: 999px;
        background: #242834;
    }

    .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6475ff, #a16df5);
    }

    .breakdown {
        margin-top: 14px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 18px;
        background: rgba(255,255,255,.022);
    }

    .prob-list {
        display: grid;
        gap: 14px;
        margin-top: 4px;
    }

    .prob-name {
        color: #d9deeb;
        font-size: 12px;
    }

    .prob-value {
        color: #858da1;
        font-size: 11px;
    }

    /* ---------- STATS ---------- */
    .stat {
        min-height: 74px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,.055);
        border-radius: 14px;
        background: rgba(255,255,255,.025);
        text-align: center;
    }

    .stat-value {
        color: #e8ebf5;
        font-size: 13px;
        font-weight: 800;
    }

    .stat-label {
        margin-top: 4px;
        color: #6f788c;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    /* ---------- RECORDING WIDGET ---------- */
    .record-zone {
        position: relative;
        margin-top: 14px;
        padding: 16px;
        border: 1px solid rgba(105, 121, 255, .20);
        border-radius: 18px;
        background: radial-gradient(circle at 50% 45%, rgba(92,108,255,.11), transparent 65%), rgba(255,255,255,.018);
        overflow: hidden;
    }

    .record-zone::before {
        content: '';
        position: absolute;
        width: 220px;
        height: 220px;
        left: 50%;
        top: 50%;
        border: 1px solid rgba(113,130,255,.12);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: record-orbit 5s linear infinite;
        pointer-events: none;
    }

    .record-zone-label {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        color: #e8ebf5;
        font-size: 13px;
        font-weight: 750;
    }

    .record-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #7182ff;
        box-shadow: 0 0 0 0 rgba(113,130,255,.55);
        animation: record-pulse 1.8s ease-out infinite;
    }

    /* Make the Streamlit recorder area feel like a primary control. */
    div[data-testid="stAudioInput"] {
        position: relative;
        z-index: 1;
        padding: 10px !important;
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 15px;
        background: rgba(255,255,255,.025);
    }

    div[data-testid="stAudioInput"] button {
        min-height: 58px !important;
        border-radius: 13px !important;
        font-size: 14px !important;
    }

    @keyframes record-pulse {
        0% { box-shadow: 0 0 0 0 rgba(113,130,255,.50); }
        70% { box-shadow: 0 0 0 9px rgba(113,130,255,0); }
        100% { box-shadow: 0 0 0 0 rgba(113,130,255,0); }
    }

    @keyframes record-orbit {
        from { transform: translate(-50%, -50%) rotate(0deg); }
        to { transform: translate(-50%, -50%) rotate(360deg); }
    }

    @media (prefers-reduced-motion: reduce) {
        .record-dot, .record-zone::before { animation: none; }
    }

    /* ---------- FOOTER ---------- */
    .footer {
        margin-top: 38px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,.05);
        color: #535b6d;
        font-size: 10px;
        line-height: 1.8;
        text-align: center;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .steps {
            grid-template-columns: 1fr;
        }

        .panel {
            min-height: auto;
        }

        .hero {
            padding-top: 18px;
        }
    }
    </style>
    """
)


# ============================================================
# EMOTIONS + PHRASES
# ============================================================

EMOTION_ICONS = {
    "angry": "😠",
    "calm": "😌",
    "disgust": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😔",
    "surprised": "😲",
}

PHRASES = [
    "I can't believe this actually happened. Everything changed so quickly.",
    "Sometimes I wonder if things would have been different.",
    "I don't know what to think about this anymore.",
    "For some reason, today just feels completely different.",
    "I thought everything was going to be fine, but I was wrong.",
]


# ============================================================
# MODEL
# ============================================================

class AdvancedEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(120, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource

def load_model():
    device = torch.device("cpu")
    model = AdvancedEmotionCNN().to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.eval()

    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # Kept with the project because they are part of the notebook pipeline.
    # The trained advanced CNN itself was evaluated without normalization,
    # so inference intentionally follows that same input convention.
    mean_adv = np.load(MEAN_PATH)
    std_adv = np.load(STD_PATH)

    return model, label_encoder, mean_adv, std_adv, device


# ============================================================
# AUDIO FEATURES
# ============================================================

def prepare_audio(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    combined = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0,
    )

    if combined.shape[1] < TARGET_FRAMES:
        padding = TARGET_FRAMES - combined.shape[1]
        combined = np.pad(
            combined,
            ((0, 0), (0, padding)),
            mode="constant",
        )
    else:
        combined = combined[:, :TARGET_FRAMES]

    return combined


def predict_emotion(audio_bytes):
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True,
    )

    original_duration = len(audio) / sr

    if len(audio) > MAX_SECONDS * SAMPLE_RATE:
        audio = audio[:MAX_SECONDS * SAMPLE_RATE]

    used_duration = len(audio) / sr
    features = prepare_audio(audio)

    model, label_encoder, _, _, device = load_model()

    input_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    emotion = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index] * 100)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "labels": label_encoder.classes_,
        "original_duration": original_duration,
        "used_duration": used_duration,
        "feature_shape": features.shape,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "phrase" not in st.session_state:
    st.session_state.phrase = random.choice(PHRASES)

if "result" not in st.session_state:
    st.session_state.result = None

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-badge">● AI · SPEECH · EMOTION</div>
        <div class="hero-title">Emotion Recognition AI</div>
        <div class="hero-subtitle">
            Record a short voice sample and let an advanced convolutional
            neural network analyze the emotional expression in your speech.
        </div>
    </div>

    <div class="steps">
        <div class="step">
            <div class="step-number">01</div>
            <div class="step-text">Read a phrase or speak naturally</div>
        </div>
        <div class="step">
            <div class="step-number">02</div>
            <div class="step-text">Record up to 10 seconds</div>
        </div>
        <div class="step">
            <div class="step-number">03</div>
            <div class="step-text">Analyze the detected emotion</div>
        </div>
    </div>
    """
)


# ============================================================
# INPUT AREA
# ============================================================

left, right = st.columns(2, gap="large")

with left:
    st.html(
        f"""
        <div class="panel">
            <div class="panel-kicker">Step 01 · Prompt</div>
            <div class="panel-title">💬 Suggested phrase</div>
            <div class="panel-subtitle">
                Use the phrase below, or ignore it and say anything naturally.
            </div>
            <div class="phrase">“{html.escape(st.session_state.phrase)}”</div>
            <div class="record-hint">Tip: expressive speech gives the model more vocal information to work with.</div>
        </div>
        """
    )

    if st.button("🔄  Generate another phrase", use_container_width=True):
        st.session_state.phrase = random.choice(PHRASES)
        st.session_state.result = None
        st.rerun()

with right:
    st.html(
        """
        <div class="panel">
            <div class="panel-kicker">Step 02 · Input</div>
            <div class="panel-title">🎙️ Test a voice sample</div>
            <div class="panel-subtitle">
                Record your voice, or upload a known audio file to verify the model.
                Files are analyzed in memory and are not added to the training dataset.
            </div>
        </div>
        """
    )

    record_tab, upload_tab = st.tabs(["🎙️ Record voice", "📁 Test audio file"])

    # IMPORTANT: keep these as two separate variables.
    # The previous version overwrote the microphone result with None
    # when the upload tab rendered, so recorded audio could never reach
    # the Analyze button.
    recorded_audio = None
    uploaded_audio = None

    with record_tab:
        st.html(
            """
            <div class="record-zone">
                <div class="record-zone-label">
                    <span class="record-dot"></span>
                    Microphone recorder
                </div>
            </div>
            """
        )
        recorded_audio = st.audio_input(
            "Record your voice",
            sample_rate=SAMPLE_RATE,
            key=f"audio_{st.session_state.audio_key}",
        )
        st.caption("Allow microphone access when your browser asks. Record a short, expressive sample, then press Analyze.")

    with upload_tab:
        uploaded_audio = st.file_uploader(
            "Upload a test audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Use a short speech sample. RAVDESS .wav files are useful for checking whether the model distinguishes emotions correctly.",
            key=f"upload_{st.session_state.audio_key}",
        )
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format=uploaded_audio.type or "audio/wav")
            st.caption("Test mode only — this file is used for prediction and is not collected for retraining.")


# ============================================================
# ACTIONS
# ============================================================

st.html('<div style="height:8px"></div>')

analyze_col, clear_col = st.columns([2.2, 1], gap="medium")

with analyze_col:
    analyze = st.button(
        "✨  Analyze emotion",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    clear = st.button(
        "🗑️  Clear recording",
        use_container_width=True,
    )

if clear:
    st.session_state.result = None
    st.session_state.audio_key += 1
    st.rerun()

if analyze:
    selected_audio = recorded_audio if recorded_audio is not None else uploaded_audio

    if selected_audio is None:
        st.warning("🎙️ Record a voice sample or upload an audio file first.")
    else:
        with st.spinner("Analyzing audio..."):
            try:
                st.session_state.result = predict_emotion(selected_audio.getvalue())
            except Exception as error:
                st.session_state.result = None
                st.error(f"Something went wrong while analyzing the audio: {error}")


# ============================================================
# RESULT
# ============================================================

if st.session_state.result is not None:
    result = st.session_state.result
    emotion = result["emotion"]
    confidence = result["confidence"]
    icon = EMOTION_ICONS.get(emotion, "🎭")

    st.html(
        """
        <div class="section-kicker">Step 03 · Model output</div>
        <div class="section-title">📊 Analysis result</div>
        <div class="section-subtitle">
            The model's probability distribution across the eight supported emotions.
        </div>
        """
    )

    st.html(
        f"""
        <div class="result-card">
            <div class="result-top">
                <div>
                    <div class="result-label">Detected emotion</div>
                    <div class="emotion">{icon} {html.escape(emotion.upper())}</div>
                </div>
                <div class="confidence-pill">{confidence:.1f}% confidence</div>
            </div>

            <div class="confidence-row">
                <span class="muted">Model confidence</span>
                <span class="muted">{confidence:.1f}%</span>
            </div>
            <div class="track">
                <div class="fill" style="width:{min(confidence, 100):.1f}%"></div>
            </div>
        </div>
        """
    )

    probability_pairs = sorted(
        zip(result["labels"], result["probabilities"]),
        key=lambda item: item[1],
        reverse=True,
    )

    probability_html = """
    <div class="section-kicker">Probability breakdown</div>
    <div class="section-title">How the model distributed its prediction</div>
    <div class="breakdown"><div class="prob-list">
    """

    for label, probability in probability_pairs:
        percentage = float(probability * 100)
        probability_html += f"""
            <div>
                <div class="prob-header">
                    <span class="prob-name">{EMOTION_ICONS.get(label, '🎭')} {html.escape(label.capitalize())}</span>
                    <span class="prob-value">{percentage:.1f}%</span>
                </div>
                <div class="track">
                    <div class="fill" style="width:{min(percentage, 100):.1f}%"></div>
                </div>
            </div>
        """

    probability_html += "</div></div>"
    st.html(probability_html)

    s1, s2, s3, s4 = st.columns(4, gap="medium")

    with s1:
        st.html(
            f"""
            <div class="stat">
                <div class="stat-value">{result['used_duration']:.2f}s</div>
                <div class="stat-label">Audio used</div>
            </div>
            """
        )

    with s2:
        st.html(
            f"""
            <div class="stat">
                <div class="stat-value">{result['feature_shape'][0]} × {result['feature_shape'][1]}</div>
                <div class="stat-label">Feature shape</div>
            </div>
            """
        )

    with s3:
        st.html(
            """
            <div class="stat">
                <div class="stat-value">8 emotions</div>
                <div class="stat-label">Output classes</div>
            </div>
            """
        )

    with s4:
        st.html(
            """
            <div class="stat">
                <div class="stat-value">Advanced CNN</div>
                <div class="stat-label">Model</div>
            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        Advanced CNN · 40 MFCC + Delta + Delta-Delta · 16 kHz audio · RAVDESS-trained
        <br>
        Built with Python, PyTorch, Librosa & Streamlit · For demonstration and educational use
    </div>
    """
)
