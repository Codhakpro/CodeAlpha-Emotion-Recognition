import gradio as gr
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
import random
import time


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
N_MFCC = 40
TARGET_FRAMES = 165
MAX_SECONDS = 10

MODEL_PATH = "emotion_cnn_advanced.pth"
LABEL_ENCODER_PATH = "label_encoder_advanced.pkl"
MEAN_PATH = "mfcc_mean.npy"
STD_PATH = "mfcc_std.npy"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


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
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD MODEL + SUPPORT FILES
# ============================================================

print("Loading model...")

model = AdvancedEmotionCNN().to(device)

try:
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )
except TypeError:
    # Compatibility fallback for older PyTorch versions
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

model.eval()

label_encoder = joblib.load(LABEL_ENCODER_PATH)

mean_adv = np.load(MEAN_PATH)
std_adv = np.load(STD_PATH)

print("Model loaded successfully!")
print("Classes:", list(label_encoder.classes_))
print("Mean shape:", np.shape(mean_adv))
print("Std shape:", np.shape(std_adv))


# ============================================================
# SUGGESTED PHRASES
# ============================================================

PHRASES = [
    "I can't believe this actually happened. Everything changed so quickly.",

    "Sometimes I wonder if things would have been different.",

    "I don't know what to think about this anymore.",

    "For some reason, today just feels completely different.",

    "I thought everything was going to be fine, but I was wrong."
]


def get_random_phrase():
    return random.choice(PHRASES)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def prepare_audio(audio):
    """
    Convert audio into exactly the same type of feature
    expected by the advanced CNN:

    40 MFCC
    + 40 delta
    + 40 delta-delta
    = 120 features

    Then pad/trim to 165 frames.
    """

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-delta
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine:
    # (40, time)
    # (40, time)
    # (40, time)
    #
    # becomes:
    # (120, time)

    combined = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0
    )

    # --------------------------------------------------------
    # PAD OR TRIM TO 165 FRAMES
    # --------------------------------------------------------

    if combined.shape[1] < TARGET_FRAMES:

        padding = TARGET_FRAMES - combined.shape[1]

        combined = np.pad(
            combined,
            (
                (0, 0),
                (0, padding)
            ),
            mode="constant"
        )

    else:

        combined = combined[:, :TARGET_FRAMES]

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------


    return combined


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(audio_path):
    """
    Predict emotion from ONE COMPLETE AUDIO FILE.

    IMPORTANT:
    We intentionally do NOT split the audio into chunks.
    The model was trained using the complete audio sample.
    """

    if audio_path is None:
        return (
            "🎙️ Please record or upload an audio file first.",
            "",
            None
        )

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        original_duration = len(audio) / sr

        # ----------------------------------------------------
        # LIMIT TO 10 SECONDS
        # ----------------------------------------------------

        if len(audio) > MAX_SECONDS * SAMPLE_RATE:

            audio = audio[
                :MAX_SECONDS * SAMPLE_RATE
            ]

        used_duration = len(audio) / sr

        # ----------------------------------------------------
        # PREPARE FEATURES
        # ----------------------------------------------------

        features = prepare_audio(audio)

        print("\n------------------------------")
        print("New prediction")
        print("------------------------------")
        print("Audio:", audio_path)
        print(f"Original duration: {original_duration:.2f}s")
        print(f"Used duration: {used_duration:.2f}s")
        print("Feature shape:", features.shape)
        print("Feature mean:", np.mean(features))
        print("Feature std:", np.std(features))

        # ----------------------------------------------------
        # CONVERT TO PYTORCH TENSOR
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            features,
            dtype=torch.float32
        )

        # Add batch dimension
        #
        # (120, 165)
        #
        # becomes
        #
        # (1, 120, 165)

        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0].cpu().numpy()

        # ----------------------------------------------------
        # GET PREDICTED CLASS
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = (
            probabilities[predicted_index] * 100
        )

        # ----------------------------------------------------
        # DEBUG OUTPUT
        # ----------------------------------------------------

        print("\nProbabilities:")

        for label, probability in zip(
            label_encoder.classes_,
            probabilities
        ):

            print(
                f"{label:10s}: "
                f"{probability * 100:.2f}%"
            )

        print(
            f"\nPrediction: {emotion}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

        # ----------------------------------------------------
        # NICE RESULT
        # --------------------------------------------------------

        emotion_icons = {
            "angry": "😠",
            "calm": "😌",
            "disgust": "🤢",
            "fearful": "😨",
            "happy": "😊",
            "neutral": "😐",
            "sad": "😔",
            "surprised": "😲",
        }

        icon = emotion_icons.get(emotion.lower(), "🎭")

        result = f"""
        <div class="result-card">
            <div class="result-topline">
                <span class="result-label">DETECTED EMOTION</span>
                <span class="result-status">● ANALYSIS COMPLETE</span>
            </div>
            <div class="emotion-display">
                <div class="emotion-icon">{icon}</div>
                <div>
                    <div class="emotion-name">{emotion.upper()}</div>
                    <div class="emotion-caption">Primary emotional expression detected in the recording</div>
                </div>
            </div>
            <div class="confidence-heading">
                <span>Confidence</span>
                <strong>{confidence:.1f}%</strong>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence:.1f}%"></div>
            </div>
        </div>
        """

        probability_rows = ""

        for label, probability in zip(label_encoder.classes_, probabilities):
            percentage = probability * 100
            label_icon = emotion_icons.get(label.lower(), "🎭")

            probability_rows += f"""
            <div class="probability-row">
                <div class="probability-label">
                    <span>{label_icon} {label.capitalize()}</span>
                    <strong>{percentage:.1f}%</strong>
                </div>
                <div class="probability-track">
                    <div class="probability-fill" style="width:{percentage:.1f}%"></div>
                </div>
            </div>
            """

        details = f"""
        <div class="details-card">
            <div class="details-header">
                <span>📊</span>
                <div>
                    <div class="details-title">Prediction breakdown</div>
                    <div class="details-subtitle">How the model distributed its confidence</div>
                </div>
            </div>

            <div class="probability-list">
                {probability_rows}
            </div>

            <div class="technical-grid">
                <div class="stat-box">
                    <span class="stat-icon">⏱</span>
                    <div><span class="stat-label">Audio duration</span>
                    <strong>{used_duration:.2f}s</strong></div>
                </div>
                <div class="stat-box">
                    <span class="stat-icon">🧬</span>
                    <div><span class="stat-label">Feature shape</span>
                    <strong>{features.shape[0]} × {features.shape[1]}</strong></div>
                </div>
                <div class="stat-box">
                    <span class="stat-icon">🧠</span>
                    <div><span class="stat-label">Model</span>
                    <strong>Advanced CNN</strong></div>
                </div>
                <div class="stat-box">
                    <span class="stat-icon">🎙</span>
                    <div><span class="stat-label">Input</span>
                    <strong>Voice recording</strong></div>
                </div>
            </div>
        </div>
        """

        return result, details, None

    except Exception as e:

        print("\nERROR:")
        print(e)

        return (
            f"""
            <div class="error-card">
                <div class="error-icon">⚠️</div>
                <div>
                    <div class="error-title">Analysis failed</div>
                    <div class="error-text">{str(e)}</div>
                </div>
            </div>
            """,
            "",
            None
        )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_all():

    return (
        None,
        get_random_phrase(),
        "",
        "",
        None
    )


# ============================================================
# TYPING EFFECT
# ============================================================

def type_phrase():

    phrase = get_random_phrase()

    current = ""

    for character in phrase:

        current += character

        yield current

        time.sleep(0.025)


# ============================================================
# POLISHED UI
# ============================================================

css = """
body {
    background:
        radial-gradient(circle at 50% -10%, rgba(99,102,241,.18), transparent 35%),
        radial-gradient(circle at 0% 50%, rgba(34,211,238,.06), transparent 30%),
        #07090d !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 24px 45px !important;
}

.hero {
    text-align: center;
    padding: 22px 10px 34px;
}

.hero-badge {
    display: inline-flex;
    padding: 7px 13px;
    border: 1px solid rgba(129,140,248,.28);
    border-radius: 999px;
    background: rgba(99,102,241,.08);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.hero-title {
    margin: 16px 0 8px;
    font-size: clamp(34px,5vw,54px);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -1.8px;
    background: linear-gradient(100deg,#f8fafc 15%,#c7d2fe 55%,#67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 660px;
    margin: 0 auto;
    color: #8f9bad;
    font-size: 15px;
    line-height: 1.7;
}

.panel {
    background: rgba(15,18,25,.82) !important;
    border: 1px solid rgba(148,163,184,.12) !important;
    border-radius: 22px !important;
    box-shadow: 0 20px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
    padding: 22px !important;
}

.panel-title {
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 750;
}

.panel-subtitle {
    color: #697586;
    font-size: 12px;
    margin-bottom: 16px;
}

.phrase-panel,.audio-panel {
    min-height: 210px;
}

.phrase-label {
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.phrase-text {
    min-height: 84px;
    display: flex;
    align-items: center;
    color: #f8fafc !important;
    font-size: 22px !important;
    line-height: 1.45 !important;
    font-weight: 600 !important;
    margin: 4px 0 14px !important;
}

.phrase-text p { margin: 0 !important; }

.analyze-btn button {
    min-height: 54px !important;
    border: 0 !important;
    border-radius: 15px !important;
    background: linear-gradient(135deg,#6366f1,#4f46e5) !important;
    box-shadow: 0 12px 30px rgba(79,70,229,.25);
    font-size: 15px !important;
    font-weight: 750 !important;
}

.analyze-btn button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 35px rgba(79,70,229,.38);
}

.clear-btn button,.new-phrase-btn button {
    min-height: 48px !important;
    border-radius: 13px !important;
    background: rgba(255,255,255,.045) !important;
    border: 1px solid rgba(148,163,184,.14) !important;
    color: #cbd5e1 !important;
    font-weight: 650 !important;
}

.result-card,.details-card {
    background: linear-gradient(145deg,rgba(17,24,39,.94),rgba(10,13,19,.96));
    border: 1px solid rgba(129,140,248,.15);
    border-radius: 22px;
    padding: 26px;
    margin-top: 10px;
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
}

.result-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}

.result-label {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-status {
    color: #67e8f9;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.emotion-display {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 6px 0 24px;
}

.emotion-icon {
    width: 76px;
    height: 76px;
    display: grid;
    place-items: center;
    border-radius: 20px;
    background: rgba(99,102,241,.11);
    border: 1px solid rgba(129,140,248,.18);
    font-size: 42px;
}

.emotion-name {
    color: #f8fafc;
    font-size: clamp(30px,4vw,44px);
    line-height: 1;
    font-weight: 850;
}

.emotion-caption {
    margin-top: 8px;
    color: #738094;
    font-size: 12px;
}

.confidence-heading,.probability-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #aab4c3;
    font-size: 12px;
}

.confidence-heading strong {
    color: #67e8f9;
    font-size: 15px;
}

.confidence-track,.probability-track {
    width: 100%;
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255,255,255,.06);
    margin-top: 9px;
}

.confidence-fill,.probability-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg,#6366f1,#22d3ee);
}

.details-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(148,163,184,.08);
}

.details-header > span {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: rgba(99,102,241,.1);
}

.details-title {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 750;
}

.details-subtitle {
    color: #697586;
    font-size: 11px;
    margin-top: 2px;
}

.probability-list {
    display: grid;
    gap: 14px;
    padding: 18px 0;
}

.probability-label strong {
    color: #e2e8f0;
}

.probability-track {
    height: 6px;
    margin-top: 7px;
}

.technical-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 10px;
}

.stat-box {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 13px;
    border: 1px solid rgba(148,163,184,.08);
    border-radius: 13px;
    background: rgba(255,255,255,.025);
}

.stat-icon { font-size: 16px; }

.stat-label {
    display: block;
    color: #687487;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.stat-box strong {
    display: block;
    color: #cbd5e1;
    font-size: 11px;
    margin-top: 3px;
}

.error-card {
    display: flex;
    gap: 14px;
    padding: 20px;
    border-radius: 17px;
    border: 1px solid rgba(248,113,113,.2);
    background: rgba(127,29,29,.12);
}

.error-icon { font-size: 24px; }
.error-title { color: #fca5a5; font-weight: 750; }
.error-text { color: #a7aeb9; font-size: 12px; margin-top: 4px; word-break: break-word; }

.footer {
    text-align: center;
    padding: 30px 0 4px;
    color: #4b5563;
    font-size: 11px;
}

@media (max-width:760px) {
    .gradio-container { padding: 18px 12px 30px !important; }
    .panel { padding: 16px !important; }
    .phrase-text { font-size: 18px !important; }
    .emotion-display { gap: 14px; }
    .emotion-icon { width: 62px; height: 62px; font-size: 34px; }
    .technical-grid { grid-template-columns: repeat(2,1fr); }
    .result-topline { align-items: flex-start; flex-direction: column; gap: 7px; }
}
"""

# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="Emotion Recognition AI",
    css=css,
    theme=gr.themes.Base(),
    delete_cache=(60,600)
) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">✦ Voice Intelligence</div>
        <div class="hero-title">🎭 Emotion Recognition AI</div>
        <div class="hero-subtitle">
            Speak naturally and let the neural network analyze
            the emotional expression carried by your voice.
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, elem_classes="panel phrase-panel"):
            gr.HTML("""
            <div class="phrase-label">💬 Suggested phrase</div>
            <div class="panel-subtitle">
                Use the phrase below or say anything naturally.
            </div>
            """)

            phrase = gr.Markdown(
                get_random_phrase(),
                elem_classes="phrase-text"
            )

            new_phrase_btn = gr.Button(
                "↻  New phrase",
                elem_classes="new-phrase-btn"
            )

        with gr.Column(scale=1, elem_classes="panel audio-panel"):
            gr.HTML("""
            <div class="panel-title">🎙 Your voice</div>
            <div class="panel-subtitle">
                Record up to 10 seconds or upload a WAV file.
            </div>
            """)

            audio_input = gr.Audio(
                sources=["microphone","upload"],
                type="filepath",
                format="wav",
                label="Voice recording"
            )

    with gr.Row():
        analyze_btn = gr.Button(
            "✦  Analyze emotion",
            variant="primary",
            elem_classes="analyze-btn"
        )

        clear_btn = gr.Button(
            "Clear recording",
            elem_classes="clear-btn"
        )

    gr.HTML("""
    <div style="margin:30px 0 10px;">
        <div class="panel-title">📊 Analysis result</div>
        <div class="panel-subtitle">
            The model's strongest detected emotional expression.
        </div>
    </div>
    """)

    result = gr.HTML("""
    <div class="result-card">
        <div class="result-topline">
            <span class="result-label">WAITING FOR AUDIO</span>
            <span class="result-status">● READY</span>
        </div>
        <div class="emotion-display">
            <div class="emotion-icon">🎙</div>
            <div>
                <div class="emotion-name">READY</div>
                <div class="emotion-caption">
                    Record your voice and press “Analyze emotion”.
                </div>
            </div>
        </div>
    </div>
    """)

    details = gr.HTML("")
    hidden = gr.State()

    gr.HTML("""
    <div class="footer">
        <span>Advanced CNN</span> · MFCC + Delta + Delta-Delta
        · RAVDESS-trained emotion classifier
    </div>
    """)

    analyze_btn.click(
        fn=predict_emotion,
        inputs=audio_input,
        outputs=[result,details,hidden]
    )

    new_phrase_btn.click(
        fn=type_phrase,
        inputs=None,
        outputs=phrase
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[audio_input,phrase,result,details,hidden]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()
