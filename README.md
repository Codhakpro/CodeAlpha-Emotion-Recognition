# 🎭 Emotion Recognition AI

A speech-emotion recognition application built with Python, PyTorch, Librosa, and Streamlit. The project extracts MFCC-based audio features from a short voice recording and uses an advanced 1D convolutional neural network (CNN) trained on the RAVDESS emotional speech dataset to predict one of eight emotions.

## ✨ Current application

The main user interface is the Streamlit app in `streamlit_app.py`.

The interface is designed around a simple three-step flow:

1. **Read a suggested phrase** or speak naturally.
2. **Record up to 10 seconds** of audio.
3. **Analyze the recording** and view the detected emotion plus the complete probability breakdown.

The interface includes:

- Dark, responsive UI
- Suggested-phrase generator
- Browser microphone recording
- Automatic 10-second maximum processing window
- Emotion result with emoji and confidence
- Probability bars for all eight emotion classes
- Audio duration and feature-shape statistics
- Clear recording control
- Cached model loading for faster repeated predictions

## 🧠 Supported emotions

The trained classifier predicts:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## 🤖 Model

The project uses an advanced 1D CNN named `AdvancedEmotionCNN`.

### Architecture

```text
Input: 120 × 165
       │
       ├── Conv1D 120 → 64
       ├── BatchNorm
       ├── ReLU
       ├── MaxPool
       │
       ├── Conv1D 64 → 128
       ├── BatchNorm
       ├── ReLU
       ├── MaxPool
       │
       ├── Conv1D 128 → 256
       ├── BatchNorm
       ├── ReLU
       ├── Adaptive Average Pool
       │
       ├── Linear 256 → 128
       ├── ReLU
       ├── Dropout 0.4
       └── Linear 128 → 8
```

The recorded audio is converted into:

- 40 MFCC features
- 40 delta features
- 40 delta-delta features

These are concatenated into **120 feature channels** and padded/truncated to **165 time frames**.

## 📊 Training

Dataset: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

The advanced model was trained using:

- 1,440 audio files
- 80/20 stratified train/test split
- 1,152 training samples
- 288 test samples
- Adam optimizer
- Learning rate: `0.0005`
- Weight decay: `1e-4`
- Cross-entropy loss
- 25 training epochs

The recorded test accuracy for the advanced CNN was **74.31%**.

> Accuracy is a dataset test result and should not be interpreted as a guarantee of the model's performance on every real-world speaker or recording environment.

## 📁 Project structure

```text
CodeAlpha_Emotion_Recognition/
│
├── data/
│   └── Audio_Speech_Actors_01-24/
│
├── emotion_recognition.ipynb
├── streamlit_app.py
├── app.py
│
├── emotion_cnn_advanced.pth
├── emotion_cnn.pth
├── label_encoder_advanced.pkl
├── label_encoder.pkl
├── mfcc_mean.npy
├── mfcc_std.npy
│
├── requirements.txt
├── README.md
└── .gitignore
```

`app.py` is the earlier Gradio version of the interface. `streamlit_app.py` is the current Streamlit interface.

## ⚙️ Installation

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run locally

From the project folder:

```bash
streamlit run streamlit_app.py
```

Streamlit will open the application in the browser, normally at:

```text
http://localhost:8501
```

## 🎙️ Using the application

1. Open the Streamlit app.
2. Read the suggested phrase or speak naturally.
3. Click the microphone recorder.
4. Record a short sample.
5. Click **Analyze emotion**.
6. View the detected emotion and confidence.
7. Inspect the probability breakdown to see how the model distributed its prediction.
8. Use **Clear recording** before making another test recording if needed.

Short test recordings are intended for inference only. The current application does **not** collect user recordings for retraining.

## 🔬 Feature extraction

Audio is loaded at **16 kHz** and converted to mono. The feature pipeline is:

```text
Audio
  ↓
40 MFCC
  ↓
Delta
  ↓
Delta-Delta
  ↓
Concatenate
  ↓
120 × time features
  ↓
Pad / truncate to 120 × 165
  ↓
Advanced CNN
  ↓
8 emotion probabilities
```

## 🗂️ Important model files

| File | Purpose |
| --- | --- |
| `emotion_cnn_advanced.pth` | Trained advanced CNN weights |
| `label_encoder_advanced.pkl` | Converts model class indices to emotion names |
| `mfcc_mean.npy` | Saved feature normalization statistics from the notebook pipeline |
| `mfcc_std.npy` | Saved feature normalization statistics from the notebook pipeline |
| `emotion_recognition.ipynb` | Training and experimentation notebook |
| `streamlit_app.py` | Current web application |

The advanced CNN was trained/evaluated using the extracted feature tensors without applying the later saved mean/std normalization to the training input. The current Streamlit inference path therefore keeps the same input convention as the successful advanced model evaluation.

## 🌐 Deployment

The Streamlit application can be deployed from the GitHub repository using a Streamlit-compatible hosting service.

The deployment should use:

```text
Main file: streamlit_app.py
Dependencies: requirements.txt
```

The model files (`.pth`, `.pkl`, and `.npy`) must also be present in the repository because the application loads them at runtime.

## 🛠️ Technologies

- Python
- PyTorch
- Librosa
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- Streamlit
- Git / GitHub

## ⚠️ Limitations

This project is a machine-learning demonstration and internship project. Speech emotion recognition is affected by factors such as:

- Speaker differences
- Accent and pronunciation
- Microphone quality
- Background noise
- Recording conditions
- Emotional intensity
- Differences between acted and natural emotion

The RAVDESS dataset contains acted emotional speech, so real-world performance can differ from the reported test accuracy.

## 🚀 Future improvements

Possible future work includes:

- Larger and more diverse speech datasets
- Better handling of background noise
- Data augmentation
- Model calibration
- More robust real-world evaluation
- Additional audio features
- Improved inference on naturally spoken emotion
- Optional user-feedback collection for a future retraining pipeline

## 👤 Project

### CodeAlpha Emotion Recognition Project

Built as a machine-learning internship project and developed as a practical exploration of speech processing, feature engineering, CNNs, model inference, and Python web application deployment.
