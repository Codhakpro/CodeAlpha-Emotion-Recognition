# 🎭 Emotion Recognition AI

A speech emotion recognition project built with **Python, PyTorch, Librosa, and Gradio**.

The application takes a short voice recording, extracts acoustic features from the speech, and uses a trained 1D Convolutional Neural Network (CNN) to predict the expressed emotion.

## ✨ Features

- 🎙️ Record or upload a voice clip
- ⏱️ Maximum analysis length of 10 seconds
- 🧠 Advanced 1D CNN emotion classifier
- 📊 Confidence/probability breakdown for all emotion classes
- 💬 Random suggested phrases to make testing more interesting
- 🧹 Clear button for starting a new test
- 🌙 Custom dark/futuristic Gradio interface
- 💻 Automatically uses CUDA when a compatible GPU is available, otherwise CPU

## 🎯 Supported Emotions

The model recognizes 8 emotion classes:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## 🧠 Model

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The advanced model uses:

- Sample rate: **16 kHz**
- MFCC features: **40**
- Delta MFCC: **40**
- Delta-delta MFCC: **40**
- Total input channels: **120**
- Target sequence length: **165 frames**
- Architecture: **1D CNN + Batch Normalization + ReLU + Max Pooling**
- Adaptive average pooling before classification
- Dropout in the classifier

The advanced model achieved approximately **74.31% test accuracy** during the project's evaluation.

## 📁 Project Structure

```text
CodeAlpha_Emotion_Recognition/
│
├── app.py
├── emotion_recognition.ipynb
│
├── emotion_cnn_advanced.pth
├── emotion_cnn.pth
│
├── label_encoder_advanced.pkl
├── label_encoder.pkl
├── scaler.pkl
│
├── mfcc_mean.npy
├── mfcc_std.npy
│
├── my_voice.wav
├── requirements.txt
├── README.md
│
└── data/
    └── Audio_Speech_Actors_01-24/
        ├── Actor_01/
        ├── Actor_02/
        └── ...
```

> The trained model files and preprocessing files are required by `app.py`. The RAVDESS dataset is used for training/evaluation and does not need to be loaded by the GUI for normal prediction.

## 🚀 Installation

### 1. Clone or download the project

Place the project folder somewhere convenient on your computer.

### 2. Open the project in VS Code

Open the `CodeAlpha_Emotion_Recognition` folder in VS Code.

### 3. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate
```

### 5. Install the dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run the Application

Make sure the virtual environment is active, then run:

```powershell
python app.py
```

Gradio will start a local web application, normally at:

```text
http://127.0.0.1:7860
```

Open the address shown in the terminal in your browser.

## 🎙️ How to Use

1. Open the application.
2. Read the suggested phrase.
3. Record your voice or upload an audio clip.
4. Keep the recording at **10 seconds or less**.
5. Click **Analyze Emotion**.
6. The application displays:
   - Detected emotion
   - Confidence
   - Probability distribution
   - Audio duration
   - Feature information
7. Click **Clear** to start another test.

## 🔬 Training Pipeline

The notebook contains the machine-learning workflow:

1. Load the RAVDESS dataset.
2. Extract audio features with Librosa.
3. Extract 40 MFCCs.
4. Calculate delta and delta-delta features.
5. Combine the three feature sets into 120 channels.
6. Pad or trim the time dimension to 165 frames.
7. Split the data into training and testing sets.
8. Train the advanced 1D CNN with PyTorch.
9. Evaluate the trained model.
10. Save the trained model and label encoder.
11. Use the saved artifacts for GUI inference.

## 📦 Important Model Files

`app.py` expects these files to be available in the same project directory:

```text
emotion_cnn_advanced.pth
label_encoder_advanced.pkl
mfcc_mean.npy
mfcc_std.npy
```

If any of these files are missing, the application will not be able to load the trained model or its preprocessing information.

## ⚠️ Notes

- This is an **emotion classification experiment**, not a reliable psychological assessment.
- Speech emotion recognition can be affected by microphone quality, background noise, accent, speaking style, volume, and the way emotions are expressed.
- A model prediction should therefore be treated as an estimate rather than a definitive statement about a person's emotional state.
- The current application does not use recorded user voices to retrain the model.

## 🛠️ Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| PyTorch | Neural network and inference |
| Librosa | Audio loading and feature extraction |
| NumPy | Numerical processing |
| Scikit-learn | Dataset splitting, label encoding and evaluation |
| Joblib | Saving/loading preprocessing objects |
| Gradio | Web-based user interface |
| Jupyter Notebook | Training and experimentation |
| SoundDevice | Optional microphone recording in the notebook |
| SciPy | Saving recorded WAV files in the notebook |

## 📚 Dataset

The model was trained using the **RAVDESS** emotional speech dataset.

RAVDESS contains acted speech recordings representing multiple emotional categories. The project uses the speech recordings to learn patterns associated with the eight emotion classes listed above.

## 👨‍💻 Project

### CodeAlpha Internship — Emotion Recognition

Built as a machine-learning project combining speech processing, deep learning, and an interactive web interface.
