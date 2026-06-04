
# 🎙️ Emotion Recognition from Speech — CodeAlpha ML Internship

> **Task 2 — Recognize human emotions from speech audio using Deep Learning.**

---

## 📌 Project Overview

This project builds an **Emotion Recognition Model** that listens to audio files and detects human emotions like Happy, Sad, Angry, Fearful, Neutral, Disgust, Calm, and Surprised. Built as part of the **CodeAlpha Machine Learning Internship**.

---

## 🎭 Emotions Detected (8 Classes)

| Code | Emotion |
|------|---------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## 📊 Dataset

- **Name:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Source:** [Zenodo — RAVDESS](https://zenodo.org/record/1188976)
- **Size:** 1,440 audio files (.wav)
- **Actors:** 24 (12 male, 12 female)
- **Format:** .wav, 16-bit, 48kHz

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Librosa | Audio processing & feature extraction |
| NumPy & Pandas | Data manipulation |
| TensorFlow / Keras | Deep learning models |
| Scikit-learn | Evaluation & preprocessing |
| Matplotlib & Seaborn | Visualization |
| Google Colab | Development environment (GPU) |

---

## 🚀 Project Pipeline

```
Audio Files (.wav)
      ↓
Feature Extraction
  • MFCCs       → 40 features
  • Chroma      → 12 features  
  • Mel Spectrogram → 128 features
  • Total       → 180 features per file
      ↓
Normalize + Train/Test Split (80/20)
      ↓
Train 2 Models
  • LSTM
  • CNN ⭐ Best
      ↓
Evaluate
  • Accuracy, F1-Score
  • Confusion Matrix
      ↓
Save Best Model (.h5)
      ↓
Live Prediction (upload audio file)
```

---

## 🎵 Feature Extraction Details

### MFCCs (Mel-Frequency Cepstral Coefficients)
- Most important feature for speech emotion
- Captures vocal tract characteristics
- 40 coefficients extracted

### Chroma Features
- Captures pitch and tonal information
- 12 pitch classes

### Mel Spectrogram
- Frequency map of audio signal
- 128 mel bands
- Represents energy at different frequencies

---

## 📈 Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| LSTM | ~13-31% ❌ | Struggled with small dataset |
| **CNN** | **57.29%** ✅ | Best performance |

### 🏆 Best Model: CNN
- **Test Accuracy: 57.29%**
- 85 epochs trained
- Early stopping applied

---

## 📉 Confusion Matrix Highlights

| Emotion | Correctly Detected | Notes |
|---------|-------------------|-------|
| Calm | 26/38 | ✅ Best |
| Angry | 21/38 | ✅ Good |
| Fearful | 21/38 | ✅ Good |
| Happy | 20/38 | ✅ Good |
| Sad | 19/38 | ⚠️ Confused with fearful |
| Neutral | 9/19 | ⚠️ Less training samples |

---

## 💡 Key Insights

- **CNN outperforms LSTM** on small audio datasets (1440 samples)
- **Happy & Fearful** are most commonly confused — similar vocal patterns
- **Neutral** has less samples (96 vs 192) — harder to detect
- Accuracy can be improved with **more data** or **data augmentation**
- 57% accuracy on 8-class emotion recognition is acceptable for this dataset size

---

## 📁 Files in This Repo

| File | Description |
|------|-------------|
| `EmotionRecognition.ipynb` | Main Colab notebook |
| `emotion_recognition_model.h5` | Saved CNN model |
| `label_encoder.pkl` | Saved label encoder |
| `ravdess_features.csv` | Extracted features (180 per file) |
| `README.md` | Project documentation |

---

## ▶️ How to Run

**Google Colab (Recommended):**

**Step 1 — Install libraries:**
```bash
!pip install librosa soundfile tqdm
```

**Step 2 — Download RAVDESS dataset:**
```bash
!wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
!unzip Audio_Speech_Actors_01-24.zip -d RAVDESS
```

**Step 3 — Run all cells in order**

---

## 🔮 Predict on New Audio File

```python
from google.colab import files
uploaded = files.upload()  # Upload your .wav file

# Model will predict emotion automatically
# Output: Emotion name + Confidence % + All probabilities
```

---

## 🏢 Internship Details

- **Company:** CodeAlpha
- **Domain:** Machine Learning
- **Task:** Task 2 — Emotion Recognition from Speech
- **Website:** [www.codealpha.tech](https://www.codealpha.tech)

---

## 👤 Author

**Ameer Hamza (Meeri)**
- LinkedIn: [your-linkedin-link]
- GitHub: [your-github-link]

---

*Built with ❤️ during CodeAlpha ML Internship*
