## 🎧 Speech Emotion Recognition Web App

A fully functional web application built using **Streamlit** that classifies emotions from `.wav` audio files using a deep learning model trained on the **RAVDESS** dataset with MFCC features.

---

## 📌 Project Description

This project focuses on detecting human emotions from speech using advanced audio processing and deep learning. The web app allows users to upload a .wav audio file, extracts MFCC (Mel-Frequency Cepstral Coefficients) features, and leverages a pre-trained neural network to predict the underlying emotion.

- User-friendly interface: Upload and classify audio in seconds.

- Real-time prediction: Get instant emotion results for your speech.

- Robust model: Trained on a diverse, high-quality dataset for reliable results.

---

## 🧪 Dataset: RAVDESS

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- 8 emotion classes:  
  - `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Contains recordings by 24 actors (12 male, 12 female) with variations in emotion and intensity.

---

## 🔍 Preprocessing Methodology

1. **File Loading:**

    - Audio files are loaded using librosa for efficient and reliable audio handling.

2. **MFCC Extraction:**

    - Extracts 40 MFCC features from each audio sample.

    - Uses a 3-second segment starting from a 0.5s offset to focus on the core speech.

    - Features are averaged across time frames, resulting in a compact (40,) feature vector per sample.

3. **Label Encoding:**

    - Emotion labels are parsed directly from filenames.

    - One-Hot Encoding is used for multi-class classification, enabling the model to distinguish all 8 emotions.

---

## 🧠 Model Architecture

A hybrid CNN + BiGRU deep learning model trained on MFCC features, designed to capture both local patterns and long-term dependencies in audio:

```python
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(180, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(512, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Bidirectional(GRU(256, return_sequences=True)),
    Bidirectional(GRU(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(8, activation='softmax')
])
```
## Key Points:
- CNN layers extract local time-frequency patterns from MFCCs.

- Bidirectional GRUs capture temporal dependencies in both directions.

- Dropout and BatchNorm help prevent overfitting and stabilize training.

- Final dense layers map extracted features to emotion probabilities.

## 📈 Classification Report

```python
                precision    recall  f1-score   support

       angry       0.88      0.84      0.86        75
        calm       0.83      0.92      0.87        75
     disgust       0.84      0.67      0.74        39
     fearful       0.76      0.67      0.71        75
       happy       0.86      0.73      0.79        75
     neutral       0.73      0.92      0.81        38
         sad       0.75      0.75      0.75        75
   surprised       0.73      0.97      0.84        39

    accuracy                           0.80       491
   macro avg       0.80      0.81      0.80       491
weighted avg       0.80      0.80      0.80       491

```

## Summary & Insights:
- Overall accuracy: 80% on the test set, showing strong generalization.

- Highest performance: calm, angry, and happy emotions, with F1-scores above 0.85.

- Challenging classes: disgust and fearful show slightly lower recall, indicating these emotions are harder to distinguish—possibly due to subtle vocal cues.

- Balanced performance: Macro and weighted averages are both at 0.80, reflecting consistent results across all classes.

