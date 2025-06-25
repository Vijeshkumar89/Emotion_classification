import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
from keras.models import load_model

# Load the trained model (ensure model.h5 is present in your app directory)
model = load_model("cnn+gru_model.h5")  # <-- replace with your actual model path

# Define class labels (adjust to match your model's training labels)
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Feature extraction using MFCC only (same as training)
def extract_mfccs(filename):
  y, sr = librosa.load(filename, duration=3, offset=0.5)
  mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
  return mfccs

# Streamlit UI
st.title("Speech Emotion Recognition 🎤")
st.write("Upload an audio file (WAV format) to classify the emotion")

uploaded_file = st.file_uploader("Choose a .wav audio file", type=[".wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_mfccs("temp.wav")
    features = np.expand_dims(features, axis=0)  # shape = (1, features)

    prediction = model.predict(features)
    predicted_label = class_labels[np.argmax(prediction)]
    st.markdown(f"### 🎧 Predicted Emotion: `{predicted_label}`")

    os.remove("temp.wav")
