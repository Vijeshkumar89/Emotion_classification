import numpy as np
import librosa
from keras.models import load_model

# Load model
model = load_model("cnn+gru_model.h5")
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs.reshape(1, -1)

# Example usage
if __name__ == "__main__":
    filepath = "Audio_Song_Actors_01-24/Actor_01/03-02-02-01-01-01-01.wav"
    features = extract_mfcc(filepath)
    prediction = model.predict(features)
    print("Predicted Emotion:", class_labels[np.argmax(prediction)])
