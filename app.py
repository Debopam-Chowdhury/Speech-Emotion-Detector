from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import joblib
import soundfile as sf
import io

app = Flask(__name__)

# Load trained model, scaler, and label encoder
model = joblib.load("emotion_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def validate_audio(file_stream):
    """Ensure the audio file is in valid PCM 16-bit format."""
    try:
        file_stream.seek(0)
        data, samplerate = sf.read(file_stream, dtype="int16")
        return True
    except Exception as e:
        print(f"Invalid WAV file: {e}")
        return False

def extract_features(file_stream, mfcc_num=40):
    """Extract MFCC features from an in-memory audio file."""
    try:
        file_stream.seek(0)
        audio, sr = librosa.load(file_stream, sr=22050, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=mfcc_num)
        mfccs = np.mean(mfccs, axis=1)
        return scaler.transform([mfccs])
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Convert file to an in-memory byte stream
    file_stream = io.BytesIO(file.read())

    # Validate WAV file format
    if not validate_audio(file_stream):
        return jsonify({"error": "Invalid WAV format"}), 500

    # Extract features and predict emotion
    features = extract_features(file_stream)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    prediction = model.predict(features)[0]
    emotion = label_encoder.inverse_transform([prediction])[0]

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
