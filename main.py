import cv2
from deepface import DeepFace
import csv
import datetime
import pyttsx3
import pyaudio
import wave
import threading
import time
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from collections import deque

# ------------------- Setup -------------------

# --- TTS Engine ---
engine = pyttsx3.init()
def speak_response(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

# --- Emotion to Response Mapping ---
def get_response(emotion):
    """Returns a pre-defined response based on the detected emotion."""
    responses = {
        "happy": "You seem in good spirits! Keep it up 🚀",
        "neutral": "All systems look stable. Stay focused 👍",
        "sad": "I sense you’re feeling low. Let’s try a short breathing exercise.",
        "angry": "I notice stress. Stretch your arms and take a short break.",
        "fear": "You seem worried. Remember, support is available.",
        "disgust": "You seem uncomfortable. Relax for a moment.",
        "surprise": "That caught you off guard! Everything okay?",
        "fatigue": "You look tired. Please hydrate or rest for a few minutes."
    }
    return responses.get(emotion, "I am here with you. Let’s keep going strong.")

# --- Audio Recording ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3  # Records audio in 3-second chunks
WAVE_OUTPUT_FILENAME = "temp_audio.wav"
audio = pyaudio.PyAudio()
audio_frames = []
is_recording = False

def start_recording():
    """Starts the audio recording in a separate thread."""
    global is_recording, audio_frames
    is_recording = True
    audio_frames = []
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while is_recording:
        data = stream.read(CHUNK)
        audio_frames.append(data)
    stream.stop_stream()
    stream.close()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(audio_frames))
    waveFile.close()

# --- Speech Emotion Recognition (SER) ---
# **NOTE**: You need to download a pre-trained SER model.
# For this example, we assume you have a model saved as 'ser_model.pkl'.
# You can find pre-trained models on GitHub or train your own.
try:
    ser_model = joblib.load("ser_model.pkl")
except FileNotFoundError:
    print("SER model 'ser_model.pkl' not found. Audio analysis will be disabled.")
    ser_model = None

def extract_features(file_name):
    """Extracts audio features (MFCC, Chroma, Mel) from an audio file."""
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=librosa.stft(X), sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        return np.hstack((mfccs, chroma, mel))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def analyze_audio_emotion():
    """Analyzes the emotion from the recorded audio file."""
    if not ser_model:
        return "neutral"
    features = extract_features(WAVE_OUTPUT_FILENAME)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = ser_model.predict(features)
        return prediction[0]
    return "neutral"

# --- Emotion Fusion ---
def get_fused_emotion(video_emotion, audio_emotion):
    """Combines video and audio emotions. Gives priority to non-neutral emotions."""
    if video_emotion != "neutral":
        return video_emotion
    if audio_emotion != "neutral":
        return audio_emotion
    return "neutral"

# --- Real-time Plotting ---
emotion_history = deque(maxlen=50) # Store last 50 emotions
emotion_map = { "happy": 5, "neutral": 3, "sad": 1, "angry": 0, "fear": 2, "disgust": 1, "surprise": 4, "fatigue": 2}

def update_plot():
    """Updates and saves the emotion history plot."""
    plt.figure(figsize=(5, 2.5))
    plt.plot(list(emotion_history), marker='o', linestyle='-', color='b')
    plt.title("Emotion Log")
    plt.xlabel("Time")
    plt.ylabel("Emotion Level")
    plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
    plt.tight_layout()
    plt.savefig("emotion_plot.png")
    plt.close()

# ------------------- Main Setup -------------------
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("Starting real-time Multimodal AI Assistant. Press 'q' to quit.")

# Open log file
log_file = open("emotion_log.csv", mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(log_file)
if log_file.tell() == 0:
    csv_writer.writerow(["Timestamp", "Video Emotion", "Audio Emotion", "Fused Emotion", "Response"])

# Start audio recording thread
recording_thread = threading.Thread(target=start_recording)
recording_thread.start()

# ------------------- Main Loop -------------------
last_analysis_time = time.time()
analysis_interval = 4  # Analyze every 4 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        last_analysis_time = current_time

        # --- Video Emotion Analysis ---
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            video_emotion = result[0]['dominant_emotion']
        except Exception as e:
            video_emotion = "neutral"

        # --- Audio Emotion Analysis ---
        is_recording = False
        time.sleep(0.1) # allow thread to save file
        audio_emotion = analyze_audio_emotion()
        # Restart recording for the next chunk
        if not recording_thread.is_alive():
            recording_thread = threading.Thread(target=start_recording)
            recording_thread.start()


        # --- Emotion Fusion & Response ---
        fused_emotion = get_fused_emotion(video_emotion, audio_emotion)
        response = get_response(fused_emotion)
        speak_response(response) # TTS response

        # --- Logging ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, video_emotion, audio_emotion, fused_emotion, response])
        log_file.flush()

        # --- Update Plot ---
        emotion_history.append(emotion_map.get(fused_emotion, 3))
        update_plot()


    # --- UI Display ---
    # Overlay emotions and response on video feed
    cv2.putText(frame, f"Emotion: {fused_emotion if 'fused_emotion' in locals() else 'Analyzing...'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Assistant: {response if 'response' in locals() else ''}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display coping prompts
    if 'fused_emotion' in locals():
        if fused_emotion == "sad":
            cv2.putText(frame, "Try 4-7-8 breathing: Inhale 4s, Hold 7s, Exhale 8s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif fused_emotion == "angry":
            cv2.putText(frame, "Stretch arms & shoulders for 30 seconds", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        elif fused_emotion == "fatigue":
            cv2.putText(frame, "Reminder: Drink water and rest briefly", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display the emotion plot
    try:
        plot_img = cv2.imread("emotion_plot.png")
        plot_img_resized = cv2.resize(plot_img, (400, 200))
        h, w, _ = frame.shape
        frame[h-200:h, w-400:w] = plot_img_resized
    except:
        pass # If plot is not ready yet

    # Show the final feed
    cv2.imshow("Crew Well-being Multimodal AI Assistant", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_recording = False # Signal recording thread to stop
        break

# ------------------- Cleanup -------------------
recording_thread.join()
cap.release()
log_file.close()
cv2.destroyAllWindows()
audio.terminate()