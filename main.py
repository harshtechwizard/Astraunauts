import cv2
from deepface import DeepFace
import csv, datetime
import pyttsx3

# ------------------- Setup -------------------
# Initialize TTS engine
engine = pyttsx3.init()

def speak_response(text):
    engine.say(text)
    engine.runAndWait()

# Emotion → assistant response mapping
def get_response(emotion):
    responses = {
        "happy": "You seem in good spirits! Keep it up 🚀",
        "neutral": "All systems look stable. Stay focused 👍",
        "sad": "I sense you’re feeling low. Let’s try a short breathing exercise.",
        "angry": "I notice stress. Stretch your arms and take a short break.",
        "fear": "You sound worried. Remember, support is available.",
        "disgust": "You seem uncomfortable. Relax for a moment.",
        "surprise": "That caught you off guard! Everything okay?",
        "fatigue": "You look tired. Please hydrate or rest for a few minutes."
    }
    return responses.get(emotion, "I am here with you. Let’s keep going strong.")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("Starting real-time AI Assistant. Press 'q' to quit.")

# ------------------- Logging -------------------
log_file = open("emotion_log.csv", mode="a", newline="",encoding="utf-8")
csv_writer = csv.writer(log_file)
# csv_writer.writerow(["Timestamp", "Emotion", "Response"])
if log_file.tell() == 0:
    csv_writer.writerow(["Timestamp", "Emotion", "Response"])


# ------------------- Main Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except:
        emotion = "neutral"

    response = get_response(emotion)

    # Timestamp for logs
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([timestamp, emotion, response])
    log_file.flush()

    # Overlay on video feed
    cv2.putText(frame, f"Emotion: {emotion}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Assistant: {response}", (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Show coping prompt
    if emotion == "sad":
        cv2.putText(frame, "Try 4-7-8 breathing: Inhale 4s, Hold 7s, Exhale 8s",
                    (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    elif emotion == "angry":
        cv2.putText(frame, "Stretch arms & shoulders for 30 seconds",
                    (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    elif emotion == "fatigue":
        cv2.putText(frame, "Reminder: Drink water and rest briefly",
                    (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Speak response
    speak_response(response)

    # Show feed
    cv2.imshow("Crew Well-being AI Assistant", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- Cleanup -------------------
cap.release()
log_file.close()
cv2.destroyAllWindows()