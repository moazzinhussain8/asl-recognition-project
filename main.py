import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
from collections import Counter
from gtts import gTTS
from playsound import playsound
import os
import tempfile
import threading

# Load model
print("Loading ASL model...")
with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded!")

def speak(text):
    def run():
        try:
            tts = gTTS(text=text, lang='en')
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(tmp.name)
            tmp.close()
            playsound(tmp.name)
            os.unlink(tmp.name)
        except Exception as e:
            print(f"Speech error: {e}")
    threading.Thread(target=run, daemon=True).start()

# Setup MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("ASL Voice App started!")
print("Controls: SPACE = speak word | C = clear | Q = quit")

# App state
word = ""
letter_counter = 0
last_accepted_letter = ""
LETTER_DELAY = 25
CONFIDENCE_THRESHOLD = 0.30
recent_predictions = []
SMOOTH_FRAMES = 12
cooldown_counter = 0
COOLDOWN = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if cooldown_counter > 0:
        cooldown_counter -= 1

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Draw dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Raw coordinates — same as training data
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])

        proba = model.predict_proba([row])[0]
        max_confidence = np.max(proba)

        # Get top 2 predictions
        top2_idx = np.argsort(proba)[-2:][::-1]
        top1_letter = model.classes_[top2_idx[0]]
        top1_conf = proba[top2_idx[0]]
        top2_letter = model.classes_[top2_idx[1]]
        top2_conf = proba[top2_idx[1]]

        # If B is top but second choice is close, prefer second
        if top1_letter == "B" and top2_conf > 0.20:
            prediction = top2_letter
            max_confidence = top2_conf
        else:
            prediction = top1_letter
            max_confidence = top1_conf

        if max_confidence >= CONFIDENCE_THRESHOLD:
            recent_predictions.append(prediction)
            if len(recent_predictions) > SMOOTH_FRAMES:
                recent_predictions.pop(0)

            stable_letter = Counter(recent_predictions).most_common(1)[0][0]

            # Show sign, confidence and second guess
            cv2.putText(frame, f"Sign: {stable_letter} ({max_confidence*100:.0f}%)",
                        (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"2nd: {top2_letter} ({top2_conf*100:.0f}%)",
                        (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            if cooldown_counter == 0:
                if stable_letter == last_accepted_letter or last_accepted_letter == "":
                    letter_counter += 1
                else:
                    letter_counter = max(0, letter_counter - 3)
                last_accepted_letter = stable_letter

            # Progress bar
            progress = int((letter_counter / LETTER_DELAY) * 200)
            cv2.rectangle(frame, (30, 80), (230, 105), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, 80), (30 + min(progress, 200), 105),
                          (0, 255, 0), -1)
            cv2.putText(frame, "Hold sign steady...", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if letter_counter >= LETTER_DELAY and cooldown_counter == 0:
                if stable_letter not in ["NOTHING", "DEL", "SPACE"]:
                    word += stable_letter
                    print(f"✅ Added: {stable_letter} | Word: {word}")
                    speak(stable_letter)
                elif stable_letter == "SPACE":
                    word += " "
                    speak("space")
                elif stable_letter == "DEL":
                    word = word[:-1]
                    print(f"🗑️ Deleted | Word: {word}")

                letter_counter = 0
                last_accepted_letter = ""
                recent_predictions = []
                cooldown_counter = COOLDOWN

        else:
            cv2.putText(frame, f"Unclear ({max_confidence*100:.0f}%)",
                        (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            letter_counter = max(0, letter_counter - 2)

    else:
        cv2.putText(frame, "Show your hand!", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        letter_counter = max(0, letter_counter - 2)
        recent_predictions = []
        last_accepted_letter = ""

    # Word display
    cv2.rectangle(frame, (0, 380), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Word: {word}", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=speak word | C=clear | Q=quit", (20, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("ASL Voice App", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word = ""
        recent_predictions = []
        letter_counter = 0
        last_accepted_letter = ""
        cooldown_counter = 0
        print("🗑️ Cleared!")
    elif key == ord(' '):
        if word.strip():
            print(f"🔊 Speaking: {word.strip()}")
            speak(word.strip())

cap.release()
cv2.destroyAllWindows()
print("App closed!")