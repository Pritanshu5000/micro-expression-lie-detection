import streamlit as st
import numpy as np
import joblib
import cv2
import mediapipe as mp
import time
from datetime import datetime
from tensorflow.keras.models import load_model
import speech_recognition as sr
import ascii_magic
import os

# Constants
MODEL_PATH = "cnn_lstm_lie_detection_model.keras"
MODEL_PATH1 = "xgboost_lie_detector_model.pkl"
SAVE_DIR = "Predictions"
FACE_COORD_SIZE = 478 * 3
PREDICTION_INTERVAL = 6
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
# model = load_model(MODEL_PATH)
model = joblib.load(MODEL_PATH1)

# FaceMesh & speech
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Streamlit App
st.title("🧠 Live Lie Detector")
run = st.checkbox("Start Webcam")
stframe = st.empty()
status_text = st.empty()

is_processing = False
last_detection_time = time.time()

cap = None

if run:
    cap = cv2.VideoCapture(0)
    status_text.success("Webcam started. Speak when face is detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                face_coords = []

                for lm in face_landmarks.landmark:
                    x, y, z = int(lm.x * w), int(lm.y * h), lm.z * 100
                    face_coords.extend([x, y, z])

                face_coords_np = np.array(face_coords).reshape(1, -1)

                if not is_processing and (time.time() - last_detection_time) > PREDICTION_INTERVAL:
                    frame_copy = frame.copy()
                    is_processing = True
                    last_detection_time = time.time()

                    try:
                        with mic as source:
                            st.info("🎙 Listening for speech...")
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                            try:
                                text = recognizer.recognize_google(audio)
                                st.write(f"🗣 You said: {text}")

                                if np.all(face_coords_np == 0):
                                    st.warning("⚠️ Empty face data, skipping prediction.")
                                else:
                                    # Normalize the face data for model prediction
                                    face_coords_np = face_coords_np / np.linalg.norm(face_coords_np)

                                    # Resize the frame to 64x64 for prediction
                                    frame_resized = cv2.resize(frame_copy, (64, 64))
                                    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
                                    frame_resized = frame_resized / 255.0  # Normalize the image to [0, 1]

                                    # Predict lie or truth using the CNN-LSTM model
                                    prediction = model.predict(frame_resized)[0][0]
                                    label = "Lie" if prediction > 0.5 else "Truth"

                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                    save_path = os.path.join(SAVE_DIR, filename)

                                    cv2.putText(frame_copy, f"Prediction: {label}", (20, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                                (0, 255, 0) if label == "Truth" else (0, 0, 255), 3)

                                    cv2.putText(frame_copy, f"Time: {timestamp}", (20, 90),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                                    cv2.imwrite(save_path, frame_copy)
                                    st.success(f"📸 Saved at: {save_path}")

                                    # Show ASCII art of the saved frame
                                    output = ascii_magic.from_image_file(save_path, columns=70, char="#")
                                    ascii_magic.to_terminal(output)

                            except sr.UnknownValueError:
                                st.warning("😕 Could not understand speech.")
                            except sr.RequestError as e:
                                st.error(f"❌ Speech recognition error: {e}")
                    except sr.WaitTimeoutError:
                        st.warning("⌛ Timeout: No speech detected.")
                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")
                    finally:
                        is_processing = False

                cv2.putText(frame, "Face detected. Speak...", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert frame to RGB before displaying it in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
else:
    if cap:
        cap.release()
        st.warning("Webcam stopped.")
