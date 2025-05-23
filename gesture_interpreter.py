import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import os
import tkinter as tk
from tkinter import ttk

# Load trained model
model = load_model("models/sign_language_model.h5")
labels = sorted(os.listdir("gesture_data"))  # Assumes training folders are named as labels

# Text-to-speech setup
engine = pyttsx3.init()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# GUI Setup
root = tk.Tk()
root.title("Sign Language Interpreter")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

label = ttk.Label(frame, text="Recognized Text:", font=("Arial", 16))
label.grid(row=0, column=0, sticky="w")

output_text = tk.StringVar()
output_display = ttk.Label(frame, textvariable=output_text, font=("Arial", 24))
output_display.grid(row=1, column=0, sticky="w")

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mode flag
typing_mode = False

def toggle_mode():
    global typing_mode
    typing_mode = not typing_mode
    mode_btn.config(text="Typing Mode: ON" if typing_mode else "Typing Mode: OFF")

mode_btn = ttk.Button(frame, text="Typing Mode: OFF", command=toggle_mode)
mode_btn.grid(row=2, column=0, sticky="w")

# Start webcam
cap = cv2.VideoCapture(0)

recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    display_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around hand
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))

            # Add padding
            padding = 40
            x1 = max(x1 - padding, 0)
            y1 = max(y1 - padding, 0)
            x2 = min(x2 + padding, w)
            y2 = min(y2 + padding, h)

            # Crop hand
            hand_img = frame[y1:y2, x1:x2]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # Resize to 64x64 with padding to keep square aspect
            desired_size = 64
            old_size = hand_img.shape[:2]
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            resized_img = cv2.resize(hand_img, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            hand_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)

            # Prepare for prediction
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=-1)  # grayscale channel
            hand_img = np.expand_dims(hand_img, axis=0)   # batch size

            prediction = model.predict(hand_img)[0]
            confidence = np.max(prediction)
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]

            if confidence > 0.80:
                cv2.putText(display_frame, f"{predicted_label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                if typing_mode:
                    if predicted_label == "space":
                        recognized_text += " "
                    elif predicted_label == "nothing":
                        pass
                    else:
                        recognized_text += predicted_label
                    output_text.set(recognized_text)

                else:
                    if predicted_label not in ["space", "nothing"]:
                        output_text.set(predicted_label)
                        speak(predicted_label)

    # Show webcam
    cv2.imshow("Sign Language Recognition", display_frame)
    root.update()

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
root.destroy()