# 🧠 Sign Language Gesture Recognition Using CNN & MediaPipe

This project is a **real-time sign language recognition system** that translates American Sign Language (ASL) gestures into text and speech using computer vision and deep learning. It leverages **MediaPipe** for hand tracking and a custom-trained **Convolutional Neural Network (CNN)** for gesture classification.

> 🔗 Ideal for: Accessibility applications, speech/hearing impairment support, and human-computer interaction.

---

## 📌 Key Features

- 🔍 **Real-time Webcam-based Recognition**  
  No need for gloves or sensors — just a camera.

- 🧠 **CNN Model Trained on ASL Dataset**  
  Trained using the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

- 🎯 **Accurate Gesture Detection with MediaPipe**  
  Uses MediaPipe Hands for precise hand landmark extraction.

- 💬 **Dual Mode Output (Text & Speech)**  
  Displays recognized gestures and optionally speaks them aloud using `pyttsx3`.

- 💻 **Simple GUI Interface**  
  Built with Python’s Tkinter for user-friendly operation.

---

## 🔧 Tech Stack

| Tool / Library       | Purpose                                |
|----------------------|----------------------------------------|
| Python 3.9           | Core programming language              |
| OpenCV               | Webcam and image processing            |
| MediaPipe            | Hand landmark detection                |
| TensorFlow/Keras     | Model training and inference           |
| NumPy                | Data manipulation                      |
| pyttsx3              | Text-to-Speech (TTS) engine            |
| Tkinter              | GUI development                        |

---

## 📁 Folder Structure

```
SignLanguageInterpreter/
│
├── gesture_interpreter.py      # Main interpreter script
├── train_model.py              # CNN model training
├── reduce_dataset.py           # Optional dataset balancing
├── gesture_cnn_model.h5        # Trained CNN model
├── dataset/                    # ASL alphabet images
├── label_encoder.pkl           # Label mapping for gestures
├── models/                     # Folder for saving models
├── README.md                   # This file
└── venv/                       # Python virtual environment
```

---

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SignLanguageInterpreter.git
   cd SignLanguageInterpreter
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the interpreter:
   ```bash
   python gesture_interpreter.py
   ```

---

## 📌 Dataset Details

- **Source**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Classes**: A-Z letters, space, delete, nothing
- **Image Size**: 64x64 grayscale, preprocessed from cropped hand regions

---

## 🎓 Use Cases

- Assisting the **speech or hearing impaired**
- Integration into **education systems**
- Prototype for **gesture-based interfaces**

---

## 🙌 Contributors

- Ambika (Final Year B.Tech Project)
- Guidance: [Add Faculty/Guide Name if needed]

---

## 📃 License

This project is open-source and available under the **MIT License**.
