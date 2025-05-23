import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# Configuration
dataset_path = 'C:/Users/Ambika/OneDrive/Desktop/SignLanguageInterpreter/dataset/asl_small'
img_size = (64, 64)
epochs = 50  # Increased from 30 to 50
batch_size = 32

# Load and preprocess data
print("🔄 Loading and processing images...")
labels = sorted(os.listdir(dataset_path))
data, target = [], []

for label in labels:
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        try:
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            data.append(img_array)
            target.append(label)
        except Exception as e:
            print(f"⚠️ Skipped image {img_name}: {e}")

if not data or not target:
    print("⚠️ No data or target labels found. Check dataset path and content.")
    exit()

data = np.array(data)
target = np.array(target)

print(f"Data shape: {data.shape}")
print(f"Target shape: {target.shape}")

encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)

print(f"Number of unique classes: {len(np.unique(target_encoded))}")
print(f"First few encoded labels: {target_encoded[:10]}")

target_categorical = to_categorical(target_encoded)

joblib.dump(encoder, 'label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(data, target_categorical, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('gesture_cnn_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

print("🚀 Training the model for 50 epochs...")
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          validation_data=(X_test, y_test),
          epochs=epochs, callbacks=[checkpoint])

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Final test accuracy: {acc * 100:.2f}%")

model.save('gesture_cnn_model_final.h5')
print("✅ Model saved as 'gesture_cnn_model_final.h5'")
