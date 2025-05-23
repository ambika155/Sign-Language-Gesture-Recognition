import os
import shutil
import random
from PIL import Image

# Absolute paths to the dataset directories
source_dir = 'C:/Users/Ambika/OneDrive/Desktop/SignLanguageInterpreter/dataset/asl_alphabet_train'
target_dir = 'C:/Users/Ambika/OneDrive/Desktop/SignLanguageInterpreter/dataset/asl_small'
images_per_class = 1000  # Set to 1000 images per class for better training

# Valid labels (A-Z, space, del, nothing)
valid_classes = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
os.makedirs(target_dir, exist_ok=True)

# Iterate through each class (label) in the source directory
for label in os.listdir(source_dir):
    if label not in valid_classes:
        continue

    src = os.path.join(source_dir, label)
    dst = os.path.join(target_dir, label)
    os.makedirs(dst, exist_ok=True)

    img_list = os.listdir(src)
    random.shuffle(img_list)

    count = 0
    for img_name in img_list:
        if count >= images_per_class:
            break

        src_img = os.path.join(src, img_name)
        dst_img = os.path.join(dst, img_name)

        try:
            with Image.open(src_img) as im:
                im.verify()  # Check for corrupted images
            shutil.copy(src_img, dst_img)
            count += 1
        except Exception as e:
            print(f"⚠️ Skipped image {img_name} due to error: {e}")

print(f"✅ Reduced dataset created in '{target_dir}' with {images_per_class} images per class.")
