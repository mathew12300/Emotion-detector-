import os

train_dir = "C:/Users/jwmat/PycharmProjects/emotiondetectionmaster/data/train"

for emotion in os.listdir(train_dir):
    emotion_path = os.path.join(train_dir, emotion)
    if os.path.isdir(emotion_path):
        images = os.listdir(emotion_path)
        print(f"📂 {emotion}: {len(images)} files")
        for img in images[:5]:  # Show first 5 images
            print(f"  - {img}")
