import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

# Convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# Define dataset directory
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Define emotion categories
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Create folders
os.makedirs(data_dir, exist_ok=True)
for outer_name in ['train', 'test']:
    outer_path = os.path.join(data_dir, outer_name)
    os.makedirs(outer_path, exist_ok=True)
    for inner_name in emotions:
        os.makedirs(os.path.join(outer_path, inner_name), exist_ok=True)

# Initialize counters
counters = {emotion: 0 for emotion in emotions}
counters_test = {emotion: 0 for emotion in emotions}

# Load dataset
csv_path = './fer2013.csv'  # Update this if needed
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)
mat = np.zeros((48,48), dtype=np.uint8)

print("Saving images...")

# Process dataset
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()

    # Convert pixel values to 48x48 image
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    emotion_index = df['emotion'][i]
    emotion_name = emotions[emotion_index]

    # Train dataset
    if i < 28709:
        img_path = os.path.join(train_dir, emotion_name, f"im{counters[emotion_name]}.png")
        counters[emotion_name] += 1
    # Test dataset
    else:
        img_path = os.path.join(test_dir, emotion_name, f"im{counters_test[emotion_name]}.png")
        counters_test[emotion_name] += 1

    img.save(img_path)

print("✅ Image conversion complete!")
