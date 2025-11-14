import os
import cv2
import matplotlib.pyplot as plt
import random

# Path to dataset
data_dir = "data/raw/GoodImgBmp"

# Pick 9 random folders (classes)
sample_folders = random.sample(os.listdir(data_dir), 9)

plt.figure(figsize=(10, 10))

for i, folder in enumerate(sample_folders):
    folder_path = os.path.join(data_dir, folder)
    img_files = os.listdir(folder_path)
    
    # pick random image from this folder
    img_file = random.choice(img_files)
    img_path = os.path.join(folder_path, img_file)
    
    # read in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    plt.subplot(3, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(folder)   # shows class folder name
    plt.axis('off')

plt.tight_layout()
plt.show()
