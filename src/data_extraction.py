import cv2
import os
import numpy as np
from PIL import Image

image_dir = "../data/raw/"
dataset = []
label = []

# ============================================================
# Load data
# ============================================================

no_tumor_images = os.listdir(image_dir + "no/")
yes_tumor_images = os.listdir(image_dir + "yes/")

# ============================================================
# Resize images
# ============================================================

for i, img in enumerate(no_tumor_images):
    if img.split(".")[1] == "jpg":
        image = cv2.imread(image_dir + "no/" + img)
        image = Image.fromarray(image, "RGB")
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)

for i, img in enumerate(yes_tumor_images):
    if img.split(".")[1] == "jpg":
        image = cv2.imread(image_dir + "yes/" + img)
        image = Image.fromarray(image, "RGB")
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)

len(dataset)
np.unique(label, return_counts=True)

dataset = np.array(dataset)
label = np.array(label)

# ============================================================
# Export data
# ============================================================
np.save('../data/processed/dataset.npy', dataset)
np.save('../data/processed/labels.npy', label)