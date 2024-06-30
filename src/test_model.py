import cv2
import numpy as np

from PIL import Image
from keras.models import load_model

# ===========================================================
# Load model
# ===========================================================

model = load_model('../model.h5')

# ===========================================================
# Load images
# ===========================================================

image = cv2.imread('../data/pred/pred1.jpg')
image = Image.fromarray(image, 'RGB')
image = image.resize((64,64))
image = np.array(image)
image = np.expand_dims(image, axis = 0)

# ===========================================================
# Make predictions
# ===========================================================

y_pred = model.predict(image)

if y_pred > 0.5:
    print("There's a tumor")
else:
    print('No brain tumor')
