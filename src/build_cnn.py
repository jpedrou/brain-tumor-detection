import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense


plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')

# ==================================================
# Load data
# ==================================================

X = np.load("../data/processed/dataset.npy")
labels = np.load("../data/processed/labels.npy")


# ==================================================
# Split into train and test
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.1, random_state=42
)

# ==================================================
# Normalize data
# ==================================================

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# ==================================================
# Build CNN (Convolucional Neural Network)
# ==================================================

nn = Sequential()

nn.add(Conv2D(32, (3, 3), strides=1, input_shape=(64, 64, 3)))
nn.add(Activation("relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
nn.add(Activation("relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Conv2D(64, (3, 3), strides=1, kernel_initializer="he_uniform"))
nn.add(Activation("relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Flatten())

nn.add(Dense(64))
nn.add(Activation("relu"))

nn.add(Dropout(0.5))

nn.add(Dense(1))
nn.add(Activation("sigmoid"))

nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

nn.summary()

# ==================================================
# Training CNN
# ==================================================

hist = nn.fit(
    X_train,
    y_train,
    batch_size=15,
    verbose=1,
    epochs=10,
    validation_data=(X_test, y_test),
    shuffle = True
)

# ==================================================
# Evaluate performance
# ==================================================

print('Train set accuracy:', hist.history['accuracy'][-1])
print('Validation set accuracy:',hist.history['val_accuracy'][-1])

#Loss plot
fig = plt.figure()
plt.plot(hist.history['loss'], color = 'orange', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'green', label = 'val_loss')
plt.title('Loss')
plt.legend(loc = 'upper right')
plt.show()

fig.savefig('../reports/loss.jpg')

#Accuracy plot
fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'orange', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'green', label = 'val_accuracy')
plt.title('Accuracy')
plt.legend(loc = 'upper right')
plt.show()

fig.savefig('../reports/accuracy.jpg')


# ==================================================
# Save model
# ==================================================
nn.save('../model.h5')

