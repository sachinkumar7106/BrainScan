import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize  # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense # type: ignore
from keras.utils import to_categorical # type: ignore
from keras.callbacks import EarlyStopping # type: ignore

# Image directory path
image_directory = 'C:/Users/Sachin/OneDrive/Desktop/MINI PROJECT/BRAIN_SCAN_AI/data/'

# Load the images from both classes
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

input_size = 64

# Load and preprocess images of "no tumor"
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size, input_size))
            dataset.append(np.array(image))
            label.append(0)
        else:
            print(f"Warning: {image_directory + 'no/' + image_name} could not be loaded.")

# Load and preprocess images of "yes tumor"
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size, input_size))
            dataset.append(np.array(image))
            label.append(1)
        else:
            print(f"Warning: {image_directory + 'yes/' + image_name} could not be loaded.")

# Convert dataset and labels to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(input_size, input_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))  
model.add(Dense(2))  
model.add(Activation('softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stopping], shuffle=True)


model.save('BrainTumor10Epochscategorical.keras')
