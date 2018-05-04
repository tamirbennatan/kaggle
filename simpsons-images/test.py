"""
Script for running Convolutional neural network model


"""

# navigating the file system
import os
import sys
import glob
import argparse

# data manipulation
import numpy as np

import matplotlib.pyplot as plt


# Tools for setting up machine learning experiments
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


# Keras stuff
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import save_model


# image processing using Pillow
from PIL import Image

import pickle as pkl
import pdb


"""
Process arguments
"""
args = sys.argv
try:
	filepath = sys.argv[1]
except:
	print("Input a file path to save the models to.")
	sys.exit()


# store paths to all images
paths = glob.glob('data/*/*')

# set the sizes for the x and y axes of image
X_SIZE, Y_SIZE = 128, 128

# create an array to store a list of all characters
all_characters = np.ndarray(shape = (0,))
for path in paths:
    # isolate the character's name from the path
    character = path.split('/')[-2]
    # add said character name to the target vector
    all_characters = np.append(all_characters, character)

# Create a seperate list for the top 20 characters
counts = np.unique(all_characters,return_counts = True)
char_count= sorted(zip(counts[0], counts[1]), key = lambda x: x[1], reverse = True)
top_characters = [character[0] for character in char_count[0:20]]

"""
Read all the images, and resize them to 128x128 sized RGB images. 
Aggreage them in a list called `data`
"""
# aggretate the images, and the labels
print("Loading pictures to memory...")
data, labels = [], []
np.random.seed(1)
for path in np.random.choice(paths, size=300):
    # isolate the character's name from the path
    character = path.split('/')[-2]
    if character in top_characters:
        # Open the current image
        img = Image.open(path)
        # reshape the image
        img = img.resize((X_SIZE, Y_SIZE))
        # flatten the image to a vector, and add to data 
        img = img_to_array(img)
        data.append(img)
        # add the character label to the label list
        labels.append(character)

# convert data and labels to numpy arrays, and re-scale to speed up training
data = np.array(data, dtype="float")
labels = np.array(labels)


print("Done.")
print

# train an encoder to convert labels to integer labels, one-hot encodings, and back
encoder = LabelEncoder().fit(labels)

def encode_onehot(lab):
    return to_categorical(encoder.transform(lab))

def decode_onehot(lab):
    return encoder.inverse_transform(np.argmax(lab, axis = 1))


'''
split data into train, developement (validation), and test splits
'''
print("Splitting data into train/dev/test splits...")

np.random.seed(1)
# create a random permutation
perm = np.random.permutation(len(labels))
# isolate training, validation and test indecies
split1, split2 = int(np.floor(.6*(len(labels)))), int(np.floor(.8*(len(labels))))


# isolate the test set, and keep the rest for training and validation
X_train_and_dev, X_test, y_train_and_dev, y_test = train_test_split(data, \
                 labels, test_size=0.2, random_state=1)
# Further split the training/dev sets into seperate training and dev sets. 
X_train, X_dev, y_train, y_dev = train_test_split(X_train_and_dev, \
                 y_train_and_dev, test_size=0.25, random_state=1)


print("Done.")
print

class LeNet:
    @staticmethod
    def build(width, height, depth, classes = 20):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # add first convolutional and max pooling layers
        model.add(Conv2D(32, (5,5),input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # add second convolutional and max pooling layers
        model.add(Conv2D(32, (5,5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # add third convolutional and max pooling layers
        model.add(Conv2D(64, (5,5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return model

"Fitting image purtubation generator..."
# construct the image generator for data augmentation
img_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# fit the generator
img_gen.fit(X_train)
print("Done.")
print


# get a model
model = LeNet.build(width = X_SIZE, height = Y_SIZE, depth = 3)

# create an Early Stopping callback
# early_stopping = EarlyStopping(patience=10)

# create a callback to save the model that performs best ont the validation set
if not os.path.isdir("models"):
    os.makedirs("models")
checkpointer = ModelCheckpoint(filepath="models/" + filepath + ".hdf5", verbose=1, save_best_only=True)

# fit the model, and save the output
history = model.fit_generator(img_gen.flow(X_train, encode_onehot(y_train), batch_size=2),
        validation_data=(X_dev, encode_onehot(y_dev)), steps_per_epoch=len(X_train),
        epochs=1, verbose=1, callbacks = [checkpointer])

# make plots of the history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# save plots
plt.savefig("models/" + filepath + ".png")

