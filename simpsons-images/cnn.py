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

# Plotting
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
from keras.optimizers import rmsprop


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
for path in paths:
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
data = np.array(data)
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

# isolate training, validation and test indecies

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
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer= rmsprop(lr=0.00001, decay=1e-6),
                      metrics=['accuracy'])
        
        return model

"Fitting image purtubation generator..."
# construct the image generator for data augmentation
img_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

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
history = model.fit_generator(img_gen.flow(X_train, encode_onehot(y_train), batch_size=32),
        validation_data=(X_dev, encode_onehot(y_dev)), steps_per_epoch=len(X_train),
        epochs=2, verbose=1, callbacks = [checkpointer])

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



