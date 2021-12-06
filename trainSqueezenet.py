import numpy as np
import pandas as pd
import cv2 as cv
import os
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot as mplplot

CATEGORY_MAP = {
    "call_me": 0,
    "fingers_crossed": 1,
    "okay": 2,
    "paper": 3,
    "peace": 4,
    "rock": 5,
    "rock_on": 6,
    "scissor": 7,
    "thumbs": 8,
    "up": 9
}

training_img = 'HandGesture/images'

def label_mapper(val):
    return CATEGORY_MAP[val]

def def_model_param():
    CATEGORIES = len(CATEGORY_MAP)
    base_model = Sequential()
    base_model.add(SqueezeNet(input_shape=(225, 225, 3), include_top=False))
    base_model.add(Dropout(0.5))
    base_model.add(Convolution2D(CATEGORIES, (1, 1), padding='valid'))
    base_model.add(Activation('relu'))
    base_model.add(GlobalAveragePooling2D())
    base_model.add(Activation('softmax'))
    base_model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return base_model

def train_model():
    # DATASET PROCESSING
    #input_data = []
    X = [] #images
    Y = [] # labels
    for sub_folder_name in os.listdir(training_img):
        path = os.path.join(training_img, sub_folder_name)
        for fileName in os.listdir(path):
            if fileName.endswith(".jpg"):
                img = cv.imread(os.path.join(path, fileName))
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # we dont need to change colours for this dataset I think
                img = cv.resize(img, (225, 225))
                #'input_data' stores the input image array and its corresponding label or category name
                #input_data.append([img, sub_folder_name])
                X.append(img)
                Y.append(label_mapper(sub_folder_name))
    xlen = len(X)
    X = np.array(X, dtype="uint8")
    X = X.reshape(len(xlen), 120, 320, 1) # Needed to reshape so CNN knows it's different images
    Y = np.array(Y)
    #img_data, labels = zip(*input_data)
    #labels = list(map(label_mapper, labels))
    #labels = np_utils.to_categorical(labels)

    # CREATE TRAINING AND VALIDATION SETS
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=.33, random_state=42)

    # NEURAL NETWORK
    model = def_model_param()
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    model.fit(X_train, y_train, epochs=15) #batch_size=64, validation_data=(X_test, y_test)) 
    model.save("gesture-model.h5")

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy: {:2.2f}%'.format(test_acc*100))

    predictions = model.predict(X_test) # Make predictions towards the test set
    y_pred = np.argmax(predictions, axis=1) # Transform predictions into 1-D array with label number

    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                                        columns=["Predicted call_me", "Predicted fingers_crossed", "Predicted okay", 
                                        "Predicted paper", "Predicted peace", "Predicted rock", "Predicted rock_on", 
                                        "Predicted scissor", "Predicted thumbs", "Predicted up"],
                                        index=["Actual call_me", "Actual fingers_crossed", "Actual okay", "Actual paper", 
                                        "Actual peace", "Actual rock", "Actual rock_on", "Actual scissor", "Actual thumbs", 
                                        "Actual up"])
    print(cm.head())
    print(cm.tail())
    cm.plot()
    mplplot.show()

train_model()
