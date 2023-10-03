import os 
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import *

from tensorflow.python.keras.utils.generic_utils import get_custom_objects

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Settings
INPUT_SHAPE = (512, 512, 1)
OUTPUT_SIZE = 16


# Create Model
def landmark_cnn(input_shape=INPUT_SHAPE, output_size=OUTPUT_SIZE):

    img_input = Input(shape=input_shape)
    

    x = Conv2D(16, (3,3), strides=(1,1), name='Conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv1')(x)
    
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool1')(x)


    x = Conv2D(32, (3,3), strides=(1,1), name='Conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv2')(x)
    
    x = Conv2D(64, (3,3), strides=(1,1), name='Conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv3')(x)
    
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool2')(x)


    x = Conv2D(32, (3,3), strides=(1,1), name='Conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv4')(x)
    
    x = Conv2D(32, (3,3), strides=(1,1), name='Conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv5')(x)
    
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool3')(x)

    x = Conv2D(64, (3,3), strides=(1,1), name='Conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Relu_conv6')(x)
    
    x = Dropout(0.2)(x)
    
    x = Flatten(name='Flatten')(x)
    x = Dense(128, activation='relu', name='FC1')(x)
    x = Dense(output_size, activation=None, name='Predictions')(x)
    
    
    model = Model([img_input], x, name='Landmark_model')
    
    return model

# Load Data
image_array = np.load('../dataset/image.npy')
label_array = np.load('../dataset/label.npy').reshape(-1, 16)
test_image_array = np.load('../dataset/test_image.npy')
test_label_array = np.load('../dataset/test_label.npy').reshape(-1, 16)

# Normalization
X_train = image_array[:180] / 255
y_train = label_array[:180] / 512

X_valid = image_array[180:240] / 255
y_valid = label_array[180:240] / 512

X_test = test_image_array[:] / 255
y_test = test_label_array[:] / 512


# Model
model = landmark_cnn()
model.compile( loss=tf.keras.losses.mean_squared_error , optimizer=tf.keras.optimizers.Adam( lr=0.0001 ) , metrics=[ 'mse' ] )


# Call Back
monitor = 'val_loss'

reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, min_lr=0.0000001,verbose=1)
earlystopper = EarlyStopping(monitor=monitor, patience=50, verbose=1)
model_checkpoint = ModelCheckpoint(filepath = '../result/model_save/landmark_model_1.h5', verbose=1, save_best_only=True)

callbacks_list = [reduce_lr, model_checkpoint, earlystopper]


# Train
history = model.fit(X_train, y_train, batch_size=20, epochs=1000, shuffle=True, verbose=1, 
                    validation_data=(X_test, y_test), callbacks=callbacks_list)


# Predict
pred = model.predict(X_test)

# Save Prediction
np.save('../result/predict/predict.npy', pred)

# Check file
pred[0]
y_test[0]
