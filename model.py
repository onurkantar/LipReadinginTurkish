import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, TimeDistributed
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, Embedding
from keras.regularizers import l2
from keras import optimizers
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D
from keras.callbacks import ReduceLROnPlateau
import keras
from keras_video import VideoFrameGenerator

import enviroment as env

classes = env.class_names

SIZE = (256, 256)
CHANNELS = 1
NBFRAME = 30
BS = 1

glob_pattern='ignoreFolder/{classname}/*.mp4'
pred_pattern='predict/8.mp4'

def generate_predict_data():
    retVal = VideoFrameGenerator(
    is_training = False,
    classes=classes, 
    glob_pattern=pred_pattern,
    nb_frames=NBFRAME,
    shuffle=False, #TODO burayı incele
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    use_frame_cache=False)

    return retVal

def generate_data():
    data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

    train = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.42, 
    shuffle=False, #TODO burayı incele
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=False)

    valid = train.get_validation_generator()

    return valid,train

def create_model():
    INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
    model = action_model(INSHAPE, len(classes))
    optimizer = optimizers.Adam(0.00000001)
    model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc'])
    return model

def train_model(my_model,train,valid):

    EPOCHS=1
    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside
    my_callbacks = [
    ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),]
    
    return my_model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=my_callbacks
    )

def build_convnet(shape=(256, 256, 3)):
    momentum = .9
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def action_model(shape=(40, 256, 256, 3), nbout=len(env.class_names)):
        # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])
    
    # then create our final model
    model = Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(LSTM(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

def build_mobilenet(shape=(256, 256, 3), nbout=len(env.class_names)):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')
    # Keep 9 layers to train﻿﻿
    trainable = 9
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = GlobalMaxPool2D()
    return keras.Sequential([model, output])