# connecting four layers of CNN
from keras.utils import np_utils
import numpy as np
import glob
import tensorflow as tf
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.utils import shuffle

import os
from keras.layers import Input, UpSampling2D, AveragePooling2D, Lambda
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, merge, BatchNormalization
from keras import backend as K

from keras.models import Model, load_model
from keras.applications import VGG16


class OMNIGLOT:
    def __init__(self):
        # set parameters
        self.IMG_ROW = 28
        self.IMG_COL = 28
        self.IMG_CHA = 1
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        
        if os.path.exists('./X_train.npy'):
            # assign to variable
            self.X_train = np.load('./X_train.npy')
            self.y_train = np.load('./y_train.npy')
            self.X_test  = np.load('./X_test.npy')
            self.y_test  = np.load('./y_test.npy')
            self.IMG_CLA = len(np.unique(self.y_train))
        else:
            X = []
            y = []
            character = {}
            counter = 0
            # loading training data
            for f in glob.glob('../images_background/*/*/*.png'):
                key = f.split('/')[2] + '_' + f.split('/')[3]
                if key not in character.keys():
                    character[key] = counter
                    counter += 1
                img = imread(f)
                img = resize(img, (28,28))
                X.append(img)
                y.append(character[key])
            X = np.array(X)
            y = np.array(y)
            X = X.reshape(-1,28,28,1)
            # split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # number of class
            self.IMG_CLA = len(character.keys())
            # 255 degree to [0,1]
            X_train = X_train.astype('float32') / 255.
            X_test = X_test.astype('float32') / 255.
            # input reshape 
            X_train = X_train.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
            X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
            # one hot encoding
            y_train = np_utils.to_categorical(y_train, self.IMG_CLA)
            y_test = np_utils.to_categorical(y_test, self.IMG_CLA)
            # assign to variable
            self.X_train = X_train
            self.y_train = y_train
            self.X_test  = X_test
            self.y_test  = y_test
            # save to file
            np.save('./X_train.npy', self.X_train)
            np.save('./y_train.npy', self.y_train)
            np.save('./X_test.npy', self.X_test)
            np.save('./y_test.npy', self.y_test)
      
        
class CNN:
    def build(self, input_shape, n_class):
        inputs = Input(shape=input_shape)
        # hidden layers
        x = Conv2D(64, (3,3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        #
        x = Flatten()(x)
        out = Dense(n_class, activation='softmax')(x)
        
        self.CNN = Model(inputs, out)
        optimizer = Adam()
        self.CNN.compile(loss="categorical_crossentropy",optimizer=optimizer)
        
        print(self.CNN.summary())
        
        
    def train(self, X, y, X_val, y_val, save_path='models/CNN_baseline'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit model
            self.CNN.fit(X, y, epochs = 50, batch_size = 128,\
                         shuffle = True, validation_data = (X_val, y_val))
        # save model
        self.CNN.save(save_path)
        
        
    def restore(self, path):
        self.CNN.load_weights(path)

        
        
if __name__ == '__main__':
    # init
    omni = OMNIGLOT()
    model = CNN()
    # build
    model.build(omni.IMG_SHAPE, omni.IMG_CLA)
    # train
    model.train(omni.X_train, omni.y_train, omni.X_test, omni.y_test)
    
    

