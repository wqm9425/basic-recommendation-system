# data generator
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
from keras.utils import Sequence

from keras.models import Model, load_model
from keras.applications import VGG16


class OMNIGLOT_train(Sequence):
    def __init__(self):
        # set parameters
        self.IMG_ROW = 64
        self.IMG_COL = 64
        self.IMG_CHA = 3
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        self.batch_size = 16
        
        if os.path.exists('./X_train_64_'+str(self.IMG_CHA)+'.npy'):
            if self.IMG_CHA == 1:
                # assign to variable
                self.X_train = np.load('./X_train_64_1.npy')
                self.y_train = np.load('./y_train_64_1.npy')
            else:
                # assign to variable
                self.X_train = np.load('./X_train_64_3.npy')
                self.y_train = np.load('./y_train_64_3.npy')
            # get same and diff
            _, self.X_train, self.y_train, self.X_same, self.X_diff = self.reform(self.X_train, self.y_train)
        else:
            X = []
            y = []
            character = {}
            counter = 0
            # loading training data
            for f in glob.glob('../Train/*/*/*/*.png'):
                key = f.split('/')[2] + '_' + f.split('/')[3]
                if key not in character.keys():
                    character[key] = counter
                    counter += 1
                img = imread(f)
                img = resize(img, (self.IMG_ROW,self.IMG_COL))
                X.append(img)
                y.append(character[key])
            X = np.array(X)
            y = np.array(y)
            X = X.reshape(-1,self.IMG_ROW,self.IMG_COL,1)
            if self.IMG_CHA == 3:
                D = np.concatenate((X,X), axis=3)
                X = np.concatenate((D,X), axis=3)
            # number of class
            self.IMG_CLA = len(character.keys())
            # 255 degree to [0,1]
            X = X.astype('float32') / 255.
            # input reshape 
            X = X.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
            # assign to variable
            self.X_train = X
            self.y_train = y
            # save to file
            np.save('./X_train_64_'+str(self.IMG_CHA)+'.npy', self.X_train)
            np.save('./y_train_64_'+str(self.IMG_CHA)+'.npy', self.y_train)
            # get same and diff
            _, self.X_train, self.y_train, self.X_same, self.X_diff = self.reform(self.X_train, self.y_train)
            
    def __len__(self):
        return int(np.floor(len(self.X_train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_l = np.concatenate((self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size], 
                                  self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size]), axis=0)
        batch_r = np.concatenate((self.X_diff[idx * self.batch_size:(idx + 1) * self.batch_size], 
                                  self.X_same[idx * self.batch_size:(idx + 1) * self.batch_size]), axis=0)
        label = np.zeros((self.batch_size * 2))
        label[self.batch_size:] = 1
        
        return [np.array(batch_l), np.array(batch_r)], np.array(label)
    
    def on_epoch_end(self):
        _, self.X_train, self.y_train, self.X_same, self.X_diff = self.reform(self.X_train, self.y_train)
        return None
    
    def reform(self, X, y):
            counter = 0
            X, y = shuffle(X, y, random_state=0)
            X_same, X_diff = [], []
            for i in y:
                # same
                index = np.argwhere(y == i).flatten()
                loc = np.random.choice(len(index), 1)
                X_same.append(X[index[loc]].reshape(self.IMG_ROW,self.IMG_COL,self.IMG_CHA))
                # diff
                index = np.argwhere(y != i).flatten()
                loc = np.random.choice(len(index), 1)
                X_diff.append(X[index[loc]].reshape(self.IMG_ROW,self.IMG_COL,self.IMG_CHA))
            X_same = np.array(X_same)
            X_diff = np.array(X_diff)
            return counter, X, y, X_same, X_diff
        
        
class OMNIGLOT_test(Sequence):
    def __init__(self):
        # set parameters
        self.IMG_ROW = 64
        self.IMG_COL = 64
        self.IMG_CHA = 3
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        self.batch_size = 16
        
        if os.path.exists('./X_test_64_'+str(self.IMG_CHA)+'.npy'):
            if self.IMG_CHA == 1:
                # assign to variable
                self.X_test = np.load('./X_test_64_1.npy')
                self.y_test = np.load('./y_test_64_1.npy')
            else:
                # assign to variable
                self.X_test = np.load('./X_test_64_3.npy')
                self.y_test = np.load('./y_test_64_3.npy')
            # get same and diff
            _, self.X_test, self.y_test, self.test_same, self.test_diff = self.reform(self.X_test, self.y_test)
        else:
            X_test = []
            y_test = []
            character = {}
            counter = 0
            for f in glob.glob('../Test/*/*/*/*.png'):
                key = f.split('/')[2] + '_' + f.split('/')[3]
                if key not in character.keys():
                    character[key] = counter
                    counter += 1
                img = imread(f)
                img = resize(img, (self.IMG_ROW,self.IMG_COL))
                X_test.append(img)
                y_test.append(character[key])
            X_test= np.array(X_test)
            y_test = np.array(y_test)
            X_test = X_test.reshape(-1,self.IMG_ROW,self.IMG_COL,1)
            if self.IMG_CHA == 3:
                D = np.concatenate((X_test,X_test), axis=3)
                X_test = np.concatenate((D,X_test), axis=3)
            # number of class
            self.IMG_CLA = len(character.keys())
            # 255 degree to [0,1]
            X_test = X_test.astype('float32') / 255.
            # input reshape 
            X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
            # assign to variable
            self.X_test  = X_test
            self.y_test  = y_test
            # save to file
            np.save('./X_test_64_'+str(self.IMG_CHA)+'.npy', self.X_test)
            np.save('./y_test_64_'+str(self.IMG_CHA)+'.npy', self.y_test)
            # get same and diff
            _, self.X_test, self.y_test, self.test_same, self.test_diff = self.reform(self.X_test, self.y_test)
            
    def __len__(self):
        return int(np.floor(len(self.X_test) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_l = np.concatenate((self.X_test[idx * self.batch_size:(idx + 1) * self.batch_size], 
                                  self.X_test[idx * self.batch_size:(idx + 1) * self.batch_size]), axis=0)
        batch_r = np.concatenate((self.test_diff[idx * self.batch_size:(idx + 1) * self.batch_size], 
                                  self.test_same[idx * self.batch_size:(idx + 1) * self.batch_size]), axis=0)
        label = np.zeros((self.batch_size * 2))
        label[self.batch_size:] = 1
        
        return [np.array(batch_l), np.array(batch_r)], np.array(label)
    
    def on_epoch_end(self):
        _, self.X_test, self.y_test, self.test_same, self.test_diff = self.reform(self.X_test, self.y_test)
        return None
    
    def reform(self, X, y):
            counter = 0
            X, y = shuffle(X, y, random_state=0)
            X_same, X_diff = [], []
            for i in y:
                # same
                index = np.argwhere(y == i).flatten()
                loc = np.random.choice(len(index), 1)
                X_same.append(X[index[loc]].reshape(self.IMG_ROW,self.IMG_COL,self.IMG_CHA))
                # diff
                index = np.argwhere(y != i).flatten()
                loc = np.random.choice(len(index), 1)
                X_diff.append(X[index[loc]].reshape(self.IMG_ROW,self.IMG_COL,self.IMG_CHA))
            X_same = np.array(X_same)
            X_diff = np.array(X_diff)
            return counter, X, y, X_same, X_diff