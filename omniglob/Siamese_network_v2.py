from keras.utils import np_utils
import numpy as np
import glob
import tensorflow as tf
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import pairwise_distances

import os
from keras.layers import Input, UpSampling2D, AveragePooling2D, Lambda
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, merge, BatchNormalization
from keras import backend as K
from keras.utils import Sequence

from keras.models import Model, load_model
from keras.applications import VGG16
from data import OMNIGLOT_train, OMNIGLOT_test


        
class siamese:
    def build(self, input_shape):
        inputs = Input(shape=input_shape)
        # hidden layers
        x = Conv2D(64, (3,3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        #
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        #
        out = Flatten()(x)
        self.CNN = Model(inputs, out)
        
        left_input = Input(shape=input_shape)
        right_input = Input(shape=input_shape)
        #encode each of the two inputs into a vector
        encoded_l = self.CNN(left_input)
        encoded_r = self.CNN(right_input)
        #merge two encoded inputs with the l1 distance
        L1_distance = lambda x: K.abs(x[0]-x[1])
        distance = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1,activation='sigmoid')(distance)
        self.siamese_net = Model([left_input,right_input], prediction)
        # compile
        optimizer = Adam(lr=1e-4)
        self.siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
        
        print(self.siamese_net.summary())
        
        
    def train(self, train, test, save_path='models/siamese_v2'):
        self.siamese_net.fit_generator(generator=train, epochs=10, validation_data=test)
#         def reform(X, y):
#             print('Start prepare data')
#             counter = 0
#             X, y = shuffle(X, y, random_state=0)
#             X_same, X_diff = [], []
#             for i in y:
#                 # same
#                 index = np.argwhere(y == i).flatten()
#                 loc = np.random.choice(len(index), 1)
#                 X_same.append(X[index[loc]].reshape(105,105,1))
#                 # diff
#                 index = np.argwhere(y != i).flatten()
#                 loc = np.random.choice(len(index), 1)
#                 X_diff.append(X[index[loc]].reshape(105,105,1))
#             X_same = np.array(X_same)
#             X_diff = np.array(X_diff)
#             print('Data prepared')
#             return counter, X, y, X_same, X_diff
#         if os.path.exists(save_path):
#             self.restore(save_path)
#         else:
#             counter, X, y, X_same, X_diff = reform(X, y)
#             _, X_test, y_test, test_same, test_diff = reform(X_test, y_test)
#             for step in range(3000):
#                 print('Running global step %d' % step)
#                 # sample batch
#                 if X.shape[0] - counter < 128:
#                     counter, X, y, X_same, X_diff = reform(X, y)
#                 batch_l = np.concatenate((X[counter:counter+64], X[counter:counter+64]), axis=0)
#                 batch_r = np.concatenate((X_diff[counter:counter+64], X_same[counter:counter+64]), axis=0)
#                 counter += 64
#                 # create label
#                 label = np.zeros((128))
#                 label[64:] = 1
#                 # shuffle
#                 batch_l, batch_r, label = shuffle(batch_l, batch_r, label, random_state=0)
#                 # train on batch
#                 self.siamese_net.train_on_batch([batch_l, batch_r], label)
#                 # evaluate 
#                 if step%100 == 0:
#                     print('Running test')
#                     # sample test batch
#                     locs = np.random.choice(X_test.shape[0], 64)
#                     test_l = np.concatenate((X_test[locs], X_test[locs]), axis=0)
#                     test_r = np.concatenate((test_diff[locs], test_same[locs]), axis=0)
#                     # predict
#                     preds = self.siamese_net.predict([test_l, test_r])
#                     y_pred = [1 if i > 0.5 else 0 for i in preds]
#                     y_pred = np.array(y_pred)
#                     # create label
#                     test_label = np.zeros((128))
#                     test_label[64:] = 1
#                     acc = len(np.argwhere(y_pred == test_label).flatten()) / 128.0
#                     print('test accuracy is %f' % acc)
#                     # save temp model
#                     if step%500 == 0:
#                         self.siamese_net.save(save_path + '_tmp_' + str(step))
#         # save model    
#         self.siamese_net.save(save_path)
        
        
    def restore(self, path):
        self.siamese_net.load_weights(path)

        
        
        
if __name__ == '__main__':
    # init
    train = OMNIGLOT_train()
    test = OMNIGLOT_test()
    model = siamese()
    # build
    model.build(train.IMG_SHAPE)
    # train
    model.train(train, test)
    
    

