from keras.utils import np_utils
import numpy as np
import glob
import tensorflow as tf
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, RMSprop
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import pairwise_distances

import os
from keras.layers import Input, UpSampling2D, AveragePooling2D, Lambda
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, merge, Reshape
from keras import backend as K

from keras.models import Model, load_model
from keras.applications import VGG16
from data import OMNIGLOT_train, OMNIGLOT_test


class siamese:
    def build(self, input_shape):
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        inp = vgg.input
        encode = Flatten()(vgg.layers[-2].output)
        self.trans = Model(inp, encode)
        #print(self.trans.summary())
        
        left_input = Input(shape=input_shape)
        right_input = Input(shape=input_shape)
        #encode each of the two inputs into a vector
        encoded_l = self.trans(left_input)
        encoded_l = Reshape((-1,1))(encoded_l)
        encoded_r = self.trans(right_input)
        encoded_r = Reshape((-1,1))(encoded_r)
#         #merge two encoded inputs with the l1 distance
#         L1_distance = lambda x: K.abs(x[0]-x[1])
#         distance = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
#         prediction = Dense(1,activation='sigmoid')(distance)
#         self.siamese_net = Model([left_input,right_input], prediction)
        # merge two encoded inputs with cosine distance
        cos_distance = merge([encoded_l, encoded_r], mode='cos', dot_axes=1) # magic dot_axes works here!
        cos_distance = Reshape((1,))(cos_distance)
        cos_similarity = Lambda(lambda x: 1-x)(cos_distance)
        self.siamese_net = Model([left_input,right_input], [cos_similarity])
        
        optimizer = Adam()
        self.siamese_net.compile(optimizer=optimizer, loss='cosine_proximity', metrics=['accuracy'])
        
        
    def train(self, train, test, save_path='models/siamese'):
        # normal training
#         # make all layers untrainable by freezing weights (except for last layer)
#         for l, layer in enumerate(self.trans.layers[:-2]):
#             layer.trainable = False
#         # ensure the last layer is trainable/not frozen
#         for l, layer in enumerate(self.trans.layers[-2:]):
#             layer.trainable = True
        optimizer = RMSprop()
        self.siamese_net.compile(optimizer=optimizer, loss='cosine_proximity', metrics=['accuracy'])
        print(self.siamese_net.summary())
        print('Fine Tune')
        self.siamese_net.fit_generator(generator=train, epochs=10, validation_data=test)
        #self.siamese_net.save(save_path)
        
        
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
    