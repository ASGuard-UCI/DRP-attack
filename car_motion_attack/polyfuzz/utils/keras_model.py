#!/usr/bin/env python

from __future__ import print_function
import os
os.environ['GLOG_minloglevel'] = '2'
import sys
import argparse
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers import Input, Lambda, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Add, ELU
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.utils import plot_model

def create_model():
    vision_input = Input(shape=(6,80,160), dtype='float32', name='vision')
    rnn_state = Input(shape=(512,), dtype='float32', name='rnn_state')

    # After permutation, the output shape will be (80, 160, 6)
    vision_permute = Permute((2,3,1), input_shape=(6,80,160), name='vision_permute')(vision_input)
    vision_conv2d = Conv2D(16, 5, strides=1, padding="same", name='vision_conv2d')(vision_permute)
    vision_elu = ELU(alpha=1.0, name='vision_elu')(vision_conv2d)
    vision_conv2d_1 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_1')(vision_elu)

    # Resnet block
    vision_max_pooling2d = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d')(vision_conv2d_1)

    vision_conv2d_2 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_2')(vision_max_pooling2d)
    vision_elu_1 = ELU(alpha=1.0, name='vision_elu_1')(vision_conv2d_2)
    vision_conv2d_3 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_3')(vision_elu_1)
    vision_add = Add(name='vision_add')([vision_max_pooling2d, vision_conv2d_3])

    # Resnet block
    vision_max_pooling2d_1 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_1')(vision_add)
    vision_conv2d_4 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_4')(vision_max_pooling2d_1)
    vision_elu_2 = ELU(alpha=1.0, name='vision_elu_2')(vision_conv2d_4)
    vision_conv2d_5 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_5')(vision_elu_2)
    vision_add_1 = Add(name='vision_add_1')([vision_max_pooling2d_1, vision_conv2d_5])

    # Resnet block
    vision_max_pooling2d_2 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_2')(vision_add_1)
    vision_conv2d_6 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_6')(vision_max_pooling2d_2)
    vision_elu_3 = ELU(alpha=1.0, name='vision_elu_3')(vision_conv2d_6)
    vision_conv2d_7 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_7')(vision_elu_3)
    vision_add_2 = Add(name='vision_add_2')([vision_max_pooling2d_2, vision_conv2d_7])

    vision_max_pooling2d_3 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_3')(vision_add_2)
    flattened = Flatten(name='flattened')(vision_max_pooling2d_3)

    # RNN layer
    dense = Dense(576, name='dense')(flattened)
    elu = ELU(alpha=1.0, name='elu')(dense)
    rnn_kernel = Dense(512, name='rnn_kernel')(elu)
    rnn_recurrent_kernel = Dense(512, name='rnn_recurrent_kernel')(rnn_state)
    rnn_add = Add(name='rnn_add')([rnn_kernel, rnn_recurrent_kernel])
    rnn_out_state = Activation('tanh', name='rnn_out_state')(rnn_add)

    # Path
    dense_1 = Dense(256, name='dense_1')(rnn_out_state)
    elu_1 = ELU(alpha=1.0, name='elu_1')(dense_1)
    dense_2 = Dense(128, name='dense_2')(elu_1)
    elu_2 = ELU(alpha=1.0, name='elu_2')(dense_2)
    path = Dense(51, name='path')(elu_2)

    # Left lane
    dense_4 = Dense(256, name='dense_4')(rnn_out_state)
    elu_3 = ELU(alpha=1.0, name='elu_3')(dense_4)
    dense_5 = Dense(128, name='dense_5')(elu_3)
    elu_4 = ELU(alpha=1.0, name='elu_4')(dense_5)
    left_lane = Dense(53, name='left_lane')(elu_4)

    # Right lane
    dense_7 = Dense(256, name='dense_7')(rnn_out_state)
    elu_5 = ELU(alpha=1.0, name='elu_5')(dense_7)
    dense_8 = Dense(128, name='dense_8')(elu_5)
    elu_6 = ELU(alpha=1.0, name='elu_6')(dense_8)
    right_lane = Dense(53, name='right_lane')(elu_6)

    # Lead
    dense_10 = Dense(256, name='dense_10')(rnn_out_state)
    elu_7 = ELU(alpha=1.0, name='elu_7')(dense_10)
    dense_11 = Dense(128, name='dense_11')(elu_7)
    elu_8 = ELU(alpha=1.0, name='elu_8')(dense_11)
    lead = Dense(4, name='lead')(elu_8)

    # Output dimension should be (1, 673)
    outputs = Concatenate(name='outputs')([path, left_lane, right_lane, lead, rnn_out_state])

    model = Model(inputs=[vision_input, rnn_state], outputs=outputs, name='driving_model')
    #print(model.summary())
    #plot_model(model, to_file='driving_model.png')
    #model.load_weights('../testdata/driving_model.h5')
    return model

INPUT_SHAPE = (1, 6, 80, 160)
RNN_INPUT_SHAPE = (1, 512)

def run_keras(model, input_, rnn_input_):
    out = model.predict([input_, rnn_input_])[0]
    return out

def main():
    model = create_model()
    model.load_weights('../testdata/driving_model.h5')

    modelin = np.load('../testdata/yuv6c_50f_highwayramp_suv.npy')
    print(modelin.shape)
    rnnin = np.zeros(RNN_INPUT_SHAPE)
    modelout = np.zeros((50, 673))
    for i,mdin in enumerate(modelin):
        mdout = run_keras(model, mdin.reshape(INPUT_SHAPE), rnnin)
        rnnin = mdout[-512:].reshape(RNN_INPUT_SHAPE)
        print(mdout.shape)
        modelout[i] = mdout
    np.save('highwayramp_modelout', modelout)

if __name__ == '__main__':
    main()
