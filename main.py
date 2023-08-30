import numpy as np
import tensorflow as tf
import random as rn
import os
from scipy.stats import sem, t
# from scipy import mean
import numpy as np
import pandas as pd

import shap

#Load Parameters
import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()

#Set Seeds for Reproducability
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))
from keras import backend as K


from keras.models import Model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Reshape, RepeatVector, Lambda
from keras.layers import Conv1D, BatchNormalization, Activation, concatenate, multiply, Dropout, Permute, LSTM
from keras.regularizers import l2
from utils.keras_utils import train_model, eval_test_model, eval_val_model


def lstm_fcn(max_seq_len, num_classes, dropout_rate=0.8, input=None, return_softmax=True):
    if input == None:
        input = Input(shape=(1, max_seq_len))

    lstm = LSTM(8)(input)
    lstm = Dropout(0.8)(lstm)

    cnn = Permute((2, 1))(input)
    cnn = conv_block(cnn, dropout_rate= dropout_rate, nodes=64)

    cnn = GlobalAveragePooling1D()(cnn)
    pool = concatenate([lstm, cnn])

    if return_softmax:
        out = Dense(num_classes, activation='softmax')(pool)
        model = Model(input, out)
        return model
    else:
        return pool

def efcn(num_classes, metadata, max_seq_len=0, embedding_dim=50, dropout_rate=0.8, input=None, return_softmax=True):
    categorical_layers = []
    if input == None:
        input = []
        for i, cat_input_dim in enumerate(metadata):
            ip = Input(shape=(1,), dtype='int32', name='input_%s' % i)
            embed = Embedding(cat_input_dim, embedding_dim)(ip)
            input.append(ip)
            categorical_layers.append(embed)
    else:
        for i, cat_input_dim in enumerate(metadata):
            embed = Embedding(cat_input_dim, embedding_dim)(input[i])
            categorical_layers.append(embed)

    #Combine Embedding Layers
    embedding = concatenate(categorical_layers, axis=1)
    #Send Embedding through Conv Block
    cnn = conv_block(embedding, dropout_rate=dropout_rate)
    pool = GlobalAveragePooling1D()(cnn)

    if return_softmax:
        out = Dense(num_classes, activation='softmax')(pool)
        model = Model(input, out)
        return model
    else:
        return pool

def lstm_fcn_efcn(max_seq_len, num_classes, metadata, embedding_dim=50, dropout_rate=0.8):
    inputs = []
    input1 = Input(shape=(1, max_seq_len), name='input_lstm')
    inputs.append(input1)

    categorical_layers = []
    for i, cat_input_dim in enumerate(metadata):
        ip = Input(shape=(1,), dtype='int32', name='input_efcn_%s' % i)
        embed = Embedding(cat_input_dim, embedding_dim)(ip)
        inputs.append(ip)
        categorical_layers.append(embed)

    pool1 = efcn(num_classes, metadata, max_seq_len=max_seq_len, embedding_dim= embedding_dim, input=inputs[1:], return_softmax=False)
    pool2 = lstm_fcn(max_seq_len, num_classes, input=input1, return_softmax=False)
    con = concatenate([pool1, pool2])
    den = Dense(128, activation='relu', name='dense_out')(con)
    out = Dense(num_classes, activation='softmax', name='output')(den)
    model = Model(inputs, out)
    return model

def conv_block(x, layers=3, dropout_rate=0.8, nodes=128):
    for _ in range(layers):
        x = Conv1D(nodes, 3, padding='same', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = squeeze_excite_block(x)
        x = Dropout(dropout_rate)(x)
    return x

def squeeze_excite_block(input):
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

if __name__ == '__main__':
    print(FIXED_PARAMETERS)

    num_non_cat = np.load(FIXED_PARAMETERS['data_path']+'/'+FIXED_PARAMETERS['dataset']+'/noncat_cols.npy')[0]

    new_cols = [313, 231, 25, 15, 407, 120, 460, 319, 88, 129]

    #train_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)

    v, _ = eval_val_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)
    t, _ = eval_test_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)








