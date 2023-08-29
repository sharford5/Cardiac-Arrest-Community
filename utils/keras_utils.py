import numpy as np
import tensorflow as tf
import random as rn
import os

#Load Parameters
import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()

#Set Seeds for Reproducability
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))
from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
import shap
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Reshape, RepeatVector, Lambda, MaxPooling1D
from keras.layers import Conv1D, BatchNormalization, Activation, concatenate, multiply, Dropout, Permute, LSTM
from keras.regularizers import l2
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from utils.data_utils import prepare_dataset_merged_keras
from utils.metrics import F1Callback
import eli5
from eli5.sklearn import PermutationImportance



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

def reduce_cols(df, new_cols):
    a = df[0]
    a = np.squeeze(a, axis = 1)

    r = a[:,new_cols]
    r = np.expand_dims(r, axis=1)
    df[0] = r

    if len(new_cols) == 0:
        del df[0]

    return df

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def train_model(model_fn, FIXED_PARAMETERS, new_cols):
    (X_train, y_train), (X_val, y_val), (_, _), metadata = prepare_dataset_merged_keras(FIXED_PARAMETERS)

    X_train = reduce_cols(X_train, new_cols)
    X_val = reduce_cols(X_val, new_cols)

    num_classes = len(np.unique(np.argmax(y_train, axis=-1)))
    try:
        max_length = X_train[0].shape[2]
    except:
        max_length = 0

    K.clear_session()
    with tf.device(FIXED_PARAMETERS['device']):
        print(FIXED_PARAMETERS['device'])
        model = model_fn(max_seq_len=max_length, num_classes=num_classes, metadata=metadata)#, FIXED_PARAMETERS=FIXED_PARAMETERS)

    optimizer = Adam(FIXED_PARAMETERS['learning_rate'])
    model.compile(optimizer, loss=focal_loss(gamma=FIXED_PARAMETERS['gamma'], alpha=FIXED_PARAMETERS['alpha']), metrics=['accuracy'])

    weights_path = './weights/%s.h5' % FIXED_PARAMETERS['NAME']
    f1_callback = F1Callback(weights_path, validation_data=[X_val, y_val])
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=6, verbose=True, min_lr=2e-5)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks = [lr, es, f1_callback]

    classes = np.unique(np.argmax(y_train, axis=-1))
    le = LabelEncoder()
    y_ind = le.fit_transform(np.argmax(y_train, axis=-1).ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))

    # class_weight = recip_freq[le.transform(classes)]
    # class_weight = [cw / min(class_weight) for cw in class_weight]
    # print("Class weights : ", class_weight)
    history = model.fit(X_train, y_train, FIXED_PARAMETERS['batchsize'], FIXED_PARAMETERS['epochs'], callbacks=callbacks, validation_data=(X_val, y_val), verbose=1, shuffle=True)# , class_weight=class_weight

    return model


def eval_val_model(model_fn, FIXED_PARAMETERS, new_cols, print_bool=True):
    (X_train, y_train), (X_val, y_val), (_, _), metadata = prepare_dataset_merged_keras(FIXED_PARAMETERS, print_bool)
    num_classes = len(np.unique(np.argmax(y_val, axis=-1)))

    X_train = reduce_cols(X_train, new_cols)
    try:
        max_length = X_train[0].shape[2]
    except:
        max_length = 0
    X_val = reduce_cols(X_val, new_cols)

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(max_seq_len=max_length, num_classes=num_classes, metadata=metadata)#, FIXED_PARAMETERS=FIXED_PARAMETERS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/%s.h5' % FIXED_PARAMETERS['NAME']
    model.load_weights(weights_path)

    preds = model.predict(X_val, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_val, axis=-1)

    cm = confusion_matrix(y_true, preds)
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))

    if print_bool: print("Avg Recall: ", avg_recall)


    # batch = [np.array(X_train[i][:10]).reshape(10,1) for i in range(len(X_train))]
    # explainer = shap.DeepExplainer(model, batch)

    # perm = PermutationImportance(model, random_state=1, scoring="accuracy").fit(X_train, y_train)
    # eli5.show_weights(perm, feature_names=X.columns.tolist())
    return avg_recall, cm


def eval_test_model(model_fn, FIXED_PARAMETERS, new_cols, print_bool=True):
    (X_train, _), (_, _), (X_test, y_test), metadata = prepare_dataset_merged_keras(FIXED_PARAMETERS, print_bool)
    num_classes = len(np.unique(np.argmax(y_test, axis=-1)))

    X_train = reduce_cols(X_train, new_cols)
    try:
        max_length = X_train[0].shape[2]
    except:
        max_length = 0
    X_test = reduce_cols(X_test, new_cols)

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(max_seq_len=max_length, num_classes=num_classes, metadata=metadata)#, FIXED_PARAMETERS=FIXED_PARAMETERS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/%s.h5' % FIXED_PARAMETERS['NAME']
    model.load_weights(weights_path)

    preds = model.predict(X_test, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, preds)
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))

    if print_bool: print("Avg Recall: ", avg_recall)
    return avg_recall, cm



