import numpy as np
import tensorflow as tf
import random as rn
import os

import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils import to_categorical
from sklearn.cluster import KMeans


def _load_processed_dataset(FIXED_PARAMETERS, year, merge_bool=True):
    X = np.load(FIXED_PARAMETERS['data_path']+'/'+FIXED_PARAMETERS['dataset']+'/X_'+year+'_'+FIXED_PARAMETERS['dataset']+'.npy')
    y = np.load(FIXED_PARAMETERS['data_path']+'/'+FIXED_PARAMETERS['dataset']+'/Y_'+year+'_'+FIXED_PARAMETERS['dataset']+'.npy')

    if merge_bool:
        y[y == 2] = 1
    return X, y


def train_val_split(X, y, test_size=0.2, random_state=0, prob_duplicate=0.0):
    train_labels, train_counts = np.unique(y, return_counts=True)
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    for label, max_cnt in zip(train_labels, train_counts):
        samples = X[y == label, :]
        train_samples, val_samples = train_test_split(samples, test_size=test_size, random_state=random_state)

        for i in range(len(train_samples)):
            X_train.append(train_samples[i])
            Y_train.append(label)
            if (rn.random() < prob_duplicate) and (label == 0):
                X_train.append(train_samples[i])
                Y_train.append(label)

        for i in range(len(val_samples)):
            X_val.append(val_samples[i])
            Y_val.append(label)

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    Y_train = np.asarray(Y_train)
    Y_val = np.asarray(Y_val)

    return X_train, X_val, Y_train, Y_val




def prepare_dataset_merged_keras(FIXED_PARAMETERS, print_bool=True):
    num_non_cat = np.load(FIXED_PARAMETERS['data_path']+'/'+FIXED_PARAMETERS['dataset']+'/noncat_cols.npy')[0]

    X, y = {}, {}
    year_total = ['14', '15','16','17','18','19']
    years = [str(i) for i in range(int(FIXED_PARAMETERS['year_start']), int(FIXED_PARAMETERS['year_end'])+1)]
    for year in year_total:
        X[year], y[year] = _load_processed_dataset(FIXED_PARAMETERS, year)

    X_train, y_train, X_val, y_val, X_test, y_test = {},{},{},{},{},{}
    for year in year_total:
        X_train[year], X_test[year], y_train[year], y_test[year] = train_test_split(X[year], y[year], test_size=0.25, random_state=0)
        X_train[year], X_val[year], y_train[year], y_val[year] = train_val_split(X_train[year], y_train[year], test_size=0.2, random_state=0, prob_duplicate=FIXED_PARAMETERS['prob_duplicate'])
        if year not in years:
            del X_train[year]
            del X_val[year]
            del y_train[year]
            del y_val[year]

    X_train = np.concatenate([X_train[y] for y in X_train.keys()], axis=0)
    X_val = np.concatenate([X_val[y] for y in X_val.keys()], axis=0)
    X_test = np.concatenate([X_test[y] for y in X_test.keys()], axis=0)
    y_train = np.concatenate([y_train[y] for y in y_train.keys()], axis=0)
    y_val = np.concatenate([y_val[y] for y in y_val.keys()], axis=0)
    y_test = np.concatenate([y_test[y] for y in y_test.keys()], axis=0)

    X_train_cat = np.asarray([X_train[i][:X_train.shape[1]-num_non_cat] for i in range(X_train.shape[0])])
    X_val_cat = np.asarray([X_val[i][:X_val.shape[1]-num_non_cat] for i in range(X_val.shape[0])])
    X_test_cat = np.asarray([X_test[i][:X_test.shape[1]-num_non_cat] for i in range(X_test.shape[0])])


    X_train_noncat = np.asarray([[X_train[i][-num_non_cat:]/100] for i in range(X_train.shape[0])])
    X_val_noncat = np.asarray([[X_val[i][-num_non_cat:]/100] for i in range(X_val.shape[0])])
    X_test_noncat = np.asarray([[X_test[i][-num_non_cat:]/100] for i in range(X_test.shape[0])])

    X = np.concatenate((X_train_cat, X_val_cat, X_test_cat), axis=0)
    num_cols = X.shape[1]
    metadata = []
    for col_id in range(num_cols):
        data = X[:, col_id]
        num_unique = int(max(np.unique(data))+1)
        metadata.append(num_unique)

    X_train_cat = X_train_cat.astype(np.int32)
    X_val_cat = X_val_cat.astype(np.int32)
    X_test_cat = X_test_cat.astype(np.int32)

    X_train_cat = [X_train_cat[:, i] for i in range(num_cols)]
    X_val_cat = [X_val_cat[:, i] for i in range(num_cols)]
    X_test_cat = [X_test_cat[:, i] for i in range(num_cols)]

    X_train = [X_train_noncat] + X_train_cat
    X_val = [X_val_noncat] + X_val_cat
    X_test = [X_test_noncat] + X_test_cat

    train_labels, train_counts = np.unique(y_train, return_counts=True)
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)

    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    if print_bool: print()
    if print_bool: print("Train set : ", (X_train[0].shape[0], num_cols), "Class Label Counts: ", train_counts)
    if print_bool: print("Val set : ", (X_val[0].shape[0], num_cols), "Class Label Counts: ", val_counts)
    if print_bool: print("Test set : ", (X_test[0].shape[0], num_cols), "Class Label Counts: ", test_counts)
    if print_bool: print()
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


