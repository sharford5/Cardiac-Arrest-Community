import numpy as np
import tensorflow as tf
import random as rn
import os
from scipy.stats import sem, t
import numpy as np
import pandas as pd
import shutil


#Load Parameters
import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()

#Set Seeds for Reproducability
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))
from keras import backend as K


from utils.keras_utils import train_model, eval_test_model, eval_val_model, lstm_fcn_efcn



grid_parameters = {
    'BOOL_LSTM':[0,1],
    'BOOL_CNN': [0,1],
    'OUT_CNN': [0],
    'lstm_nodes': [8,12,16],
    'conv_layers': [3,3,3,3,2,4],
    'conv_nodes': [16,32,64],
    'conv_stride': [3,5,8],
    'embedding_dim': [10,20,50],
    'dropout': [0.6,0.7,0.8,0.9],
    'batchsize': [32,64,128],
    'dense_out':[32,64,128],
    'learning_rate': [2e-3, 2e-4, 2e-5],
    'gamma': [0.5,0.75,1,1.5,2,2.5,3],
    'alpha': [0.5,0.7,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5]
}


col_set1 = [313,231,25,15,407,120,460,319,88,129]
col_set2 = [421,276,512,267,32,135,221,96,83]
col_set3 = [306,71,440,101,110,178,287,75,369,421]

if __name__ == '__main__':
    print(FIXED_PARAMETERS)

    num_non_cat = np.load(FIXED_PARAMETERS['data_path']+'/'+FIXED_PARAMETERS['dataset']+'/noncat_cols.npy')[0]
    holder = [['New Cols', 'Val_Eval', 'Test_Eval', 'New Best']]
    best = 0
    while True:
        new_best = 'No'
        new_cols = rn.choice([col_set1, col_set2, col_set3])

        for k in grid_parameters.keys():
            value = rn.choice(grid_parameters[k])
            FIXED_PARAMETERS[k] = value


        train_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)
        v, _ = eval_val_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)
        t, _ = eval_test_model(lstm_fcn_efcn, FIXED_PARAMETERS, new_cols)

        if v > best:
            best = v
            new_best = 'Yes'
            shutil.copy('./weights/temp.h5', './weights/grid_best.h5')



        row = [new_cols, v, t, str(FIXED_PARAMETERS), new_best]
        holder.append(row)

        Holder = pd.DataFrame(holder)
        Holder.to_csv('grid_search.csv')








