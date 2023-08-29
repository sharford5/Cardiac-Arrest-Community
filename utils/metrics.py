import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score


class F1Callback(Callback):

    def __init__(self, path, validation_data):
        self.path = path
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.best_val = -1.

    def on_epoch_end(self, epoch, logs={}):

        # Specific to this model. Change for other models
        val_predict = self.model.predict(self.validation_data[0])
        val_targ = np.argmax(self.validation_data[1], axis=-1)

        val_predict = np.argmax(val_predict, axis=-1)
        _val_f1 = fbeta_score(val_targ, val_predict, beta=3., average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print(' val_f1: % f  val_precision: % f  val_recall % f' % (_val_f1, _val_precision, _val_recall))

        if (_val_f1 > self.best_val):
            print("Validation F1 score improved from %0.6f to %0.6f. Saving Weights." % (self.best_val, _val_f1))
            print("")
            self.best_val = _val_f1
            self.model.save_weights(self.path)

        return
