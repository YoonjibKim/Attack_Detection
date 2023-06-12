import numpy as np
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import Constant_Parameters


class DNN:
    @classmethod
    def dnn_run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn = training_feature_array
        y_tn = to_categorical(training_label_array)

        X_te = testing_feature_array
        y_te = testing_label_array

        n_feat = X_tn.shape[1]
        epo = 50
        n_class = 2

        model = Sequential()
        model.add(Dense(20, input_dim=n_feat))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(n_class))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        try:
            model.fit(X_tn, y_tn, epochs=epo, batch_size=5)
            pred_x = model(X_te)  # predict
            temp_predict_array = np.asarray(np.round(pred_x), dtype=int)
            temp_predict_list = []
            for data in temp_predict_array:
                if data[0] > 0:
                    temp_predict_list.append(int(Constant_Parameters.NORMAL_LABEL))
                else:
                    temp_predict_list.append(int(Constant_Parameters.ATTACK_LABEL))
            predict_array = np.array(temp_predict_list)
            class_report = classification_report(y_te, predict_array, zero_division=0, output_dict=True)
        except ValueError:
            class_report = None

        return class_report
