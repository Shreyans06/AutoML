import os
import pickle
import re
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def test_columns(dataset, c_columns):  # method to test the non-numerical columns for unique values
    le = LabelEncoder()
    dict_all = dict(zip([], []))
    for c_column in c_columns:
        if re.search('^[-+]?\d+(\.\d+)?$', str(dataset[c_column][0])):
            continue
        else:
            temp_keys = dataset[c_column].values
            temp_values = le.fit_transform(dataset[c_column])
            dict_temp = dict(zip(temp_keys, temp_values))
            dict_all[c_column] = dict_temp
            dataset.replace(dict_all[c_column], inplace=True)
            # dataset[[c_column]] = le.fit_transform(np.ravel(dataset[[c_column]]))
            # print(le.classes_)
            # label_encoder_dict = dict(zip(le.classes_, range(len(le.classes_))))
            # mapping_dict[c_column]=label_encoder_dict

            # fit=dataset[[c_column]].apply(lambda x:d[x.name].fit_transform(np.ravel(dataset[[c_column]])))
            # label_encoder_dict=fit.apply(lambda x:d[x.name].inverse_transform(dataset[[c_column]]))
    filehandler = open(
        os.path.dirname(os.path.abspath(__file__)) + '/Download/encoded_file/encoding_file.obj', "wb")
    pickle.dump(dict_all, filehandler)
    filehandler.close()
    return dataset


def file_input(file):
    dataset = pd.read_csv(file)
    return dataset


def feature_target_split(input_df, target):
    y = input_df.pop(target)
    X = input_df
    return X, y


def encoded_input_df_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    return X_train, X_test, y_train, y_test


def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    filehandler = open(
        os.path.dirname(os.path.abspath(__file__)) + '/Download/scaling/scaling.pkl', "wb")
    pickle.dump(sc,filehandler)
    filehandler.close()
    X_test = sc.transform(X_test)
    return X_train, X_test


def model_with_tuning(nn_units, unique_classes, inp_dim, X_train, y_train):
    def create_model(init_mode='glorot_uniform'):
        model = Sequential()
        model.add(Dense(nn_units, input_dim=inp_dim, kernel_initializer=init_mode))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(nn_units, kernel_initializer=init_mode))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=np.size(unique_classes), kernel_initializer=init_mode,
                        activation='softmax'))  # Note: no activation beyond this point

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(
                     filepath=os.path.dirname(os.path.abspath(__file__)) + '/Download/models/best_model.hdf5',
                     monitor='val_loss',
                     save_best_only=True, mode='max')]
    model = KerasClassifier(build_fn=create_model, verbose=1)

    param_grid = dict(batch_size=[32, 64], epochs=[500],
                      init_mode=['he_normal', 'he_uniform'])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train, callbacks=callbacks, validation_split=0.15)
    return grid


def encode_test_df(test_df,columns,dict_all_loaded):
    if not dict_all_loaded:
        return test_df
    else:
        for col in test_df.columns:
            if col in dict_all_loaded.keys():
                test_df[col] = test_df[col].map(dict_all_loaded[col])
        return test_df


def check_prediction_column(y_pred,dict_all_loaded,columns):
    target_column=list(set(dict_all_loaded.keys())-set(columns))
    def get_key(val):
        for key, value in dict_all_loaded[target_column[0]].items():
            if val == value:
                return key
    if len(target_column)==0:
        return y_pred
    else:
        y_pred=list(map(get_key,y_pred))
        return y_pred

