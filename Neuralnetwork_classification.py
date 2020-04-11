from sklearn.metrics import confusion_matrix
import Class_hyp as ch
import pandas as pd
import numpy as np
import pickle
from threading import Thread, Event
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import time
from keras.models import model_from_json, load_model
import os
from keras.layers import Dropout
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

file_name = 'C:/Users/Shreyans/Desktop/Churn_Modelling.csv'
target = 'Exited'


pd.set_option('display.max_rows', 20)


def main(file, target):
    # Input the dataset into inp_df dataframe
    input_df = ch.file_input(file)

    # Encoding Categorical data and feature engineering
    columns = input_df.columns.values.tolist()  # columns of input_df
    encoded_input_df= ch.test_columns(input_df, columns)


    dict_file = open("C:/Users/Shreyans/PycharmProjects/Flask/dict_all.obj", 'rb')
    dict_all_loaded = pickle.load(dict_file)
    dict_file.close()


    input_df1 = ch.file_input(file)
    for col in input_df1.columns:
        if col in dict_all_loaded.keys():
            input_df1.replace(dict_all_loaded[col],inplace=True)
    # for c_columns in columns:
    #     if c_columns in mapping.keys():
    #         input_df1[c_columns]=input_df1[c_columns].map(mapping[c_columns])
    df1 = pd.DataFrame(data=input_df1, columns=columns)
    df1.to_csv('C:/Users/Shreyans/Desktop/test.csv', index=False)



    inp_test=pd.read_csv('C:/Users/Shreyans/Desktop/Churn_Modelling1.csv')
    columns_test = inp_test.columns.values.tolist()
    print(columns_test)
    print(list(set(mapping.keys())-set(columns_test)))


    # Preparing target column and features column
    X, y = ch.feature_target_split(encoded_input_df, target)

    # Number of output Classes
    unique_classes = np.unique(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = ch.encoded_input_df_train_test_split(X, y)


    #
    # # Input dimension
    # inp_dim = X_train.shape[1]
    #
    # # Number of units in hidden layer
    # nn_units = int((X_train.shape[1] * 2 / 3) + np.size(unique_classes))
    #
    # # Feature Scaling
    # X_train, X_test = ch.feature_scaling(X_train, X_test)
    #
    # # Training and hyperparameter tuning
    # model = ch.model_with_tuning(nn_units, unique_classes, inp_dim, X_train, y_train)
    #
    # # Predicting the Test set results
    # y_pred = model.predict(X_test)
    # print("Predicted", y_pred)
    # score = accuracy_score(y_test, y_pred)
    # print(score * 100)

    # Calculate the Accuracy

    # final_mod = decoded_model.predict(X_test)
    # # print(final_mod.best_params_)
    # # print(final_mod.best_estimator_)
    # #
    # # # Function to predict class
    # class_predictor = lambda x: np.argmax(x)
    # predicted_class = list(map(class_predictor, final_mod))
    # #
    # score1 = accuracy_score(predicted_class, y_test)
    #
    # print(score1 * 100)


if __name__ == '__main__':
    main(file_name, target)
