import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Activation,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers import Activation
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

d = defaultdict(LabelEncoder)

le = LabelEncoder()


def test_columns(dataset, c_columns):  # method to test the non-numerical columns for unique values
    for c_column in c_columns:
        if re.search('^[-+]?\d+(\.\d+)?$', str(dataset[c_column][0])):
            continue
        else:
            dataset[[c_column]] = le.fit_transform(np.ravel(dataset[[c_column]]))
            print(le.classes_)
            # label_encoder_dict = dict(zip(le.classes_, range(len(le.classes_))))
            # print(label_encoder_dict)
            # fit=dataset[[c_column]].apply(lambda x:d[x.name].fit_transform(np.ravel(dataset[[c_column]])))
            # label_encoder_dict=fit.apply(lambda x:d[x.name].inverse_transform(dataset[[c_column]]))
    return dataset


def file_input(file):
    file_read = file
    dataset = pd.read_csv(file_read)
    return dataset


# c_columns=dataset.columns.values.tolist()
# n=dataset[c_columns[0]].count()
# c_columns=test_columns(dataset,c_columns,n,target)

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
    X_test = sc.transform(X_test)
    return X_train, X_test


# print(X)
# print(y)
#
# # Importing the dataset
#
#
# # Number of output Classes
# unique_classes = np.unique(y)
#
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
#
# # Number of units in hidden layer
# nn_units = int((X_train.shape[1] * 2 / 3) + np.size(unique_classes))
#
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# start_time = time.time()
#
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.callbacks import EarlyStopping, ModelCheckpoint
#
#
def model_with_tuning(nn_units, unique_classes, inp_dim, X_train, y_train):
    def create_model():
        model = Sequential()
        model.add(Dense(nn_units, input_dim=inp_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(nn_units))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=np.size(unique_classes), kernel_initializer='glorot_uniform',
                        activation='softmax'))  # Note: no activation beyond this point

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath=os.path.join("models", 'best_model1.hdf5'), monitor='val_loss',
                                 save_best_only=True, mode='max')]
    model = KerasClassifier(build_fn=create_model, verbose=1)

    param_grid = dict(batch_size=[32, 64, 128, 256], epochs=[1000])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train, callbacks=callbacks, validation_split=0.15)
    return grid

    # def create_model(activation='relu'):
    #     model = Sequential()
    #     model.add(Dense(units=nn_units, activation=activation, kernel_initializer='glorot_uniform', input_dim=inp_dim))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=nn_units, kernel_initializer='glorot_uniform', activation=activation))
    #     model.add(Dropout(0.3))
    #
    #     model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation=tf.nn.sigmoid))
    #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     return model
    #
    # epochs = 400
    # callbacks = [EarlyStopping(monitor='val_loss', patience=2),
    #              ModelCheckpoint(filepath=os.path.join("models", 'best_model1.hdf5'), monitor='val_loss',
    #                              save_best_only=True, mode='max')]
    # model_CV = KerasClassifier(build_fn=create_model, verbose=1)
    # # define the grid search parameters
    # # init_mode = ['lecun_uniform', 'glorot_uniform', 'he_normal', 'he_uniform']
    #
    # batch_size = [4, 8, 16, 32]
    # param_grid = dict(batch_size=batch_size, epochs=[epochs],nn_units)
    # grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, cv=5)
    # grid.fit(X_train, y_train, callbacks=callbacks, validation_split=0.15)
    # return grid

#
# def model_with_tuning1(X_train, y_train,X_test,y_test):
#     def model1(X_train, y_train, X_test, y_test, params):
#         model = Sequential()
#         model.add(Dense(params['first_neuron'], input_dim=X_train.shape[1],
#                         activation=params['activation'],
#                         kernel_initializer=params['kernel_initializer']))
#
#         model.add(Dropout(params['dropout']))
#
#         model.add(Dense(1, activation=params['last_activation'],
#                         kernel_initializer=params['kernel_initializer']))
#
#         model.compile(loss=params['losses'],
#                       optimizer=params['optimizer'],
#                       metrics=['acc', talos.utils.metrics.f1score])
#
#         history = model.fit(X_train, y_train,
#                             validation_data=[X_test, y_test],
#                             batch_size=params['batch_size'],
#                             callbacks=[talos.utils.live()],
#                             epochs=params['epochs'],
#                             verbose=0)
#
#         return history, model
#     p = {'first_neuron':[9,10,11],
#          'hidden_layers':[0, 1, 2],
#          'batch_size': [30],
#          'epochs': [100],
#          'dropout': [0],
#          'kernel_initializer': ['uniform','normal'],
#          'optimizer': [ 'Adam'],
#          'losses': ['sparse_categorical_crossentropy'],
#          'activation':['relu', 'elu'],
#          'last_activation': ['softmax']}
#     t = talos.Scan(x=X_train,
#                    y=y_train,
#                    model=model1,
#                    params=p,
#                    experiment_name='breast_cancer',
#                    round_limit=10)
#     talos.Deploy(scan_object=t,model_name='model1',metric='val_acc')
#
#     breast=talos.Restore('model1.zip')
#
#     print(breast.model.predict(X_test))

# # Part 3 - Making the predictions and evaluating the model
#
# # Predicting the Test set results
# y_pred = grid.predict(X_test)
# print(y_pred)
# print(y_test)
#
# # Function to predict class
# # class_predictor = lambda x: np.argmax(x)
# # predicted_class = list(map(class_predictor, y_pred))
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)
#
# # Calculate the Accuracy
# from sklearn.metrics import accuracy_score
#
# score = accuracy_score(y_pred, y_test)
#
# end_time = time.time()
#
# print("************************************")
# print("total time taken", end_time - start_time, "seconds")
# print("Accuracy", score * 100, "%")
