import os
import pandas as pd
from tensorflow.keras.models import load_model
import Class_hyp as ch

decoded_model = load_model(os.path.join("models", 'best_model1.hdf5'))

test_df = pd.read_csv('C:/Users/Shreyans/Desktop/breast1.csv')

columns = test_df.columns.values.tolist()  # columns of input_df
encoded_input_df = ch.test_columns(test_df, columns)
X_train, X_test = ch.feature_scaling(encoded_input_df, encoded_input_df)

y_pred = decoded_model.predict_classes(X_train)

test_df['Target']=y_pred
test_df.to_csv('C:/Users/Shreyans/Desktop/breast11.csv', index=False)

print(y_pred)
