import os
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect, flash, send_from_directory
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import Class_hyp as ch
from forms import InputdataForm, PredictdataForm

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = {'csv'}
ALLOWED_PREDICTION_EXTENSIONS = {'pkl', 'hdf5'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['SECRET_KEY'] = '0c86e28fdca1e098f7116aaaa70a407a'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_file_prediction(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_PREDICTION_EXTENSIONS


def prediction_func(model, file):
    decoded_model = load_model(model)

    test_df = pd.read_csv(file)

    columns = test_df.columns.values.tolist()  # columns of input_df
    encoded_input_df = ch.test_columns(test_df, columns)
    X_train, X_test = ch.feature_scaling(encoded_input_df, encoded_input_df)

    y_pred = decoded_model.predict_classes(X_train)

    test_df['Target'] = y_pred
    test_df.to_csv(app.config['DOWNLOAD_FOLDER'] + '/predicted_files/'+'/breast1.csv', index=False)


def nn_classification(filename, target):
    input_df = ch.file_input(os.path.join(app.config['UPLOAD_FOLDER'] + '/datasets/', filename))

    # Encoding Categorical data and feature engineering
    columns = input_df.columns.values.tolist()  # columns of input_df
    encoded_input_df = ch.test_columns(input_df, columns)

    # Preparing target column and features column
    X, y = ch.feature_target_split(encoded_input_df, target)

    # Number of output Classes
    unique_classes = np.unique(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = ch.encoded_input_df_train_test_split(X, y)

    # Input dimension
    inp_dim = X_train.shape[1]

    # Number of units in hidden layer
    nn_units = int((X_train.shape[1] * 2 / 3) + np.size(unique_classes))

    # Feature Scaling
    X_train, X_test = ch.feature_scaling(X_train, X_test)

    # Training and hyperparameter tuning
    model = ch.model_with_tuning(nn_units, unique_classes, inp_dim, X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)
    print("Predicted", y_pred)
    score = accuracy_score(y_test, y_pred)
    return score


# @app.route("/")
# @app.route("/home", methods=['GET', 'POST'])
# def home():
#     return render_template('home.html')

# @app.route("/")
# @app.route("/home")
@app.route("/register", methods=['POST', 'GET'])
def register():
    form = InputdataForm()
    project_name = form.project_name.data
    target = form.target_column.data
    # if input_form.validate_on_submit():
    #     # flash(f'Training Started for {input_form.project_name.data}! ','success')
    if form.validate_on_submit() and request.method == 'POST':
        # check if the post request has the file part
        if 'inp_data' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['inp_data']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'] + '/datasets/', filename))
            # flash(f'File Uploaded Successfully! ', 'success')
        return redirect(url_for('machine_learning', filename=filename, target=target))
    return render_template('register.html', form=form)


@app.route('/machine_learning/<path:filename>/<target>', methods=['GET', 'POST'])
def machine_learning(filename, target):

    score = nn_classification(filename, target)
    score = round(score, 2)
    return render_template('output.html', score=score)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form = PredictdataForm()
    if form.validate_on_submit() and request.method == 'POST':
        if 'predict_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['predict_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            prediction_file = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'] + '/prediction_files/', prediction_file))

        if 'model_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['model_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file_prediction(file.filename):
            model_file = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'] + '/uploaded_models/', model_file))

        return redirect(url_for('prediction', model_file=model_file, prediction_file=prediction_file))
        # flash(f'File Uploaded Successfully! ', 'success')
    return render_template('predict.html', form=form)


@app.route('/prediction/<path:model_file>/<prediction_file>', methods=['GET', 'POST'])
def prediction(model_file, prediction_file):
    prediction_func(os.path.join(app.config['UPLOAD_FOLDER'] + '/uploaded_models/', model_file),
               os.path.join(app.config['UPLOAD_FOLDER'] + '/prediction_files/', prediction_file))
    return render_template('prediction_ouput.html',value='Prediction done')


@app.route('/downloads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'] + '/models/', filename, as_attachment=True)


@app.route('/predict_downloads/<filename>')
def predicted_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'] + '/predicted_files/', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
