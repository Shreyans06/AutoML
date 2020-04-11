from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from flask_wtf.file import FileField, FileRequired, FileAllowed


class InputdataForm(FlaskForm):
    project_name = StringField('Project Name', validators=[DataRequired(), Length(min=2, max=20)])
    inp_data = FileField('Input File', validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')])
    target_column = StringField('Target Column', validators=[DataRequired()])
    submit = SubmitField('Train')


class PredictdataForm(FlaskForm):
    model_file = FileField('Model File',
                           validators=[FileRequired(), FileAllowed(['pkl', 'hdf5'], 'Pkl/HDF5 files only!')])
    encoding_file = FileField('Encoding File',
                           validators=[FileRequired(), FileAllowed(['obj'], 'obj files only!')])
    scaling_file = FileField('Scaling File',
                              validators=[FileRequired(), FileAllowed(['pkl'], 'Pkl files only!')])
    predict_file = FileField('Prediction File', validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Predict')
