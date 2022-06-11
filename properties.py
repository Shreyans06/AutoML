import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
import re
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC


lim=linear_model
reg={'Lasso':lim.Lasso,'Ridge':lim.Ridge}
prg={'Lasso':{'alpha':[0.1,0.2,0.3,0,1]},'Ridge':{'alpha':[0.1,0.2,0.3,0,1]}}
fea_eng={'corr':0.7,'uniqueness':0.7}

