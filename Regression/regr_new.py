#importing classes
import math
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
from AutoML import properties as pr
import pickle as p
import os,sys,inspect
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
import warnings
from AutoML import Data_clean as dc
#end of importing

warnings.filterwarnings("ignore")

model={}
le = LabelEncoder()
vif=pd.DataFrame()
colm_list=[]

def train_model(X,Y,report,n):       #method to calculate the scores of different algorithms
    h=1
    report['Algorithms'] = {}  #generating a report for the user to provide the results of each algorithm
    maxi=-9999999
    for reg_model in pr.reg:
        res=pr.reg[reg_model]()
        grid = GridSearchCV(res, pr.prg[reg_model], cv=2, scoring=None,refit=True)
        grid.fit(X, Y)
        s_score = grid.score(X, Y)
        report['Algorithms'][h] = {}
        report['Algorithms'][h]['Name'] = reg_model
        report['Algorithms'][h]['Score'] = s_score
        report['Algorithms'][h]['Hyperparameters Used'] = grid.best_params_
        h += 1
        if(s_score>maxi):
            maxi=s_score
            model['best_model'] = grid.best_estimator_
    return maxi

def test_corr(df,c_columns, n,target):
    c_cols=[]
    df1 = df.drop([target], inplace=False, axis=1)
    if(df1.shape[1]>1):
        vif["VIF Factor"] = [vf(df1.values, i) for i in range(df1.shape[1])]
        vif["features"] = df1.columns
        mn = 999
        for i, j in zip(vif["VIF Factor"], vif["features"]):
            if (i < mn):
                mn = i
                mnc = j
            if (i < 10):
                c_cols.append(j)
        if len(c_cols) == 0:
            c_cols.append(mnc)
        return c_cols
    return c_columns

def excecute_regr(df,target):
    report = {}
    c_columns=df.columns.values.tolist()
    c_cols=[]
    colm_list = df.columns.values.tolist()
    n=df[c_columns[0]].count()
    df = dc.data_clean(df, n)
    c_columns = dc.test_columns(df, c_columns, n, target)
    c_columns=test_corr(df,c_columns, n,target)
    train_model(df[c_columns],df[target],report,n)
    report['Features used for scoring']=c_columns
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    with open(sys.path[0]+'\models\Regression\\features', 'wb') as f:
        p.dump(c_columns, f)
    f.close()
    with open(sys.path[0]+'\models\Regression\\best_model', 'wb') as f:
        p.dump(model['best_model'], f)
    f.close()
    return report

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', None)
df = pd.read_csv('F:\MLData\\wea1dat.csv')
print (df)
target = 'FR_windspeed_10m'
print(excecute_regr(df,target))





