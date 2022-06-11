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
import properties as pr
import pickle as p
import os,sys,inspect
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
import warnings
import Data_clean as dc

warnings.filterwarnings("ignore")
#end of importing
model={}
le = LabelEncoder()
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

def test_columns(df,c_columns,report,n,target):  #method to test the non-numerical columns for unique values
    unwanted=[]
    for c_column in c_columns:
            if(re.search('^[-+]?\d+(\.\d+)?$',str(df[c_column][0]))):
                continue
            else:
                df[[c_column]]=le.fit_transform(df[[c_column]])
                uniqueness=len(df[c_column].unique())
                unique_ratio=uniqueness/n
                if(unique_ratio > pr.fea_eng['uniqueness'] and c_column!=target):   #if uniqueness of values of a column is greater
                    df.drop([c_column], inplace=True,axis=1)   #than a threshold, discard the column
                    unwanted.append(c_column)
    c1_columns = [c for c in c_columns if c not in unwanted]
    return c1_columns

def data_clean(df,n):
    df=df.dropna(thresh=0.2)
    df=df.fillna(method="ffill")
    return df

def test_corr(df,c_columns,report,n,target):
    c_cols=[]
    df1 = df.drop([target], inplace=False, axis=1)
    print("df1")
    print(df1)
    print(df1.shape[1])

    if(df1.shape[1]>1):
        vif["VIF Factor"] = [vf(df1.values, i) for i in range(df1.shape[1])]
        vif["features"] = df1.columns
        print(vif)
        mn = 999
        for i, j in zip(vif["VIF Factor"], vif["features"]):
            if (i < mn):
                mn = i
                mnc = j
            if (i < 10):
                print(i, j)
                # df.drop([j], inplace=True, axis=1)
                c_cols.append(j)
        if len(c_cols) == 0:
            c_cols.append(mnc)
        print(df)
        print(c_cols)
        return c_cols
    return c_columns

def excecute_regr(df,target):
    report = {}

    c_columns=df.columns.values.tolist()
    c_cols=[]
    colm_list = df.columns.values.tolist()
    n=df[c_columns[0]].count()
    print(df)
    #df = data_clean(df,n)
    df = dc.data_clean(df, n)
    print(df)
    print('1:',c_columns)

    #c_columns=test_columns(df,c_columns,report,n,target)
    c_columns = dc.test_columns(df, c_columns, n, target)
    print(df)
    c_columns=test_corr(df,c_columns,report,n,target)
    print('3:', c_columns)

    #c_columns=c_cols

    #c_columns.remove(target)
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
df = pd.read_csv('D:\\New folder\\MLData\\wea1dat.csv')
#df = pd.read_csv('F:\MLData\\Churn_Modelling.csv')
print (df)
target = 'FR_windspeed_10m'
#target = 'Exited'

#print(df1)
vif=pd.DataFrame()
#print(df.values)
#print(df.shape[1])
#print(df.columns)


print(excecute_regr(df,target))

#'Regression\\features'
#'Regression\\best_model'




