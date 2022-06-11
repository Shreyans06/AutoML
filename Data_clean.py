from sklearn.preprocessing import LabelEncoder
import re
import properties as pr


le = LabelEncoder()

def test_columns(df,c_columns,n,target):  #method to test the non-numerical columns for unique values
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