import pandas as pd
import numpy as np

def colWithMissing(df):
    return phydf.isnull().sum()[(phydf.isnull().sum()>0)]
    

def repOutlierQuantile(df,low=0.05, high=0.95, repwith='limit'): 
    # params:
    # repwith:{'limit','median'}
    
    outlier_bound=df.quantile([low,high])
    
    outlier_low = (df < outlier_bound.loc[low])
    outlier_high = (df > outlier_bound.loc[high])
    
    if repwith == 'median':
        df.median()
        df=df.mask(outlier_low,df.median(),axis=1)
        df=df.mask(outlier_high,df.median(),axis=1)
    elif repwith == 'limit':  
        df=df.mask(outlier_low,outlier_bound.loc[low],axis=1)
        df=df.mask(outlier_high,outlier_bound.loc[high],axis=1)
    return df
    
    
from sklearn.preprocessing import MinMaxScaler,Normalizer,RobustScaler
def normbygroup(dfm,dfnormalizor):
    normdfls=[]
    gps = dfm.groupby(['sujet','len'])
    #dfnormalizor = MinMaxScaler()
    #dfnormalizor = Normalizer()
    for (sub,leng),gp in gps:
        gpd=gp.drop(['len','sujet','idx','arousal','valence'],axis=1)
        gpd=gpd.fillna(gpd.median())
        gp_trans = dfnormalizor.fit_transform(gpd)
        normdfls.append(pd.concat([gp[['len','sujet','idx','arousal','valence']].reset_index(drop=True),pd.DataFrame(gp_trans,columns=gpd.columns)],axis=1))
    normdf=pd.concat(normdfls)
    return normdf
