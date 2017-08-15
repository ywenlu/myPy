import numpy as np
import pandas as pd
# For binary class


def FisherDiscriminant(X, y):  # (X: feature dataframe, y:label)
    # return feature ranking (most->least important)
    # used in DEAP database
    classlabel = y.unique()
    c1 = X[y == classlabel[0]]
    c2 = X[y == classlabel[1]]
    u1=c1.mean()
    var1=c1.var()
    u2=c2.mean()
    var2=c2.var()
    J=np.abs(u1-u2)*1.0/(var1+var2)
    return J.sort_values(ascending=False)

# For multiple class
def FeatureFScore(X,y):#(X: feature dataframe, y:label)
	#return feature ranking (most->least important)
	#used in <<EEG-Based Emotion Recognition in Music Listening>>
    featurenames=X.columns.values
    classlabel=y.unique()
    nbc=len(classlabel)
    Xl_mean = np.vstack([X[y==l].mean() for l in classlabel])
    X_mean = X.mean().reshape(-1,X.shape[1])
    nl=np.array([len(X[y==l]) for l in classlabel]).reshape((nbc,1))
    F = sum(nl*(Xl_mean-X_mean)*(Xl_mean-X_mean))/sum((np.array(X)-X_mean)*(np.array(X)-X_mean))
    return pd.Series(F,index=featurenames).sort_values(ascending=False)
