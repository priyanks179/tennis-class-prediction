# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:46:46 2018

@author: Kartik
"""

import pandas as pd
import numpy as np
import time
m_train=pd.read_csv('mens_train_file.csv')
X=m_train.iloc[:,:24]
X['outside.sideline']=X['outside.sideline'].astype(int)
X['outside.baseline']=X['outside.baseline'].astype(int)
X['same.side']=X['same.side'].astype(int)
X['server.is.impact.player']=X['server.is.impact.player'].astype(int)
Y=m_train.outcome

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1=LabelEncoder()
X.iloc[:,2]=labelencoder_X1.fit_transform(X.iloc[:,2])
 
labelencoder_X2=LabelEncoder()
X.iloc[:,21]=labelencoder_X2.fit_transform(X.iloc[:,21])

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

onehotencoder=OneHotEncoder(categorical_features=[0])
Y=onehotencoder.fit_transform(Y.reshape(5000,1)).toarray()



from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.05,random_state=0)

#model rando forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',
                                    random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred=classifier.predict(X_test)


Y_pred[:,0][Y_pred[:,0]==1]=8
Y_pred[:,1][Y_pred[:,1]==1]=1
Y_pred[:,2][Y_pred[:,2]==1]=2
Y_pred=Y_pred[:,0]+Y_pred[:,1]+Y_pred[:,2]
Y_pred[Y_pred[:]==8]=0


Y_test[:,0][Y_test[:,0]==1]=8
Y_test[:,1][Y_test[:,1]==1]=1
Y_test[:,2][Y_test[:,2]==1]=2
Y_test=Y_test[:,0]+Y_test[:,1]+Y_test[:,2]
Y_test[Y_test[:]==8]=0

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,np.float64(Y_pred))
ac=(cm[0,0]+cm[1,1]+cm[2,2])/250
print(ac)