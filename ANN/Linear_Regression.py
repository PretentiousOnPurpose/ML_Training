"""
Created on Sun Sep 17 18:24:47 2017
@author: chawat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('USA_Housing.csv')
X = data.iloc[: , :5].values
y = data.iloc[: , 5].values

X = np.append(arr=np.ones((5000 , 1)) ,values=X, axis=1)

from sklearn.model_selection import train_test_split
xtr , xts , ytr, yts = train_test_split(X , y , test_size=0.001 , random_state=101)

#--------Using Scikit Learn

#from sklearn.linear_model import LinearRegression
#lm = LinearRegression()
#lm.fit(xtr , ytr)
#resLM = lm.predict(test_var.reshape((1 , 6)))
#--------END sklearn---------

#------Using ANN-----------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scy = StandardScaler()

xtr = sc.fit_transform(xtr)
xts = sc.transform(xts)

ytr = scy.fit_transform(np.array(ytr).reshape((-1 , 1)))
yts = scy.transform(np.array(yts).reshape((-1 , 1)))

#------Low Accuray Code-------
#ANN = Sequential()
#ANN.add(Dense(units=3 , kernel_initializer='normal' , activation='linear' , input_dim=6)) #Hidden Layer
#ANN.add(Dense(units=1 , kernel_initializer='normal' , activation='linear')) #Output Layer
#ANN.compile(optimizer='sgd' , loss='mean_squared_error' , metrics=['accuracy'])
#ANN.fit(xtr , ytr , batch_size=35 , epochs=101)
#---------END LAC------------

#---------High Accuracy---------------
ANN = Sequential()
ANN.add(Dense(units=3 , kernel_initializer='normal' , activation='relu' , input_dim=6)) # First HL
ANN.add(Dense(units=3 , kernel_initializer='normal' , activation='linear')) #HL
ANN.add(Dense(units=3 , kernel_initializer='normal' , activation='linear')) #HL
ANN.add(Dense(units=1 , kernel_initializer='normal' , activation='linear')) # Output Layer
ANN.compile(optimizer='sgd' , loss='mean_squared_error' , metrics=['accuracy'])
ANN.fit(xtr , ytr , batch_size=35 , epochs=101)

test_var = np.array([1 , 68700 , 5.38286 , 8.55 , 2.45 , 33655]).reshape((-1 , 1))

res= scy.inverse_transform(ANN.predict(sc.transform(test_var.reshape((1 ,6)))))

#resLM = 1284040.83 (using sklearn)
#res = 1292539.125 (using ANN)
#y_test = 1291330 (from Data)
