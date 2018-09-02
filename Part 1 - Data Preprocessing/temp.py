#we are importing libraries for our machine learning program
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets for machine learning
dataset= pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#taking care og missing values 
#we are applying maen for missing data values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#this data contains categeorical strings //countries
#so we need to encode them into number since machine learning is mathematical
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#all country data encoded into number but our machine learning model may think
#the encoded number as priority so we need to create dummy variabeles to 
#prevent errors in our predictions
from sklearn.preprocessing import OneHotEncoder
onehotencoder= OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

# so x is encoded then its time to Y
#its a has only 2 categories so not need of onehotencoding
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#SPLITTING DATA into training and test set 
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#so we splitted these guys into two categeories 
#Feature Scaling
#its because Euclidean Distance between two categeories should not be more
#bringing these values into normal state or reducing distance
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler
#X_train = sc_x.fit_transform(X_train)
#X_test = sc_x.transform(X_test)

#sc_y = StandardScaler()
#Y_train = sc_y.fit_transform(Y_train)





