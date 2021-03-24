#!!!!!!!with the help of laboratory code:
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz


fertility = pd.read_csv('ppfertility.csv',header=None,delimiter=",")

x=fertility.iloc[:,0:8].values
t = fertility[9]

le = preprocessing.LabelEncoder()
y_encoded=le.fit_transform(t)
#print('T',y_encoded)

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded,test_size=0.2) #80% training, 20% test
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#tree = DecisionTreeRegressor(min_samples_leaf=30,max_leaf_nodes=70,min_samples_split=3,splitter='best',criterion='mse')
#tree.fit(X_train,y_train) #Build a decision tree classifier from the training set (X_train, y_train).
#predictions = tree.predict(X_test)
error=[]
r2=[]
for i in range(1,50):
    tree = DecisionTreeRegressor(min_samples_leaf=i,max_leaf_nodes=70,min_samples_split=3,splitter='best',criterion='mse')
    tree.fit(X_train,y_train) #Build a decision tree classifier from the training set (X, y).
    print('min samples leaf = ', i)
    predictions = tree.predict(X_test)
    err1 = np.sum(abs(y_test - predictions)**2)/len(y_test)
    error.append(err1)
    r=r2_score(y_test, predictions)
    r2.append(r)
    print('Minimum numbers of leafs: ', err1)
    print('R SQUARED: ',r)
    
    
    
print(tree.decision_path(X_test))
df=pd.DataFrame({'Actual':y_test, 'Predicted':np.round(predictions)})
print(df)

err = np.sum(abs(y_test - predictions))/len(y_test)
print('Mean absolute error for test =', err)

err1 = np.sum(abs(y_test - predictions)**2)/len(y_test)
print('Mean square error for test =', err1)

mse = np.sum(abs(y_test - predictions)**2)
print('MSE =', mse)

rmse=np.sqrt(mse/len(y_test))
print('RMSE Root mean squared error =', rmse)  

ssr = np.sum((predictions - y_test)**2)
print('ssr =', ssr)

sst = np.sum((y_test - np.mean(y_test))**2)
print('sst =', sst)

r2_score1 = 1 - (ssr/sst)
print('r2_score =', r2_score1)

plt.figure()
plt.plot(y_test,'.-g', label='Actual Data')
plt.plot(predictions,'.-r', label='Predicted Data')
plt.legend()

plt.figure()
plt.xlabel('Minimum numbers of leafs')
plt.ylabel('Mean absolute error')
plt.plot(error, color='blue')
plt.show()

plt.figure()
plt.xlabel('Minimum numbers of leafs')
plt.ylabel('R SQUARED')
plt.plot(r2, color='blue')
plt.show()










