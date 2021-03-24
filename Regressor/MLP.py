#!!!with the help from the lab code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

fertility=pd.read_csv('ppfertility.csv',header=None,delimiter=",")

#x=A.as_matrix(columns=[0,1,2,3,4,5,6,7,8,])
x=fertility.iloc[:,0:8].values #features
t = fertility[9] #9th column represents the dependent data (the class)

le = preprocessing.LabelEncoder()
t_encoded=le.fit_transform(t)

x_train, x_test, t_train, t_test= train_test_split(x,t_encoded,test_size=0.2,random_state=42) #80% training, 20% test
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#net= MLPRegressor(activation = 'logistic',solver='lbfgs', max_iter=1000, verbose=True, hidden_layer_sizes=(10,10,10), random_state=1)
net= MLPRegressor(activation = 'logistic', max_iter=1000, verbose=True, hidden_layer_sizes=(10,10,10), random_state=1)

net.fit(x_train,t_train)

y_test=net.predict(x_test)
#y_test=np.round(y_test)
#t_test=np.round(t_test)

err = np.sum(abs(t_test - y_test))/len(t_test)
print('Mean absolute error for test =', err)

err1 = np.sum(abs(t_test - y_test)**2)/len(t_test)
print('Mean square error for test =', err1)

mse = np.sum(abs(t_test - y_test)**2)
print('MSE =', mse)

rmse=np.sqrt(mse/len(t_train))
print('RMSE Root mean squared error =', rmse)


ssr = np.sum((y_test - t_test)**2)
print('ssr =', ssr)

sst = np.sum((t_test - np.mean(t_test))**2)
print('sst =', sst)

r2_score1 = 1 - (ssr/sst)
print('r2_score =', r2_score1)

print('R SQAURED METRICS LIB: ',r2_score(t_test, y_test))
plt.close()

plt.figure()
plt.plot(t_test,'.-g',label='Actual data')
plt.plot(y_test,'.-r',label='Predicted data')
plt.legend()

plt.figure()
plt.plot(net.loss_curve_)





