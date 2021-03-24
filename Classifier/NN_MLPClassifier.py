#!!!!!!!with the help of laboratory code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


balance= pd.read_csv('ppbalance-scale.data.csv', header=None)
#balance.head()
X=balance.as_matrix(columns=[1,2,3,4, ])
#X = balance.drop([0],axis=1)

y = balance[0]
t=['B']
x=[[1,1,1,1]]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#mlp = MLPClassifier(activation='relu',verbose=True,hidden_layer_sizes=(5,5,5),max_iter=5000)
#mlp = MLPClassifier(activation='relu',verbose=1,hidden_layer_sizes=(5,5,5),max_iter=5000,learning_rate='adaptive',solver='sgd',learning_rate_init = 5)
mlp = MLPClassifier(verbose=1,hidden_layer_sizes=(13,13,13),max_iter=5000)
mlp.fit(X_train,y_train)

print('For train taget',y_train,'we get',mlp.predict(X_train))
print('For taget ', t, 'we get ', mlp.predict(x))
predictions = mlp.predict(X_test)
err=mlp.loss_curve_
print('Test set ',y_test)
print("Network's set ",predictions)
print(confusion_matrix(y_test,predictions))
plt.plot(err)
plt.title('Error')
plt.xlabel('Iterations')
plt.ylabel('Loss')
acc=100 * np.sum((y_test == predictions )/len( y_test ))
print('Accuracy for test = ', acc)
acc1=accuracy_score(y_test, predictions)*100
print('Another accuracy for test = ', acc1)
print(classification_report(y_test,predictions))