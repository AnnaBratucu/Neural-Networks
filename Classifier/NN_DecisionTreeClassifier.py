#!!!!!!!with the help of laboratory code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#gini method: measures impurity of node ( = when all of its records belong to the same class = leaf node)

balance= pd.read_csv('ppbalance-scale.data.csv',header=None)

accs = []
accs_training = []
#print(balance.head())
X=balance.as_matrix(columns=[1,2,3,4, ])
y = balance[0]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.05) #95% training, 5% test

#tree = DecisionTreeClassifier(min_samples_split=3, splitter='random',criterion='gini')
#tree = DecisionTreeClassifier(min_samples_leaf=30,max_leaf_nodes=70,min_samples_split=3,splitter='random',criterion='gini')


for i in range(1,15):
    tree = DecisionTreeClassifier(max_depth=i,random_state=0,min_samples_split=10,splitter='random')
    tree.fit(X_train,y_train) #Build a decision tree classifier from the training set (X, y).
    print('max depth = ', i)
    predictions = tree.predict(X_test)
    acc=100 * np.sum((y_test == predictions )/len( y_test ))
    predictions_train = tree.predict(X_train)
    acc_train=accuracy_score(y_train, predictions_train)*100
    accs_training.append(acc_train)
    accs.append(acc)
    #how often is the neural network (classifier) correct
    print('Finished classifying. Accuracy for test =', acc)
    
 

plt.xlabel('Size of tree')
plt.ylabel('Accuracy')
plt.plot(accs, color='blue',label='Accuracy for test set')
plt.plot(accs_training, color='red',label='Accuracy for training set')
plt.legend(loc=4, prop={'size' : 8})
plt.show()
acc1=accuracy_score(y_test, predictions)*100
print('Another accuracy for test =', acc1)
print(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
print(df)  
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(tree.decision_path(X_test))

    
    





