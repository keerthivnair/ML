import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression

bc =datasets.load_breast_cancer()
# df=pd.DataFrame(data=bc.data, columns=bc.feature_names)
# df['target']=bc.target 
# print(df.head)
# print(df.info)
X,y=bc.data,bc.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123) 

clf=LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)  

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc=accuracy(y_pred,y_test)

print(acc)
 