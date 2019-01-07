import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data_path='~/data/credit_processed.csv'
data_path_hivae='~/data/credit_processed_hivae.csv'
dest_path='/h/shalmali/data/defaultCredit/'
data=np.asarray(pd.read_csv(data_path))
X=data[:,1:]
y=data[:,0]

data_hivae=np.asarray(pd.read_csv(data_path_hivae,header=None))
X_hivae=data_hivae[:,1:]

indices=np.arange(X.shape[0])
xtrain,xtest,ytrain,ytest,idx_train,idx_test = train_test_split(X,y,indices,test_size=0.2,random_state=31)

xtrain_hivae = X_hivae[idx_train,:]
xtest_hivae = X_hivae[idx_test,:]

np.savetxt(dest_path+'xtrain.csv',xtrain,fmt='%d',delimiter=',')
np.savetxt(dest_path+'ytrain.csv',ytrain,fmt='%d',delimiter=',')

np.savetxt(dest_path+'xtrain_hivae.csv',xtrain_hivae,fmt='%d',delimiter=',')
np.savetxt(dest_path+'xtest_hivae.csv',xtest_hivae,fmt='%d',delimiter=',')

np.savetxt(dest_path+'xtest.csv',xtest,fmt='%d',delimiter=',')
np.savetxt(dest_path+'ytest.csv',ytest,fmt='%d',delimiter=',')

kf = KFold(n_splits=5)
k=0
for train_index,val_index in kf.split(xtrain):
    xtrain0,xval0 = xtrain[train_index], xtrain[val_index]
    ytrain0,yval0 = ytrain[train_index], ytrain[val_index]
    np.savetxt(dest_path+'xtrain'+str(k)+'.csv',xtrain0,fmt='%d',delimiter=',')
    np.savetxt(dest_path+'ytrain'+str(k)+'.csv',ytrain0,fmt='%d',delimiter=',')
    np.savetxt(dest_path+'xval'+str(k)+'.csv',xval0,fmt='%d',delimiter=',')
    np.savetxt(dest_path+'yval'+str(k)+'.csv',yval0,fmt='%d',delimiter=',')
    k+=1
