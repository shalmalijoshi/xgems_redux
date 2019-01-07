import numpy as np
import scipy
import os, sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
import pandas as pd

BATCH_SIZE=1
class DataSampler(object):
    def __init__(self,one_hot=True):
        self.n_labels = 2
        self.xtrain=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtrain.csv',header=None))
        self.xtest=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtest.csv',header=None))
        self.ytrain=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/ytrain.csv',header=None))
        print(np.where(self.ytrain==1)[0].shape,self.ytrain.shape)
        self.ytest=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/ytest.csv',header=None))
        self.scaler=MinMaxScaler()
        self.scaler.fit(self.xtrain)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.ytrain=self.enc.fit_transform(self.ytrain).toarray()
        self.ytest = self.enc.transform(self.ytest).toarray()
        #print(self.ytest.shape,type(self.ytest))
        self.xtrain = self.scaler.transform(self.xtrain)
        self.xtest = self.scaler.transform(self.xtest)
        self.n_samples,self.n_features=self.xtrain.shape
        self.n_samples_test = self.xtest.shape[0]
        self.shape=[self.n_features]

    def __call__(self, batch_size,batch_index,normalized=True):
        idx_start=batch_index*batch_size
        idx_end=(batch_index+1)*batch_size
        return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:]

    def get_test(self):
        return self.xtest, self.ytest

    def get_test_i(self,i):
        return np.reshape(self.xtest[i,:],[1,-1]),np.reshape(self.ytest[i,:],[1,-1])


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.normal(size=[batch_size, z_dim])


if __name__ == '__main__':
    #xtrain, ytrain, xtest, ytest = load_mnist()
    x_sampler = DataSampler()
    #print('sizes:', np.shape(xtrain), np.shape(ytrain))
    print('n labels:', len(np.where(np.argmax(x_sampler.ytest,1)==1)[0]),len(np.where(np.argmax(x_sampler.ytest,1)==0)[0]))
