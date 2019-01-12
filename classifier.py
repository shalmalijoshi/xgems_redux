# Class definition that takes
# the required classifer and
# data, corresponding parameters
# and learns classifier and saves
# results.

import matplotlib.pyplot as plt
import numpy as np
import os,sys
import time
import argparse
import tensorflow as tf
import importlib
import urllib
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
BATCH_SIZE=64

class classifier(object):
    def __init__(self, estimator, data, modelpath, resultpath):
        self.model = estimator
        self.data = data
        self.n_samples=self.data.n_samples
        self.modelpath=modelpath
        self.resultpath=resultpath
        self.x_dim = self.model.x_dim
        self.n_labels = self.model.n_labels
        self.x = tf.placeholder(tf.float32,[None] + self.data.shape)
        self.target=tf.placeholder(tf.float32,[None,self.n_labels],name='y_')

        self.modelop = self.model(self.x,self.target)
        self.modeloptimizer = self.model.train(0.1)
        self.corrpred = tf.equal(tf.argmax(self.modelop,1),tf.argmax(self.target,1))
        #print('modelop',self.modelop.shape,self.target.shape)
        #self.corrpred=tf.equal(tf.to_int32(tf.greater(self.modelop,0.5)),tf.to_int32(self.target))
        self.pred_labels = tf.argmax(self.modelop,1)
        self.true_labels = tf.argmax(self.target,1)
        self.accuracy = tf.reduce_mean(tf.cast(self.corrpred,tf.float32))
        
        self.f1 = tf.contrib.metrics.f1_score(self.true_labels,self.pred_labels)
        #self.fp = tf.metrics.false_positives(self.true_labels,self.modelop)
        #self.fn = tf.metrics.false_negatives(self.true_labels,self.modelop)
        #self.tp = tf.metrics.true_positives(self.true_labels,self.modelop)
        #self.tn = tf.metrics.true_negatives(self.true_labels,self.modelop)

        all_vars = tf.global_variables()
        var_dict = {v.op.name: v for v in all_vars}
        print('classifier variable names:',var_dict)
        self.saver = tf.train.Saver(var_dict)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self,batch_size=64,epochs=50):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        start_time = time.time()
        n_batches=int(np.floor(self.n_samples/batch_size))
        xtest,ytest = self.data.get_test()
        for t in range(epochs):
            for i in range(n_batches):
                xs,ys = self.data(batch_size,i)
                #print(ys)
                _,y_pred,loss,acc = self.sess.run([self.modeloptimizer,self.model.y,self.model.crossentropy,self.accuracy],feed_dict={self.x: xs, self.target:ys})
                #if i%1==0:
                    #print('Iter [%8d] Time [%5.4f] loss [%.4f]' % \
                #(i,time.time()-start_time,loss))

            #test_acc,f1,fp,fn,tp,tn = self.sess.run([self.accuracy,self.f1,self.fp,self.fn,self.tp,self.tn],feed_dict={self.x:xtest,self.target:ytest})
            test_acc,f1 = self.sess.run([self.accuracy,self.f1],feed_dict={self.x:xtest,self.target:ytest})
            if t % 1 == 0:
                #print('Epoch [%8d] Time [%5.4f] loss [%.4f] test acc [%.4f] f1 [%.4f] fp [%.4f] fn [%.4f]  tp [%.4f] tn [%.4f]' % \
                #(t,time.time()-start_time,loss,test_acc,f1,fp,fn,tp,tn))
                #print('Epoch ',t,'Time ',time.time()-start_time,' loss',loss,' test acc',test_acc,'f1 ',fp,'fp ',fp,'fn ',fn,' tp ',tp,'tn',tn)
                print('Epoch ',t,'Time ',time.time()-start_time,' loss',loss,' test acc',test_acc,'f1 ',f1)

    def test(self):
        xtest, ytest = self.data.get_test()
        acc = self.sess.run(self.accuracy,feed_dict={self.x:xtest,self.target:ytest})
        print('Accuracy: [%.4f]' % (acc))
        with open(self.resultpath,'w') as res:
            res.write('Accuracy: [%.4f]' % (acc))

    def save_model(self):
        self.saver.save(self.sess,self.modelpath)

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='defaultCredit')
    parser.add_argument('--estimator', type=str, default='softmax')
    parser.add_argument('--reg', type=str,default=0.1)
    parser.add_argument('--modelpath', type=str,default='/scratch/gobi1/shalmali/defaultCredit/classifier')
    parser.add_argument('--resultpath', type=str,default='/scratch/gobi1/shalmali/defaultCredit/classifier')
    parser.add_argument('--gpus', type=str, default='0')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data +'.'+args.estimator)
    
    reg_penalty = float(args.reg)
    print(reg_penalty)
    estobj = model.estimator(reg_penalty=reg_penalty)
    estobj.name = args.data + '/' + args.estimator
    #estobj.set_params(args.params)
    
    dataobj = data.DataSampler()

    modelpath=args.modelpath
    resultpath=args.resultpath

    # Using only defaults for now
    modelpath=modelpath+'/' + args.estimator + '_' + str(reg_penalty)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    resultpath=modelpath

    modelpath += '/' + args.estimator + '.classifier'
    
    resultpath += '/' + args.estimator + '.res'
 
    classifierObj = classifier(estimator=estobj, data=dataobj, \
                    modelpath=modelpath, resultpath=resultpath)
    classifierObj.train()
    classifierObj.test()
    classifierObj.save_model()

