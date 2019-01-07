#Main explainer class that implements
#Main explainer class that implements
# all necessary functions for generating
# adversarial examples

import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
import os,sys
import time
import argparse
import tensorflow as tf
import importlib
from tensorflow.python.framework import ops
sys.path.append(os.path.join(os.path.dirname(__file__),'hi_vae'))
#from hi_vae import VAE_functions
ops.reset_default_graph()

BATCH_SIZE=1

ftol = 1e-15
gtol = 1e-5


class explainer(object):
    def __init__(self,generator,data,x_sampler,classifierpath,generatorpath,classobj_path,genobj_path,classname,genname,class_vsname,gen_vsname,resultpath,classifier=None,mode='explainer',reg=1.):
        self.generator = generator
        self.classifier = classifier
        self.x_sampler = x_sampler
        self.classifierpath=classifierpath
        self.generatorpath=generatorpath
        self.resultpath=resultpath
        self.data = data
        self.mode = mode
        self.lambdaf = reg

        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.classobj = self.classifier(classobj_path,class_vsname)
        self.genobj = self.generator(genobj_path,gen_vsname)
        self.x_dim = self.x_sampler.shape
        self.n_labels = self.x_sampler.n_labels
        self.z_dim = 10
 
        with tf.variable_scope('explainer'):
            self.z = tf.get_variable('z',shape=[BATCH_SIZE,self.z_dim],initializer=tf.zeros_initializer)
       
       #placeholders for sample
        self.x_star = tf.placeholder(tf.float32,[BATCH_SIZE]+self.x_dim,name='x_star')
        self.y_star = tf.placeholder(tf.float32,shape=[BATCH_SIZE,self.n_labels],name='y_star')
        self.y_tar = tf.placeholder(tf.float32,shape=[BATCH_SIZE,self.n_labels],name='y_tar')
        self.lambda_reg = tf.placeholder(tf.float32, shape=(),name='lambda')

        #_,self.bz_op,_,_,_ = self.genobj.autoencoder(self.x_star,self.x_star,np.prod(self.x_dim),self.z_dim,n_hidden=10,keep_prob=1)
        self.bz_op,_ = self.genobj.gaussian_MLP_encoder(self.x_star,n_output=self.z_dim,n_hidden=100,keep_prob=1)

        self.f_ = self.genobj.decoder(self.z,np.prod(self.x_dim),n_hidden=100,reuse=False)
        self.ypred = self.classobj(self.f_,self.y_star)
        self.ypred_orig = self.classobj(self.x_star,self.y_star,reuse=True)

        self.reg = tf.square(self.x_star - self.f_)
        self.reg = self.lambda_reg*tf.reduce_sum(self.reg)
        self.loss1 = -tf.reduce_sum(self.y_tar*tf.log(self.ypred))
        self.loss = self.loss1 + self.reg

        self.op_grad = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.all_vars = tf.trainable_variables()
        c_vars = [self.all_vars[0]]
        print('variable:', c_vars) 
        self.gvs = tf.gradients(self.loss,c_vars)
        self.grad_map = self.op_grad.apply_gradients([(g,v) for g,v in zip(self.gvs,c_vars)])
 
        self.sess = self.classifier.load_model(self.sess,classifierpath,classname,class_vsname)
        self.sess = self.generator.load_model(self.sess,generatorpath,genname,gen_vsname)

        n_defaults=0.
        n_defaults_correct=0.
        n_match=0.
        n_total=0.
        for n in range(0,self.x_sampler.n_samples):
            x_in,y_star = self.x_sampler(batch_size=BATCH_SIZE,batch_index=n)
            #x_in,y_star = self.x_sampler.get_test_i(n)
            x_star=x_in
            #print(y_star.shape)
            y_star_label=np.argmax(y_star,1)

            #1: no default,0:default
            #print('x_star min max: ',np.min(x_star,1),np.max(x_star,1))
            bz = self.sess.run(self.bz_op,feed_dict={self.x_star:x_star,self.y_star:y_star})
            self.z.load(bz,session=self.sess)
            
            x_gen0,ypred0 = self.sess.run([self.f_, self.ypred],feed_dict={self.x_star:x_star,self.y_star:y_star})
 
            ypredorig = self.sess.run(self.ypred_orig,feed_dict={self.x_star:x_star,self.y_star:y_star})
            
            #print('*** predicted label for original sample:',ypredorig[0,0])

            y_tar = np.reshape(np.asarray([0,0]),[BATCH_SIZE,self.n_labels])
            y_tar[0,1-y_star_label]=1.


            if np.argmax(ypredorig[0,:])==0:
                n_total+=1
                if np.argmax(ypredorig[0,:])==np.argmax(ypred0[0,:]):
                    n_match+=1
                print('********* correct reconstruction:', n_match/float(n_total+1))
            
            #continue
           
            str_n=''
            if np.argmax(ypredorig[0,:])==np.argmax(y_star[0,:]):
                print('Classifier classified correctly')
                str_n += '_' + '1'
            else:
                print('Classifier classified incorrectly, skipping sample')
                str_n += '_' + '0'
                continue

            if y_star_label==1: #person does not default
                continue
            else:
                print('*** sample defaults on credit')
                n_defaults+=1

            print('Decoder at init:', np.min(x_gen0), np.max(x_gen0), 'label:' ,ypred0,'original pred:', ypredorig)

            if np.argmax(ypred0[0,:])==np.argmax(y_star[0,:]):
                print('Reconstruction classified correctly')
                str_n = str(n) + '_' + '1'
            else:
                print('Reconstruction classified incorrectly')
                str_n = str(n) + '_' + '0'
                continue

            print('Sample [%d] True label [%d] Pred label [%d] Pred label gen [%d] Target label [%d]'% (n,y_star_label,np.argmax(ypredorig),np.argmax(ypred0) ,np.argmax(y_tar)))
            
            dest_path ='/scratch/gobi1/shalmali/defaultCredit/explainer/logs/{}/{}/n{}'.format('recourse','no_reg',str_n) 
            dir_path = dest_path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            filename=dest_path+'/input.txt'
            np.savetxt(filename,x_in)

            file_log = open(dest_path + '/confidence_prediction.txt','w+')
            filename =  (dest_path +'/t{}_pred{}.txt').format(1,np.argmax(ypred0[0,:]))
            np.savetxt(filename,x_gen0)

            file_log.write('0' + ',' + str(ypredorig[0,0]) + ',' + str(ypredorig[0,1]) +'\n')
            file_log.write('1 ' + ',' + str(ypred0[0,0]) + ',' + str(ypred0[0,1]) + '\n')

            print('Sample [%d] True label [%d] Pred label [%d] Target label [%d]'% (n,y_star_label,np.argmax(ypredorig[0,:]), np.argmax(y_tar[0,:])))

            #iter_max = 20000
            iter_max = 1
            break_t=iter_max
            loss_old = 0.
            for t in range(0,iter_max):
                
                _, loss1, reg, gvs, y_pred, f_in = self.sess.run([self.grad_map,self.loss1,self.reg,self.gvs,self.ypred,self.f_],feed_dict={self.x_star:x_star,self.y_star:y_star,self.y_tar:y_tar,self.lambda_reg:0.0001})

                if np.argmax(y_pred[0,:])==np.argmax(y_tar[0,:]) and break_t==iter_max:
                    print('predicted change in label', y_pred[0,0], t, y_tar[0,0])
                    break_t = t+10

                if np.linalg.norm(gvs[0])<gtol and t>=5000:
                    print('exiting since gradient too small: %.6f' % (np.linalg.norm(gvs[0])))
                    break

                if abs(loss1+reg - loss_old)<ftol and t>=5000:
                    print('exiting since change in loss too small: %.6f' % (abs(loss1+reg-loss_old)))
                    break

                loss_old = loss1+reg

                if t %100 == 0 or t%1==0:
                    if t%100==0:
                        print('Iter [%8d] loss_entropy [%.4f] reg [%.4f] gradient Norm [%.4f] predicted_label [%.4f]'% (t,loss1,reg,np.linalg.norm(gvs[0]), np.argmax(y_pred[0,:])))
                        #print('ypred',y_pred[0,:],'y_tar', y_tar[0,:])
                        filename =  (dest_path + '/t{}_pred{}.txt').format(t+2,int(y_pred[0,0]>0.5))
                        np.savetxt(filename,f_in)
                        file_log.write(str(t+2) + ',' + str(y_pred[0,0]) + ',' + str(y_pred[0,1]) + '\n')

                if t==break_t:
                    break

            file_log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(' ',allow_abbrev=False)
    parser.add_argument('--data', type=str, default='defaultCredit')
    parser.add_argument('--generator',type=str,default='reg_vae')
    parser.add_argument('--classifier',type=str,default='softmax')
    parser.add_argument('--mode',type=str,default='explainer')
    parser.add_argument('--resultpath',type=str,default='/scratch/gobi1/shalmali/defaultCredit/explainer/')
    parser.add_argument('--gpus',type=str,default='0')
    parser.add_argument('--l',type=float,default=0.0)
    #parser.add_argument('--ckpt_iter',type=int,default=40000)
    #parser.add_argument('--con_ckpt_iter',type=int,default=80000)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    #iter_str = str(args.ckpt_iter)
    #condetect_iter_str = str(args.con_ckpt_iter)

    data = importlib.import_module('data_utils')
    model = importlib.import_module('explainer_utils')
    #con_detector = importlib.import_module(args.data + '.entities_celebA_vae_resnet_bn_gender',package='..')

    xs = data.DataSampler()
    generator = model.generator(args.data + '/' + args.generator)

    if args.classifier==None:
        raise ValueError('Classifier cannot be none')
    else:
        classifier = model.classifier(args.data + '/' + args.classifier)
        classifierpath='/scratch/gobi1/shalmali/'+args.data + '/classifier/'
        classobj_path= args.classifier 
        class_vsname = args.data  + '/' + args.classifier


    generatorpath='/scratch/gobi1/shalmali/' +args.data + '/' + args.generator + '/'
    #genobj_path = args.data + '.hi_vae.model_HIVAE_inputDropout'
    genobj_path = args.generator + '.vae'
    gen_vsname = args.data + '/' + args.generator

    explainerobj = explainer(generator,data,xs,classifierpath,\
                   generatorpath,classobj_path,genobj_path,args.classifier,\
                   args.generator,class_vsname,gen_vsname,args.resultpath,\
                   classifier,args.mode,args.l)
