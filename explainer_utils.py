import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import importlib

class classifier(object):
    def __init__(self,name):
        self.name = name

    def __call__(self,wrapperobjpath,scopename):
        print('classifier',wrapperobjpath)
        model = importlib.import_module(wrapperobjpath)
        classifierobj = model.estimator()
        classifierobj.name = scopename
        return classifierobj

    def load_model(self,sess,tfmodelpath,modelname,scopename):
        print('classifier loading model from:',tfmodelpath+modelname+'.classifier.meta')
        #saver = tf.train.import_meta_graph(tfmodelpath + modelname + '.classifier.meta')
        all_vars = tf.global_variables()
        #print("all vars", all_vars) 
        class_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scopename)
        #print("class vars",class_vars)
        saver = tf.train.Saver(class_vars)
        saver.restore(sess,tf.train.latest_checkpoint(tfmodelpath))
        print('classifier: restored saved model at',tfmodelpath)
        return sess

    # Returns variables of the classifier
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class generator(object):
    def __init__(self,name):
        self.name = name

    def __call__(self,wrapperobjpath,scopename):
        print('generator',wrapperobjpath)
        model = importlib.import_module(wrapperobjpath)
        genobj = model
        return genobj

    def load_model(self,sess,tfmodelpath,modelname,scopename):
        print('generator loading model from:',tfmodelpath)
        all_vars = tf.global_variables()
        class_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scopename)
        saver = tf.train.Saver(class_vars)
        saver.restore(sess,tf.train.latest_checkpoint(tfmodelpath))
        print('generator: restored saved model at',tfmodelpath)
        return sess

    # Returns variables of the generator
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
