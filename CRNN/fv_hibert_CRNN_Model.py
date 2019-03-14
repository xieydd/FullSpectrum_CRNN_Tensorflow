# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:07:33 2019

E-mail: xieydd@gmail.com

@author: xieydd

@description: Full vector specturm hilbert 2-D CNN model
"""

import os  
import numpy as np  
import tensorflow as tf   
import pandas as pd  
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
import keras.backend.tensorflow_backend as KTF

class FV_Hilbert_CNNConfig(object):
    """CNN配置参数"""
    kernel_size1 = 5    # Convolution layer 1 kernel size
    kernel_size2 = 6    # Convolution layer 2 kernel size
    kernel_size3 = 5    # Convolution layer 3 kernel size
    kernel_size4 = 4    # Convolution layer 3 kernel size
    channels1 = 1       # conv1 chennels
    kernel_num1 = 8     # conv1 kernel nums
    kernel_num2 = 16    # conv2 kernel nums
    kernel_num3 = 32    # conv3 kernel nums
    kernel_num4 = 64    # conv4 kernel nums
    num_fc1 = 10       # full connection 1 nums
    num_fc2 = 3         # full connection 2 nums

    signal_length = 4096    # signal size
    signal_classes = 3     # signal fault kind nums
    signal_reshaped = 1024   # Input signal reshaped size

    batch_size = 64         # batch-size
    num_epochs = 1000         # epochs
    #learning_rate = 1e-4    # learning_rate
    drop_out = 0.8          # dropout params

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 50      # 每多少轮存入tensorboard
    train = True             # 是否使用DropOut
    Gpu_used = 0.8           # Gpu使用量
    moving_average_decay = 0.99 #滑动平均衰减率
    learning_rate_base = 0.001 #学习率
    learning_rate_decay = 0.95  #学习率衰减率

    cell_num = 100
    n_step = 1024
    n_input = 1


class FV_Hilbert_CNN(object):
    def __init__(self,config):
        self.config = config
        self.input_x = tf.placeholder(tf.float32,shape=[None,self.config.signal_reshaped,1],name='input_x')
        self.input_y = tf.placeholder(tf.float32,shape=[None,self.config.signal_classes],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.iter = tf.placeholder(tf.int32)
        self.cnn()

    def batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
       exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
       bnepsilon = 1e-5
       if convolutional:
          mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
       else:
          mean, variance = tf.nn.moments(Ylogits, [0])
       update_moving_averages = exp_moving_avg.apply([mean, variance])
       m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
       v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
       Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
       return Ybn, update_moving_averages 

    def compatible_convolutional_noise_shape(self,Y):
       noiseshape = tf.shape(Y)
       noiseshape = noiseshape * tf.constant([1,0,1]) + tf.constant([0,1,0])
       return noiseshape

    def cnn(self):
        KTF.set_session(FV_Hilbert_CNN.get_session(0.7))  # using 80% of total GPU Memory
        #TODOwith tf.device('/cpu:0'):
        regularizer = tf.contrib.layers.l2_regularizer(self.config.learning_rate_base)

        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weights",[self.config.kernel_size1,self.config.channels1,self.config.kernel_num1],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("biases",[self.config.kernel_num1],initializer=tf.constant_initializer(0.1))
            conv1 = tf.nn.conv1d(self.input_x,conv1_weights,stride=2,padding = 'SAME')
            bn1, update_ema1 = self.batchnorm(conv1, tf.convert_to_tensor(self.config.train), self.iter, conv1_biases, convolutional=True)
            relu1 = tf.nn.relu(bn1)
            Y1 = tf.nn.dropout(relu1, self.keep_prob, self.compatible_convolutional_noise_shape(relu1))

        with tf.variable_scope('layer2-conv2'):
            conv2_weights = tf.get_variable("weights",[self.config.kernel_size2,self.config.kernel_num1,self.config.kernel_num2],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("biases",[self.config.kernel_num2],initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.conv1d(Y1,conv2_weights,stride=2,padding="SAME")
            bn2, update_ema2 = self.batchnorm(conv2, tf.convert_to_tensor(self.config.train), self.iter, conv2_biases, convolutional=True)
            relu2 = tf.nn.relu(bn2)
            Y2 = tf.nn.dropout(relu2, self.keep_prob, self.compatible_convolutional_noise_shape(relu2))
        
        with tf.variable_scope('layer3-conv3'):
            conv3_weights = tf.get_variable("weights",[self.config.kernel_size3,self.config.kernel_num2,self.config.kernel_num3],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("biases",[self.config.kernel_num3],initializer=tf.constant_initializer(0.1))
            conv3 = tf.nn.conv1d(Y2,conv3_weights,stride=2,padding="SAME")
            bn3, update_ema3 = self.batchnorm(conv3, tf.convert_to_tensor(self.config.train), self.iter, conv3_biases, convolutional=True)
            relu3 = tf.nn.relu(bn3)
            Y3 = tf.nn.dropout(relu3, self.keep_prob, self.compatible_convolutional_noise_shape(relu3))

        with tf.variable_scope('layer4-conv4'):
            conv4_weights = tf.get_variable("weights",[self.config.kernel_size4,self.config.kernel_num3,self.config.kernel_num4],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("biases",[self.config.kernel_num4],initializer=tf.constant_initializer(0.1))
            conv4 = tf.nn.conv1d(Y3,conv4_weights,stride=2,padding="SAME")
            bn4, update_ema4 = self.batchnorm(conv4, tf.convert_to_tensor(self.config.train), self.iter, conv4_biases, convolutional=True)
            relu4 = tf.nn.relu(bn4)
            Y4 = tf.nn.dropout(relu4, self.keep_prob, self.compatible_convolutional_noise_shape(relu4))
 
            YY = tf.reshape(Y4,[-1,self.config.signal_length])

        with tf.variable_scope("layer5-fc1"):
            #fc1_weights = tf.Variable("weights",tf.truncated_normal_initializer([self.config.signal_length,self.config.num_fc1],stddev=0.1))
            fc1_weights = tf.get_variable("weights",[self.config.signal_length,self.config.num_fc1],initializer = tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses",regularizer(fc1_weights))
            #fc1_biases = tf.Variable("biases",tf.constant(0.1, tf.float32, [self.config.num_fc1]))
            fc1_biases = tf.get_variable("biases",[self.config.num_fc1],initializer=tf.constant_initializer(0.1))
            Yfc1 = tf.matmul(YY, fc1_weights)
            fc1, update_ema5 = self.batchnorm(Yfc1, tf.convert_to_tensor(self.config.train), self.iter, fc1_biases)
            if self.config.train: Yfcr = tf.nn.relu(fc1)
            Y5 = tf.nn.dropout(Yfcr, self.keep_prob)

        with tf.variable_scope("lstm"):
            x_input = tf.reshape(self.input_x,[-1,self.config.n_input])  
            layer_input = tf.layers.dense(x_input, self.config.cell_num) 
            self.layer_input = tf.reshape(layer_input, [-1, self.config.n_step, self.config.cell_num])
            self.cell = tf.contrib.rnn.BasicLSTMCell(self.config.cell_num)
            self.init_state = self.cell.zero_state(self.config.batch_size, dtype=tf.float32)
            self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.layer_input, initial_state=self.init_state, time_major=False)
            self.layer_out = tf.layers.dense(self.state[1], self.config.signal_classes)
            

        with tf.variable_scope("layer6-fc2"):
            #fc2_weights = tf.Variable("weights",tf.truncated_normal([self.config.num_fc1,self.config.num_fc2],stddev=0.1))
            fc2_weights = tf.get_variable("weights",[self.config.num_fc1,self.config.num_fc2],initializer = tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses",regularizer(fc2_weights))
            #fc2_biases = tf.Variable("biases",tf.constant(0.1, yf.float32, [self.config.num_fc2]))
            fc2_biases = tf.get_variable("biases",[self.config.num_fc2],initializer=tf.constant_initializer(0.1))
            self.logit = tf.matmul(fc1,fc2_weights)+fc2_biases
            self.logit_1 = 0.5*tf.nn.softmax(self.logit) + 0.5*self.layer_out
            self.y_pred_cls = tf.argmax(self.logit_1,1)
        self.update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5)

        #使用滑动平均输出
        global_step =  tf.Variable(0.0,trainable=True)
        with tf.name_scope('moving_average'):
            variable_average = tf.train.ExponentialMovingAverage(self.config.moving_average_decay,global_step)
            variable_average_op = variable_average.apply(tf.trainable_variables())

        with tf.name_scope('loss_function'):
            cross_entrypy_mean  = 100*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logit_1))
            self.loss = cross_entrypy_mean + tf.add_n(tf.get_collection('losses'))


        with tf.name_scope('train_step_optimize'):
            self.learning_rate = tf.train.exponential_decay(self.config.learning_rate_base,global_step,1000,self.config.learning_rate_decay,staircase=True)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            with tf.control_dependencies([self.train_step,variable_average_op]):
                self.config.train_op = tf.no_op(name='train')

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    ######################################
    # TODO: 设置GPU使用量
    '''
    在Tensorflow中可以直接通过
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True #设置最小GPU使用量
    session = tf.Session(config=config)
    '''
    #####################################
    def get_session(gpu_fraction=0.6):
        """
        This function is to allocate GPU memory a specific fraction
        Assume that you have 6GB of GPU memory and want to allocate ~2GB
        """
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

