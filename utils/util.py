# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:30:20 2019

E-mail: xieydd@gmail.com

@author: xieydd

@description:
"""
import scipy.io as sio
import numpy as np
import time
import os
import sys
sys.path.append(os.path.abspath("../"))
from FullSpectrum import fv_hibert

def mat2pb(abspath,output):
    """
    *Args: abspath - mat abspath xxx/xxx/xxx/1.mat
           output - npy path xxx/xxx/xxx/
    *output: null
    """
    name = abspath.split('/')[-1].split('.')[0]
    data = sio.loadmat(abspath)[name]
    print(data.shape)
    np.save(output+name,data)

def get_time(start_time):
    end_time = time.time()
    return timedelta(seconds=int(round(end_time-start_time)))

# Read the bear data
def readfile(path):
    files = os.listdir(path)
    E = []
    nums = 0
    for filename in files:
        E.append(sio.loadmat(path+"/"+filename)[filename.split('.',-1)[0]])
        nums += 1
        print(filename.split('.',-1)[0])
    return E[0],E[1],E[2],E[3],E[4],E[5]

def loadmat(abs_filename):
    return sio.loadmat(abs_filename)[abs_filename.split('/')[-1].split('.')[0]]

def test(argv):
    e3,e4,e5,e6,e7,e8 = util.readfile(argv[0])
    del e3,e4,e7,e8
    gc.collect()
    shape = e5.shape
    FvHibert = fv_hibert.FvHibert(shape[1],shape[1])
    e5_1 = e5[0,:]
    e6_1 = e6[0,:]
    del e5,e6
    gc.collect()
    x_amplitude_envelope_inner,y_amplitude_envelope_inner = FvHibert.hibert(e5_1,e6_1)  
    #x_amplitude_envelope_roll,y_amplitude_envelope_roll = FvHibert.hibert(e7,e8)  
    #Plot = plot.Plot(shape[1],shape[1])
    #Plot.plot_amplitude_envelope(e5_1,e6_1)
    RL_inner = FvHibert.fv_hibert(x_amplitude_envelope_inner,y_amplitude_envelope_inner)
    Plot = plot.Plot(RL_inner.shape[0],RL_inner.shape[0]*2)
    Plot.plot_fv_hibert(RL_inner)
    print(RL_inner.shape)

def mat2npy(argv):
    """
    Get the full vector specturm and compress the 
    memory usage by convert the float64 to float32
    """
    x = util.loadmat('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/E5.mat') 
    y = util.loadmat('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/E6.mat') 
    shape =  x.shape
    FvHibert = fv_hibert.FvHibert(shape[1],shape[1])
    output = np.zeros((shape[0],int(shape[1]/2)),dtype=np.float32)
    for i in range(shape[0]):
       x_amplitude_envelope, y_amplitude_envelope = FvHibert.hibert(x[i,:],y[i,:])
       RL = FvHibert.fv_hibert(x_amplitude_envelope, y_amplitude_envelope)       
       RL = RL.astype(np.float32)
       output[i,:] = RL 
    outputs = output.reshape(shape[0]*10,-1)
    print(outputs.shape)
    np.save('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/inner',outputs) 
   
def loaddata(argv):
    sample = np.load("J:/FullSpectrum_CRNN_Tensorflow/IMS_data/inner.npy")
    roll = np.load("J:/FullSpectrum_CRNN_Tensorflow/IMS_data/roll.npy")
    inner = np.load("J:/FullSpectrum_CRNN_Tensorflow/IMS_data/inner.npy")
    all_data = np.concatenate((sample,roll,inner))
    print(all_data[21560,0])
    print(all_data.shape)
    print(roll[0,0])
    np.save('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/all', all_data)

def createlabel(argv):
    label = np.zeros((64680,3), dtype=np.float32)
    label[0:39559,0] = 1
    label[39560:43119,1] = 1
    label[43120:59119,0] = 1
    label[59120:64680,2] = 1 
    print(label[59120,:])
    np.save('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/label', label)

#def create_label():
if __name__ == '__main__':
    #mat2pb('J:/FullSpectrum_CRNN_Tensorflow/IMS_data/E3.mat','J:/FullSpectrum_CRNN_Tensorflow/IMS_data/')
    #createlabel(sys.argv[1:])
    loaddata(sys.argv[1:])
