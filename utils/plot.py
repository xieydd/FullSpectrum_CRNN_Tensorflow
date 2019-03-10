# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:15:21 2019

E-mail: xieydd@gmail.com

@author: xieydd

@description:
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import fftpack
import matplotlib
import scipy.signal as signals
import os
import sys
sys.path.append(os.path.abspath("../FullSpectrum"))
from FullSpectrum import fv_hibert
#import fv_hibert

class Plot():
    def __init__(self,N,fs):
       matplotlib.rcParams['axes.unicode_minus']=False
       plt.rc('font', family='SimHei', size=13)

       self.N = N 
       self.fs = fs
       self.N_half = int(self.N/2)
    def plotxy(self,path_x,path_y):
       """Plot the time domain signal of two channels from x and y dircection
        *Args: path_x, path_y data of the two channels singal data
        NOTICE: Just using half data in dataset, sample frequecy is 20Khz, only show 10000 points
       """
       matplotlib.rcParams['axes.unicode_minus']=False
       plt.rc('font', family='SimHei', size=13)
       E3 = sio.loadmat(path_x)["E3"]
       E4 = sio.loadmat(path_y)["E4"]
       x_data = E3[0,:self.N_half]
       y_data = E4[0,:self.N_half]

       t = np.linspace(0/self.fs,self.N/self.fs,self.N)
       plt.subplot(2,1,1)
       plt.plot(t[0:self.N_half],x_data.flatten())
       plt.title("x方向时域信号 ")
       plt.ylabel('加速度  m/s^2')


       plt.subplot(2,1,2)
       plt.plot(t[0:self.N_half],y_data.flatten())
       plt.title("y方向时域信号 ")
       plt.ylabel('加速度  m/s^2')
       plt.xlabel('时间 t/s')
       plt.show()

    def plotfv(self,path_x,path_y):
      
       """Plot the frequecey domain signal of two channels from x and y dircection and full vector specturm main vibration
        *Args: path_x, path_y data of the two channels singal data
        NOTICE: Just using half data in dataset, sample frequecy is 20Khz, only show 10000 points
       """
       matplotlib.rcParams['axes.unicode_minus']=False
       plt.rc('font', family='SimHei', size=13)
       E3 = sio.loadmat(path_x)["E3"]
       E4 = sio.loadmat(path_y)["E4"]
       x_data = E3[0,:self.N_half]
       y_data = E4[0,:self.N_half]
       xf = np.fft.rfft(E3[0,:self.N_half])/self.N_half
       yf = np.fft.rfft(E4[0,:self.N_half])/self.N_half
       xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
       yfp = 20*np.log10(np.clip(np.abs(yf), 1e-20, 1e100))
       FvHibert = fv_hibert.FvHibert(self.N_half,self.fs)
       fullvector = FvHibert.fv_hibert(x_data, y_data)
       freqs = np.linspace(0, self.N_half, self.N_half/2+1)
       print(freqs.shape)
       plt.subplot(3,1,1)
       plt.plot(freqs,xfp.flatten())
       plt.title("x方向频域信号 ")


       plt.subplot(3,1,2)
       plt.plot(freqs,yfp.flatten())
       plt.title("y方向频域信号 ")


       plt.subplot(3,1,3)
       plt.plot(freqs[2:],fullvector.flatten()[1:])
       plt.title("全矢谱主振矢图 ")
       plt.ylabel('加速度  m/s^2')
       plt.xlabel('频率(Hz)')
       plt.show()

    def plot_amplitude_envelope(self,x,y):
       matplotlib.rcParams['axes.unicode_minus']=False
       plt.rc('font', family='SimHei', size=13)

       #X_Hilbert包络谱
       x_signal = np.array(x).flatten()#展成一维
       t = np.linspace(0/self.fs,self.N/self.fs,self.N)
       x_analytic_signal = signals.hilbert(x_signal)#希尔伯特变换
       x_amplitude_envelope = np.abs(x_analytic_signal)
       #x_amplitude_envelope = np.sqrt(x_analytic_signal**2+x_signal**2)
       x_instantaneous_phase = np.unwrap(np.angle(x_analytic_signal))#瞬时相位
       x_instantaneous_frequency = (np.diff(x_instantaneous_phase)/(2.0*np.pi) * self.fs)#瞬时频率

       x_signal_fft = np.abs(fftpack.fft(x_analytic_signal)/self.N_half)
       f = [i*self.fs/self.N for i in range(self.N_half)]

       fig1 = plt.figure(figsize=(12,12))
       ax0 = fig1.add_subplot(211)
       #ax0.plot(t[0:100], x_signal[0:100], label='signal')
       ax0.plot(t, x_signal, label='signal')
       #ax0.plot(t[0:100], x_amplitude_envelope[0:100], label='envelope')
       ax0.plot(t, x_amplitude_envelope, label='envelope')
       ax0.set_xlabel("时间/s")
       ax0.set_ylabel('加速度m/s^2')
       ax0.set_title('X通道希尔伯特包络')
       ax0.legend()
       #Y_Hilbert包络谱
       y_signal = np.array(y).flatten()#展成一维
       y_analytic_signal = signals.hilbert(y_signal)#希尔伯特变换
       y_amplitude_envelope = np.abs(y_analytic_signal)
       y_instantaneous_phase = np.unwrap(np.angle(y_analytic_signal))#瞬时相位
       y_instantaneous_frequency = (np.diff(y_instantaneous_phase)/(2.0*np.pi) * self.fs)#瞬时频率

       y_signal_fft = np.abs(fftpack.fft(y_analytic_signal)/self.N_half)
       f = [i*self.fs/self.N for i in range(self.N_half)]

       ax1 = fig1.add_subplot(212)
       ax1.plot(t, y_signal, label='signal')
       ax1.plot(t, y_amplitude_envelope, label='envelope')
       ax1.set_xlabel("时间/s")
       ax1.set_ylabel('加速度m/s^2')
       ax1.set_title('Y通道希尔伯特包络')
       ax1.legend()

       fig2 = plt.figure(figsize=(12,12))
       ax0 = fig2.add_subplot(211)
       ax0.plot(t[1:], x_instantaneous_frequency)
       ax0.set_xlabel("时间/s")
       ax0.set_ylabel("瞬时频率/Hz")
       ax0.set_title('X通道瞬时频率')

       ax1 = fig2.add_subplot(212)
       ax1.plot(t[1:], y_instantaneous_frequency)
       ax1.set_xlabel("时间/s")
       ax1.set_ylabel("瞬时频率/Hz")
       ax1.set_title('Y通道瞬时频率')

       fig3 = plt.figure(figsize=(12,12))
       ax0 = fig3.add_subplot(211)
       ax0.plot(f[1:2000],x_signal_fft[1:2000])
       ax0.set_ylim(0.0,0.1)
       ax0.set_xlabel("频率/Hz")
       ax0.set_ylabel("加速度m/s^2")
       ax0.set_title('X通道Hilbert频谱')

       ax1 = fig3.add_subplot(212)
       ax1.plot(f[1:2000],y_signal_fft[1:2000])
       ax1.set_ylim(0.0,0.1)
       ax1.set_xlabel("频率/Hz")
       ax1.set_ylabel("加速度m/s^2")
       ax1.set_title('Y通道Hilbert频谱')
       plt.show()

    def plot_fv_hibert(self,RL):
       df = np.arange(0,self.N,self.N/self.fs)
       fig1 = plt.figure(figsize=(12,12))
       ax0 = fig1.add_subplot(111)
       ax0.plot(df[1:2000],RL[1:2000])
       ax0.set_xlabel("频率/Hz")
       ax0.set_ylabel("加速度m/s^2")
       ax0.set_title('全矢Hilbert解调信号')
       plt.show()
if __name__  == '__main__':
    Plot = Plot(20000,20000)
    Plot.plotfv('../IMS_data/E3.mat','../IMS_data/E4.mat')
    #Plot.plotxy('../IMS_data/E3.mat','../IMS_data/E4.mat')

