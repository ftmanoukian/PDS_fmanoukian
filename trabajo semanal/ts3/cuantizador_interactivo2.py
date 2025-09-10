#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 20:22:27 2025

@author: franciscomanoukian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from numpy.fft import fft, fftfreq

class cuantizador_interactivo():

    __N_DEFAULT = 8
    __SNR_DEFAULT = 60
    __TEXTOKCOLOR = (161/255, 240/255, 141/255, 1)
    __TEXTWRONGCOLOR = (240/255, 141/255, 141/255, 1)

    def __init__(self):

        self.N = self.__N_DEFAULT
        self.SNR = self.__SNR_DEFAULT
        
        self.a_max = np.sqrt(2)
        self.dc = 0 
        self.ph = 0 
        self.ff = 10
        self.fs = 1000 
        self.nn = 1000
        
        self.calculate_signal()
        
        self.__init_plot()
        self.plot()
        
    def __init_plot(self):
        self.fig = plt.figure(figsize=[16,8])
        
        self.fs_tbx = TextBox(self.fig.add_axes([0.025,0.87,0.04,0.03]), '$f_s$  ', initial=f'{self.fs}', color=self.__TEXTOKCOLOR, hovercolor=self.__TEXTOKCOLOR)
        self.ff_tbx = TextBox(self.fig.add_axes([0.025,0.82,0.04,0.03]), '$f_f$  ', initial=f'{self.ff}', color=self.__TEXTOKCOLOR, hovercolor=self.__TEXTOKCOLOR)
        self.fs_tbx.on_submit(self.__fs_update)
        self.ff_tbx.on_submit(self.__ff_update)
        
        self.SNR_slider = Slider(self.fig.add_axes([0.08,0.66,0.02,0.25]),'SNR',0,200,valinit=self.__SNR_DEFAULT,valstep=1,orientation='vertical')
        self.SNR_slider.on_changed(self.__SNR_update)
        
        self.N_slider = Slider(self.fig.add_axes([0.11,0.66,0.02,0.25]),'N',1,24,valinit=self.__N_DEFAULT,valstep=1,orientation='vertical')
        self.N_slider.on_changed(self.__N_update)

        self.signal_ax = self.fig.add_axes([0.18,0.66,0.79,0.25])
        self.signal_ax.set_title('Señal')
        self.signal_ax.set_ylabel('Amplitud')
        self.signal_ax.set_xlabel('Tiempo [seg]')
                
        self.analog_noise_ax = self.fig.add_axes([0.075,0.37,0.4,0.16])
        self.quantization_noise_ax = self.fig.add_axes([0.075,0.1,0.4,0.16])
        self.analog_noise_ax.set_title('Ruido analógico')
        self.analog_noise_ax.set_ylabel('Amplitud')
        self.analog_noise_ax.set_xlabel('Tiempo [seg]')
        self.quantization_noise_ax.set_title('Ruido de cuantización')
        self.quantization_noise_ax.set_ylabel('Amplitud')
        self.quantization_noise_ax.set_xlabel('Tiempo [seg]')
        
        self.noise_pow_ax = self.fig.add_axes([0.55,0.1,0.42,0.43])
        
        self.signal_ax.plot(self.tt, self.xx, label='Señal analógica')
        
    def PDS_sin(self, a_max = 1, dc = 0, ph = 0, ff = 100, fs = 1000, nn = 1000):
        
        tt = np.linspace(0, (nn - 1) / fs, nn)
        
        xx = a_max * np.sin(2 * np.pi * ff * tt + ph) + dc
        
        return tt,xx
    
    def PDS_noise_sin(self, SNR_db = __SNR_DEFAULT, a_max = 1, dc = 0, ph = 0, ff = 100, fs = 1000, nn = 1000):
        
        P_A = a_max / np.sqrt(2) # potencia normalizada
        SNR = 10 ** (SNR_db / 10)
        theta_sq = P_A / SNR
        
        tt,xx_a = self.PDS_sin(a_max = a_max, dc = dc, ph = ph, ff = ff, fs = fs, nn = nn)
        
        xx_n = np.random.normal(loc = 0, scale = np.sqrt(theta_sq), size = nn)
        
        xx = xx_a + xx_n
        
        return tt,xx,xx_a,xx_n
    
    def PDS_quantize(self, xx, vref, N):
        
        q = vref / (np.pow(2,N))
        xx_q = np.floor(xx / q) * q
        xx_q_n = xx - xx_q

        return xx_q, xx_q_n
    
    def __SNR_update(self, SNR):
        self.SNR = SNR
        self.plot()
    
    def __N_update(self, N):
        self.N = N
        self.plot()
        
    def __fs_update(self, fs):
        try:
            fs = float(fs)
            if fs < 1:
                raise ValueError()
            self.fs = fs
            if self.ff >= self.fs / 2:
                self.ff = self.fs / 2 - 1
                self.ff_tbx.set_val(f'{self.ff}')
            self.fs_tbx.color = self.__TEXTOKCOLOR
            self.fs_tbx.hovercolor = self.__TEXTOKCOLOR
            self.plot()
        except ValueError:
            self.fs_tbx.color = self.__TEXTWRONGCOLOR
            self.fs_tbx.hovercolor = self.__TEXTWRONGCOLOR

    def __ff_update(self, ff):
        try:
            ff = float(ff)
            if ff >= self.fs / 2:
                raise ValueError()
            self.ff = ff
            self.ff_tbx.color = self.__TEXTOKCOLOR
            self.ff_tbx.hovercolor = self.__TEXTOKCOLOR
            self.plot()
        except ValueError:
            self.ff_tbx.color = self.__TEXTWRONGCOLOR
            self.ff_tbx.hovercolor = self.__TEXTWRONGCOLOR
                
    def calculate_signal(self):
        self.nn = int(self.fs) # normalización de nn
        
        # señal con ruido
        self.tt, self.xx, self.xx_a, self.xx_n = self.PDS_noise_sin(
             SNR_db = self.SNR, 
             a_max = self.a_max, 
             dc = self.dc, 
             ph = self.ph, 
             ff = self.ff, 
             fs = self.fs, 
             nn = self.nn
        )
        
        # señal cuantizada
        vref = self.a_max * 2 / 0.8
        self.xx_q, self.xx_q_n = self.PDS_quantize(xx = self.xx, vref = vref, N = self.N)
        
        # transformadas y ruido
        self.XX         = np.abs(fft(self.xx)[:self.nn//2])
        self.XX_N       = np.abs(fft(self.xx_n)[:self.nn//2])
        self.XX_QN      = np.abs(fft(self.xx_q_n)[:self.nn//2])
        self.FF = fftfreq(n = self.nn, d=self.tt[1]-self.tt[0])[:self.nn//2]
        
        self.XX_pow     = 20 * np.log10(self.XX)
        dB_bias = max(self.XX_pow)
        
        self.XX_N_pow   = 20 * np.log10(self.XX_N) - dB_bias
        self.XX_QN_pow  = 20 * np.log10(self.XX_QN) - dB_bias
        
    def plot_noise_time(self, selected):
        self.selected_noise = selected
        if selected == 'analógico':
            self.active_nn = self.xx_n
        elif selected == 'cuantización':
            self.active_nn = self.xx_q_n
        self.noise_ax.clear()
        self.noise_ax.set_title('Ruido')
        self.noise_ax.set_ylabel('Amplitud')
        self.noise_ax.set_xlabel('Tiempo [seg]')
        self.noise_ax.plot(self.tt,self.active_nn)
    
    def plot(self):
        self.calculate_signal()
 
        self.signal_ax.clear()
        self.signal_ax.set_title('Señal')
        self.signal_ax.set_ylabel('Amplitud')
        self.signal_ax.set_xlabel('Tiempo [seg]')
        self.signal_ax.plot(self.tt, self.xx, label='Señal analógica')
        self.signal_ax.step(self.tt, self.xx_q, label='Señal cuantizada')
        self.signal_ax.legend(loc='upper right')

        self.noise_pow_ax.clear()
        self.noise_pow_ax.set_ylim([min(np.min(self.XX_N_pow),np.min(self.XX_QN_pow)),0])
        self.noise_pow_ax.set_title('Potencia de ruido')
        self.noise_pow_ax.set_ylabel('Potencia normalizada [dB]')
        self.noise_pow_ax.set_xlabel('Frecuencia [Hz]')
        self.noise_pow_ax.plot(self.FF, self.XX_N_pow, label='Ruido analógico')
        self.noise_pow_ax.plot(self.FF, self.XX_QN_pow, label='Ruido de cuantización')
        self.noise_pow_ax.legend(loc='upper right')
        
        self.analog_noise_ax.clear()
        self.analog_noise_ax.set_title('Ruido analógico')
        self.analog_noise_ax.set_ylabel('Amplitud')
        self.analog_noise_ax.set_xlabel('Tiempo [seg]')
        self.analog_noise_ax.plot(self.tt,self.xx_n)

        self.quantization_noise_ax.clear()
        self.quantization_noise_ax.set_title('Ruido de cuantización')
        self.quantization_noise_ax.set_ylabel('Amplitud')
        self.quantization_noise_ax.set_xlabel('Tiempo [seg]')
        self.quantization_noise_ax.plot(self.tt,self.xx_q_n)
        

if __name__ == "__main__":
    
    app = cuantizador_interactivo()