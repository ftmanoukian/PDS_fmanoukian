import numpy as np
import matplotlib.pyplot as plt

fs = 8#1000
ff = 3#10
nn = fs

tt = np.linspace(start = 0, stop = (nn - 1) / fs, num = nn)

xx = np.sin(2 * np.pi * ff * tt)

#plt.plot(tt,xx)

#%% DFT

N = len(xx)

XX = []

for k in range(N):
    nn = np.linspace(start = 0, stop = N - 1, num = N)
    
    arg = 2 * np.pi * k * nn / N

    twiddle = np.cos(arg) + 1j * np.sin(arg)

    XX.append(np.sum(xx * twiddle))

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(XX))
plt.subplot(2,1,2)
plt.plot(np.angle(XX))

#%% Implementación usando únicamente operadores de numpy

N = len(xx)

nn = np.arange(N)
kk = nn.reshape((N,1))
W = np.exp(-2j * np.pi * kk * nn / N)

XX = np.dot(W, xx)

plt.subplot(2,1,1)
plt.plot(np.abs(XX))
plt.subplot(2,1,2)
plt.plot(np.angle(XX))