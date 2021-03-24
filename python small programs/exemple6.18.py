import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def mySawTooth(A,f,ph,t):
    return A*sig.sawtooth(2*np.pi*f*t+ph)
t1=1
fs=50
t0=0
tf=3
t=np.arange(t0,t1+1/fs,1/fs)
A=4
T=1
ph=0
x=mySawTooth(4,1,0,t)
plt.subplot(3,1,1)
plt.plot(t,x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original signal')
noiseProb = 0.1
noiseA=3
noise=(np.random.rand(np.size(x))<noiseProb)*mySawTooth(noiseA,1,0,t)
x=x+noise
plt.subplot(3,1,2)
plt.plot(t,x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal with noise')
n=3
x=sig.medfilt(x,n)
plt.subplot(3,1,3)
plt.plot(t,x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Median filtered signal')