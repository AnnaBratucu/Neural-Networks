import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import sounddevice as sd
import dspLib as dsp

fs=44100
fc=np.array([5000,10000])
ft=2000
w=fc/(fs/2)
ft=ft/(fs/2)
db=30

t0=0
t1=1
t=np.arange(t0,t1+1/fs,1/fs)

A=1
f=1000
ph0=0

x=dsp.mySine(A,f,ph0,t)
sd.play(x,fs) #before filtering

N,beta =sp.kaiserord(db,ft)
b=sp.firwin(N,cutoff = 0.3,window=('kaiser' , beta) )
a=np.array(1.)
w,h=sp.freqz(b, a, worN=512, whole=False, plot=None)
f=(fs/2)*w/(np.pi)
q=(fs/2)*x
z=np.fft.fft(b)
dsp.plotInFrequency(b,fs) 
dsp.plotInTime(b,fs)
sd.play(q,fs)  #after filtering
plt.figure()
plt.plot(f,abs(h))
plt.grid()
a=np.fft.fft(b)
print(a)
plt.figure()
plt.plot(z) 

