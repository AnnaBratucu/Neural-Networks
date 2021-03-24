import scipy.signal as sig
import matplotlib.pyplot as plt

N=2 #order 2
fs=10000 #sampling frequency #transformed in Hz
wc=1000 #cut off frequency #transformed in Hz
wn=wc/(fs/2) #normalized frequency
[b,a]=sig.butter(N,wn) #analog
plt.plot(sig.bilinear(b,a)) #digital signal