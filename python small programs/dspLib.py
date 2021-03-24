import numpy as np
import matplotlib.pyplot as plt
def mySine(A,f,ph0,t):
    return A*np.sin(2*np.pi*f*t+ph0)

def myStep(A,t0,t):
        return A*(t>t0)
    
def myChirp(A,f,t):
    return A*np.sin(2*np.pi*f*t*t)

def plotInTime(y,fs):
    plt.figure()
    N=len(y)
    t=np.arange(0,N/fs,1/fs)
    
    plt.plot(t,y)
    plt.grid()
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')

def plotInFrequency(y,fs):
    plt.figure()
    N=len(y)
    #y=y[:,0]
    print(N)
    
    Y=np.fft.fft(y)
    Y=abs(Y)
    N=int(np.floor(N/2))
    Y=Y[:N]
    f=np.arange(0,fs/2,fs/2/N)
    plt.plot(f,Y)
    plt.grid()
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Magnitude')

def myWhiteNoise(noise,t):
    return 0.1*noise*t
    
