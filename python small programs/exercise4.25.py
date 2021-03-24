import numpy as np
import matplotlib.pyplot as plt
import dspLib as dsp


def myStep(A,t0,t):
        return A*(t>t0)
    
def plotStep():
    A=1
    t0=0
    t=np.linspace(-5,5,100)
    fs=100
    
    x=myStep(A,t0,t)
    plt.figure()
    plt.plot(x)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.figure()
    plt.phase_spectrum(x,fs)
    
    plt.figure()
    plt.magnitude_spectrum(x,fs)

plotStep()



