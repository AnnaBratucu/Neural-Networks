import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import dspLib as dsp

#1
def mySawTooth(A,f,ph0,t):
    return A*sig.sawtooth(2*np.pi*f*t+ph0)


def plotSawTooth():
    t0=0
    t1=1
    fs=100
    t=np.arange(t0,t1+1/fs,1/fs)

    A=1
    f=4
    ph0=0
    x=mySawTooth(A,f,ph0,t)
    plt.plot(x)
    plt.title('Saw Tooth')
    
plotSawTooth()

#2
def myStep(A,t0,t):
        return A*(t>t0)
    
def plotStep():
    A=1
    t0=0
    t1=5
    fs=100
    t=np.arange(t0,t1+1/fs,1/fs)
    x=myStep(A,t0,t)
    plt.plot(x)
    plt.title('Step')

plt.figure()
plotStep()

#3
def myChirp(A,f,t):
    return A*np.sin(2*np.pi*f*t*t)

def plotChirp():
    A=1
    t0=0
    t1=3
    fs=200
    f=10
    t=np.arange(t0,t1+1/fs,1/fs)
    x=myChirp(A,f,t)
    plt.plot(x) 
    plt.title('Chirp')
    dsp.plotInFrequency(x,fs)

plt.figure()
plotChirp()

#4
def myTriangle(A,f,ph0,t):
    return A*sig.sawtooth(2*np.pi*f*t+ph0,1/2)

def plotTriangle():
    t0=0
    t1=1
    fs=100
    t=np.arange(t0,t1+1/fs,1/fs)

    A=1
    f=4
    ph0=0
    x=myTriangle(A,f,ph0,t)
    plt.plot(x)
    plt.title('Triangle')

plt.figure()
plotTriangle()

#5
def myCosine(A,f,ph0,t):
    return A*np.cos(2*np.pi*f*t+ph0)

def plotCosine():
    t0=0
    t1=1
    fs=100
    t=np.arange(t0,t1+1/fs,1/fs)

    A=1
    f=4
    ph0=0
    x=myCosine(A,f,ph0,t)
    plt.plot(x)
    plt.title('Cosine')
  
plt.figure()
plotCosine()

#6
def mySine(A,f,ph0,t):
    return A*np.sin(2*np.pi*f*t+ph0)

def plotSine():
    t0=0
    t1=1
    fs=100
    t=np.arange(t0,t1+1/fs,1/fs)

    A=1
    f=4
    ph0=0
    x=mySine(A,f,ph0,t)
    plt.plot(x)
    plt.title('Sine')
  
plt.figure()
plotSine()

#7
def mySquare(A,f,ph0,t):
    return A*sig.square(2*np.pi*f*t+ph0)

def plotSquare():
    t0=0
    t1=1
    fs=1000
    t=np.arange(t0,t1+1/fs,1/fs)

    A=1
    f=4
    ph0=0
    x=mySquare(A,f,ph0,t)
    plt.plot(x)
    plt.title('Square')
  
plt.figure()
plotSquare()

#8
def myCircle(A,f,ph0,t):
    return A*sig.square(2*np.pi*f*t+ph0)

def plotCircle():
    t = np.linspace(0,1,201)
    wt = 2*np.pi*t
    plt.plot(np.cos(wt), np.sin(wt))
    plt.title('Circle')
  
plt.figure()
plotCircle()

#9
def myPulse(A,t,f):
    return A*sig.gausspulse(t,f)


def plotPulse():
    t0=-4
    t1=4
    fs=400
    t=np.linspace(t0, t1, 2 * fs)

    A=1
    f=10
    x=myPulse(A,t,f)
    plt.plot(x)
    plt.title('Pulse')
    
plt.figure()
plotPulse()

#10
def mySinc(A,x):
    return A*np.sin(np.pi*x)/(np.pi*x)

def plotSinc():
    t0=-7
    t1=7
    fs=100
    t=np.linspace(t0, t1, 2 * fs)

    A=1
    x=mySinc(A,t)
    plt.plot(x)
    plt.title('Sinc')
    
plt.figure()
plotSinc()










