import numpy as np
import matplotlib.pyplot as plt

def sampleAndHoldD(t0,t1,fs):
    y=[]
    t=[]
    tVector=np.arange(t0,t1+1/fs,1/fs)
    #tVector=[0,1,2,3,4]
    for index in range(0,len(tVector)):
        t.append(tVector[index])
        a=t[index]
        y.append(eval('sin(pow(a,5))/pow(2,exp(a))'))
    
    plt.stem(y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

t0=0
t1=1.5
fs=50
sampleAndHoldD(t0,t1,fs)