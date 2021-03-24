
import numpy as np
import matplotlib.pyplot as plt


def extendThroughPeriodicity():
    a=[1,2,4,5,6,7,8,9,5,3]
    aPeriodic=np.matlib.repmat(a, 1,10)
    aPeriodic=np.reshape(aPeriodic,np.size(a)*10,1)
    plt.stem(aPeriodic)
    

extendThroughPeriodicity()
