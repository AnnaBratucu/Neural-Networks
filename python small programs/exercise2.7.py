import math
import numpy as np

#0/0
x=[0]
y=[0]
print('0/0:',np.float64(x)/y)

#1000^1000
print('1000^1000:',pow(1000,1000))

#infinity-1
x=math.inf
y=1
print('infinity-1:',x-y)

#infinity+infinity
x=math.inf
y=math.inf
print('infinity+infinity:',x+y)

#infinity-infinity
x=math.inf
y=math.inf
print('infinity-infinity:',x-y)

#infinity/infinity
x=math.inf
y=math.inf
print('infinity/infinity:',x/y)

#infinity/0
x=math.inf
print('infinity/0:',np.float64(x)/0)

#0/infinity
x=math.inf
print('0/infinity:',0/x)

#1^infinity
x=math.inf
print('1^infinity:',pow(1,x))

#0.1^infinity
x=math.inf
print('0.1^infinity:',pow(0.1,x))

#-1^infinity
x=math.inf
print('-1^infinity:',pow(-1,x))





