import random as rd
import numpy as np

x = rd.randint(1,20)
y = rd.randint(1,10)

a = np.convolve(x, y)
print(a)
b = np.convolve(y, x)
print(b)

if a == b:
    print('They are equal!!!')
    

print('Proof:')

errAccepted=10^-10
if np.abs(a-b)<errAccepted:
    print('Not commutative')
else:
    print('Commutative')
