import numpy as np

A=np.array([[1.,2.,3.,4.],[5.,6.,7.,8.]])
print('First matrix')
print(A)

A = np.lib.pad(A, ((0,3),(0,1)), 'constant', constant_values=(0))
print('Second matrix')
print(A)

A[4][4] = np.pi
np.set_printoptions(precision=4)
print('Third matrix')
print(A)