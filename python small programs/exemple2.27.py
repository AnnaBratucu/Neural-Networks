import numpy as np

A=np.array([[1.,2.,3.,4.],[5.,6.,7.,8.]])
print('First matrix')
print(A)

A = np.lib.pad(A, ((0,3),(0,1)), 'constant', constant_values=(0))
print('Second matrix')
print(A)

print('Element 2nd row - 2nd column:')
print(A[1][1])