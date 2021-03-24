import numpy as np
  


def bits(n):
    bitss=[]
    if n<0:
        bitss.append(1)
        n=-n
    else:
        bitss.append(0)
    bitss.append(np.binary_repr(n,4))
    return ''.join([str(i) for i in bitss])

print(bits(-5))
print(bits(5))