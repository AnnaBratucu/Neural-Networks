import numpy as np

x=[1,2,3,4,5,6,7]
y=[2,3,1,4,5,6,7]

z=np.polyfit(x,y,5)

print(z[0],'x^5',z[1],'x^4+',z[2],'x^3+',z[3],'x^2+',z[4],'x',z[5])