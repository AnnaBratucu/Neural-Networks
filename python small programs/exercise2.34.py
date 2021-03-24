import time
start = time.time()
a=[]


def prime1(x):
    for i in range(2,x):
        if x%i==0:
            return False
    return True

def lstPrime1(x):
    for i in range(2,x):
        if prime1(i):
            a.append(i)
    return a


                
z=lstPrime1(10000)
print(z)

end = time.time()
print('Time needed for the run: ',end - start)
