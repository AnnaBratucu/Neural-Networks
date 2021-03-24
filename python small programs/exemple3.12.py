
t=list(range(1, 101))

print('input data: ', t)

f = open('file.txt', 'w')
for i in t:
    f.write(str(i)+ ' ')
f.close()

with open('file.txt', 'r') as file:
    data = file.read()
print('output data: ',data)