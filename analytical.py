#Para este programa se deben de generar las matricen con el programa simple_exameple_X.py
#usar seed = 3

import numpy as np

def relu (v):
    if v <=0:
        return 0
    else:
        return v

b1=np.load('b1.npy')
b2=np.load('b2.npy')
w1=np.load('w1.npy')
w2=np.load('w2.npy')
w2=w2.T
x = np.load('input.npy')
dim = len(x)
#x=np.array([0,0,0])
lf = np.load('lf.npy')

#w1=np.transpose(w1)
print('Lipschitz = ',lf)

print('x = ',x)
'''
print('b1 = ',b1)
print('b2 = ',b2)
print('w1 = ',w1)
print('w2 = ',w2)

print(w2.shape)
'''

d=dim #usaré un ejemplo con una red de 3 entradas, 3 neuronas ocultas y una de salida

D=np.identity(d)
R=np.zeros((d,d))
Lnet=1
epsi=0.1
yo=np.matmul(w1,x)+b1
#print(yo)
#calcular y sombrero
yh=np.zeros(d)


for i in range (d):
    ai=w1[i,:]
    #print(ai, ai.shape)
    ait=np.transpose(ai)
    mul=np.matmul(ait,D)
    norma=epsi* np.linalg.norm(mul)
    #print(norma)
    yh[i]= norma+yo[i]
    
    if yh[i] != yo[i]:
        R[i,i]=(relu(yh[i])-relu(yo[i]))/(yh[i]-yo[i])


print('D = ', D)
print('R = ', R)    
L=np.linalg.norm(np.matmul(np.matmul(R,w1),D),2)
#L=np.linalg.norm(w1)
Lnet=L*Lnet
print(Lnet)

epsi=epsi*L
x=yo

for i in range(d):
    if yh[i] <= 0:
        D[i,i]=0
    else:
        D[i,i]=1


R=0
#w2=np.transpose(w2)
#print(w2)
yo=np.matmul(w2,x)+b2
#print('yo=',yo)
yh=0

ai=w2
#ait=np.transpose(ai)
mul=np.matmul(ait,D)
norma=epsi* np.linalg.norm(mul)
yh= norma+yo

print(yh,yo)
if yh != yo:
    R=(relu(yh)-relu(yo))/(yh-yo)

print('D = ', D)
print('R = ', R)

LG=np.linalg.norm(np.matmul(w2,w1),2)
L=np.linalg.norm(np.matmul(np.matmul(R,w2),D),2)
Lnet=L*Lnet
print('Final Global Lipschitz ',LG,'\n')
print('Final Lipschitz ',Lnet)

    
