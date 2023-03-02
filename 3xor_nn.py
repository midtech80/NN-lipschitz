

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(3) # con tres no es excta

def binario(d,dim):
    binary_list = []
    while d > 0:
        binary_list.append(d % 2)
        d //= 2

    for i in range(dim-len(binary_list)):
        binary_list.append(0)
    # Reverse the order of the binary digits in the list
    binary_list.reverse()
    return binary_list


def inpu(dim):
    X=[]
    for k in range(2**dim):
        X.append(binario(k,dim))
    x=np.array(X)
    return x
    
def outpu(inpu):
    Y=[]
    for k in range(len(inpu)):
        if np.sum(inpu[k,:])%2==0:
            Y.append(0)
        else:
            Y.append(1)
    y=np.array(Y)
    return y

def sigmoid(x):
    return (1/(1+np.exp(-x)))

dim=6

#aca estoy cambiando la entada por una funcion que me permita cambiar la dimension
#x=np.vstack(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]))
#t=np.array([0,1,1,0,1,0,0,1]).reshape(-1,1)
x=inpu(dim)
t=outpu(x)
t=np.reshape(t,(-1,1))

#print(t)
w1=np.random.rand(dim,dim+1)
w2=np.random.rand(dim+1,1)

alpha=0.5

loss=[]
for i in range(10000):
    if i%1000==0:
        print(i)
    y1=sigmoid(np.dot(x,w1))
    ys=sigmoid(np.dot(y1,w2))
    E=1/4*np.sum((ys-t)**2)
    dEdw2=2*(np.dot(y1.T,(ys-t)*ys*(1-ys)))
    dEdw1=2*np.dot(x.T,np.dot((ys-t)*ys*(1-ys),w2.T)*y1*(1-y1))
    w2=w2-alpha*dEdw2
    w1=w1-alpha*dEdw1
    loss.append(E)

plt.plot(loss)
plt.title('gráfica del valor de la función de perdida')
plt.xlabel('épocas')
plt.ylabel('loss')
plt.grid()

"""La función de predicción es entonces"""

def pred(x,w1,w2):
    y1=sigmoid(np.dot(x,w1))
    ys=sigmoid(np.dot(y1,w2))
    return ys

def pred2(x,w1,w2):
    y1=sigmoid(np.dot(x,w1))
    ys=sigmoid(np.dot(y1,w2))
    if ys>=0.5:
        return 1
    else:
        return 0


"""y "predecimos" su valor"""
'''
for i in range(2**dim):
    print('y = %i' % pred2(x[i,:],w1,w2))
'''

def lips(f1,f2,x1,x2):
    ns=np.abs(f2-f1)
    ni=np.linalg.norm(x2-x1,2)
    l=ns/ni
   # print('l= ',l)
    return l

lip=[]
k=0
for i in range (2**dim):
    for j in range (2**dim):
        if np.array_equal(x[i], x[j]):
            n=0
        else:
            f1=pred(x[i],w1,w2)
            f2=pred(x[j],w1,w2)
            lip.append(lips(f1,f2,x[i],x[j]))

print("Exact GLOBAL = ",np.max(lip))
print(lip)


b1=w1[:,dim]
w1=w1[:,0:dim]
b2=w2[dim]
w2=w2[0:dim]

LG=np.linalg.norm(np.matmul(w1,w2),2)
print('Final Global Lipschitz ',LG,'\n')

np.save('w1.npy',w1)
np.save('w2.npy',w2)
np.save('b1.npy',b1)
np.save('b2.npy',b2)
np.save('input.npy', x[2])

plt.show()

