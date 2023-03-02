#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Para ejecutar, primero generar la red con simple_example_X.py


import numpy as np
from collections import deque
from queue import PriorityQueue
from cvxopt import matrix, solvers
from copy import deepcopy
import time
import math

# In[2]:


#rounding down/up for lower/upper bounds of intervals

def rdn(a):
    return round(a-(5e-16),15)
def rup(a):
    return round(a+(5e-16),15)


# In[3]:


#matrix and interval operations

def ii_add(a,b):
    return [rdn(a[0]+b[0]),rup(a[1]+b[1])]

def ii_mul(a,b):
    mn=min(rdn(a[0]*b[0]),rdn(a[0]*b[1]),rdn(a[1]*b[0]),rdn(a[1]*b[1]))
    mx=max(rup(a[0]*b[0]),rup(a[0]*b[1]),rup(a[1]*b[0]),rup(a[1]*b[1]))
    return [mn,mx]

def ic_mul(a,b):    
    return [min(rdn(b*a[0]),rdn(b*a[1])),max(rup(b*a[0]),rup(b*a[1]))]

def mm_mul(m1,m2):
    m=[[0 for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m[i][j]+=m1[i][k]*m2[k][j]
    return m

def mv_spmul(m1,m2):
    m=[[0 for j in range(len(m2))] for i in range(len(m1))]
    for i in range(len(m2)):
        for j in range(len(m1)):
            m[j][i]=m1[j][i]*m2[i][0]
    return m

def mim_mul(m1,m2):
    m=[[[0,0] for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m[i][j]=ii_add(m[i][j], ic_mul(m2[k][j],m1[i][k]))
    return m

def ivim_spmul(m1,m2):
    m=[[[0,0] for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            m[i][j]=ii_mul(m1[i],m2[i][j])
    return m


# In[4]:


#calculation of the lower/upper bounds of expressions
#expressions are represented using coefficient vectors and constants

def e_bounds(a,b):
    lb,ub=0,0
    for i in range(len(a)):
        if a[i]>0:
            lb=rdn(lb+rdn(a[i]*var[i][0]))
            ub=rup(ub+rup(a[i]*var[i][1]))
        elif a[i]<0:
            lb=rdn(lb+rdn(a[i]*var[i][1]))
            ub=rup(ub+rup(a[i]*var[i][0]))
    lb=rdn(lb+b)
    ub=rup(ub+b)
    return lb,ub


# In[5]:


#symbolic interval propagation
#identifies initial activation pattern
#also calculates output bounds of the network


def symprop():
    ev=[np.identity(len(X)),np.zeros(len(X))]
    for l in range(1,L+1):
        ev=[np.dot(weights[l],ev[0]), np.dot(weights[l],ev[1])+biases[l]]
        if l==L:
            break
        for i in range(len(ev[0])):
            lb,ub=e_bounds(ev[0][i],ev[1][i])
            if lb>=0:
                act_pat[l][i]=[1,1]
            elif ub<=0:
                act_pat[l][i]=[0,0]
                ev[0][i]=np.zeros(len(ev[0][i]))
                ev[1][i]=0
            else:
                act_pat[l][i]=[0,1]
                ast_neurons.append([l,i])
                ev[0]=np.hstack((ev[0], np.zeros((len(ev[0]), 1))))
                ev[0][i]=np.zeros(len(ev[0][i]))
                ev[1][i]=0
                ev[0][i][len(ev[0][i])-1]=1
                var.append([0,ub])
    
    for i in range(len(ev[0])):
        lb,ub=e_bounds(ev[0][i],ev[1][i])
        output_bounds.append([lb,ub])
    


# In[6]:


#defines the structure of the subproblems

class LipNet:
    def __init__(self):
        self.Lub=0
        self.actvp=[]
        self.ast_ns=deque()
        self.H=[]
        self.t=0
        self.tev=[]


# In[7]:


#calculation of Lipschitz upper-bounds using interval matrix multiplication

def lip_bound(p):
        global pnorm
        if len(p.ast_ns)==0:
            J=deepcopy(weights[L])
            for l in range(L-1,0,-1):
                J=mv_spmul(J,p.actvp[l])
                J=mm_mul(J,weights[l])
            return np.linalg.norm(J,pnorm)
        else:
            J=[[[0,0] for j in range(len(weights[1][0]))] for i in range(len(weights[1]))]
            for i in range(len(J)):
                for j in range(len(J[0])):
                    J[i][j]=[weights[1][i][j],weights[1][i][j]]
            for l in range(2,L+1):
                J=ivim_spmul(p.actvp[l-1],J)
                J=mim_mul(weights[l],J)             
            U=[[0 for j in range(len(J[0]))] for i in range(len(J))]
            for i in range(len(J)):
                for j in range(len(J[0])):
                    U[i][j]=max(abs(J[i][j][0]),abs(J[i][j][1]))
            return np.linalg.norm(U,pnorm)


# In[8]:


#propagation of linear expressions for generation of half-space constraints

def linprop(p):
    l=p.t
    ev=p.tev
    while l<L:
        if not p.ast_ns or p.ast_ns[0][0]==l:
            break
        for i in range(len(ev[0])):
            if p.actvp[l][i]==[0,0]:
                ev[0][i]=np.zeros(len(ev[0][i]))
                ev[1][i]=0
        l+=1
        ev=[np.dot(weights[l],ev[0]), np.dot(weights[l],ev[1])+biases[l]]
    p.t=l
    p.tev=ev


# In[9]:


#feasibility filter
#reduces undecided-neurons to active/inactive-neurons

def ffilter(p):
    A,b=deepcopy(p.H[0]),deepcopy(p.H[1])
    while len(p.ast_ns)>0:
        l,i=p.ast_ns[0][0],p.ast_ns[0][1]
        A.append(p.tev[0][i].tolist())
        b.append(-p.tev[1][i])
        sol=solvers.lp(matrix([0.0]*len(X)),matrix(np.array(A)),matrix(b),solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        if sol['status']!='optimal':            
            p.actvp[l][i]=[1,1]
            p.ast_ns.popleft()
            linprop(p)
            A.pop()
            b.pop()
            continue
        A.pop()
        b.pop()
        A.append((-p.tev[0][i]).tolist())
        b.append(p.tev[1][i])
        sol=solvers.lp(matrix([0.0]*len(X)),matrix(np.array(A)),matrix(b),solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        if sol['status']!='optimal':            
            p.actvp[l][i]=[0,0]
            p.ast_ns.popleft()
            linprop(p)
            A.pop()
            b.pop()
            continue
        A.pop()
        b.pop()
        break


# In[10]:


#branching of sub-problems to generate new sub-problems

def branch(p,sgn):
    global pcnt
    global glb
    l,i=p.ast_ns[0][0],p.ast_ns[0][1]    
    A,b=deepcopy(p.H[0]),deepcopy(p.H[1])
    A.append(-sgn*p.tev[0][i])
    b.append(sgn*p.tev[1][i])
    res=solvers.lp(matrix([0.0]*len(X)),matrix(np.array(A)),matrix(b),solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    if res['status']=='optimal':
        pb=LipNet()
        pcnt+=1
        pb.H=[A,b]
        pb.actvp=deepcopy(p.actvp)        
        pb.ast_ns=deepcopy(p.ast_ns)
        pb.t=p.t
        pb.tev=deepcopy(p.tev)
        pb.actvp[l][i]=[max(0,sgn),max(0,sgn)]
        pb.ast_ns.popleft()                
        linprop(pb)
        ffilter(pb)
        pb.Lub=lip_bound(pb)
        pq.put((-pb.Lub,pcnt,pb))
        if len(pb.ast_ns)==0:
            glb=max(glb,pb.Lub)


#random network
#"""


b1=np.load('b1.npy')
b2=np.load('b2.npy')
#w1=np.load('w1.npy')
#w2=np.load('w2.npy')
x = np.load('input.npy')
dim = len(x)
lf = np.load('lf.npy')

w1=np.load('w1.npy')
w2=np.load('w2.npy')
#x=np.array([1,1,1])
w2=w2.T

ls=[dim,dim,1]
L=len(ls)-1

#x[dim]=1
X=x.tolist()
#x=np.transpose(x)

print('\n estos son los valores de entrada \n', X)
weights,biases=[None],[None]

weights.append(w1)
weights.append(w2)
biases.append(b1)
biases.append(b2)

def create_random_network(ls):
    for i in range(1,len(ls)):
        weights.append((np.random.randn(ls[i], ls[i-1]) * np.sqrt(2.0 / ls[i-1])).tolist())
        biases.append(np.random.uniform(low=1,high=2,size=(ls[i],)).tolist())

#create_random_network(ls)


act_pat=[]
for e in ls:
    act_pat.append([[1,1]for i in range(e)])

bnds=[[0.,0.1] for i in range(ls[0])]

#print('\n estos son los pesos y bias originales \n',weights,'\n',biases,'\n')


#main algorithm

start_time = time.time()

#initialization
output_bounds=[]
ast_neurons=deque()
var=[]
pcnt=0
pq=PriorityQueue()

for b in bnds:
    var.append([b[0],b[1]])
    
HC=[[],[]]

for i in range(len(X)):
    b=bnds[i]
    a=[0]*len(X)
    a[i]=-1.
    HC[0].append(a)
    HC[1].append(-b[0])
    a=[0]*len(X)
    a[i]=1.
    HC[0].append(a)
    HC[1].append(b[1])

#print ('\n Este es X \n',X,'Este es HC \n',HC,'\n')
    
pnorm=2 #norm of choice (1,2,inf)
#af=1 #approximation factor of choice
af=1    
#start
symprop()
glb=0.0

p=LipNet()
pcnt+=1
p.H=deepcopy(HC)
p.actvp=deepcopy(act_pat)
p.ast_ns=deepcopy(ast_neurons)
p.tev=[np.identity(len(X)),np.zeros(len(X))]
linprop(p)
p.Lub=lip_bound(p)
pq.put((-p.Lub,pcnt,p))

if len(p.ast_ns)==0:
    glb=p.Lub

while(True):
    #if (time.time() - start_time)>=300: #(setting execution time limit (in seconds))
        #break
    if(pq.empty()==True):
        break
    tp=pq.get()[2]
    print(tp.Lub)
    if tp.Lub<=af*glb:
        break
    branch(tp,-1)
    branch(tp,1)
       
print("Final LiBaB Lipschitz estimation:", tp.Lub)
execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))

#"""