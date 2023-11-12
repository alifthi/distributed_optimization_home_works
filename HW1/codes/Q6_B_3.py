import numpy as np
from utils import utils
def project_on_set(x,C,D):
    dist=C.T@x-D
    if dist>0:
        x=x-(dist)*C/(np.linalg.norm(C)**2)
    return x

def calculate_gradient(x,A,B):
    losses=np.abs(A@x+B)
    argmax=np.argmax(losses)
    isNeg=A[argmax]@x+B[argmax]<0
    grad=A[argmax]*((-1)**int(isNeg))
    return grad,B[argmax]

def gradient_descent(A,B,C,D,n=50,x0=None,iteration=None):
    if x0==None:
        x0=np.random.normal(size=[n,1])
    counter=1
    current_x = x0
    function=lambda A,x,b:np.abs(A.T@x+b)
    learning_rate=lambda step:1/step
    eps=0.001
    steps=[]
    value=[]
    x=[]
    while True:
        steps.append(counter)
        current_x=project_on_set(current_x,C,D)
        grad,b=calculate_gradient(current_x,A,B)
        x.append(current_x)
        value.append(function(grad,current_x,b))
        alpha=learning_rate(counter)
        grad=grad[:,None]
        next_x=current_x-alpha*grad
        if iteration==None:
            if counter%1000==0:
                print(function(grad,current_x,b))
            if np.linalg.norm(function(grad,current_x,b)-function(grad,next_x,b)) < eps:
                return next_x,function(grad,current_x,b), {'x':np.array(x),
                                                 'values':np.array(value),
                                                 'iterations':steps}
        else:
            if counter==iteration:
                return next_x,function(grad,current_x,b), {'x':np.array(x),
                                                 'values':np.array(value),
                                                 'iterations':steps}
        counter+=1
        current_x=next_x

if __name__=='__main__':
    n=50
    m=500
    A=np.random.normal(0,4,[m,n])
    B=np.random.normal(0,4,[m,1])
    C=np.random.normal(0,4,[n,1])
    D=np.random.normal(0,4,1)
    next_x,_,diction=gradient_descent(A,B,C,D)