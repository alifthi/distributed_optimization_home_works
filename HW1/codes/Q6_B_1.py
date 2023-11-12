import numpy as np

def project_on_set(x,C,D):
    dist=C.T@x-D
    if dist>0:
        x=x-(dist)*C/(np.linalg.norm(C)**2)
    return x

def calculate_gradient(x,A,B):
    losses=np.abs(A@x+B)
    argmax=np.argmax(losses)
    grad=A[argmax]
    return grad
if __name__=='__main__':
    n=50
    m=500
    A=np.random.normal(0,4,[m,n])
    B=np.random.normal(0,4,[m,1])
    C=np.random.normal(0,4,[n,1])
    D=np.random.normal(0,4,1)
    x=np.ones([n,1])
    print(calculate_gradient(x,A,B).shape)
    