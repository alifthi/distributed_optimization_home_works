import numpy as np
n=50
m=500


def project_on_set(x,C,D):
    dist=C.T@x-D
    if dist>0:
        x=x-(dist)*C/(np.linalg.norm(C)**2)
    return x
if __name__=='__main__':
    A=np.random.normal(0,4,[m,n])
    B=np.random.normal(0,4,m)
    C=np.random.normal(0,4,[n,1])
    D=np.random.normal(0,4,1)
    