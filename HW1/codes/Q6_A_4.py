from utils import utils 
import numpy as np

def calculate_hessian(function, x0, n):
    eps=0.0001
    grad=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                x1=x0.copy()
                x2=x0.copy()
                x1[i]=x0[i]+eps
                x2[i]=x0[i]-eps
                y0=function(x0)
                y1=function(x1)
                y2=function(x2)
                grad[i,j]=(y1-2*y0+y2)/(eps**2)
            elif i>j:
                x1=x0.copy()
                x2=x0.copy()
                x3=x0.copy()
                x4=x0.copy()
                x1[i]=x1[i]+eps
                x1[j]=x1[j]+eps
      
                x2[i]=x2[i]-eps
                x2[j]=x2[j]-eps
                
                x3[i]=x3[i]+eps
                x3[j]=x3[j]-eps
                
                x4[i]=x4[i]-eps
                x4[j]=x4[j]+eps
                
                y1=function(x1)
                y2=function(x2)
                y3=function(x3)
                y4=function(x4)
                
                y=y1-y3-y2+y4
                grad[i,j]=y/(4*eps**2)
            else:
                grad[j,i]=grad[i,j]
                
        return grad        
fun=lambda x:x[1]**2+x[0]**2
print(calculate_hessian(fun,[0,0],2))