import numpy as np

def calculate_gradient(function,x0,n=2, iterations=None):
    eps=0.00001
    grad=[]
    for i in range(n):
        x1=x0.copy()
        x1[i]=x0[i]+eps
        y0=function(x0)
        y1=function(x1)
        grad.append((y1-y0)/eps)
    return grad
def gradient_descent():
    pass
    
if __name__=='__main__':
    function=lambda x : x[0]**2+x[1]**2
    grad=calculate_gradient(function=function)
    print(grad)