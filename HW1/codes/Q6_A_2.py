import numpy as np

def calculate_gradient(function,x0,n=2, iterations=None):
    eps=0.00001
    grad=np.zeros(n)
    for i in range(n):
        x1=x0.copy()
        x1[i]=x0[i]+eps
        y0=function(x0)
        y1=function(x1)
        grad[i]=(y1-y0)/eps
    return grad

def gradient_descent(function,n=2,learning_dist=0.01,x0=None,iteration=None):
    if x0==None:
        x0=np.random.normal(size=n)
    learning_rate=lambda grad:learning_dist/np.linalg.norm(grad)
    counter=0
    current_x = x0
    eps=0.0001
    while True:
        grad=calculate_gradient(function=function,x0=current_x,n=n)
        next_x=current_x-learning_rate(grad)*grad
        if iteration==None:
            if np.linalg.norm(function(next_x)-function(current_x)) < eps:
                return next_x , function(next_x)
        else:
            if counter==iteration:
                return next_x,function(next_x)
            counter+=1
        current_x=next_x
    
if __name__=='__main__':
    function=lambda x : x[0]**2+x[1]**2
    a=gradient_descent(function=function)
    print(a)
