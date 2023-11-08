import numpy as np
from utils import utils

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
        x0=np.random.normal(size=n)*5
    learning_rate=lambda grad:learning_dist/np.linalg.norm(grad)
    counter=0
    current_x = x0
    eps=0.0001
    steps=[]
    value=[]
    x=[]
    while True:
        x.append(current_x)
        value.append(function(current_x))
        steps.append(counter)
        grad=calculate_gradient(function=function,x0=current_x,n=n)
        next_x=current_x-learning_rate(grad)*grad
        if iteration==None:
            if np.linalg.norm(function(next_x)-function(current_x)) < eps:
                return next_x,function(next_x), {'x':np.array(x),
                                                 'values':np.array(value),
                                                 'iterations':steps}
        else:
            if counter==iteration:
                return next_x,function(next_x), {'x':np.array(x),
                                                 'values':np.array(value),
                                                 'iterations':steps}
        counter+=1
        current_x=next_x
if __name__=='__main__':
    function=lambda x : x[0]**4+x[1]**6+15
    final_argument, final_value, history =gradient_descent(function=function,learning_dist=0.01)
    print(final_value)
    utils = utils(function,history=history)
    utils.draw_cotours()
    utils.draw_optim_path()
    utils.plot_hist()
