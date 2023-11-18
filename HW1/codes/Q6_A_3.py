from utils import utils 
import numpy as np

def backtrackining_linesearch(function,x,grad,alpha=0.5,beta=0.1):
    t=1
    while function(x-t*grad) > function(x)-alpha*t*(np.linalg.norm(grad)**2):
       t*=beta
    return t

def calculate_gradient(function,x0,n=2):
    eps=0.0001
    grad=np.zeros(n)
    for i in range(n):
        x1=x0.copy()
        x1[i]=x0[i]+eps
        y0=function(x0)
        y1=function(x1)
        grad[i]=(y1-y0)/eps
    return grad

def gradient_descent(function,n=2,x0=None,iteration=None):
    if x0==None:
        x0=np.random.normal(size=n)*5
    # else:
    #     x0=np.array(x0)
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
        alpha=backtrackining_linesearch(function,current_x,grad=grad)
        next_x=current_x-alpha*grad
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
    # to use problem 10 uncomment this part and comment part 11
    b=150
    a=-2
    n=2
    x0=[23,37]
    function=lambda x : sum([b*(x[i+1]**2-x[i])**2 + (x[i]-a) for i in range(0,n-1)])
    #################################
    # to use problem 11 uncomment this part and comment part 10
    # n=4
    # x0=[1,2,2,2]
    # function=lambda x : (x[0]-10*x[2])**2+5*(x[2]-x[3])**2+(x[1]-2*x[2])**4+10*(x[0]-x[3])**4
    ####################################################################
    final_argument, final_value, history =gradient_descent(function=function,x0=x0,n=n)
    util = utils(function,history=history,name='Q6_A_3')
    util.draw_cotours()
    util.draw_optim_path()
    util.plot_hist()