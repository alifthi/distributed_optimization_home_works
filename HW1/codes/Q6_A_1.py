import numpy as np
from utils import utils

def gradient_descent(Q,q,p,x0=None,iteration=None,learning_rate=None):
    Q=np.array(Q)
    p=np.array(p)
    q=np.array(q)
    if x0==None:
        x0=np.random.normal(size=q.shape)
    else:
        x0=np.array(x0)
    if not np.all(np.linalg.eigvals(Q) > 0):
        raise TypeError('[Error] Q is negative definite therefore there is no solution for this minimization (loss function is minimum in inf)!')
    loss=lambda x: 0.5*(x.T@Q@x) + q.T@x + p
    d_f = lambda x: Q@x+q
    if learning_rate==None:
       learning_rate=lambda x: -(d_f(x).T@d_f(x))/(d_f(x).T@Q@d_f(x))
    else:
        learning_rate=lambda x: learning_rate
    counter=0
    current_x = x0
    print(current_x.shape)
    eps=0.000001
    steps=[]
    value=[]
    x=[]
    while True:
        x.append(current_x)
        value.append(loss(current_x)[0,0])
        steps.append(counter)
        next_x=current_x+learning_rate(current_x)*d_f(current_x)
        if iteration==None:
            if np.linalg.norm(next_x-current_x) < eps:
                return next_x , loss(next_x),{'x':np.array(x),
                                            'values':np.array(value),
                                            'iterations':steps}
        else:
            if counter==iteration:
                return next_x,loss(next_x),{'x':np.array(x),
                                            'values':np.array(value),
                                            'iterations':steps}
        counter+=1
        current_x=next_x                
if __name__=='__main__':
    Q = [[48,12],[8,8]]
    q=[[13],[23]]
    p=4
    x0=[[23],[37]]
    x_optim,val, history=gradient_descent(Q,q,p,x0=x0)
    print(f'optimum point is: {x_optim}\n optimum value is: {val}')
    Q=np.array(Q)
    q=np.array(q)
    function=lambda x:0.5*(x[0].T@Q[0]+Q[1]@x) + q.T@x + p
    utils = utils(function,history=history,name='Q6_A_1')
    utils.plot_hist()


