import numpy as np
def gradient_descent(Q,q,p,x0=None,iteration=None,learning_rate=None):
    Q=np.array(Q)
    p=np.array(p)
    q=np.array(q)
    if x0==None:
        x0=np.random.normal(size=q.shape)
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
    while True:
        next_x=current_x+learning_rate(current_x)*d_f(current_x)
        if iteration==None:
            if np.linalg.norm(next_x-current_x) < eps:
                return next_x , loss(next_x)
        else:
            if counter==iteration:
                return next_x,loss(next_x)
            counter+=1
        current_x=next_x                
if __name__=='__main__':
    Q = [[48,12],[8,2.092]]
    q=[[13],[23]]
    p=4
    print(gradient_descent(Q,q,p))

