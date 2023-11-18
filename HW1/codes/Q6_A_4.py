from utils import utils 
import numpy as np



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

def calculate_hessian(function, x0, n):
    eps=0.1
    grad=np.zeros([n,n])
    for j in range(n):
        for i in range(n):
            if i==j:
                x1=x0.copy()
                x2=x0.copy()
                x1[i]=x0[i]+eps
                x2[i]=x0[i]-eps
                y0=function(x0)
                y1=function(x1)
                y2=function(x2)
                grad[i,i]=(y1-2*y0+y2)/(eps**2)
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
            elif i<j:
                grad[j,i]=grad[i,j]
    return grad        

def newton(function,n=2,x0=None,iteration=None):
    if x0==None:
        x0=np.random.normal(size=n)
    else:
        x0=np.array(x0)
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
        hessian=calculate_hessian(function,current_x,n)
        next_x=current_x-np.linalg.inv(hessian)@grad
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
    # b=150
    # a=-2
    # n=2
    # x0=[0.2,0.2]
    # function=lambda x : sum([b*(x[i+1]**2-x[i])**2 + (x[i]-a) for i in range(0,n-1)])
    #################################
    # to use problem 11 uncomment this part and comment part 10
    n=4
    x0=[0.1,0.2,0.2,0.2]
    function=lambda x : (x[0]-10*x[2])**2+5*(x[2]-x[3])**2+(x[1]-2*x[2])**4+10*(x[0]-x[3])**4
    ####################################################################
    final_argument, final_value, history =newton(function=function,x0=x0,n=n)
    print(final_value)
    util = utils(function,history=history,name=f'Q6_A_x1_{x0[0]}_x2_{x0[1]}_4')
    util.draw_cotours()
    util.draw_optim_path()
    util.plot_hist()