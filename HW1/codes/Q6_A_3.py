import utils 
import numpy as np
def backtrackining_linesearch(function,x,grad,alpha=0.5,beta=0.9):
    t=1
    while function(x+t*grad) > function(x)+alpha*t*grad:
       t*=beta
    return t