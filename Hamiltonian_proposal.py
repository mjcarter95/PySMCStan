import autograd.numpy as np
from autograd import elementwise_grad    

"""
A collection of methods that relate to Hamiltonian proposals.

LJ Devlin

"""

def HMC_proposal( x, v, gradObj):
    
    #Precompute gradient to reduce computation
    grad_x = gradObj(x) 

    # Main leapfrog loop, fixed to 5 s steps
    for k in range(0,5):
        x, v, grad_x = Leapfrog(x, v, grad_x, gradObj)

    return x, v


def Leapfrog(x, v, grad_x, gradObj):
    
    # We will hard-code the step-size for the moment
    h=0.1
    
    v = np.add(v, (h/2)*grad_x)
    x = np.add(x, h*v)
    grad_x = gradObj(x)
    v = np.add(v, (h/2)*grad_x)
    
    return x, v, grad_x

def propose_step_size(x):
    pass