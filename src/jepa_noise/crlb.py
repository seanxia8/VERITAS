"""Full-rank Gaussian known-noise Fisher bounds; never a pseudoinverse."""
import numpy as np
from .linear import invsqrt


def fisher(jacobian,noise):
    j=np.asarray(jacobian,float); invsqrt(noise)
    info=j.T@np.linalg.solve(noise,j)
    if np.linalg.matrix_rank(info)!=len(info): raise ValueError('Fisher rank ineligible')
    invsqrt(info)
    return info


def crlb(jacobian,noise):
    return np.linalg.inv(fisher(jacobian,noise))


def of_amplitude(template,noise,data=None):
    template=np.asarray(template,float); invsqrt(noise)
    direction=np.linalg.solve(noise,template)
    information=float(template@direction)
    if information<=0: raise ValueError('amplitude information absent')
    return {'variance':1/information,'estimate':None if data is None else np.asarray(data)@direction/information}


def central_jacobian(function,theta,ranges,half=False):
    theta=np.asarray(theta,float); h=1e-4*np.asarray(ranges,float)*(0.5 if half else 1.)
    if np.any(h<=0): raise ValueError('positive declared parameter ranges required')
    columns=[]
    for i,step in enumerate(h):
        e=np.zeros_like(theta); e[i]=step
        columns.append((np.asarray(function(theta+e))-np.asarray(function(theta-e)))/(2*step))
    return np.stack(columns,axis=-1)
