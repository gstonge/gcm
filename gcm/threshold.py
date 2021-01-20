# -*- coding: utf-8 -*-
"""
This module calculates the invasion threshold and the secondary bifurcation
(bistability threshold or tricritical point) for general contagion model
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
from scipy.optimize import brentq

def _cumulative_infection_product(beta, nmax, args=tuple()):
    mat = np.zeros((nmax+1,nmax+1))
    nvec = np.arange(2,nmax+1)
    mat[2:,0] = np.ones(nmax-1)
    for i in range(1,nmax+1):
        mat[2:,i] = mat[2:,i-1]*beta(nvec,i,*args)
    return mat

def _sum(beta, pn, args=tuple()):
    nmax = len(pn)-1
    cum_prod = _cumulative_infection_product(beta,nmax,args=args)
    mat = np.zeros((nmax+1,nmax+1))
    ivec = np.arange(1,nmax+1)
    for n in range(2,nmax+1):
        mat[n,1:] = gamma(n+1)/(gamma(n-ivec)*gamma(ivec+1))
        mat[n,1:] *= cum_prod[n,1:]*pn[n]
    return np.sum(mat)

def _get_hl(beta, nmax, args=tuple()):
    h = np.zeros((nmax+1,nmax+1))
    l = np.zeros((nmax+1,nmax+1))
    #calculate the hni and lni
    for n in range(2,nmax+1):
        #hni
        h[n][1] = n
        for i in range(1,n):
            h[n][i+1] = h[n][i]*(n-i)*beta(n,i,*args)/(i+1)
        h[n][0] = -np.sum(h[n])
        #lni
        l[n][1] = 2*n*h[n][0]
        for i in range(1,n):
            l[n][i+1] = (2*((n-i)*h[n][i]-(n-i+1)*h[n][i-1]) \
                    +(i+(n-i)*beta(n,i,*args))*l[n][i] \
                    -(n-i+1)*beta(n,i-1,*args)*l[n][i-1])/(i+1)
        l[n][0] = -np.sum(l[n])
    return h,l

def invasion_threshold(beta, gm, pn, fixed_args=tuple(), initial_param=0.1):
    """invasion_threshold calculates the invasion threshold for the dynamics

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param gm: array for the membership distribution of length mmax+1
    :param pn: array for the group size distribution of length nmax+1
    :args: tuple of extra arguments to specify the infection rate function
    :initial_param: float for the initial value of the parameter for which we
                    calculate the invasion threshold
    """
    nmax = len(pn)-1
    mmax = len(gm)-1
    n = np.arange(nmax+1,dtype=float)
    m = np.arange(mmax+1,dtype=float)
    const = np.sum(m*(m-1)*gm)/(np.sum(m*gm)*np.sum(n*pn))
    func = lambda param: const*_sum(beta,pn,args=(param,*fixed_args)) - 1
    return fsolve(func, initial_param)[0]

def invasion_threshold_safe(beta, gm, pn, fixed_args=tuple(),
                            min_param=10**(-14), max_param=1):
    """invasion_threshold_safe calculates the invasion threshold for the dynamics

    Note : uses the brentq method.

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param gm: array for the membership distribution of length mmax+1
    :param pn: array for the group size distribution of length nmax+1
    :args: tuple of extra arguments to specify the infection rate function
    :initial_param: float for the initial value of the parameter for which we
                    calculate the invasion threshold
    """
    nmax = len(pn)-1
    mmax = len(gm)-1
    n = np.arange(nmax+1,dtype=float)
    m = np.arange(mmax+1,dtype=float)
    const = np.sum(m*(m-1)*gm)/(np.sum(m*gm)*np.sum(n*pn))
    func = lambda param: const*_sum(beta,pn,args=(param,*fixed_args)) - 1
    return brentq(func,min_param,max_param)


def bistability_threshold(beta, gm, pn, initial_params=(0.1,1.1)):
    """bistability threshold calculates the bistability threshold for the
    dynamics

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param gm: array for the membership distribution of length nmax+1
    :param pn: array for the group size distribution of length nmax+1
    :initial_params: tuple of floats for the initial value of the parameters,
                     the first one is associated with the invasion threshold,
                     the second one with the bistability threshold.
    """
    nmax = len(pn)-1
    mmax = len(gm)-1
    n = np.arange(nmax+1,dtype=float)
    m = np.arange(mmax+1,dtype=float)
    c1 = np.sum(m*(m-1)*gm)/np.sum(m*gm)
    c2 = np.sum(n*pn)
    c3 = (np.sum(m**2*gm)**2/(np.sum(m*gm)**2)
          - np.sum(m**3*gm)/np.sum(m*gm))
    def func(x):
        args=tuple(x)
        h,l = _get_hl(beta,nmax,args=args)
        mat1 = np.zeros((nmax+1,nmax+1))
        mat2 = np.zeros((nmax+1,nmax+1))
        for n_ in range(2,nmax+1):
            for i in range(n_+1):
                mat1[n_][i] = (n_-i)*pn[n_]
                mat2[n_][i] = beta(n_,i,*args)*(n_-i)*pn[n_]
        vp = np.sum(mat1*h)
        up = np.sum(mat2*h)
        upp = np.sum(mat2*l)
        y = np.zeros(2)
        y[0] = c1*up/c2 - 1
        y[1] = 2*c3/(c1**3) + (upp/c2 - 2*vp*up/c2**2)
        return y
    sol = fsolve(func, np.array(initial_params))
    return sol[1]

def bistability_threshold_safe(beta, gm, pn, min_params=(10**(-14),1),
                              max_params=(1,7)):
    """bistability threshold calculates the bistability threshold for the
    dynamics using brentq method

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param gm: array for the membership distribution of length nmax+1
    :param pn: array for the group size distribution of length nmax+1
    :min_params: tuple of floats for the minimal value of the parameters,
                     the first one is associated with the invasion threshold,
                     the second one with the bistability threshold.
    :max_params: tuple of floats for the maximal value of the parameters,
                     the first one is associated with the invasion threshold,
                     the second one with the bistability threshold.
    """
    nmax = len(pn)-1
    mmax = len(gm)-1
    n = np.arange(nmax+1,dtype=float)
    m = np.arange(mmax+1,dtype=float)
    c1 = np.sum(m*(m-1)*gm)/np.sum(m*gm)
    c2 = np.sum(n*pn)
    c3 = (np.sum(m**2*gm)**2/(np.sum(m*gm)**2)
          - np.sum(m**3*gm)/np.sum(m*gm))
    def func(x):
        param1 = invasion_threshold_safe(beta, gm, pn, fixed_args=(x,),
                                         min_param=min_params[0],
                                         max_param=max_params[0])
        args=tuple([param1,x])
        h,l = _get_hl(beta,nmax,args=args)
        mat1 = np.zeros((nmax+1,nmax+1))
        mat2 = np.zeros((nmax+1,nmax+1))
        for n_ in range(2,nmax+1):
            for i in range(n_+1):
                mat1[n_][i] = (n_-i)*pn[n_]
                mat2[n_][i] = beta(n_,i,*args)*(n_-i)*pn[n_]
        vp = np.sum(mat1*h)
        up = np.sum(mat2*h)
        upp = np.sum(mat2*l)

        y = 2*c3/(c1**3) + (upp/c2 - 2*vp*up/c2**2)
        return y
    try:
        sol = brentq(func, min_params[1], max_params[1])
    except ValueError:
        print(f"f(a) = {func(min_params[1])}, f(b) = {func(max_params[1])}")
        sol = None

    return sol



if __name__ == '__main__':
    #structure
    mmin = 3
    nmax = 5
    mmax = 3
    pn = np.zeros(nmax+1)
    pn[4] += 1
    gm = np.zeros(mmax+1)
    gm[mmin:mmax+1] += 1/(mmax+1-mmin) #uniform
    beta = lambda n,i,trate,alpha: trate*i**alpha
    alpha = 1.908
    print(invasion_threshold(beta,gm,pn,fixed_args=(alpha,)))
    print(bistability_threshold(beta,gm,pn))

