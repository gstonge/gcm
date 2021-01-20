# -*- coding: utf-8 -*-
"""
This module provides functions to find the optimal initial conditions
to maximize the early spread of the contagion, given a fixed initial
fraction of infected.

We make use of the linear programming algorithms provided by scipy.
"""

import numpy as np
from scipy.optimize import linprog

def objective_function_vector(inf_mat, state_meta):
    """objective_function_vector returns the canonical c vector for
    optimization using linear programming. The objective function is
    therefore

    F(x) = c @ x

    with @ the dot product. The vector x is a flatten version of the fni
    matrix.

    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.

    :return c: array representing the c vector of the objective function.
    """
    nmax = state_meta[1]
    pn = state_meta[4]
    c = -inf_mat.copy()
    for n in range(c.shape[0]):
        for i in range(c.shape[1]):
            if i <= n:
                c[n][i] *= pn[n]*(n-i)
            else:
                c[n][i] = 0.
    return c.reshape((nmax+1)**2)

def constraint_arrays(sm,state_meta):
    """constraint_arrays returns the canonical arrays for implementing equality
    constraints with linear programming. Constraints take the form

    A @ x = b

    with @ the dot product. The vector x is a flatten version of the fni
    matrix.

    Note : although A is a 2d array, in our case we only have 1 equality
    constraint. Similarly, b is a vector of a single element.

    :param sm: array for the probability that a node of membership m is
               susceptible.
    :param state_meta: tuple of arrays encoding information of the structure.

    :return A,b: tuple of arrays (2d,1d) for the equality constraints
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    #constraints on normalization
    b = [1 for n in range(nmax+1)]
    A = []
    for n in range(nmax+1):
        v = np.zeros((nmax+1,nmax+1))
        v[n,0:n+1] += 1.
        A.append(v.reshape((nmax+1)**2))
    #constraint on number of infected stubs
    b.append(np.sum(np.arange(nmax+1)*pn)*np.sum(m*sm*gm)/np.sum(m*gm))
    M = np.ones((nmax+1,nmax+1),dtype=np.float64)
    for n in range(M.shape[0]):
        for i in range(M.shape[1]):
            if i <= n:
                M[n][i] *= pn[n]*(n-i)
            else:
                M[n][i] = 0.
    M = M.reshape((nmax+1)**2)
    A.append(M)
    return np.array(A),np.array(b)

def optimize_fni(initial_density,inf_mat,state_meta):
    """optimize_fni returns the fni matrix that optimize the early spread,
       assuming randomly chosen nodes (irrespective of their membership).

    :param initial_density: float for the initial fraction of infected nodes
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    sm = np.ones(mmax+1)*(1.-initial_density)
    c = objective_function_vector(inf_mat,state_meta)
    A,b = constraint_arrays(sm,state_meta)
    bounds = (0.,1.)
    res = linprog(c,A_eq=A,b_eq=b,bounds=bounds)
    if res.success:
        fni = np.array(res.x).reshape((nmax+1,nmax+1))
        #clean up elements that should be 0 and normalize
        for n in range(nmax+1):
            fni[n][n+1:] = 0.
            fni[n] /= np.sum(fni[n])
        return fni
    else:
        raise RuntimeError('optimization failed')


def optimize_sm(initial_density,state_meta):
    """optimize_sm returns the sm vector that optimize the early spread,
       assuming that nodes within groups are infected at random.

       We assume that the initial density of infected node is small,
       in which case we can use the optimal heuristic to infect nodes
       with largest membership.

    :param initial_density: float for the initial fraction of infected nodes
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    mmax = state_meta[0]
    gm = state_meta[3]
    sm = np.ones(mmax+1,dtype=np.float64)
    density_allowed = 0.
    current_m = mmax
    while density_allowed < initial_density:
        if gm[current_m] >= (initial_density - density_allowed):
            sm[current_m] = 1 - (initial_density - density_allowed)/gm[current_m]
            density_allowed = initial_density
        else:
            sm[current_m] = 0
            density_allowed += gm[current_m]
            current_m -= 1
    return sm

if __name__ == '__main__':
    from ode import *
    nmax = 10
    mmax = 10
    pn = np.arange(nmax+1,dtype=np.float64)
    pn[2:] = pn[2:]**(-2.5)
    pn[0:2] = 0.
    pn /= np.sum(pn)
    gm = np.arange(mmax+1,dtype=np.float64)
    gm[2:] = gm[2:]**(-2.5)
    gm[0:2] = 0.
    gm /= np.sum(gm)
    state_meta = get_state_meta(mmax, nmax, gm, pn)
    beta = lambda n,i: 0.5*i**2.
    inf_mat = infection_matrix(beta,nmax)
    initial_density = 0.02
    print(optimize_sm(initial_density,state_meta))
    print(optimize_fni(initial_density,inf_mat,state_meta))
