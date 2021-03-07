# -*- coding: utf-8 -*-
"""
This module provides functions to find the optimal initial conditions
to maximize the early spread of contagions, given a fixed initial
fraction of infected nodes.
"""

import warnings
import numpy as np
import heapq
from scipy.optimize import brentq
from scipy.stats import binom
from .ode import *
from numba import jit

# def _objective_function_vector(inf_mat, state_meta):
    # """_objective_function_vector returns the canonical c vector for
    # optimization using linear programming. The objective function is
    # therefore

    # F(x) = c @ x

    # with @ the dot product. The vector x is a flatten version of the fni
    # matrix.

    # :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    # :param state_meta: tuple of arrays encoding information of the structure.

    # :return c: array representing the c vector of the objective function.
    # """
    # nmax = state_meta[1]
    # pn = state_meta[4]
    # c = -inf_mat.copy()
    # for n in range(c.shape[0]):
        # for i in range(c.shape[1]):
            # if i <= n:
                # c[n][i] *= pn[n]*(n-i)
            # else:
                # c[n][i] = 0.
    # return c.reshape((nmax+1)**2)

# def _constraint_arrays(sm,state_meta):
    # """_constraint_arrays returns the canonical arrays for implementing equality
    # constraints with linear programming. Constraints take the form

    # A @ x = b

    # with @ the dot product. The vector x is a flatten version of the fni
    # matrix.

    # Note : although A is a 2d array, in our case we only have 1 equality
    # constraint. Similarly, b is a vector of a single element.

    # :param sm: array for the probability that a node of membership m is
               # susceptible.
    # :param state_meta: tuple of arrays encoding information of the structure.

    # :return A,b: tuple of arrays (2d,1d) for the equality constraints
    # """
    # mmax = state_meta[0]
    # nmax = state_meta[1]
    # m = state_meta[2]
    # gm = state_meta[3]
    # pn = state_meta[4]
    # #constraints on normalization
    # b = [1 for n in range(nmax+1)]
    # A = []
    # for n in range(nmax+1):
        # v = np.zeros((nmax+1,nmax+1))
        # v[n,0:n+1] += 1.
        # A.append(v.reshape((nmax+1)**2))
    # #constraint on number of infected stubs
    # b.append(np.sum(np.arange(nmax+1)*pn)*np.sum(m*sm*gm)/np.sum(m*gm))
    # M = np.ones((nmax+1,nmax+1),dtype=np.float64)
    # for n in range(M.shape[0]):
        # for i in range(M.shape[1]):
            # if i <= n:
                # M[n][i] *= pn[n]*(n-i)
            # else:
                # M[n][i] = 0.
    # M = M.reshape((nmax+1)**2)
    # A.append(M)
    # return np.array(A),np.array(b)

# def optimize_fni_lp(initial_density,inf_mat,state_meta):
    # """optimize_fni_lp returns the state with the fni matrix that optimize the
    # early spread, assuming randomly chosen nodes (irrespective of their
    # membership).

    # Note: this function use linear programming to solve the problem

    # :param initial_density: float for the initial fraction of infected nodes
    # :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    # :param state_meta: tuple of arrays encoding information of the structure.
    # """
    # mmax = state_meta[0]
    # nmax = state_meta[1]
    # m = state_meta[2]
    # gm = state_meta[3]

    # sm = (1-initial_density*m/np.sum(m*gm))
    # c = _objective_function_vector(inf_mat,state_meta)
    # A,b = _constraint_arrays(sm,state_meta)
    # bounds = (0.,1.)

    # # options = {'tol':10**(-15)}
    # res = linprog(c,A_eq=A,b_eq=b,bounds=bounds)#, options=options)
    # if res.success:
        # fni = np.array(res.x).reshape((nmax+1,nmax+1))
        # #clean up elements that should be 0 and normalize
        # for n in range(nmax+1):
            # fni[n][n+1:] = 0.
            # fni[n] /= np.sum(fni[n])
    # else:
        # raise RuntimeError('optimization failed')

    # return sm,fni

def _get_eta(initial_density,state_meta):
    m = state_meta[2]
    gm = state_meta[3]
    func = lambda eta: np.sum(gm*(1.-eta)**m) - (1 - initial_density)
    eta = brentq(func,0,1)
    assert eta > 0 and eta < 1
    return eta

def _get_u(eta, state_meta):
    m = state_meta[2]
    gm = state_meta[3]
    sm = (1.-eta)**m
    u = np.sum(m*(1-sm-eta)*gm)/np.sum((1-eta)*m*gm)
    assert u > 0 and u < 1
    return u

def _get_binomial(u,state_meta):
    nmax = state_meta[1]
    B = np.zeros((nmax+1,nmax+1))
    for k in range(0,nmax+1):
        B[k,0:k+1] = binom.pmf(np.arange(k+1,dtype=int), k, u)
    return B

@jit(nopython=True)
def _get_Rni(inf_mat,B,state_meta):
    nmax = state_meta[1]
    Rni = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(1,n+1):
            jvec = np.arange(n-i+1)
            Rni[n,i] = np.sum(inf_mat[n,i:n+1]*(n-i-jvec)*B[n-i,0:n-i+1])/i
    return Rni

@jit(nopython=True)
def _get_fni_from_fni_tilde(fni_tilde,B,state_meta):
    nmax = state_meta[1]
    fni = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(0,n+1):
            for j in range(0,i+1):
                fni[n,i] += fni_tilde[n,i-j]*B[n-i+j,j]
    return fni



def _optimize_fni_core(Rni,eta,state_meta):
    nmax = state_meta[1]
    pn = state_meta[4]

    #initialize at all 0 nodes infected
    fni_tilde = np.zeros((nmax+1,nmax+1))
    fni_tilde[:,0] = 1.
    #empty iopt dictionary for optimal i conf
    iopt = dict()
    #identify infected budget, in terms of i*fni_tilde*pn
    psi = np.sum(np.arange(nmax+1)*pn)*eta
    #initialize priority queue with cost-efficiency ratio
    Q = []
    heapq.heapify(Q)
    counter = 0
    for n in range(2,nmax+1):
        if pn[n] > 0:
            for i in range(1,n+1):
                heapq.heappush(Q, (-Rni[n,i], counter, (n,i)))
                                    #counter is used to break ties
                counter += 1

    #while Q is not empty, and budget not expanded
    while Q and not np.isclose(psi,0):
        item = heapq.heappop(Q)
        n,i = item[2]
        if n in iopt:
            #there is already a fni_tilde configuration chosen
            if i > iopt[n]:
                i_ = iopt[n]
                #its possible that the new config is better
                Rni_hat = (Rni[n,i]*i-Rni[n,i_]*i_)/(i-i_)
                if len(Q) == 0 or -Rni_hat <= Q[0][0]:
                    #apply the configuration
                    fni_tilde[n,i] = min((1,psi/(pn[n]*(i-i_))))
                    fni_tilde[n,i_] -= fni_tilde[n,i]
                    assert np.isclose(1,np.sum(fni_tilde[n]))
                    psi -= fni_tilde[n,i]*pn[n]*(i-i_)
                    #if we are not finished, new i is 'optimal'
                    iopt[n] = i
                else:
                    #push again in the queue
                    heapq.heappush(Q, (-Rni_hat, counter, (n,i)))
                    counter += 1
        else:
            #first time n is encountered
            fni_tilde[n,i] = min((1,psi/(pn[n]*i)))
            fni_tilde[n,0] -= fni_tilde[n,i]
            assert np.isclose(1,np.sum(fni_tilde[n]))
            psi -= fni_tilde[n,i]*pn[n]*i
            iopt[n] = i

    return fni_tilde


def optimize_fni(initial_density, inf_mat, state_meta):
    """optimize_fni returns the state with the fni matrix that optimize the
    early spread, assuming randomly chosen nodes.

    :param initial_density: float for the initial fraction of infected nodes
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]

    eta = _get_eta(initial_density,state_meta)
    u = _get_u(eta, state_meta)
    B = _get_binomial(u,state_meta)
    Rni = _get_Rni(inf_mat,B,state_meta)
    sm = (1.-eta)**m
    fni_tilde = _optimize_fni_core(Rni,eta,state_meta)
    fni = _get_fni_from_fni_tilde(fni_tilde,B,state_meta)

    return sm, fni, fni_tilde


def optimize_sm(initial_density,state_meta):
    """optimize_sm returns the state with the sm vector that optimize the
    early spread, assuming that nodes within groups are infected at random.

       We assume that the initial density of infected node is small,
       in which case we can use the optimal heuristic to infect nodes
       with largest membership.

    :param initial_density: float for the initial fraction of infected nodes
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    mmax = state_meta[0]
    m = state_meta[2]
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
    #identify q, the density within groups, and get fni
    q = 1 - np.sum(gm*m*sm)/np.sum(m*gm)
    fni = initialize(state_meta, q)[1]
    if q > 1/2:
        #for contagions with supralinear (or linear) exponents
        warnings.warn("The sm solution might be suboptimal")

    return sm,fni

def objective_function(fni, inf_mat, state_meta):
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    return np.sum(inf_mat*(nmat-imat)*fni*pnmat)

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
