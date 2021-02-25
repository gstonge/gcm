# -*- coding: utf-8 -*-
"""
This module provides functions for the solution of the ODE system for general
infection function.
"""

import numpy as np
from numba import jit
from scipy.stats import binom
from scipy.integrate import odeint

@jit(nopython=True)
def infected_fraction(sm, gm):
    return np.sum((1-sm)*gm)

@jit(nopython=True)
def excess_susceptible_membership(m,gm,sm):
    """excess_susceptible_membership return the average membership of a
    node, following a random susceptible node within a group

    :param m: array for the membership.
    :param gm: array for the membership distribution.
    :param sm: array for the probability that a node of membership m is
               susceptible.
    """
    return np.sum(m*(m-1)*sm*gm)/np.sum(m*sm*gm)

def get_state_meta(mmax, nmax, gm, pn):
    """Return a tuple that encapsulate all meta information about the structure

    :param mmax: maximal membership
    :param nmax: maximal group size >= 2
    :param gm: array for the membership distribution of length mmax+1
    :param pn: array for the group size distribution of length nmax+1

    :return state_meta: tuple of useful arrays describing the structure
    """
    m = np.arange(0,mmax+1)
    imat = np.zeros((nmax+1,nmax+1))
    nmat = np.zeros((nmax+1,nmax+1))
    for n in range(2, nmax+1):
        imat[n,0:n+1] = np.arange(n+1)
        nmat[n,0:n+1] = np.ones(n+1)*n
    pnmat = np.outer(pn,np.ones(nmax+1))
    return (mmax,nmax,m,np.array(gm),np.array(pn),imat,nmat,pnmat)


def get_state_meta_corr(mmax, nmax, Pm_n, Pn_m):
    """Return a tuple that encapsulate all meta information about the structure

    :param mmax: maximal membership
    :param nmax: maximal group size >= 2
    :param Pm_n: array for the membership following an edge starting from a
                 group of size n, (nmax,mmax)
    :param Pn_m: array for the group size following an edge starting from a
                 node of membership m, (mmax,nmax).

    :return state_meta: tuple of useful arrays describing the structure
    """
    mvec = np.arange(0,mmax+1)
    imat = np.zeros((nmax+1,nmax+1))
    nmat = np.zeros((nmax+1,nmax+1))
    for n in range(2, nmax+1):
        imat[n,0:n+1] = np.arange(n+1)
        nmat[n,0:nmax+1] = np.ones(nmax+1)*n
    Pn_m_mat = np.array([np.outer(Pn_m[m],np.ones(nmax+1)) for m in mvec])
    return (mmax,nmax,mvec,np.array(Pm_n),np.array(Pn_m),imat,nmat,Pn_m_mat)



@jit(nopython=True)
def flatten(sm,fni,state_meta):
    nmax = state_meta[1]
    return np.concatenate((sm,fni.reshape((nmax+1)**2)))


def unflatten(v,state_meta):
    mmax = state_meta[0]
    nmax = state_meta[1]
    return v[:mmax+1],v[mmax+1:].reshape((nmax+1,nmax+1))


def initialize(state_meta, initial_density=0.5):
    """initialize returns an array representing the state of the
    system at t=0 and state meta information, assuming uniformly distributed
    infected nodes.

    :param state_meta: tuple of arrays encoding information of the structure.
    :param initial_density: float for initial fraction of infected

    :returns (sm,fni): tuple of arrays of shape mmax+1 and
                                  (nmax+1, nmax+1) representing the state of
                                  nodes and groups, and the state meta data
    """
    mmax = state_meta[0]
    nmax = state_meta[1]

    sm = np.zeros(mmax+1)
    fni = np.zeros((nmax+1,nmax+1))
    #initialize nodes
    sm += 1-initial_density
    #initialize groups
    for n in range(2, nmax+1):
        pmf = binom.pmf(np.arange(n+1,dtype=int),n,initial_density)
        fni[n][:n+1] = pmf
    return sm,fni


def infection_matrix(beta, nmax, args=()):
    """infection_matrix returns an array for the infection rate at each (n,i)

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param nmax: int for maximal group size >= 2
    :args: tuple of extra arguments to specify the infection rate function

    :returns: array of shape (nmax+1,nmax+1) for the infection rate at each
              (n,i) values.
    """
    inf_mat = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(n):
            inf_mat[n][i] = beta(n,i,*args)
    return inf_mat


def advance(sm, fni, tvar, inf_mat, state_meta, corr=False):
    """advance integrates the ODE starting from a certain initial state and
    returns the new state.

    :param sm: array of shape (1,mmax+1) representing the nodes state.
    :param fni: array of shape (nmax+1,nmax+1) representing the groups state.
    :param tvar: float for time variation.
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.
    :param corr: bool to determine if there are correlations.

    return (sm,fni): tuple of state arrays later in time
    """
    v = flatten(sm,fni,state_meta)
    t = np.linspace(0,tvar)
    if corr:
        vvec = odeint(vector_field_corr,v,t,args=(inf_mat,state_meta))
    else:
        vvec = odeint(vector_field,v,t,args=(inf_mat,state_meta))
    return unflatten(vvec[-1],state_meta)


@jit(nopython=True)
def vector_field(v, t, inf_mat, state_meta):
    """vector_field returns the temporal derivative of a flatten state vector

    :param v: array of shape (1,mmax+1+(nmax+1)**2) for the flatten state vector
    :param t: float for time (unused)
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.

    :returns vec_field: array of shape (1,(nmax+1)**2) for the flatten
                        vector field.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]
    sm = v[:mmax+1]
    fni = v[mmax+1:].reshape(nmax+1,nmax+1)
    fni_field = np.zeros(fni.shape) #matrix field
    sm_field = np.zeros(sm.shape)

    #calculate mean-field quantities
    r = np.sum(inf_mat[2:,:]*(nmat[2:,:]-imat[2:,:])*fni[2:,:]*pnmat[2:,:])
    r /= np.sum((nmat[2:,:]-imat[2:,:])*fni[2:,:]*pnmat[2:,:])
    rho = r*excess_susceptible_membership(m,gm,sm)

    #contribution for nodes
    #------------------------
    sm_field = 1 - sm - sm*m*r

    #contribution for groups
    #------------------------
    #contribution from above
    fni_field[2:,:nmax] += imat[2:,1:]*fni[2:,1:]
    #contribution from equal
    fni_field[2:,:] += (-imat[2:,:]
                        -(nmat[2:,:] - imat[2:,:])
                        *(inf_mat[2:,:] + rho))*fni[2:,:]
    #contribution from below
    fni_field[2:,1:nmax+1] += ((nmat[2:,:nmax] - imat[2:,:nmax])
                               *(inf_mat[2:,:nmax] + rho))*fni[2:,:nmax]
    return np.concatenate((sm_field,fni_field.reshape((nmax+1)**2)))

# @jit(nopython=True)
def vector_field_corr(v, t, inf_mat, state_meta):
    """vector_field returns the temporal derivative of a flatten state vector

    :param v: array of shape (1,mmax+1+(nmax+1)**2) for the flatten state vector
    :param t: float for time (unused)
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.

    :returns vec_field: array of shape (1,(nmax+1)**2) for the flatten
                        vector field.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    Pm_n = state_meta[3]
    Pn_m = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    Pn_m_mat = state_meta[7]
    sm = v[:mmax+1]
    fni = v[mmax+1:].reshape(nmax+1,nmax+1)
    fni_field = np.zeros(fni.shape) #matrix field
    sm_field = np.zeros(sm.shape)

    #calculate mean-field quantities
    rm = np.zeros(mmax+1)
    rm = np.sum(inf_mat[2:,:]*(nmat[2:,:]-imat[2:,:])*fni[2:,:]*\
               Pn_m_mat[:,2:,:]/nmat[2:,:],axis=(1,2))
    norm1 = np.sum((nmat[2:,:]-imat[2:,:])*fni[2:,:]*\
               Pn_m_mat[:,2:,:]/nmat[2:,:],axis=(1,2))
    rm = np.divide(rm,norm1, where=norm1!=0)
    rhon = np.sum((m-1)*rm*sm*Pm_n,axis=1)
    norm2 = np.sum(sm*Pm_n,axis=1)
    rhon = np.divide(rhon,norm2, where=norm2!=0)
    rhon_mat = np.outer(rhon,np.ones(nmax+1))

    #contribution for nodes
    #------------------------
    sm_field = 1 - sm - sm*m*rm

    #contribution for groups
    #------------------------
    #contribution from above
    fni_field[2:,:nmax] += imat[2:,1:]*fni[2:,1:]
    #contribution from equal
    fni_field[2:,:] += (-imat[2:,:]
                        -(nmat[2:,:] - imat[2:,:])
                        *(inf_mat[2:,:] + rhon_mat[2:,:]))*fni[2:,:]
    #contribution from below
    fni_field[2:,1:nmax+1] += ((nmat[2:,:nmax] - imat[2:,:nmax])
                               *(inf_mat[2:,:nmax] + rhon_mat[2:,:nmax]))*fni[2:,:nmax]

    return np.concatenate((sm_field,fni_field.reshape((nmax+1)**2)))



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    #structure
    mmin = 5
    nmax = 7
    mmax = 5
    pn = np.zeros(nmax+1)
    gm = np.zeros(mmax+1)
    pn[4] += 1
    gm[mmin:mmax+1] += 1/(mmax+1-mmin) #uniform
    state_meta = get_state_meta(mmax,nmax,gm,pn)

    #infection
    trate = 0.05
    alpha = 3.
    inf_mat = infection_matrix(lambda n,i: trate*i**alpha,nmax)

    #initialize
    sm,fni = initialize(mmax,nmax,gm,pn,initial_density=0.016)

    #advance some tvar
    tvar = 1
    sm,fni = advance(sm,fni,tvar, inf_mat, state_meta)
    v = np.concatenate((sm,fni.reshape((nmax+1)**2)))

    #integrate manually
    t = np.linspace(0,1000,100)
    vvec = odeint(vector_field,v,t,args=(inf_mat,state_meta))
    t1 = time.time()
    vvec = odeint(vector_field,v,t,args=(inf_mat,state_meta))
    t2 = time.time()
    print(t2-t1)
    I = [infected_fraction(v[:mmax+1],gm) for v in vvec]

    #plot infected fraction
    plt.plot(t,I)
    plt.show()
