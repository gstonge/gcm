# -*- coding: utf-8 -*-
"""
This module provides functions to obtain the stationary state solutions.
"""
import numpy as np
from .ode import *
from scipy.integrate import ode
from scipy.optimize import brentq
from numba import jit
import numba as nb

def stable_branch(beta, state_meta, param_init, param_var, fixed_args=(),
                  tvar=100, h=0.01, rtol=10**(-8), Jtol=10**(-2),
                  min_density=10**(-4),max_iter=1000, verbose=True):
    """stable_branch captures the non-trivial stable branch starting from
       some initial parameter until the bifurcation.

    IMPORTANT : it is assumed that the direction of the parameter variation
                is such that the system goes toward a bifurcation. At the
                bifurcation, it is expected that the Jacobian J_r = 1.
                It is also assumed that J_r < 1, since we are targetting
                a stable branch.

    We converge the stationary state at the current parameter value. We
    then vary the parameter. If the Jacobian J_r varies by a value more than
    Jtol/2, we adapt the parameter variation value until it respects it.
    With this adaptive parameter variation, we ensure that we will not miss
    the bifurcation, which serves as a stopping criterion.

    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param state_meta: StateMeta object encoding the information of the
                       structure.
    :param param_init: float for the initial value of the parameter to vary
    :param param_var: float (positive or negative) for the parameter variation
    :param fixed_args: tuple of extra arguments for beta that are fixed.
    :param tvar: float for the initial time relaxation to stationary state
    :param h: float for the step size of the linearly implicit Euler.
    :param rtol: float for the relative variation desired on the convergence.
    :param Jtol: float for tolerance on Jacobian variation for stability
    :param min_density: float for the minimal infected fraction allowed
    :param max_iter: int for the maximum number of iteration to converge.
    :param verbose: bool indicating wheter or not to print Jacobian & parameter

    :returns stable_branch: tuple of parameter list, state list and infected
                            fraction list for the stable branch
    """
    if verbose:
        print("Entering the stable_branch method")
        print("---------------------------------")
    nmax = state_meta[1]
    gm = state_meta[3]
    param_list = []
    stationary_state_list = []
    infected_fraction_list = []
    #for the first state, we use the ODE directly
    param = param_init
    sm,fni = initialize(state_meta, initial_density=0.9)
    inf_mat = infection_matrix(beta,nmax,args=(param_init,*fixed_args))
    # sm,fni = advance(sm, fni, tvar, inf_mat, state_meta)
    r0 = mf_from_state(fni,inf_mat,state_meta)
    r = stationary_state(r0,inf_mat,state_meta,stable=True,
                         h=h,rtol=rtol,max_iter=max_iter,
                         verbose=verbose)
    sm,fni = unflatten(state_from_mf(r,inf_mat,state_meta),state_meta)
    param_list.append(param)
    stationary_state_list.append((sm,fni))
    infected_fraction_list.append(infected_fraction(sm,gm))
    #hereafter we use the stationary_state method
    branch_stable = True
    current_param_var = param_var*1.
    while branch_stable:
        r0 = mf_from_state(fni,inf_mat,state_meta)
        J0 = jac_mf_map(r0,inf_mat,state_meta)
        if verbose:
            print(f"Jacobian value at : {J0}, parameter : {param}, mean-field :{r0}")

        I0 = infected_fraction(sm,gm)
        if (-(J0 - 1) >= Jtol and I0 > min_density) : #far enough from bifurcation
            step_small_enough = False
            while not step_small_enough:
                param += current_param_var
                inf_mat = infection_matrix(beta,nmax,args=(param,*fixed_args))
                r = stationary_state(r0,inf_mat,state_meta,stable=True,
                                     h=h,rtol=rtol,max_iter=max_iter,
                                     verbose=verbose)
                J = jac_mf_map(r,inf_mat,state_meta)
                if abs(J - J0) <= Jtol/2:
                    step_small_enough = True
                    if abs(J - J0) <= Jtol/10:
                        #step probably too small
                        current_param_var *= 2
                else:
                    param -= current_param_var
                    current_param_var /= 2
                    if verbose:
                        print(f"Jacobian difference too big : {abs(J-J0)}")
                        print(f"Reducing parameter variation : {current_param_var}")
            sm,fni = unflatten(state_from_mf(r,inf_mat,state_meta),state_meta)
            param_list.append(param)
            stationary_state_list.append((sm,fni))
            infected_fraction_list.append(infected_fraction(sm,gm))
        else:
            branch_stable = False
    return (param_list,stationary_state_list,infected_fraction_list)


def unstable_branch(fni, beta, state_meta, param_init, param_var, fixed_args=(),
                  init_iter=10, h=0.01, rtol=10**(-8), Jtol=10**(-2),
                  min_density=10**(-4),max_iter=1000, verbose=True):
    """unstable_branch captures the non-trivial unstable branch starting from
       some initial parameter until the bifurcation.

    IMPORTANT : it is assumed that the direction of the parameter variation
                is such that the system goes toward a bifurcation. At the
                bifurcation, it is expected that the Jacobian J_r = 1.
                It is also assumed that J_r > 1, since we are targetting
                an unstable branch.

    We converge the stationary state at the current parameter value. We
    then vary the parameter. If the Jacobian J_r varies by a value more than
    Jtol/2, we adapt the parameter variation value until it respects it.
    With this adaptive parameter variation, we ensure that we will not miss
    the bifurcation, which serves as a stopping criterion.

    Note : it is recommended to seed the algo with fni obtained from the
           nearby stable branch

    :param fni: array for the initial state of the groups
    :param beta: function of two argument (n,i) or more (parameters) for the
                 infection rate
    :param state_meta: StateMeta object encoding the information of the
                       structure.
    :param param_init: float for the initial value of the parameter to vary
    :param param_var: float (positive or negative) for the parameter variation
    :param fixed_args: tuple of extra arguments for beta that are fixed.
    :param init_iter: int for the number of initial iteration without checking
                      the Jacobian. It is necessary, since we already start
                      near a bifurcation and wouldn't want an early stop.
    :param h: float for the step size of the linearly implicit Euler.
    :param rtol: float for the relative variation desired on the convergence.
    :param Jtol: float for tolerance on Jacobian variation for stability
    :param min_density: float for the minimal infected fraction allowed
    :param max_iter: int for the maximum number of iteration to converge.
    :param verbose: bool indicating wheter or not to print Jacobian & parameter

    :returns unstable_branch: tuple of parameter list, state list and infected
                              fraction list for the unstable branch
    """
    if verbose:
        print("Entering the unstable_branch method")
        print("-----------------------------------")
    nmax = state_meta[1]
    gm = state_meta[3]
    param_list = []
    stationary_state_list = []
    infected_fraction_list = []
    #for the first state, we use the provided fni as proxy
    param = param_init
    inf_mat = infection_matrix(beta,nmax,args=(param_init,*fixed_args))
    r0 = mf_from_state(fni, inf_mat, state_meta)*0.95 #we perturbe it a little
    r = stationary_state(r0,inf_mat,state_meta,stable=False,
                         h=h,rtol=rtol,max_iter=max_iter)
    # r = stationary_state_safe(inf_mat, state_meta, r_upp=r0, r_low=10**(-14))
    sm,fni = unflatten(state_from_mf(r,inf_mat,state_meta),state_meta)
    param_list.append(param)
    stationary_state_list.append((sm,fni))
    infected_fraction_list.append(infected_fraction(sm,gm))

    branch_unstable = True
    current_param_var = param_var*1.
    it = 0
    while branch_unstable or it < init_iter:
        it += 1
        r0 = mf_from_state(fni,inf_mat,state_meta)
        J0 = jac_mf_map(r0,inf_mat,state_meta)
        if verbose:
            print(f"Jacobian value at : {J0}, parameter : {param}, mean-field :{r0}")
        I0 = infected_fraction(sm,gm)
        if ((J0 - 1) >= Jtol and I0 > min_density) or it < init_iter:
            step_small_enough = False
            while not step_small_enough:
                param += current_param_var
                inf_mat = infection_matrix(beta,nmax,args=(param,*fixed_args))
                r = stationary_state(r0,inf_mat,state_meta,stable=False,
                                     h=h,rtol=rtol,max_iter=max_iter,
                                     verbose=verbose)
                # r = stationary_state_safe(inf_mat, state_meta, r_upp=r0, r_low=10**(-14))
                J = jac_mf_map(r,inf_mat,state_meta)
                if abs(J - J0) <= Jtol/2:
                    step_small_enough = True
                    if abs(J - J0) <= Jtol/10:
                        #step probably too small
                        current_param_var *= 2
                else:
                    param -= current_param_var
                    current_param_var /= 2
                    if verbose:
                        print(f"Jacobian difference too big : {abs(J-J0)}")
                        print(f"Reducing parameter variation : {current_param_var}")
            sm,fni = unflatten(state_from_mf(r,inf_mat,state_meta),state_meta)
            param_list.append(param)
            stationary_state_list.append((sm,fni))
            infected_fraction_list.append(infected_fraction(sm,gm))
        else:
            branch_unstable = False
    return (param_list,stationary_state_list,infected_fraction_list)



@jit(nopython=True)
def stationary_state(r0, inf_mat, state_meta, stable=True,
                    h=0.01, rtol=10**(-12), max_iter=5000, verbose=True):
    """stationary_state uses the self-consistent relation to find a fixed
    point (stationary state).

    We consider a transformed system of ODE

    dr/dt = sgn*(M(r) - r)

    where M(r) is the map used to calculate the stationary state value
    of the mean-field r. The sgn (-1, 1) is used to target unstable/stable
    fixed points of the system.

    We use a linearly implicit Euler method to solve the above transformed ODE

    r_{t+h} ~ r_t + sgn*(M(r) - r)/(1/h - sgn*(J_r(r) - 1))

    In this case, a fixed point is stable if the jacobian of the mean-field
    self-consistent equation J_r < 1, whereas it is unstable if J_r > 1.

    :param r0: float for the initial mean-field value
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.
    :param stable: bool indicating if the targeted stationary state is stable
                   or unstable.
    :param h: float for the step size of the linearly implicit Euler.
    :param rtol: float for the relative variation desired on the convergence.
    :param max_iter: int for the maximum number of iteration to converge.

    return r: float for the converged mean-field
    """
    if stable:
        sgn = 1
    else:
        sgn = -1
    converged = False
    it = 0
    r = r0
    while not converged and it < max_iter:
        it += 1
        diff = mf_map(r, inf_mat, state_meta) - r
        rnew = r + sgn*diff/\
                ((abs(diff)+10**(-12))/h - sgn*(jac_mf_map(r, inf_mat, state_meta) - 1))
        if abs(rnew/r-1) <= rtol:
            converged = True
        r = rnew
    return r

def stationary_state_safe(inf_mat, state_meta, r_upp=None, r_low=10**(-12)):
    """stationary_state uses the self-consistent relation to find a fixed
    point (stationary state). It uses a safe brentq routine.

    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.

    return r: float for the converged mean-field
    """
    if r_upp is None:
        sm,fni = initialize(state_meta, initial_density=0.99)
        r_upp_ = mf_from_state(fni,inf_mat,state_meta)
    else:
        r_upp_ = r_upp
    r = brentq(mf_root_equation,r_low,r_upp_,args=(inf_mat,state_meta))
    return r



@jit(nopython=True)
def mf_from_state(fni,inf_mat,state_meta):
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    r = np.sum(inf_mat[2:,:]*(nmat[2:,:]-imat[2:,:])*fni[2:,:]*pnmat[2:,:])
    r /= np.sum((nmat[2:,:]-imat[2:,:])*fni[2:,:]*pnmat[2:,:])
    return r


@jit(nopython=True)
def state_from_mf(r,inf_mat,state_meta):
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    #get node state
    sm = (1/(1+m*r))
    rho = r*excess_susceptible_membership(m,gm,sm)
    #get groups state
    fni = np.zeros((nmax+1,nmax+1),dtype=np.float64)
    for n in range(2, nmax+1):
        if pn[n] > 0:
            fni[n][0] = 1
            for i in range(n):
                #unnormalized assignation
                fni[n][i+1] += ((n-i)*(rho+inf_mat[n][i])+i)*fni[n][i]/(i+1)
                if i > 0:
                    fni[n][i+1] -= (n-i+1)*(inf_mat[n][i-1]+rho)*fni[n][i-1]/(i+1)
            #normalize
            fni[n] /= np.sum(fni[n])
    return flatten(sm,fni,state_meta)

@jit(nopython=True)
def mf_map(r, inf_mat, state_meta):
    """mf_map takes a mean-field r as argument and return a mean-field r

    :param r: float representing a mean-field value
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection
                    rate
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    v = state_from_mf(r,inf_mat,state_meta)
    mmax = state_meta[0]
    nmax = state_meta[1]
    fni = v[mmax+1:].reshape((nmax+1,nmax+1))
    return mf_from_state(fni,inf_mat,state_meta)


@jit(nopython=True)
def mf_root_equation(r, inf_mat, state_meta):
    """mf_root_equation is the function that needs to be equal 0 for a
    stationary state

    :param r: float representing a mean-field value
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection
                    rate
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    return mf_map(r, inf_mat, state_meta) - r

@jit(nopython=True)
def jac_mf_map_(r, inf_mat, state_meta, eps=10**(-8)):
    r2 = mf_map(r+eps,inf_mat,state_meta)
    r1 = mf_map(r,inf_mat,state_meta)
    return (r2-r1)/eps



@jit(nopython=True)
def jac_mf_map(r, inf_mat, state_meta):
    """jac_mf_map takes a mean-field r as argument and return the jacobian

    Note : it assumes we are at a fixed point (stationary state)

    :param r: float representing a mean-field value
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection
                    rate
    :param state_meta: tuple of arrays encoding information of the structure.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    v = state_from_mf(r,inf_mat,state_meta)
    sm = v[:mmax+1]
    fni = v[mmax+1:].reshape((nmax+1,nmax+1))
    rho = r*excess_susceptible_membership(m,gm,sm)

    #calculate dfni and drdrho
    dni = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(1,n+1):
            dni[n][i] = np.sum(1/(inf_mat[n][:i]+rho))
    dfni = -fni*np.outer(np.sum(fni[:,1:]*dni[:,1:],axis=1),np.ones(nmax+1)) + fni*dni
    u = np.sum(inf_mat*(nmat-imat)*fni*pnmat)
    v = np.sum((nmat-imat)*fni*pnmat)
    du = np.sum(inf_mat*(nmat-imat)*dfni*pnmat)
    dv = np.sum((nmat-imat)*dfni*pnmat)
    drdrho = du/v-u*dv/v**2

    #calculate dsm and drhodr
    dsm = -m/(1+m*r)**2
    u = np.sum(m*(m-1)*sm*gm)
    v = np.sum(m*sm*gm)
    du = np.sum(m*(m-1)*dsm*gm)
    dv = np.sum(m*dsm*gm)
    drhodr = u/v + r*(du/v - u*dv/v**2)

    return drdrho*drhodr
