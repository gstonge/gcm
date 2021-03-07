#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the initializaiton, insertion, sampling, etc. methods for samplable set

Author: Guillaume St-Onge <guillaume.st-onge.4@ulaval.ca>
"""

import pytest
import numpy as np
from gcm import *
from scipy.special import loggamma


def constraint_prob(sm,fni,fni_tilde,state_meta):
    assert (fni <= 1).all() and (fni >= 0).all()

def constraint_norm(sm,fni,fni_tilde,state_meta):
    for n in range(2,len(fni)):
        assert np.isclose(np.sum(fni[n]),1)

def constraint_norm_tilde(sm,fni,fni_tilde,state_meta):
    for n in range(2,len(fni_tilde)):
        assert np.isclose(np.sum(fni_tilde[n]),1)

def constraint_stubs(sm,fni,fni_tilde,state_meta):
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]
    nmean = np.sum(pn*np.arange(nmax+1))
    mmean = np.sum(m*gm)
    assert np.isclose(np.sum(fni*(nmat-imat)*pnmat),
                      np.sum(m*sm*gm)*nmean/mmean)

def constraint_eta(sm,fni,fni_tilde,state_meta):
    nmax = state_meta[1]
    m = state_meta[2]
    pn = state_meta[4]
    imat = state_meta[5]
    pnmat = state_meta[7]
    nmean = np.sum(pn*np.arange(nmax+1))

    eta = np.sum(imat*fni_tilde*pnmat)/nmean
    assert np.isclose((1.-eta)**m,sm).all()

constraint_list = [constraint_prob,constraint_norm,constraint_stubs,
                   constraint_eta, constraint_norm_tilde]

class TestFniOptimization:
    def test_reg_net_1(self):
        #structure
        nmax = 5
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[5] = 1
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 3.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)




    def test_reg_net_2(self):
        #structure
        nmax = 5
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[5] = 1
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 3.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 10**(-1)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_reg_net_3(self):
        #structure
        nmax = 5
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[5] = 1
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 0.5
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 5*10**(-1)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_hom_net_1(self):
        #structure
        nmax = 10
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[2:nmax+1] = np.exp(-0.1*np.arange(2,nmax+1))
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 3.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_hom_net_2(self):
        #structure
        nmax = 10
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[2:nmax+1] = np.exp(-0.5*np.arange(2,nmax+1))
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 3.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 5*10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_hom_net_3(self):
        #structure
        nmax = 20
        mmax = 20
        m = np.arange(mmax+1)
        n = np.arange(nmax+1)
        pn = np.zeros(nmax+1)
        param = 5
        gm = np.exp(m*np.log(param) - loggamma(m+1))
        gm[0:1] = 0
        gm /= np.sum(gm)
        pn = np.exp(n*np.log(param) - loggamma(n+1))
        pn[0:2] = 0
        pn /= np.sum(pn)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 0.5
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 10**(-3)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)

    def test_het_1(self):
        #structure
        nmax = 20
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[2:nmax+1] = (1.*np.arange(2,nmax+1))**(-3)
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 3.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 5*10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_het_2(self):
        #structure
        nmax = 20
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[2:nmax+1] = (1.*np.arange(2,nmax+1))**(-3)
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 2.
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 5*10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)


    def test_het_3(self):
        #structure
        nmax = 20
        mmax = 5
        pn = np.zeros(nmax+1)
        pn[2:nmax+1] = (1.*np.arange(2,nmax+1))**(-3)
        pn /= np.sum(pn)
        gm = np.zeros(mmax+1)
        gm[5] = 5
        gm /= np.sum(gm)
        state_meta = get_state_meta(mmax, nmax, gm, pn)

        #infection
        nu = 0.5
        beta = lambda n,i,trate,nu: trate*i**nu
        trate = invasion_threshold_safe(beta, gm, pn, fixed_args=(nu,))*1.1
        inf_mat = infection_matrix(beta,nmax,args=(trate,nu))
        initial_density = 5*10**(-2)

        sm,fni,fni_tilde = optimize_fni(initial_density,inf_mat,state_meta)

        for const in constraint_list:
            const(sm,fni,fni_tilde,state_meta)

