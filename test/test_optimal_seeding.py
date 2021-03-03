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



def constraint(fni,state_meta):
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]
    return np.sum(fni*(nmat-imat)*pnmat)


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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]

        assert np.isclose(fni_lp[nmax],fni[nmax]).all()

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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        assert np.isclose(fni_lp[nmax],fni[nmax],rtol=10**(-3),atol=10**(-5)).all()

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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        assert np.isclose(fni_lp[nmax],fni[nmax],rtol=10**(-3),atol=10**(-5)).all()


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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        assert np.isclose(fni_lp[2:nmax],fni[2:nmax],rtol=10**(-3),atol=10**(-5)).all()

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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        assert np.isclose(fni_lp[2:nmax],fni[2:nmax],rtol=10**(-3),atol=10**(-5)).all()

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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        assert np.isclose(objective_function(fni_lp,inf_mat,state_meta),
                          objective_function(fni,inf_mat,state_meta),
                          rtol=10**(-3))


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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]

        assert np.isclose(fni_lp[2:nmax],fni[2:nmax],rtol=10**(-3),atol=10**(-5)).all()


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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]

        assert np.isclose(fni_lp[2:nmax],fni[2:nmax],rtol=10**(-3),atol=10**(-5)).all()

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

        fni_lp = optimize_fni_lp(initial_density,inf_mat,state_meta)[1]
        fni = optimize_fni(initial_density,inf_mat,state_meta)[1]
        # print(fni_lp)
        # print(fni)
        # print(objective_function(fni_lp,state_meta,inf_mat))
        # print(objective_function(fni,state_meta,inf_mat))
        # print(constraint(fni_lp,state_meta))
        # print(constraint(fni,state_meta))

        assert np.isclose(fni_lp[2:nmax],fni[2:nmax],rtol=10**(-3),atol=10**(-5)).all()

