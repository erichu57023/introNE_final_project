# -*- coding: utf-8 -*-
# @Author: Eric Hu
# @Email: eric.hu@nyulangone.org
# @Date:  2025-07-06 16:16:17
# @Last Modified by:   Eric Hu
# @Last Modified time: 2025-07-09 00:22:38

import numpy as np
import scipy

from ..core import PointNeuron, addSonicFeatures

@addSonicFeatures
class QuadraticHNap(PointNeuron):
    ''' Quadratic model of subthreshold-resonant neurons, composed of a persistent Na current and 
        hyperpolarization-gated inward (h) current.
    '''

    # Neuron name
    name = 'quadratic_HNap'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0   =  1e-2   # Membrane capacitance (F/m2)
    Vm0   =  None   # Membrane potential (mV)
    Iapp  = -25     # Applied current (mA/m2)

    # Threshold parameters
    Vspk  = 20
    Vrst  = -80

    # Reversal potentials (mV)
    ENap  =  55.0   # Sodium
    Eh    = -20.0   # Hyperpolarization-inward
    ELeak = -65.0   # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gp    =   5.0   # Sodium
    gh    =  15.0   # Hyperpolarization-inward
    gLeak =   5.0   # Non-specific leakage

    # Adjusted variables
    def __new__(cls):
        gh = cls.gh
        gp = cls.gp
        gL = cls.gLeak
        Eh = cls.Eh
        ENap = cls.ENap
        ELeak = cls.ELeak
        Iapp = cls.Iapp

        f_Ve = lambda V: gh * (V-Eh) * (-gp * cls.dpinf_dVm(V) * (V-ENap) - gp * cls.pinf(V) - gL) - gh * (Iapp - gp * cls.pinf(V) * (V-ENap) - gL * (V-ELeak))
        rt = scipy.optimize.root_scalar(f_Ve, x0=-50, x1=-40, xtol=1e-16)
        Ve = rt.root
        cls.Ve = Ve
        re = (Iapp - gp * cls.pinf(Ve) * (Ve-ENap) - gL * (Ve-ELeak)) / (gh * (Ve-Eh))
        cls.re = re
        cls.Fe = -gp * cls.pinf(Ve) * (Ve - ENap) - gh * re * (Ve - Eh) - gL * (Ve - ELeak) + Iapp
        cls.tau1_bar = cls.taur(Ve)
        cls.g_1 = gh * (Ve - Eh) * cls.drinf_dVm(Ve)
        cls.g_2 = gp * (Ve - ENap) * cls.dpinf_dVm(Ve)
        cls.g_L = gL + gh * re + gp * cls.pinf(Ve) + cls.g_2
        g_c = - gp * (cls.d2pinf_dVm2(Ve) * (Ve - ENap) + 2 * cls.dpinf_dVm(Ve)) / 2
        cls.g_c = np.abs(g_c)
        cls.sign = np.sign(g_c)
        cls.beta = (cls.rinf(Ve) - re) / cls.drinf_dVm(Ve)
        cls.xi = cls.beta * cls.dtaur_dVm(Ve) / cls.taur(Ve)
        cls.uss = cls.calculate_ss_u()
        cls.Vm0 = cls.Vss = cls.uss + Ve
        return super(QuadraticHNap, cls).__new__(cls)

    # ------------------------------ States names & descriptions ------------------------------
    
    states = {
        'w': 'adjusted dynamic gating constant'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def pinf(Vm):
        return 1 / (1 + np.exp(-(Vm + 38)/6.5))

    @staticmethod
    def dpinf_dVm(Vm):
        exc = np.exp((Vm + 38)/6.5)
        return exc / (6.5 * (1 + exc)**2)  # mV^-1

    @staticmethod
    def d2pinf_dVm2(Vm):
        exc = np.exp((Vm + 38)/6.5)
        return (4 * (1 - exc) * exc) / (169 * (1 + exc)**3)  #mV^-2
    
    @staticmethod
    def taup(_):
        return 0.15e-3 # s

    @staticmethod
    def rinf(Vm):
        return 1 / (1 + np.exp((Vm + 79.2)/9.78))

    @staticmethod
    def drinf_dVm(Vm):
        exc = np.exp(50*(Vm + 79.2)/489)
        return (-50 * exc) / (489 * (1 + exc)**2)  # mV^-1

    @staticmethod
    def taur(Vm):
        return 0.51e-3 / (np.exp((Vm - 1.7)/10) + np.exp(-(Vm + 340)/52)) + 1e-3  # s
    
    @staticmethod
    def dtaur_dVm(Vm):
        ex1 = np.exp((Vm - 1.7)/10)
        ex2 = np.exp(-(Vm + 340)/52)
        return -0.51e-3 * (ex1 / 10 - ex2 / 52) / ((ex1 + ex2) ** 2)  # s/mV

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def dw_dt(cls, Vm, w):
        return ((1 - cls.xi) * (Vm - cls.Ve) - w + cls.beta) / cls.tau1_bar

    @classmethod
    def derStates(cls):
        return {
            'w': lambda Vm, x: cls.dw_dt(Vm, x['w'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def calculate_ss_u(cls):
        f_uss = lambda u: cls.Fe - cls.g_L * u + cls.sign * cls.g_c * u**2 - cls.g_1 * (cls.beta + (1-cls.xi) * u)
        rt = scipy.optimize.root_scalar(f_uss, x0=0, x1=-10, xtol=1e-16)
        return rt.root

    @classmethod
    def steadyStates(cls):
        return {
            'w': lambda Vm: (cls.beta + (1-cls.xi) * cls.uss)
        }

    # ------------------------------ Membrane currents ------------------------------
    
    @classmethod
    def iFe(cls):
        ''' Linear model current '''
        return -cls.Fe
    
    @classmethod
    def iV(cls, Vm):
        ''' Voltage-dependent current '''
        u = Vm - cls.Ve
        return (cls.g_L * u) - (cls.sign * cls.g_c * u**2)

    @classmethod
    def iW(cls, w):
        ''' Quadratized gate-dynamic current '''
        return w * cls.g_1 # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iFe': lambda Vm, _: cls.iFe(),
            'iV': lambda Vm, _: cls.iV(Vm),
            'iW': lambda _, x: cls.iW(x['w'])
        }


@addSonicFeatures
class QuadraticKsNap(PointNeuron):
    ''' Quadratic model of subthreshold-resonant neurons, composed of a persistent Na current and 
        slow K current.
    '''

    # Neuron name
    name = 'quadratic_KsNap'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0   =  1e-2   # Membrane capacitance (F/m2)
    Vm0   =  np.inf   # Membrane potential (mV)
    Iapp  = -6      # Applied current (mA/m2)
    
    # Threshold parameters
    Vspk  = -40
    Vrst  = -80

    # Reversal potentials (mV)
    ENap  =  55.0   # Sodium
    EKs   = -90.0   # Potassium
    ELeak = -54.0   # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gp    =   2.1   # Sodium
    gKs   =  20.0   # Potassium
    gLeak =   1.0   # Non-specific leakage


    # Adjusted variables
    def __new__(cls):
        gKs = cls.gKs
        gp = cls.gp
        gL = cls.gLeak
        EKs = cls.EKs
        ENap = cls.ENap
        ELeak = cls.ELeak
        Iapp = cls.Iapp

        f_Ve = lambda V: gKs * (V-EKs) * (-gp * cls.dpinf_dVm(V) * (V-ENap) - gp * cls.pinf(V) - gL) - gKs * (Iapp - gp * cls.pinf(V) * (V-ENap) - gL * (V-ELeak))
        rt = scipy.optimize.root_scalar(f_Ve, x0=-60, x1=-70, xtol=1e-16)
        Ve = rt.root
        cls.Ve = Ve
        re = (Iapp - gp * cls.pinf(Ve) * (Ve-ENap) - gL * (Ve-ELeak)) / (gKs * (Ve-EKs))
        cls.re = re
        cls.Fe = -gp * cls.pinf(Ve) * (Ve - ENap) - gKs * re * (Ve - EKs) - gL * (Ve - ELeak) + Iapp
        cls.tau1_bar = cls.tauq(Ve)
        cls.g_1 = gKs * (Ve - EKs) * cls.dqinf_dVm(Ve)
        cls.g_2 = gp * (Ve - ENap) * cls.dpinf_dVm(Ve)
        cls.g_L = gL + gKs * re + gp * cls.pinf(Ve) + cls.g_2
        g_c = - gp * (cls.d2pinf_dVm2(Ve) * (Ve - ENap) + 2 * cls.dpinf_dVm(Ve)) / 2
        cls.g_c = np.abs(g_c)
        cls.sign = np.sign(g_c)
        cls.beta = (cls.qinf(Ve) - re) / cls.dqinf_dVm(Ve)
        cls.xi = cls.beta * cls.dtauq_dVm(Ve) / cls.tauq(Ve)
        cls.uss = cls.calculate_ss_u()
        cls.Vm0 = cls.Vss = cls.uss + Ve
        return super(QuadraticKsNap, cls).__new__(cls)

    # ------------------------------ States names & descriptions ------------------------------
    
    states = {
        'w': 'adjusted dynamic gating constant'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def pinf(Vm):
        return 1 / (1 + np.exp(-(Vm + 38)/6.5))

    @staticmethod
    def dpinf_dVm(Vm):
        exc = np.exp((Vm + 38)/6.5)
        return exc / (6.5 * (1 + exc)**2)  # mV^-1

    @staticmethod
    def d2pinf_dVm2(Vm):
        exc = np.exp((Vm + 38)/6.5)
        return (4 * (1 - exc) * exc) / (169 * (1 + exc)**3)  #mV^-2
    
    @staticmethod
    def taup(_):
        return 0.15e-3 # s

    @staticmethod
    def qinf(Vm):
        return 1 / (1 + np.exp(-(Vm + 35)/6.5))

    @staticmethod
    def dqinf_dVm(Vm):
        exc = np.exp((Vm + 35)/6.5)
        return exc / (6.5 * (1 + exc)**2)  # mV^-1

    @staticmethod
    def tauq(Vm):
        return 90e-3  # s
    
    @staticmethod
    def dtauq_dVm(Vm):
        return 0  # s/mV

    # ------------------------------ States derivatives ------------------------------
    
    @classmethod
    def dw_dt(cls, Vm, w):
        return ((1 - cls.xi) * (Vm - cls.Ve) - w + cls.beta) / cls.tau1_bar

    @classmethod
    def derStates(cls):
        return {
            'w': lambda Vm, x: cls.dw_dt(Vm, x['w'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def calculate_ss_u(cls):
        f_uss = lambda u: cls.Fe - cls.g_L * u + cls.sign * cls.g_c * u**2 - cls.g_1 * (cls.beta + (1-cls.xi) * u)
        rt = scipy.optimize.root_scalar(f_uss, x0=0, x1=-10, xtol=1e-16)
        return rt.root

    @classmethod
    def steadyStates(cls):
        return {
            'w': lambda Vm: (cls.beta + (1-cls.xi) * cls.uss)
        }

    # ------------------------------ Membrane currents ------------------------------
    
    @classmethod
    def iFe(cls):
        ''' Linear model current '''
        return -cls.Fe
    
    @classmethod
    def iV(cls, Vm):
        ''' Voltage-dependent current '''
        u = Vm - cls.Ve
        return (cls.g_L * u) - (cls.sign * cls.g_c * u**2)

    @classmethod
    def iW(cls, w):
        ''' Quadratized gate-dynamic current '''
        return w * cls.g_1 # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iFe': lambda Vm, _: cls.iFe(),
            'iV': lambda Vm, _: cls.iV(Vm),
            'iW': lambda _, x: cls.iW(x['w'])
        }