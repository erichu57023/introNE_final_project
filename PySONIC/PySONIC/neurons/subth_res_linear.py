# -*- coding: utf-8 -*-
# @Author: Eric Hu
# @Email: eric.hu@nyulangone.org
# @Date:  2025-07-05 18:04:26
# @Last Modified by:   Eric Hu
# @Last Modified time: 2025-07-07 02:19:47

import numpy as np
import scipy

from ..core import PointNeuron, addSonicFeatures

@addSonicFeatures
class LinearHNap(PointNeuron):
    ''' Linear model of subthreshold-resonant neurons, composed of a persistent Na current and 
        hyperpolarization-gated inward (h) current.
    '''

    # Neuron name
    name = 'linear_HNap'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0   =  1e-2   # Membrane capacitance (F/m2)
    Vm0   =  None   # Membrane potential (mV)
    Iapp  = -25     # Applied current (mA/m2)

    # Threshold parameters
    Vspk  = -40
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
        f_Vss = lambda V: -cls.gp * cls.pinf(V) * (V - cls.ENap) - cls.gh * cls.rinf(V) * (V - cls.Eh) - cls.gLeak * (V - cls.ELeak) + cls.Iapp
        rt = scipy.optimize.root_scalar(f_Vss, x0=-50, x1=-60)
        Vm0 = rt.root
        cls.Vm0 = Vm0
        cls.Vss = Vm0
        cls.tau1_bar = cls.taur(Vm0)
        cls.g_1 = cls.gh * (Vm0 - cls.Eh) * cls.dr_dVm(Vm0)                                 # Adjusted conductance of h
        cls.g_2 = cls.gp * (Vm0 - cls.ENap) * cls.dp_dVm(Vm0)                               # Adjusted conductance of Nap
        cls.g_L = cls.gLeak + cls.gh * cls.rinf(Vm0) + cls.gp * cls.pinf(Vm0) + cls.g_2     # Adjusted conductance of leak
        return super(LinearHNap, cls).__new__(cls)

    # ------------------------------ States names & descriptions ------------------------------
    
    states = {
        'w': 'adjusted dynamic gating constant'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def pinf(Vm):
        return 1 / (1 + np.exp(-(Vm + 38)/6.5))

    @staticmethod
    def dp_dVm(Vm):
        exc = np.exp((Vm + 38)/6.5)
        return exc / (6.5 * (1 + exc)**2)

    @staticmethod
    def taup(Vm):
        return 0.15e-3 # s

    @staticmethod
    def rinf(Vm):
        return 1 / (1 + np.exp((Vm + 79.2)/9.78))

    @staticmethod
    def dr_dVm(Vm):
        exc = np.exp(50*(Vm + 79.2)/489)
        return (-50 * exc) / (489 * (1 + exc)**2)

    @staticmethod
    def taur(Vm):
        return 0.51e-3 / (np.exp((Vm - 1.7)/10) + np.exp(-(Vm + 340)/52)) + 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def dw_dt(cls, Vm, w):
        if (Vm >= cls.Vspk):
            dt = 5e-4   # s
            target_dV_dt = (cls.Vrst - Vm) / dt
            target_w = (-cls.g_L * (Vm - cls.Vss) - cls.Cm0 * target_dV_dt) / cls.g_1
            return (target_w - w) / dt
        else:
            return (Vm - cls.Vss - w) / cls.tau1_bar

    @classmethod
    def derStates(cls):
        return {
            'w': lambda Vm, x: cls.dw_dt(Vm, x['w'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'w': lambda _: 0
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iLeak(cls, Vm):
        ''' Nonspecific leakage current '''
        return cls.g_L * (Vm - cls.Vm0)  # mA/m2

    @classmethod
    def iW(cls, w):
        ''' Linearized dynamic current '''
        return cls.g_1 * w  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iW': lambda Vm, x: cls.iW(x['w']),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }


@addSonicFeatures
class LinearKsNap(PointNeuron):
    ''' Linear model of subthreshold-resonant neurons, composed of a persistent Na current and 
        slow K current.
    '''

    # Neuron name
    name = 'linear_KsNap'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0   =  1e-2   # Membrane capacitance (F/m2)
    Vm0   =  None   # Membrane potential (mV)
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
        f_Vss = lambda V: -cls.gp * cls.pinf(V) * (V - cls.ENap) - cls.gKs * cls.qinf(V) * (V - cls.EKs) - cls.gLeak * (V - cls.ELeak) + cls.Iapp
        rt = scipy.optimize.root_scalar(f_Vss, x0=-50, x1=-60)
        Vm0 = rt.root
        cls.Vm0 = Vm0
        cls.Vss = Vm0
        cls.tau1_bar = cls.tauq(cls.Vm0)
        cls.g_1 = cls.gKs * (Vm0 - cls.EKs) * cls.dqinf_dVm(Vm0)                      # Adjusted conductance of h
        cls.g_2 = cls.gp * (Vm0 - cls.ENap) * cls.dpinf_dVm(Vm0)                    # Adjusted conductance of Nap
        cls.g_L = cls.gLeak + cls.gKs * cls.qinf(Vm0) + cls.gp * cls.pinf(Vm0) + cls.g_2     # Adjusted conductance of leak
        return super(LinearKsNap, cls).__new__(cls)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'w': 'adjusted dynamic gating state'
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
    def taup(Vm):
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

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def dw_dt(cls, Vm, w):
        return (Vm - cls.Vss - w) / cls.tau1_bar

    @classmethod
    def derStates(cls):
        return {
            'w': lambda Vm, x: cls.dw_dt(Vm, x['w'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'w': lambda _: 0
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iLeak(cls, Vm):
        ''' Nonspecific leakage current '''
        return cls.g_L * (Vm - cls.Vm0)  # mA/m2

    @classmethod
    def iW(cls, w, Vm):
        ''' Linearized dynamic current '''
        return cls.g_1 * w  # mA/m2

    @classmethod
    def iRst(cls, w, Vm):
        ''' Reset current '''
        if (Vm >= cls.Vspk):
            dt = 5e-4   # s
            target_dV_dt = (cls.Vrst - Vm) / dt
            target_Iin = cls.Cm0 * target_dV_dt + cls.g_L * (Vm - cls.Vss) + cls.g_1 * w
            return target_Iin
        else:
            return 0

    @classmethod
    def currents(cls):
        return {
            'iW': lambda Vm, x: cls.iW(x['w'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }
    