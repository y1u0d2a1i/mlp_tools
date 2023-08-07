import pandas as pd
import numpy as np
from typing import List

from mlptools.utils.utils import get_mass_atom, get_atom_number

class YamamuraTawaraSputteringYieldCalculator():
    """
    Z1, M1 : ion, or projectile atom

    Z2, M2: target atom, or substate atom 

    """
    def __init__(self, projectile, target):
        df = pd.read_csv("/Users/y1u0d2/desktop/Project/mlp_tools/data/Table_I.csv")
        self.param = self.make_param(df)
        self.flag = False
        self.projectile = projectile
        self.target = target
        self.Z1 = get_atom_number(projectile)
        self.Z2 = get_atom_number(target)
        self.M1 = get_mass_atom(projectile)
        self.M2 = get_mass_atom(target)
        
    def make_param(self, df):
        z2list = df.loc[:,"Z2"].values
        uslist = df.loc[:,"Us"].values
        qlist = df.loc[:,"Q"].values
        wlist = df.loc[:,"W"].values
        slist = df.loc[:,"s"].values

        Us = {}
        Q = {}
        W = {}
        S = {}
        for z2,us,q,w,s in zip(z2list,uslist,qlist,wlist,slist):
            #print(z2,us,q,w,s)
            Us[z2] = us
            Q[z2] = q
            W[z2] = w
            S[z2] = s

        param = {}
        param["s"]= S
        param["Us"] = Us
        param["W"] = W
        param["Q"] = Q
        return param

    #


    def snTF(self,ep):
        """ eq 4 """
        flag = self.flag
        A = 3.441 * np.sqrt(ep) * np.log(ep+2.718)
        B = 1+6.355*np.sqrt(ep) + ep*(6.882*np.sqrt(ep)-1.708)
        r = A/B
        if flag:
            print("snTF",r)
        return r

    def epsilon(self,Z1,Z2,M1,M2,E):
        """ eq.22 """
        param = self.param
        flag = self.flag
        tw = 2.0/3.0
        A = Z1*Z2*np.sqrt(Z2**tw+Z1**tw)
        B = M2/(M1+M2)
        ep = 0.03255/A*B*E
        if flag:
            print("ep",ep)
        return ep

    def ke(self,Z1,Z2,M1,M2,E):
        """ eq. 20 """
        param = self.param
        flag = self.flag
        ft = 3.0/4.0
        tw = 2.0/3.0
        A = (M1+M2)**(3.0/2.0)
        B = M1**(3.0/2.0)*np.sqrt(M2)
        C = Z1**tw*np.sqrt(Z2)
        D = (Z2**tw+Z1**tw)**ft
        r = 0.079*A/B*C/D 
        #print(A,B,C,D)
        if flag:
            print("ke",r)
        return r

    def Sn(self,Z1,Z2,M1,M2,E):
        """ eq. 21 """
        param = self.param
        flag = self.flag
        tw = 2.0/3.0
        z12 = np.sqrt(Z1**tw+Z2**tw)
        A = Z1*Z2/z12
        #print("Z1*Z2",Z1*Z2,z12)
        B = M1/(M1+M2)
        sntfv = self.snTF(self.epsilon(Z1,Z2,M1,M2,E))
        r = 84.78*A*B*sntfv
        if flag:
            print("Sn",r)
        return r

    def Eth(self,Z1,Z2,M1,M2):
        param = self.param
        flag = self.flag
        s = param["s"][Z2]
        Q = param["Q"][Z2]
        Us = param["Us"][Z2]
        W = param["W"][Z2]    
        gamma = 4*M1*M2/(M1+M2)**2
        """ eq. 18 """
        if M1>=M2:
            eth = 6.7/gamma*Us
        else:
            eth = (1+5.7*(M1/M2))/gamma* Us
        return eth
    
    def get_sputtering_yield(self, energy_list:List[int]):
        energy_list = np.array(energy_list)
        sp = self.SY(
            Z1=self.Z1,
            Z2=self.Z2,
            M1=self.M1,
            M2=self.M2,
            E=energy_list
        )
        return sp

    def SY(self,Z1,Z2,M1,M2,E):
        """ eq. 15 """
        flag = self.flag
        param = self.param
        s = param["s"][Z2]
        Q = param["Q"][Z2]
        Us = param["Us"][Z2]
        W = param["W"][Z2]


        """ eq. 17 """
        if M1<=M2:
            alpha = 0.249*(M2/M1)**0.56 + 0.0035*(M2/M1)**1.5
        else:
            alpha = 0.0875*(M2/M1)**(-0.15) + 0.165*(M2/M1)
        if flag:
            print("alpha",alpha)


        eth = self.Eth(Z1,Z2,M1,M2)
        if flag:
            print("Eth",eth)

        """ eq. 16 """
        Gamma = W/ (1+(M1/7)**3) 

        A = Q*alpha/Us
        B = self.Sn(Z1,Z2,M1,M2,E)/(1.0+Gamma*self.ke(Z1,Z2,M1,M2,E)*self.epsilon(Z1,Z2,M1,M2,E)**0.3)
        C = (1-np.sqrt(eth/E))**s
        Y = 0.042*(A*B*C)
        return Y