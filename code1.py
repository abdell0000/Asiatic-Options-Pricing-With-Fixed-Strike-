#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:29:55 2023


"""

import numpy as np                    # bibliotheque pour vecteurs et matrices
import matplotlib.pyplot as plt       # pour le plot
from scipy.stats import norm          # distribution gaussienne
import numpy.random as npr
from tqdm import tqdm
 
class marché:
    def __init__(self,K,T,r,sigma,payoff):
        # Parametres principaux
        self.K = K        # strike
        self.T = T       # echeance
        self.r = r    # taux de l'actif sans risque
        self.sigma = sigma # volatilite du sous-jacent
        self.payoff= payoff
    def AC(self,X0,m=100000,n=225):
            delta=float(self.T/n)
            G=npr.normal(0,1,size=(m,n))
            #Log returns
            LR=(self.r-0.5*self.sigma**2)*delta+np.sqrt(delta)*self.sigma*G
            # concatenate with log(X0)
            LR=np.concatenate((np.log(X0)*np.ones((m,1)),LR),axis=1)
            # cumsum horizontally (axis=1)
            LR=np.cumsum(LR,axis=1)
            Spaths=np.exp(LR)
            Spaths=Spaths[:,0:len(Spaths[0,:])-1]
            #take the average over each row
            Sbar=np.mean(Spaths,axis=1)
            payoff=np.exp(-self.r*self.T)*np.maximum(Sbar-self.K,0) #call function
            Asian_MC_price=np.mean(payoff)
            return Asian_MC_price
 
class schema_EE:
    def __init__(self,N,J,X_min,bar_x,market):
        # Parametres de discretisation
        self.market = market    
        self.K = market.K        # strike
        self.T = market.T       # echeance
        self.r = market.r    # taux de l'actif sans risque
        self.sigma = market.sigma # volatilite du sous-jacent
        self.g= market.payoff
 
        self.N = N      # pas dans le maillage en space
        self.J = J      # pas dans le maillage en temps
 
        # essayer avec N = 10, J = 225 ou J = 220 pour apprecier la stabilite 
        self.X_min = -2.5
        self.bar_x = 5 # bord du domaine coupe
        self.X_max = self.bar_x
 
    def mat_A(self, x):
        sigma=self.sigma
        r=self.r
        N = x.size
        h = x[1] -x[0]
 
        A1 = - 1 / (2.*h*h) * np.diag(np.power(sigma*x, 2)).dot(np.eye(N,k=-1) - 2. * np.eye(N) + np.eye(N,k=1) )
        A1[0] = 0
        A2 =  1 / (2*h) * np.diag(1./self.T+r*x).dot( -np.eye(N, k=-1) + np.eye(N, k=1) )
        A2[0] = 0
        return (A1+A2)
 
    def résoudre(self,disp=True,matrixsol=False):
        # point du maillage en espace, endpoint = False parce que l'on sait deja
        # que dans le dernier point x_(N+1) = bar_x la solution approchee vaut 0
        x = np.linspace(self.X_min, self.bar_x, self.N, endpoint=False)
 
        # pas de discretisation en espace et en temps
        h = x[1] -x[0]
        k = self.T / self.J
 
        # matrice de discretisation de l'operateur en espace
        A = self.mat_A(x)
 
        # matrice du schema numerique, dans ce cas Euler Implicite
        B = np.eye(self.N) - k*A
 
 
        # condition initiale (payoff de l'option)
        U0 = self.g(x)
        matU=U0
        # Formule exacte de Black et Scholes, cas put
        # Methode numerique
 
        Uj = U0
        if disp:
            plt.figure(figsize=(12,9))
 
 
        for j in tqdm(range(self.J)):
 
             # Pas de l'algorithme, pour Euler Explicte
             # U_(j+1) = B U_j
             Uj = B.dot(Uj)
             if matrixsol:
                matU=np.vstack((matU,Uj))
             # visualisation de la solution approchee
             if (j) % 20 == 0 and disp :
                plt.plot(x, Uj, color='blue', linestyle='dashed', linewidth=1)
 
        if disp :
            # reglages de la visualisation
            plt.plot(x, U0,  color='orange', linestyle='solid', linewidth=2, label="Valeur Limite")
            plt.plot(x, Uj,  color='blue', linestyle='solid', linewidth=2, label="Valeur (approx)" )
 
            plt.xlabel("x")
            plt.ylabel("f(t,x)")
 
            plt.xlim((self.X_min,self.bar_x))
 
            plt.legend() 
        if matrixsol:
            return matU
        return Uj
       
    def erreur(self,n,pas): 
        x = np.linspace( self.X_min, self.bar_x, 1000 , endpoint=False)
        U1=schema_CN( N = 1000, J = self.J,X_min = self.X_min , bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))] 
        for i in range(10,n,pas):
           x = np.linspace(self.X_min, self.bar_x, i, endpoint=False)
           U2=schema_EI( N = i, J = self.J,X_min = self.X_min, bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))]
           yield np.max(np.abs(U1-U2)), x[1]-x[0]     
    
    
    
    
    
    
    
class schema_EI:
    def __init__(self,N,J,X_min,bar_x,market):
        # Parametres de discretisation
        self.market=market
        self.K = market.K        # strike
        self.T = market.T       # echeance
        self.r = market.r    # taux de l'actif sans risque
        self.sigma = market.sigma # volatilite du sous-jacent
        self.g= market.payoff
 
        self.N = N      # pas dans le maillage en space
        self.J = J      # pas dans le maillage en temps
 
        # essayer avec N = 10, J = 225 ou J = 220 pour apprecier la stabilite 
        self.X_min = X_min
        self.bar_x = bar_x # bord du domaine coupe
        self.X_max = self.bar_x
        self.x= np.linspace(self.X_min, self.bar_x, self.N, endpoint=False)
        self.h= self.x[1] - self.x[0]

    def mat_A(self, x):
        sigma=self.sigma
        r=self.r
        N = x.size
        h = x[1] -x[0]
 
        A1 = - 1 / (2.*h*h) * np.diag(np.power(sigma*x, 2)).dot(np.eye(N,k=-1) - 2. * np.eye(N) + np.eye(N,k=1) )
        A1[0] = 0
        A2 =  1 / (2*h) * np.diag(1./self.T+r*x).dot( -np.eye(N, k=-1) + np.eye(N, k=1) )
        A2[0] = 0
        #A3 = r * np.eye(N)
        return (A1+A2)
    def résoudre(self,disp=True, matrixsol=False):
        # point du maillage en espace, endpoint = False parce que l'on sait deja
        # que dans le dernier point x_(N+1) = bar_x la solution approchee vaut 0
        x = np.linspace(self.X_min, self.bar_x, self.N, endpoint=False)
 
        # pas de discretisation en espace et en temps
        h = x[1] -x[0]
        k = self.T / self.J
 
        # matrice de discretisation de l'operateur en espace
        A = self.mat_A(x)
 
        # matrice du schema numerique, dans ce cas Euler Implicite
        B = np.eye(self.N) + k*A
 
 
        # condition initiale (payoff de l'option)
        U0 = self.g(x)
        matU=U0
        # Methode numerique
 
        Uj = U0
        if disp:
            plt.figure(figsize=(12,9))
 
 
        for j in tqdm(range(self.J)):
 
             # Pas de l'algorithme, pour Euler Explicte
             # U_(j+1) = B U_j
             Uj = np.linalg.solve(B, Uj)
             if matrixsol:
                matU=np.vstack((matU,Uj))
             # visualisation de la solution approchee
             if (j) % 20 == 0 and disp :
                plt.plot(x, Uj, color='blue', linestyle='dashed', linewidth=1)
 
        if disp :
            # reglages de la visualisation
            plt.plot(x, U0,  color='orange', linestyle='solid', linewidth=2, label="Valeur Limite")
            plt.plot(x, Uj,  color='blue', linestyle='solid', linewidth=2, label="Valeur (approx)" )
 
            plt.xlabel("x")
            plt.ylabel("f(t,x)")
 
            plt.xlim((self.X_min,self.bar_x))
 
            plt.legend() 
        if matrixsol:
            return matU
        return Uj
    def erreur(self,n,pas): 
        x = np.linspace( self.X_min, self.bar_x, 1000 , endpoint=False)
        U1=schema_CN( N = 1000, J = self.J,X_min = self.X_min , bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))] 
        for i in range(10,n,pas):
           x = np.linspace(self.X_min, self.bar_x, i, endpoint=False)
           U2=schema_EI( N = i, J = self.J,X_min = self.X_min, bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))]
           yield np.max(np.abs(U1-U2)), x[1]-x[0]   
 
 
class schema_CN:
    def __init__(self,N,J,X_min,bar_x,market):
        # Parametres de discretisation
        self.market = market
        self.K = market.K        # strike
        self.T = market.T       # echeance
        self.r = market.r    # taux de l'actif sans risque
        self.sigma = market.sigma # volatilite du sous-jacent
        self.g= market.payoff
 
        self.N = N   # pas dans le maillage en space
        self.J = J      # pas dans le maillage en temps
 
        # essayer avec N = 10, J = 225 ou J = 220 pour apprecier la stabilite 
        self.X_min = X_min
        self.bar_x = bar_x # bord du domaine coupe
        self.X_max = self.bar_x
 
    def mat_A(self, x):
        N = x.size
        h = x[1] -x[0]
 
        A1 = - 1 / (2.*h*h) * np.diag(np.power(self.sigma*x, 2)).dot(np.eye(N,k=-1) - 2. * np.eye(N) + np.eye(N,k=1) )
        A2 = + 1 / (2*h) * np.diag(1./self.T+self.r*x).dot( -np.eye(N, k=-1) + np.eye(N, k=1) )
        A1[0] = 0
        A2[0] = 0
 
        return (A1+A2)
    def résoudre(self,disp=True, matrixsol=False):
        # point du maillage en espace, endpoint = False parce que l'on sait deja
        # que dans le dernier point x_(N+1) = bar_x la solution approchee vaut 0
        x = np.linspace(self.X_min, self.bar_x, self.N, endpoint=False)
 
        # pas de discretisation en espace et en temps
        h = x[1] -x[0]
        k = self.T / self.J
 
        # matrice de discretisation de l'operateur en espace
        A = self.mat_A(x)
 
        # matrice du schema numerique, dans ce cas Euler Explicite
        B1 = np.eye(self.N) + k/2.*A
        B2 = np.eye(self.N) - k/2.*A
 
        # condition initiale (payoff de l'option)
        U0 = self.g(x)
        matU=U0
        # Formule exacte de Black et Scholes, cas put
        # Methode numerique
 
        Uj = U0
        if disp:
           plt.figure(figsize=(12,9))
 
 
        for j in tqdm(range(self.J)):
 
             # Pas de l'algorithme, pour Euler Explicte
             # U_(j+1) = B U_j
             Uj = np.linalg.solve(B1, B2.dot(Uj))
             if matrixsol:
                 matU=np.vstack((matU,Uj))
             # visualisation de la solution approchee
             if (j) % 20 == 0 and disp :
                plt.plot(x, Uj, color='blue', linestyle='dashed', linewidth=1)
 
        if disp :
            # reglages de la visualisation
            plt.plot(x, U0,  color='orange', linestyle='solid', linewidth=2, label="Valeur Limite")
            plt.plot(x, Uj,  color='blue', linestyle='solid', linewidth=2, label="Valeur (approx)" )
 
            plt.xlabel("x")
            plt.ylabel("f(t,x)")
 
            plt.xlim((self.X_min,self.bar_x))
 
            plt.legend() 
            plt.show()
        if matrixsol:
            return matU
        return Uj
    def erreur(self,n,pas): 
        x = np.linspace( self.X_min, self.bar_x, 1000 , endpoint=False)
        U1=schema_CN( N = 1000, J = self.J,X_min = self.X_min , bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))] 
        for i in range(10,n,pas):
           x = np.linspace(self.X_min, self.bar_x, i, endpoint=False)
           U2=schema_CN( N = i, J = self.J,X_min = self.X_min, bar_x = self.bar_x, market=self.market ).résoudre(disp=False,matrixsol=True)[:,np.argmin(np.abs(x))]
           yield np.max(np.abs(U1-U2)), x[1]-x[0]
 
#%%
if __name__=="__main__":
    plt.close("all")
    market=marché(K = 100,        
                  T = 1.0, 
                  r = 0.05,    
                  sigma = 0.30,
                  payoff= lambda x : np.maximum(-x,0))
 
    schemaEE=schema_EE( N = 99,
                  J = 220,      
                  X_min = -2.5,
                  bar_x = 5, market=market)
    schemaEE.résoudre()
 
    schemaEI=schema_EI( N = 150,
                  J = 220,      
                  X_min = -2.5,
                  bar_x = 5, market=market)
    schemaEI.résoudre()
 
 
    schemaCN=schema_CN( N = 150,
                  J = 220,      
                  X_min = -2.5,
                  bar_x = 5, market=market)
    U=schemaCN.résoudre()
 
    plt.figure()
    itr=10
    x = np.linspace(schemaCN.X_min, schemaCN.bar_x, schemaCN.N, endpoint=False)
    x_axis = x[schemaCN.N//2 + itr : ]
    S = market.K/x_axis
    plt.plot(S, S*U[schemaCN.N//2 + itr: ],  color='blue', linestyle='solid', linewidth=2)
    plt.xlabel("Actif sous-jacent")
    plt.ylabel("Valeur de l'option")
    plt.xlim((min(S),max(S)))
    plt.legend( ["Valeur (approx)"])
    #plt.title(nom_methode)
 
 
    ans_CN = S*U[schemaCN.N//2 + itr: ]
    ans_CN = ans_CN [::-1]
    S = S[::-1]
    ind = np.where(S==100.)
    print('\n',ans_CN[ind])
    
#%%
    plt.figure("erreur EE")
    loger,logh= zip(*schemaEE.erreur(100,10))
    loger,logh= np.log(loger),np.log(logh)
    plt.plot(logh,loger,label="Log(erreur)")
    plt.plot(logh,logh,label="y=x")
    plt.xlabel("Log(h)")
    plt.legend()
    
    plt.figure("erreur EI")
    loger,logh= zip(*schemaEI.erreur(1000,100))
    loger,logh= np.log(loger),np.log(logh)
    plt.plot(logh,loger,label="Log(erreur)")
    plt.plot(logh,logh,label="y=x")
    plt.xlabel("Log(h)")
    plt.legend()
    
    plt.figure("erreur CN")
    loger,logh= zip(*schemaCN.erreur(1000,100))
    loger,logh= np.log(loger),np.log(logh)
    plt.plot(logh,loger,label="Valeur Limite")
    plt.plot(logh,logh,label="y=x")
    plt.xlabel("Log(h)")
    plt.legend()
    
    