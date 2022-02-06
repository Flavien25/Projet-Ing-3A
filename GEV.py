# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:31:01 2021

@author: Louis

Ce code a pour principal objectif de donner la distribution et le mode du quantile à 99% d'un jeu de données. Pour ce faire, il s'appuie 
sur la librairie Openturns et la théorie des valeurs extrèmes.
On importe d'abord les données (en entier ou seulement 200 valeurs choisies aléatoirement en fonction de la section commentée), puis on utilise la 
méthode des "block maxima" couplée à celle du "bootstrap" pour dientifier les paramètres de la GEV et son mode (valeur la plus probable du quantile).
Ici, on étudie des données de roulage, mais en changeant l'adresse du fichier et les légendes des graphes, ce code peut être facilement adapté pour 
n'importe quel usage.
"""

##Import des librairies et fonctions nécessaires
import pandas as pd
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import random as rd

ot.Log.Show(ot.Log.NONE)                                            # Desactivation de l'affichage des warnings de la console


## Import des données standard
rd.seed(1)
D = np.sort(np.array(pd.read_csv('data_roulage_km.csv'))[:, 0])     # Lecture et conversion en tableau du fichier .csv (adresse à modifier si besoin)
                                                                    # puis classement des données (les fonctions suivantes ne fonctionnent qu'avec 
data = np.zeros((len(D),1))                                         # des données croissantes)
for i in range(len(data)):                                          # retransformation en tableau à deux dimensions pour les besoins du traitement
    data[i] = np.array(D[i])


# ## Import des données avec reduction à 200 valeurs aléatoires
# rd.seed(1)                                                          # "graine" pour le générateur de nombres aléatoires (pour obtenir des résutats constants)
# dataP = np.sort(np.array(pd.read_csv('data_roulage_km.csv'))[:, 0]) # Lecture et conversion en tableau du fichier .csv (adresse à modifier si besoin)
# D = []                                                              # Futur tableau de l'échantillon de 200 valeurs

# for b in range(200):                                                # Boucle de génération des nouvelles données avec:
#     new = rd.randint(0,len(dataP)-1)                                # création d'un entier aléatoire entre 0 et longueur du tableau de données - 1
#     D.append(dataP[new])                                            # ajout à l'échantillon de la donnée de position correspondante
    
# D = np.sort(D)                                                      # classement des données (les fonctions suivantes ne fonctionnent qu'avec des 
# data = np.zeros((len(D),1))                                         # données croissantes)

# for i in range(len(data)):                                          # retransformation en tableau pour les besoins du traitement
#     data[i] = np.array(D[i])


## Initialisation du Bootstrap
b = 1000                                                            # Nombre de boucles du bootstrap
p = 60001                                                           # Nombre de valeurs d'échantillonnage des GEV du bootstrap
start = 1E4                                                         # Début de l'échantillonnage des GEV
finish = 7E4                                                        # Fin de l'échantillonnage des GEV

Q = []                                                              # Future liste des quantiles (pour moyenne après bootstrap)
Mu = []                                                             # Future liste des valeurs de mu (pour moyenne après bootstrap)
Sigma = []                                                          # Future liste des valeurs de sigma (pour moyenne après bootstrap)
Xi = []                                                             # Future liste des valeurs de xi (pour moyenne après bootstrap)
x = np.linspace(start, finish, p)                                   # Tableau d'abscisses pour le GEV
F = np.zeros((b, p))                                                # Futur tableau de stockage des GEV calculées

n = 100                                                             # Taille des sous-échantillons de données
l = round(len(data)/100)                                            # Nombre d'échantillons à générer
    
for i in range(b):                                                  # Début du bootstrap
    
    
    ## Génération des échantillons et relevé des max des échantillons ("block maxima")
    S = len(data) * np.random.rand(l, n)                            # Génération des entiers qui donneront la position des données à échantillonner dans le
    S = S.astype(int)                                               # tableau initial
    
    M = np.zeros(len(S))                                            # Future liste des "block maxima"
    
    for k in range(len(S)):
        S[k] = data[S[k]][:, 0]                                     # Sous-échantillonnage
        M[k] = max(S[k])                                            # Relevé du maximum des échantillons


    ## Fit de la GEV sur les max relevés
    sample = ot.Sample(M.reshape(-1,1))                             # Conversion du type d'échantillon pour utilisation avec openturns
    dist = ot.GeneralizedExtremeValueFactory().buildAsGeneralizedExtremeValue(sample)
                                                                    # "fit" de la GEV
    
    # print('\n')                                                     # Affichage des paramètres et du domaine d'attraction de la GEV fittée
    # print('paramètres de la GEV')
    # print(dist)
    # print('\n')
    # print('domaine attraction GEV et paramètres')
    # print(dist.getActualDistribution())
    # print('\n')
    
    
    ## Mode de la GEV (estimation du quantile à 99% par méthode analytique et empirique)
    mu = dist.getMu()                                               # Extraction des paramètres et ajout dans les listes concernées
    sigma = dist.getSigma()
    xi = dist.getXi()
    Mu.append(mu)
    Sigma.append(sigma)
    Xi.append(xi)
    
    if xi == 0:                                                     # Début du calcul de la GEV et du quantile
        t = np.exp(-(x - mu)/sigma)
        m = mu
    else:
        t = (1 + xi * (x - mu)/sigma)**(-1/xi)
        m = mu + sigma*(((1 + xi)**-xi - 1)/xi)
    
    Q.append(m)                                                     # Stockage du quantile dans la liste prévue à cet effet
    f = (1/sigma)*(t**(1 + xi))*np.exp(-t)                          # Fin du calcul de la GEV
    F[i] = f                                                        # Stockage de la GEV calculée dans la liste prévue à cet effet
                                                                    # Fin du bootstrap
    
## Calcul et affichage du quantile moyen
q = round(np.mean(Q))
med = round(np.percentile(Q, 50))
print('\n')
print('quantile à 99% : ', q, ' (médiane = ', med, ')')


## Affichage de la superposition de toutes les GEV calculées
plt.figure()
for i in range(b):
    plt.plot(x, F[i])
plt.title('Courbes de toutes les GEV calculées')
plt.xlabel('Roulage (km)')


## Affichage des paramètres de la distribution GEV estimée
mu = round(np.mean(Mu))
sigma = round(np.mean(Sigma))
xi = round(np.mean(Xi), 3)
print('\n')
print('--- Paramètres de la GEV ---')
print('mu = ', mu)
print('sigma = ', sigma)
print('xi = ', xi)
