# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:31:01 2021

@author: Louis

Ce code a pour principal objectif de donner la meilleure distribution correspondant à la tendance centrale d'un jeu de données. Pour ce faire, il s'appuie 
sur la librairie Reliability.
On importe d'abord les données (en entier ou seulement 200 valeurs choisies aléatoirement en fonction de la section commentée), puis on regarde quelles 
distributions leur correspondent le mieux (d'abord parmi tous types de distribution, puis plus finement sur les trois meilleures).
Ici, on étudie des données de roulage, mais en changeant l'adresse du fichier et les légendes des graphes, ce code peut être facilement adapté pour 
n'importe quel usage.
"""

## Import des librairies et fonctions nécessaires
import pandas as pd
import numpy as np
from reliability.Fitters import Fit_Weibull_3P, Fit_Lognormal_2P, Fit_Gamma_3P
from reliability.Fitters import Fit_Everything
from reliability.Probability_plotting import plot_points
from reliability.Other_functions import histogram, crosshairs
import matplotlib.pyplot as plt
import random as rd

## Import des données standard
data = pd.read_csv('data_roulage_km.csv')                           # Lecture du fichier .csv (nom ou adresse à modifier en fonction des données prises)
data = np.sort(np.array(data)[:, 0])                                # Transfert des données dans un tableau et classement (les fonctions suivantes ne 
                                                                    # fonctionnent qu'avec des données classées dans l'ordre croissant)

# ## Import des données avec reduction à 200 valeurs aléatoires
# rd.seed(1)                                                          # "graine" pour le générateur de nombres aléatoires (pour obtenir des résutats constants)
# dataP = pd.read_csv('data_roulage_km.csv')                          # Lecture du fichier .csv (nom ou adresse à modifier en fonction des données prises)
# dataP = np.array(dataP)                                             # Transfert des données dans un tableau
# data = []                                                           # Futur tableau de l'échantillon de 200 valeurs

# for b in range(200):                                                # Boucle de génération des nouvelles données avec:
#     new = rd.randint(0,len(dataP)-1)                                # création d'un entier aléatoire entre 0 et longueur du tableau de données - 1
#     data.append(dataP[new])                                         # ajout à l'échantillon de la donnée de position correspondante
    
# data = np.sort(np.array(data))                                      # Classement de l'échantillon obtenus (les fonctions suivantes ne fonctionnent qu'avec 
#                                                                     # des données classées dans l'ordre croissant)

## Tests de fit de tous les types de lois
                                                                    # La ligne suivante utilise la fonction Fit_Everything de la librairie Reliability pour 
                                                                    # obtenir les paramètres correspondant au mieux aux données, pour diverses lois de 
                                                                    # probabilités, à l'aide de la méthode du maximum de vraissemblance ainsi que des 
                                                                    # indicateurs AIC et BIC (cf. documentation pour tout changement dans les champs)
result = Fit_Everything(failures=data, show_histogram_plot=True, show_probability_plot=False, show_PP_plot=True, show_best_distribution_probability_plot=False)
Parameters=result.best_distribution                                 # Extraction des paramètres de la meilleure loi
Name=result.best_distribution_name                                  # Extraction du nom de la meilleure loi
print('\n')                                                         # Affichage du nom et des paramètres de la meilleure loi
print('---------------RESULTATS DU CALCUL---------------')
print('\n', 'Nom de la loi :', Name, '\n','\n', 'Paramètres de la loi : ', '\n', 'Alpha = ',Parameters.alpha, '\n', 'Beta = ',Parameters.beta, '\n', 'Gamma = ',Parameters.gamma)


## Fit des "meilleures" lois sur les données de roulage
P = []                                                              # Future liste des 'upper estimates" des quantiles à 99% pour chaque loi
P.append(data[round(0.99* len(data))])                              # Calcul et stockage de la valeur empirique du quantile à 99% (pour comparaison future)

plt.figure()                                                        # Nouvelle figure pour la courbe de probabilité de la fonction "fittée" ci-dessous
g3 = Fit_Gamma_3P(failures=data, percentiles=[99], CI=0.99, show_probability_plot=True)
                                                                    # La loi Gamma était la meilleure des distributions testées plus haut; on refait donc un 
                                                                    # "fit" de cette dernière pour obtenir son quantile à 99% (cf. documentation pour toute 
                                                                    # modification dans les champs)
P.append(np.array(g3.percentiles)[0, 3])                            # Stockage de la valeur du quantile à 99% de cette loi (pour comparaison future)
crosshairs()                                                        # Permet d'avoir les coordonnées d'un point du graphe en placant le curseur dessus
plt.show()

                                                                    # Les deux blocs de code suivants remplissent la même focntion pour les lois de Weibull et 
                                                                    # lognormale (deuxième et troisième meilleures des lois testées par le "Fit_everything")
plt.figure()
wb = Fit_Weibull_3P(failures=data, percentiles=[99], CI=0.99, show_probability_plot=True)                                                                    
P.append(np.array(wb.percentiles)[0, 3])
crosshairs()
plt.show()

plt.figure()
lN = Fit_Lognormal_2P(failures=data, percentiles=[99], CI=0.99, show_probability_plot=True)
P.append(np.array(lN.percentiles)[0, 3])
crosshairs()
plt.show()


## Affichages des quantiles à 99% et calcul des écarts entre réel et théorique
E = []                                                              # Future liste des écarts entre quantiles "fittés" et quantile empirique
for i in range(1, len(P)):
    E.append(100* abs(P[i] - P[0])/P[0])                            # Calcul des écarts au quantile empirique (en %)

print('\n')                                                         # Affichage des différentes valeurs du quantile
print('--- "Upper estimates" des quantiles à 99% ---')
print('    Réel          Gamma (3P)         Weibull (3P)      logNormale (2P)')
print(P)
print('--- Ecarts entre quantiles théoriques et qauntile réel ---') # Affiche des écarts au quantile empirique
print('     Gamma (3P)         Weibull (3P)       logNormale (2P)')
print(E)
print('\n')


## Comparaison des distributions après Fit aux données réelles
                                                                    # Affichage des distributions "fittées" et comparaison à l'histogramme des données
plt.figure()
histogram(data)
g3.distribution.PDF(label='Loi gamma (3 paramètres)', color='red', linestyle='--')
wb.distribution.PDF(label='Loi de Weibull (3 paramètres)', color='steelblue', linestyle='--')
lN.distribution.PDF(label='Loi lognormale (2 paramètres)', color='purple', linestyle='--')
plt.title('Comparaison des lois estimées aux données réelles')
plt.xlabel('Roulage (km)')
plt.legend()
plt.show()


## Affichage des fonctions de survie
                                                                    # Affichage des fonctions de survie pour les distributions "fittées" et comparaison 
                                                                    # à la fonction de survie empirique
plt.figure()
lN.distribution.SF(label='Fitted Lognormal (2P)', color='purple', linestyle='--')
wb.distribution.SF(label='Fitted Weibull (3P)', color='steelblue', linestyle='--')
g3.distribution.SF(label='Fitted Gamma (3P)', color='red', linestyle='--')
plot_points(failures=data,func='SF',label='failure data', alpha=0.3)
plt.xlabel('Roulage (km)')
plt.legend()
plt.show()
