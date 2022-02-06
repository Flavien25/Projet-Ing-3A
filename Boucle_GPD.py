# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:46:15 2022

@author: flavien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import time
import random

#On importe les données
data = pd.read_csv('data_roulage_km.csv', delimiter=',',header=None)  
data = np.array(data)
quantileR=28263

## On importe les données puis on en prend que 200 aléatoirement
random.seed(1)
#On importe les données
dataP = pd.read_csv('data_roulage_km.csv', delimiter=',',header=None)  
dataP = np.array(dataP)
quantileR=28263 #Valeur trouvé avec le code reliability
data=[]
new=[]
for b in range(200):
    new.append(random.randint(0,len(dataP)-1))
new=np.sort(new)    
for i in range(200):
    data.append(dataP[new[i]])
data=np.array(data) 

def MEF(data, u): # On définit la Mean Excess Function
    somme=0
    compteur=0
    res=[]
    res=np.array(res)
    resUP=[]
    resDOWN=[]
    for i in range(len(u)):
        Liste=data[data>U[i]]-U[i]
        somme=np.sum(Liste)
        compteur=len(Liste)
        res=np.append(res,somme/compteur)
        resUP=np.append(resUP,somme/compteur+(0.99*(np.std(Liste)/np.sqrt(len(Liste)))))
        resDOWN=np.append(resDOWN,somme/compteur-(0.99*(np.std(Liste)/np.sqrt(len(Liste)))))
        somme=0
        compteur=0
        Liste=np.array([])
    return [np.array(res).T,resUP,resDOWN]

    

start=time.time()
U=np.linspace(0,data[-1],10000) #On crée un vecteur U des seuils avec des valeurs entre les 2 extrémités 
MEP=MEF(data,U)[0]
MEPUP=MEF(data,U)[1] # ça sera la limite sup de l'intervalle de confiance
MEPDOWN=MEF(data,U)[2] # ça sera la limite inf de l'intervalle de confiance
plt.figure()
plt.plot(U,MEP,label="MEF") # On trace le Mean Excess Plot
plt.plot(U,MEPUP,'r--',label="Limite supérieur de l'intervalle de confiance") # On trace l'intervalle de confiance
plt.plot(U,MEPDOWN,'b--',label="Limite inférieur de l'intervalle de confiance")
plt.legend()
plt.xlabel("Seuil U", fontsize=11)
plt.ylabel("Fonction des excès moyens", fontsize=11)
plt.show()
end=time.time()
print('\n')
print('Le temps d éxécution est de :', round(end-start),'secondes')

E=[]
S=[]
valeur_seuil=5003 # Valeur initiale des seuils qu'on va tester
pas = 200 # Valeur du pas pour tester les seuils
for s in range(60): # On fait la boucle ici pour tester les différents seuils
            
    #On prend les valeurs avec le seuil précedemment trouvé 
    T=valeur_seuil
    Seuil=[]
    for a in range(len(data)): # On crée une nouvelle liste avec le seuil qu'on a trouvé
        if data[a]>=T:
            Seuil.append(data[a])
    Seuil=np.array(Seuil)
    Seuil=ot.Sample(Seuil) # On le transforme en sample avec openturns pour utiliser cette librairie
    # Attention, ici il faut bien respecter l'ordre des conversions de données i.e. Liste -> Array -> Sample sinon on a des problèmes de dimension

    myFittedDist = ot.GeneralizedParetoFactory().buildAsGeneralizedPareto(Seuil)
    print('\n')
    print('---------- Generalized Pareto Distribution ----------')
    print('Les paremètres de la loi sont :','\n',myFittedDist)
    
    ## On trace la fonction déterminée et l'histogramme ici cette partie est commentée pour ne pas avoir tous les graphiques affichés
    # graph = myFittedDist.drawPDF()
    # graph.add(ot.HistogramFactory().build(Seuil).drawPDF())
    # graph.setTitle("Generalized Pareto distribution fitting on the data")
    # graph.setColors(["black", "red"])
    # graph.setLegends(["GPD fitting", "histogram"])
    # graph.setLegendPosition("topright")
    # view = viewer.View(graph)
    # axes = view.getAxes()
    # # On trace le QQ plot
    # graph = ot.VisualTest.DrawQQplot(Seuil, myFittedDist)
    # view = viewer.View(graph)
    
    ### ON cherche le quantile à 99%
    #On veut voir 99% du TOTAL et pas juste de la GPD donc il faut évaluer quelles proportions il y a avant le seuil
    AvantSeuil=len(data)-len(Seuil)
    PourcentageAvantSeuil=AvantSeuil/len(data) #Pourcentage des valeurs avant seuil
    #On veut 99% des valeurs totales donc on va chercher à quoi cela revient pour le pourcentage de la GPD
    PourcentageGPD= (0.99-PourcentageAvantSeuil)/(1-PourcentageAvantSeuil)
    #Ici on a le % qu'il faut prendre de la GPD pour avoir 99% des valeurs totales
    print('\n')
    print('------------Seuil à 99% sur GPD-------------')
    quantileGPD=np.array(myFittedDist.computeQuantile(PourcentageGPD))[0]
    print(quantileGPD)
    print('\n')
    print('------------Écart avec la valeur réelle en %-----')
    ecart=(abs(quantileGPD-quantileR)/quantileR)*100
    print(ecart)
    E.append(ecart) # On stocke toutes les valeurs des écarts qu'on a calculé
    S.append(valeur_seuil) # On stocke toutes les valeurs des seuils qu'on a essayé
    valeur_seuil=valeur_seuil+pas # On augmente la valeur du seuil avec le pas précédemment défini
    
    
E=np.array(E)
S=np.array(S)
Emin=min(E) # On récupère le plus petit écart calculé
indice=E.argmin() # On récupère pour quelle indice on a cet écart 
valeur_seuil=S[indice] # On récupère enfin le seuil pour lequel on a l'écart le plus faible

plt.figure()
plt.plot(S,E) # On plot les écarts en fonction du seuil 
plt.legend()
plt.title('Écarts en fonction du seuil')
plt.xlabel("Seuil U", fontsize=11)
plt.ylabel("Écart du quantile à 99% avec la valeur empirique (en %)", fontsize=11)
plt.show()

print('\n')
print('====================RÉSULTATS====================')
print('---------Écart avec la valeur réelle---------')
print(Emin,'%')
print('---------Valeur du seuil enregistré---------')
print(valeur_seuil)
