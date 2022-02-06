# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:09:10 2021

@author: flavien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import openturns.viewer as viewer
import time
import random

## Ce code a pour but de trouver un seuil pour utiliser une GPD, après avoir trouvé ce seuil on ira trouver les paramètres
## de la GPD et comparer le quantile à 99% 

# ## On importe les données
# data = pd.read_csv('data_roulage_km.csv', delimiter=',',header=None)  
# data = np.array(data)
# quantileR=28263 #Valeur trouvée avec le code Tendance_centrale

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

###On Calcule la fonction MEF

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


k=0
while MEP[k]+4>MEP[k+1]: # Ici, on veut trouver le moment où la courbe se stabilise avec une sensibilité
    k=k+1
valeur_seuil=U[k]
print('\n')
print('Valeur du seuil =',valeur_seuil)
#On prend les valeurs avec le seuil précedemment trouvé 
T=valeur_seuil
Seuil=[]
for a in range(len(data)): # On crée une nouvelle liste avec le seuil qu'on a trouvé
    if data[a]>=T:
        Seuil.append(data[a])
Seuil=np.array(Seuil)
Seuil=ot.Sample(Seuil) # On le transforme en sample avec openturns pour utiliser cette librairie
# Attention, ici il faut bien respecter l'ordre des conversions de données i.e. Liste -> Array -> Sample sinon on a des problèmes de dimension

myFittedDist = ot.GeneralizedParetoFactory().buildAsGeneralizedPareto(Seuil) # On fit les valeurs sur une GPD
print('\n')
print('---------- Generalized Pareto Distribution ----------')
print('Les paremètres de la loi sont :','\n',myFittedDist)

# On trace la fonction déterminée et l'histogramme
graph = myFittedDist.drawPDF()
graph.add(ot.HistogramFactory().build(Seuil).drawPDF())
graph.setTitle("Generalized Pareto distribution fitting on the data")
graph.setColors(["black", "red"])
graph.setLegends(["GPD fitting", "histogram"])
graph.setLegendPosition("topright")
view = viewer.View(graph)
axes = view.getAxes()
# On trace le QQ plot
graph = ot.VisualTest.DrawQQplot(Seuil, myFittedDist)
view = viewer.View(graph)

### ON cherche le quantile à 99%
#On veut voir 99% du TOTAL et pas juste de la GPD donc il faut évaluer quelles proportions il y a avant le seuil
AvantSeuil=len(data)-len(Seuil)
PourcentageAvantSeuil=AvantSeuil/len(data) #Pourcentage des valeurs avant seuil ça fait environ 56% de toutes les valeurs
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
