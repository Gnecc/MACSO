#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACSO - K-Means clustering.

Author: Bryan Cisneros
Date: January 2024

Dataset: conjunto_de_datos_normalizados.xlsx

Summary:
    This file contains an experiment script for the MACSO repository.
    It was standardized to remove external web references and Spanish-only
    header metadata, and to use a consistent English documentation header.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, roc_curve, roc_auc_score, auc
from pandas import read_csv
from sklearn import preprocessing
 
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


dataframe = read_csv("alumnos.csv")
dataframe.head()
dataset = dataframe.values

#dataframe['Etiqueta'].value_counts()

dataframe.describe()

print(dataframe.groupby('Etiqueta').size())

dataframe.drop(['Etiqueta'],1).hist()
plt.show()

X = dataset[:,0:9].astype(float)
#Y = dataset[:,9]

et = preprocessing.LabelEncoder()
et.fit(dataframe['Etiqueta'])
y = et.transform(dataframe['Etiqueta'])

kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

# Getting the values and plotting it
f1 = dataframe['calculo'].values
f2 = dataframe['Literario'].values


copy =  pd.DataFrame()
copy['Etiqueta']=dataframe['Etiqueta'].values
copy['label'] = labels;
cantidadGrupo =  pd.DataFrame()
cantidadGrupo['color']=colores
cantidadGrupo['cantidad']=copy.groupby('label').size()
print(cantidadGrupo)

group_referrer_index = copy['label'] ==0
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['Etiqueta']=[0,1,2,3]
diversidadGrupo['cantidad']=group_referrals.groupby('Etiqueta').size()
diversidadGrupo

n_classes = 4

hot_y = np.zeros((y.size, y.max()+1))
hot_y[np.arange(y.size),y] = 1

hot_labels = np.zeros((labels.size, labels.max()+1))
hot_labels[np.arange(labels.size),labels] = 1

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(hot_y[:, i], hot_labels[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(hot_y.ravel(), hot_labels.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Kmeans ROC AUC")
plt.legend(loc="lower right")
plt.show()