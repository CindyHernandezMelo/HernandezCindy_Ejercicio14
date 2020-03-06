import pandas as pd
import sklearn.metrics
import numpy as np
import os, glob
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import f1_score

lista_train = os.listdir("train")
anotacion = np.zeros(len(lista_train))
img_gris = np.zeros([100,100,len(lista_train)], dtype = float)

for i in range(len(lista_train)):
    if int(lista_train[i][:-4])%2 == 1:
        anotacion[i] = 0
    else:
        anotacion[i] = 1       
    img= plt.imread('train/%s'%lista_train[i])
    img_gris[:,:,i] = np.mean(img, axis = 2) 

img_gris = img_gris.reshape(100*100,-1)

#def Estimar(canal, anotacion):
x_train, x_test, y_train, y_test = train_test_split(img_gris.T, anotacion, train_size=0.7)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf_hombre = svm.SVC(kernel = 'linear')
clf_hombre.fit(x_train, y_train)
y_predicted_hombre= clf_hombre.predict(x_test)

F1_mujer = f1_score(y_test, y_predicted_hombre, pos_label=0)
F1_hombre = f1_score(y_test, y_predicted_hombre, pos_label=1)

lista_test = glob.glob("test/*.jpg")
truth = pd.read_csv("test/truth_test.csv")
img_gris_test = np.zeros([100,100,len(lista_test)])

for i in range(len(lista_test)):     
    img= plt.imread(lista_test[i])
    img_gris_test[:,:,i] = np.mean(img, axis = 2) 
    lista_test[i] = lista_test[i][5:]

img_gris_val = img_gris_test.reshape(100*100,-1)

x_val = scaler.fit_transform(img_gris_val.T)
y_test_val_predicted_hombre= clf_hombre.predict(x_val)

data = {'Name':lista_test, 'Target':y_test_val_predicted_hombre}
df = pd.DataFrame(data) 
df.to_csv("test/predict_test.csv")