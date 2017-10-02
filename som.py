# Self Organizing Map

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Credit_Card_Applications.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(10,10, input_len = 15, sigma =1, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X,100)

from pylab import colorbar,pcolor,bone
bone()
pcolor(som.distance_map().T)
colorbar()

mappings = som.win_map(X)
