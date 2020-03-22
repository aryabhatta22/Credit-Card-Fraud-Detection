
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

                            # Part 1 - SOM

from minisom import MiniSom 

som = MiniSom(x=10, y=10, input_len = 15, sigma = 1.0, learning_rate = 0.5) 
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T) 
colorbar()
markers = ['o', 's'] 
colors = ['r','g']
for i,x in enumerate(X): 
    w = som.winner(x)  
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]],  
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)   
    
show()
    
mapping = som.win_map(X) 
frauds = np.concatenate((mapping[(2,7)], mapping[(3,4)], mapping[(1,6)]), axis = 0) # Change the value of coordinates based on the grid image for correct values
frauds = sc.inverse_transform(frauds) 

                            # Part 2 - ANN

customer = df.iloc[:, 1:].values
is_fraud = np.zeros(len(df))
for i in range(len(df)):
    if(df.iloc[i,0] in frauds):
        is_fraud[i] = 1
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customer = sc.fit_transform(customer)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()                   
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15 ))                                                 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 15 ))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

classifier.fit(customer, is_fraud, batch_size = 1, epochs = 2)

y_pred = classifier.predict(customer)
y_pred = np.concatenate((df.iloc[:, 0:1].values, y_pred), axis = 1) 
y_pred = y_pred[y_pred[:, 1].argsort()]
