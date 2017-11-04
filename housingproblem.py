'''

Alex Todorovic CMPS-3240 housingproblem.py 2017

'''

import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#reading in samples with pandas
data = pd.read_csv('data.csv', header=None, names=['size', 'price', 'city', 'bedrooms', 'baths'])

#selecting relevant features. Leaving cities out so I can convert each instance
#to a one-hot binary vector
x_data = data[['size','bedrooms','baths']].values
y_data = data[['price']].values
cities = data[['city']]

#initializing encoder from sklearn
lab_enc = LabelEncoder()
int_enc = lab_enc.fit_transform(cities)
#initializing binary encoder
onehot_encoder = OneHotEncoder(sparse=False)
int_enc = int_enc.reshape(len(int_enc), 1)
hotbin_enc = onehot_encoder.fit_transform(int_enc)
#print(hotbin_enc[0])


#normalizing x data
x_min = np.min(x_data,0)
x_max = np.max(x_data,0)
x_data = (x_data - x_min) / (x_max - x_min)
#print(x_data)

#normalising y data
y_min = np.min(y_data,0)
y_max = np.max(y_data,0)
y_data = (y_data - y_min) / (y_max - y_min)
#print(y_data)


#x_data = np.hstack((x_data, hotbin_enc))
#print(x_data)


'''Gradient descent algorithm returns weights'''
def sgd(x_train, y_train):
    #taking 70% of all examples
    n = len(x_train[0:58])
    #learning rate
    r = 0.001
    w = np.zeros((1, x_train.shape[1]))
    total_err = 0

    for x in range(0,n):
        mean_err = mean_squared_error(np.dot(w, x_train[x]),y_train[x])
        total_err += mean_err

    #print('total error at: ' + str(total_err / n))
    iters = 200
    gradient = 0
    for i in range(0, iters):
        for i in range(0,n):
            error = y_train[i][0] - (np.dot(w, x_train[i]))[0]
            gradient += error * x_train[i]
        w = w + r*gradient
        gradient = 0

    #print('weights at: ' + str(w[0]))
    return w[0]

#returns mean squared error of stochastic gradient descent algorithm
def get_error(x_test, y_test, weights):
    x_test = x_test[58:]
    predictions = np.dot(x_test, weights.T)

    return mean_squared_error(predictions, y_test[58:])

def perceptron(x, weights,threshold=0.5):
    return 1 if np.dot(x,weights) >= threshold else 0

#calculating weights
weights = sgd(x_data,y_data)

#list of 1s and 0s
targets = []
for x in x_data:
    y = perceptron(x,weights)
    targets.append(y)

actuals = []
for y in y_data:
    if y >= 0.5:
        actuals.append(1)
    else:
        actuals.append(0)



print("Mean squared error for gradient descent: " + str(get_error(x_data, y_data, weights)))

print("Perceptron accuracy: " + str(metrics.accuracy_score(targets,actuals)))







