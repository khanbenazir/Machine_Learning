import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#loading dataset
data = pd.read_csv('C:/Users/benuk/Downloads/headbrain.csv')
print(data.shape)
data.head()

#dependent & independent variable
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

#calculating coefficients
mean_X = np.mean(X)
mean_Y = np.mean(Y)

n = len(X)
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_X)*(Y[i] - mean_Y)
    denom += (X[i] - mean_X)**2

b1 = numer/denom
b0 = mean_Y - b1*mean_X
print(b0, b1)

#predicting Y
predicted_Y = []
for i in range(n):
    yp = b1*X[i] + b0
    predicted_Y.append(yp)

#plotting scatter plot of original points and predicted Regression Line
plt.plot(X, predicted_Y, color = 'red', label = 'Regression Line')
plt.scatter(X, Y, color = 'green', label = 'Scatter Plot')
plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.legend()
plt.show()

#calculating error
nu = 0
deno = 0
for i in range(n):
    nu += (predicted_Y[i] - mean_Y)**2
    deno += (Y[i] - mean_Y)**2

squared_R = nu/deno
print(squared_R)


