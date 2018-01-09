
# coding: utf-8

# # Fitting curve to data
# Within this notebook we do some data analytics on historical data to feed some real numbers into the model. Since we assume the consumer data to be resemble a sinus, due to the fact that demand is seasonal, we will focus on fitting data to this kind of curve.

# In[69]:

import numpy as np
from scipy.optimize import leastsq
import pylab as plt
import pandas as pd

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

guess_mean = np.mean(data)
guess_std = 3*np.std(data)/(2**0.5)
guess_phase = 0

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - data
est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_std*np.sin(t+est_phase) + est_mean

plt.plot(data, '.')
plt.plot(data_fit, label='after fitting')
plt.plot(data_first_guess, label='first guess')
plt.legend()
plt.show()


# ## import data for our model
# This is data imported from statline CBS webportal.

# In[70]:

importfile = 'CBS Statline Gas Usage.xlsx'
df = pd.read_excel(importfile, sheetname='Month', skiprows=1)
df.drop(['Onderwerpen_1', 'Onderwerpen_2', 'Perioden'], axis=1, inplace=True)

df


# In[71]:

# transpose
df = df.transpose()



# In[72]:

new_header = df.iloc[0]
df = df[1:]
df.rename(columns = new_header, inplace=True)


# In[73]:

df


# In[74]:

df['Via regionale netten'].plot()
plt.show()


# In[99]:

#print(data)
N = 84
t = np.linspace(1, 84, N)
b = 603
m = 3615
data = b + m*(.5 * (1 + np.cos((t/6)*np.pi))) + 100*np.random.randn(N) # create artificial data with noise

#print(t)
print(type(data[0]))
print(data)
plt.plot(t, data)
plt.show()
#print(est_std, est_phase, est_mean)
guess_std = 3*np.std(data)/(2**0.5)
print(guess_std)
data2 = df['Via regionale netten'].values
data3 = np.array(data2)
data3.astype(np.float64)
print(type(data3[0]))
print(data2)
print((len(data2)))


# In[102]:

#b = self.base_demand
#m = self.max_demand
#y = b + m * (.5 * (1 + np.cos((x/6)*np.pi)))
b = 603
m = 3615

N = 84 # number of data points
t = np.linspace(1, 84, N)
#data = b + m*(.5 * (1 + np.cos((t/6)*np.pi))) + 100*np.random.randn(N) # create artificial data with noise
#data = df['Via regionale netten'].values
data = data3
guess_mean = np.mean(data)
guess_std = 3*np.std(data)/(2**0.5)
guess_phase = 0

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_mean + guess_std*(.5 * (1 + np.cos((t/6)*np.pi + guess_phase)))

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*(.5 * (1 + np.cos((t/6)*np.pi+x[1]))) + x[2] - data
est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_mean + est_std*(.5 * (1 + np.cos((t/6)*np.pi + est_phase)))

plt.plot(data, '.')
plt.plot(data_fit, label='after fitting')
plt.plot(data_first_guess, label='first guess')
plt.legend()
plt.show()
print(est_std, est_phase, est_mean)


# In[ ]:



