import numpy as np
import matplotlib.pyplot as plt
import pandas
from cvxpy import *
from xlrd import open_workbook
import functools
import math
import scipy
from scipy.optimize import minimize

df = pandas.read_excel('Assets_3.xlsx')

print(df.columns)
stocks = df['Stock'].values
#print(len(stocks))
bonds = df['Bond'].values
tbills = df['T-bill'].values
#print(bonds)
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))

stocks_mean = np.mean(np.array(stocks))
print("Stocks mean monthly = %f" % stocks_mean)
stocks_annual_mean = math.pow((1 + stocks_mean), 12) - 1
print("Stocks mean annual = %f" % stocks_annual_mean)
bonds_mean = np.mean(np.array(bonds))
print("Bonds mean monthly = %f" % bonds_mean)
bonds_annual_mean = math.pow((1 + bonds_mean), 12) - 1
print("Bonds mean annual = %f" % bonds_annual_mean)
tbills_mean = np.mean(np.array(tbills))
print("T-Bills mean monthly = %f" % tbills_mean)
tbills_annual_mean = math.pow((1 + tbills_mean), 12) - 1
print("Tbills mean annual = %f" % tbills_annual_mean)

stocks_std = np.std(np.array(stocks))
print("Std for stocks monthly = %f" % stocks_std)
stocks_annual_std = math.sqrt(12) * stocks_std
print("Std for stocks annual = %f" % stocks_annual_std)
bonds_std = np.std(np.array(bonds))
print("Std for bonds monthly = %f" % bonds_std)
bonds_annual_std = math.sqrt(12) * bonds_std
print("Std for bonds annual = %f" % bonds_annual_std)
tbills_std = np.std(np.array(tbills))
print("Std for t-bills monthly = %f" % tbills_std)
tbills_annual_std = math.sqrt(12) * tbills_std
print("Std for tbills annual = %f" % tbills_annual_std)

stocks_variance = stocks_std * stocks_std
bonds_variance = bonds_std * bonds_std
tbills_variance = tbills_std * tbills_std
print("Variance for stocks monthly = %f" % stocks_variance)
print("Variance for bonds monthly = %f" % bonds_variance)
print("Variance for t-bills monthly = %f" % tbills_variance)

#stocks - asset1, bonds - asset2, tbills - asset3
#now calculate the covariance of asset1 and 2
#cov_12 = np.cov(np.array(np.array(stocks), np.array(bonds), np.array(tbills)))
#print(cov_12)
#correlation = np.corrcoef(stocks, bonds, tbills)
c = list()
c.append(np.array(stocks))
c.append(np.array(bonds))
c.append(np.array(tbills))
correlation = np.corrcoef(np.array(c))
#print(correlation)
cov = np.cov(np.array(c))
#print(cov)
covariance_annual = 12 * cov #annualized covariance matrix
#print(covariance_annual)

#stocks - asset1, bonds - asset2, tbills - asset3
#now calculate the covariance of asset1 and 2
#cov_12 = np.cov(np.array(np.array(stocks), np.array(bonds), np.array(tbills)))
#print(cov_12)
#correlation = np.corrcoef(stocks, bonds, tbills)
c = list()
c.append(np.array(stocks))
c.append(np.array(bonds))
c.append(np.array(tbills))
#correlation = np.corrcoef(np.array(c))
#print(correlation)
cov = np.cov(np.array(c))
#print(cov)
covariance_annual = 12 * cov #annualized covariance matrix
#print(covariance_annual)
'''c2 = list()
c2.append(stocks)
c2.append(bonds)
c2.append(tbills)
correlation2 = np.corrcoef(np.array(c2))
print(correlation2)'''

mu = np.array([stocks_annual_mean, bonds_annual_mean, tbills_annual_mean]) #vectorized form for the
#mu = np.reshape(mu, (mu.shape[0], 1)) #reshape the average returns vector - (3, 1)
np.random.seed(1)
n = 3 #the number of assets in our portfolio

def objective(w):
    ret = w.dot(mu)
    risk_free = tbills_annual_mean  # the risk free rate acc to the question
    risk = np.sqrt(w.dot(covariance_annual).dot(w.T))
    '''var = 0.0
    # to speed up sum covariance over i < j and variance over i
    for i in range(n):
        for j in range(n):
            var += w[i] * w[j] * covariance_annual[i] * covariance_annual[j] * correlation[i, j]'''
    #print(risk)
    #print(var)
    num = ret - risk_free
    denom = risk
    #print(num)
    #print(denom)
    sharpe_ratio = num / denom # the equation to be minimized which is Sharpe Ratio
    sharpe_ratio = -1 * sharpe_ratio
    #print(sharpe_ratio)
    return sharpe_ratio




#w = np.array([np.ones(n)]) #the variable of weights for each asset
#w = np.reshape(w, (w.shape[0], 1))
equal_weight = float (1 / n)
w = list()
for i in range(n):
    w.append(equal_weight)
w = np.array(w)
#print(w)
#w = np.array([1/n, 1/n, 1])
#print('before going to solver printing ws')
#print(w)
bound = [(0.0, 1.0)] * n
#target_returns = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.109]) #we will optimize for each one
# of these target returns
#print(len(target_returns))
#returns_data = np.zeros(len(target_returns))
#risk_data = np.zeros(len(target_returns))

#returns_data = 0
#risk_data = 0

weights = list()
sol = minimize(objective, w, constraints=(
    {'type': 'eq', 'fun': lambda w:  np.sum(w) - 1. }),
                method='SLSQP', bounds = bound)
#print(i)
#print(sol.x)
weights_returned = sol.x
#print(weights_returned)
#weights.append(weights_returned)
#weights[i] = sol.x
#print(w)
ret_data_sharpe = weights_returned.dot(mu)
vol_data_sharpe = np.sqrt(weights_returned.dot(covariance_annual).dot(weights_returned.T))
print(ret_data_sharpe)
print(vol_data_sharpe)

#np.random.seed(1)
#n = 3 #the number of assets in our portfolio
w = Variable(n) #the variable of weights for each asset
mu = np.reshape(mu, (mu.shape[0], 1)) #reshape the average returns vector - (3, 1)
ret = mu.T * w #this will be the expected return
'''print(ret.size)
print(ret)'''
risk = quad_form(w, covariance_annual) #w.T * Q * w
#print(covariance_annual)
target_returns = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.109]) #we will optimize for each one
# of these target returns
#print(len(target_returns))
returns_data = np.zeros(len(target_returns))
risk_data = np.zeros(len(target_returns))
for i in range(len(target_returns)):
    prob = Problem(Minimize(risk),
               [sum_entries(w) == 1,
                w >= 0,
                ret >= target_returns[i]])
    prob.solve() #solve the problem for the particular target expected return
    returns_data[i] = ret.value
    risk_data[i] = sqrt(risk).value

print(returns_data)
print(risk_data)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, returns_data, 'bs-',)
for xy in zip(risk_data, returns_data):
    #plt.plot(xy, 'bs')
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.plot(ret_data_sharpe, vol_data_sharpe, 'x')
X = np.ones(1)
X[0] = ret_data_sharpe
Y = np.ones(1)
Y[0] = vol_data_sharpe
for xy in zip(X, Y):
    #plt.plot(xy, 'bs')
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
plt.show()


