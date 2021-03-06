# Generate data for long only portfolio optimization.
import numpy as np
import matplotlib.pyplot as plt
import pandas
from cvxpy import *
from xlrd import open_workbook
import functools
import math
import cvxopt as opt
from cvxopt import blas, solvers
import numpy
import itertools

def is_sum_one(tup):
    sum = 0
    for i in range(len(tup)):
        sum += tup[i]
    if sum == 1:
        return True
    else:
        return False


df = pandas.read_excel('Assets_3.xlsx')

print(df.columns)
stocks = df['Stock'].values
#print(len(stocks))
bonds = df['Bond'].values
tbills = df['T-bill'].values
for i in range(len(tbills)):
    stocks[i] = stocks[i] + 1
    bonds[i] = bonds[i] + 1
    tbills[i] = tbills[i] + 1 #R
#print(tbills)
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
'''def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))'''
#print(functools.reduce(lambda x, y: x*y, stocks)**(1.0/len(stocks)))
gmean_stocks_monthly = geo_mean(stocks)
print('Geo-mean monthly stocks = %f' % gmean_stocks_monthly)
gmean_bonds_monthly = geo_mean(bonds)
print('Geo-mean monthly bonds = %f' % gmean_bonds_monthly)
gmean_tbills_monthly = geo_mean(tbills)
print('Geo-mean monthly tbills = %f' % gmean_tbills_monthly)

gmean_stocks_annually = math.pow(gmean_stocks_monthly, 12)
gmean_bonds_annually = math.pow(gmean_bonds_monthly, 12)
gmean_tbills_annually = math.pow(gmean_tbills_monthly, 12)

'''print(geo_mean_overflow(stocks))
print(geo_mean_overflow(bonds))
print(geo_mean_overflow(tbills))'''



'''stocks_mean = np.mean(np.array(stocks))
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
print("Tbills mean annual = %f" % tbills_annual_mean)'''

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
#correlation = np.corrcoef(np.array(c))
#print(correlation)
covariance_monthly = np.cov(np.array(c)) #monthly covariance matrix
#print(cov)
covariance_annual = 12 * covariance_monthly #annualized covariance matrix

geomean_monthly_stocks_r = gmean_stocks_monthly - 1
geomean_monthly_bonds_r = gmean_bonds_monthly - 1
geomean_monthly_tbills_r = gmean_tbills_monthly - 1

mu_r = np.array([gmean_stocks_monthly, gmean_bonds_monthly, gmean_tbills_monthly])
#mu_r = np.array([gmean_stocks_annually, gmean_bonds_annually, gmean_tbills_annually])
#mu_r = np.array([geomean_monthly_stocks_r, geomean_monthly_bonds_r, geomean_monthly_tbills_r])
#mu_r = np.reshape(mu_r, (mu_r.shape[0], 1)) #reshape the average returns vector - (3, 1)
#vectorized form for expected monthly geometric mean returns
#print(mu)
scenario_matrix = np.random.multivariate_normal(mu_r, covariance_monthly, 10000) #10000 scenarios
# of expected monthly returns have been generated - (10000, 3) shape
#scenario_matrix = np.random.multivariate_normal(mu_r, covariance_annual, 10000) #10000 scenarios
#print(scenario_matrix[0])
#print(scenario_matrix)

np.random.seed(1)
n = 3 #the number of assets in our portfolio

p = np.asmatrix(np.mean(scenario_matrix.T, axis=1)) #(1, 3)
print(p.shape)

y = np.linspace(0.0, 1.0, num=100, endpoint = False)
w1 = list()
w2 = list()
w3 = list()
#print(y)
for i in range(y.size):
    w1.append(float(round(y[i], 2)))
    w2.append(float(round(y[i], 2)))
    w3.append(float(round(y[i], 2)))
w1.append(1)
w2.append(1)
w3.append(1)

w = list()
w.append(w1)
w.append(w2)
w.append(w3)
#print(w)
d = list(itertools.product(*w))

sum_to_one = list()
for i in range(len(d)):
    if(is_sum_one(d[i]) is True):
        sum_to_one.append(d[i])
    else:
        continue

ws = list()
ws.append(np.array(sum_to_one))
print(ws)
w_s = np.array(np.array(ws))
print(w_s)
t = w_s[0][2]
f = np.asmatrix(t)
print(f)
print(f.T)
print(p)
print(scenario_matrix.shape)
print(scenario_matrix[0])

l = p * f.T
m = scenario_matrix[0] * f.T
print(m)
print(float(m))
print(l)
print(float(l))

max_value_so_far = 0
max_weights_so_far = w_s[0][1]
RT = 1.04 #target rate of monthly return
for i in range(w_s.shape[1]): #for each of the weights from the grid matrix
    t2 = w_s[0][i] #the current weight's under consideration
    f2 = np.asmatrix(t2)
    denom = 0
    for j in range(scenario_matrix.shape[0]):
        retj = float(scenario_matrix[j] * f.T) #the returni in the equation
        max = 0
        diff = RT - retj
        if diff > 0:
            max = diff
        sq = float (math.pow(max, 2))
        denom = float (denom + sq)
    fraction = (denom / scenario_matrix.shape[0])
    frac = float (fraction)
    l = float(p * f2.T) #E[R]
    num = float (l - RT)
    d = math.sqrt(frac)
    sortino = num / d
    #print(sortino)
    #sortino_f = float(sortino)
    if sortino > max_value_so_far:
        max_value_so_far = sortino
        max_weights_so_far = t2

print(max_value_so_far)
print(max_weights_so_far)








#print(w_s[0][2])

'''print(sum_to_one)
print(len(sum_to_one))'''