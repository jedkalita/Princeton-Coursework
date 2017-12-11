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

#mu_r = np.array([gmean_stocks_monthly, gmean_bonds_monthly, gmean_tbills_monthly])
mu_r = np.array([gmean_stocks_annually, gmean_bonds_annually, gmean_tbills_annually])
#mu_r = np.array([geomean_monthly_stocks_r, geomean_monthly_bonds_r, geomean_monthly_tbills_r])
#mu_r = np.reshape(mu_r, (mu_r.shape[0], 1)) #reshape the average returns vector - (3, 1)
#vectorized form for expected monthly geometric mean returns
#print(mu)
#scenario_matrix = np.random.multivariate_normal(mu_r, covariance_monthly, 10000) #10000 scenarios
# of expected monthly returns have been generated - (10000, 3) shape
scenario_matrix = np.random.multivariate_normal(mu_r, covariance_annual, 10000) #10000 scenarios
#print(scenario_matrix)
#print(scenario_matrix)

np.random.seed(1)
n = 3 #the number of assets in our portfolio
w = Variable(n) #the variable of weights for each asset

target_returns = np.array([1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.109]) #we will optimize for each one
'''target_returns = list()
target_returns = np.ones(len(targets))
for i in range(len(targets)):
    target_returns[i] = (targets[i] * 100) + 100'''
# of these target returns
#print(len(target_returns))
returns_data = np.zeros(len(target_returns))
risk_data = np.zeros(len(target_returns))

'''ret = 0
for i in range(len(scenario_matrix)):
    ret += scenario_matrix[i].T * w
ret = (ret / len(scenario_matrix)) #this will generate the return'''
p = np.asmatrix(np.mean(scenario_matrix.T, axis=1))
#print(p.shape) #(1, 3) - shape of p in python
ret = p * w
#print(ret.size) - size of ret is (1, 1)

#risk = 0
cov = opt.matrix(np.cov(scenario_matrix.T))
print(cov)
covar = np.array(cov)
print(covar)
risk = quad_form(w, covar) #w.T * Q * w
'''print(cov)
for i in range(len(scenario_matrix)):
    exp_ret = 0
    for j in range(n):
        exp_ret += scenario_matrix[i].T * w
    for j in range(n):
        #risk = risk + math.pow((scenario_matrix[i][j] - exp_ret), 2)
        risk += (scenario_matrix[i][j] - exp_ret) * (scenario_matrix[i][j] - exp_ret)
risk = (risk / len(scenario_matrix))'''
#risk = quad_form(w, scenario_matrix.T)
#risk = risk_t

for i in range(len(target_returns)):
    prob = Problem(Minimize(risk),
               [sum_entries(w) == 1,
                w >= 0,
                ret >= target_returns[i]])
    prob.solve() #solve the problem for the particular target expected return
    #print(prob)
    print(risk.value)
    returns_data[i] = ret.value
    #print(ret)
    risk_data[i] = sqrt(risk).value
    #ret = 0
    #risk = 0

print(returns_data)
print(risk_data)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, returns_data, 'bs-',)
for xy in zip(risk_data, returns_data):
    #plt.plot(xy, 'bs')
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
plt.show()






