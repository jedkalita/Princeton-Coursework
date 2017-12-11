import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
# Generate data for long only portfolio optimization.
import numpy as np
import matplotlib.pyplot as plt
import pandas
from cvxpy import *
from xlrd import open_workbook
import functools
import math

df = pandas.read_excel('Assets_3.xlsx')
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
'''print(geo_mean_overflow(stocks))
print(geo_mean_overflow(bonds))
print(geo_mean_overflow(tbills))'''



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
#correlation = np.corrcoef(np.array(c))
#print(correlation)
covariance_monthly = np.cov(np.array(c)) #monthly covariance matrix
#print(cov)
covariance_annual = 12 * covariance_monthly #annualized covariance matrix

geomean_monthly_stocks_r = gmean_stocks_monthly - 1
geomean_monthly_bonds_r = gmean_bonds_monthly - 1
geomean_monthly_tbills_r = gmean_tbills_monthly - 1
#mu_R = np.array([gmean_stocks_monthly, gmean_bonds_monthly, gmean_tbills_monthly])
mu_r = np.array([geomean_monthly_stocks_r, geomean_monthly_bonds_r, geomean_monthly_tbills_r])






np.random.seed(123)

# Turn off progress printing
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 3

## NUMBER OF OBSERVATIONS
n_obs = 1000

#return_vec = np.random.randn(n_assets, n_obs)
return_vec = np.random.multivariate_normal(mu_r, covariance_monthly, 10000)
return_vec = return_vec.T
print(return_vec.shape)

'''fig = plt.figure()
plt.plot(return_vec.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
plt.show()'''

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

'''print(rand_weights(n_assets))
print(rand_weights(n_assets))'''


def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    '''print("Inside random_portfolios method")
    print(returns)
    print(C)'''

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 10000
means, stds = np.column_stack([
    random_portfolio(return_vec)
    for _ in range(n_portfolios)
])

'''plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()'''


def optimal_portfolio(returns):
    n = len(returns)
    print(n)
    returns = np.asmatrix(returns)
    #print(returns)

    #N = 10
    #mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]
    mus = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.109]
    print(mus)

    # Convert to cvxopt matrices
    print(returns)
    S = opt.matrix(np.cov(returns))
    print(S)
    #print(returns, axis=1)
    pbar = opt.matrix(np.mean(returns, axis=1))
    print(pbar)

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1)) #Gh = wi >= 0 (or -wi <= 0)
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0) #Ab = sum(wi) = 1
    '''print(G*h)
    print(A*b)'''

    # Calculate efficient frontier weights using quadratic programming
    '''t = [solvers.qp(mu * S, -pbar, G, h, A, b)
                  for mu in mus]
    print(len(t))'''
    print(mus[0] * S)
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    for i in range(len(portfolios)):
        print(portfolios[i])
    #print(portfolios[2])
    #print(portfolios) #vector of weights (n x 1) for total no of expected returns
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    print(returns)
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    print(risks)
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    '''m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks'''
    return returns, risks


#weights, returns, risks = optimal_portfolio(return_vec)
returns, risks = optimal_portfolio(return_vec)
#print(weights)
print(returns)
print(risks)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
plt.show()