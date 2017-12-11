import numpy as np
import matplotlib.pyplot as plt
import pandas
import cvxpy as cvx
from xlrd import open_workbook
import functools
import math
import cvxopt as opt
from cvxopt import blas, solvers
import numpy
from scipy.stats import norm
import scipy as scipy
import cvxopt as cvxopt

import matplotlib.pyplot as plt

#load in the data
df = pandas.read_excel('Homework_7.xlsx')
#print(df.columns)

#get all 7 assets
pvt_equity = df['Private Equity'].values
ind_return = df['Independent Return'].values
real_assets = df['Real Assets'].values
us_domestic = df['US Domestic Equity'].values
int_equity_dev = df['International Equity - Developed'].values
int_equity_em = df['International Equity - Emerging'].values
fixed_income = df['Fixed Income'].values
#print(fixed_income)

#get S&P 500 growth
sp500 = df['S&P 500 Growth'].values

#convert r to R by adding 1
for i in range(len(sp500)):
    pvt_equity[i] = pvt_equity[i] + 1
    ind_return[i] = ind_return[i] + 1
    real_assets[i] = real_assets[i] + 1
    us_domestic[i] = us_domestic[i] + 1
    int_equity_dev[i] = int_equity_dev[i] + 1
    int_equity_em[i] = int_equity_em[i] + 1
    fixed_income[i] = fixed_income[i] + 1
    sp500[i] = sp500[i] + 1  # R

#function to calculate monthly geometric mean of a series
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

#calculate the quarterly geometric means (R)
gmean_pvt_equity_quarterly_R = geo_mean(pvt_equity)
print('Geo-mean quarterly Private Equities (R) = %f' % gmean_pvt_equity_quarterly_R)
gmean_ind_return_quarterly_R = geo_mean(ind_return)
print('Geo-mean quarterly Independent Return (R) = %f' % gmean_ind_return_quarterly_R)
gmean_real_assets_quarterly_R = geo_mean(real_assets)
print('Geo-mean quarterly Real Assets (R) = %f' % gmean_real_assets_quarterly_R)
gmean_us_domestic_quarterly_R = geo_mean(us_domestic)
print('Geo-mean quarterly US Domestic (R) = %f' % gmean_us_domestic_quarterly_R)
gmean_int_equity_dev_quarterly_R = geo_mean(int_equity_dev)
print('Geo-mean quarterly International Equity Developed (R) = %f' % gmean_int_equity_dev_quarterly_R)
gmean_int_equity_em_quarterly_R = geo_mean(int_equity_em)
print('Geo-mean quarterly International Equity Emerging (R) = %f' % gmean_int_equity_em_quarterly_R)
gmean_fixed_income_quarterly_R = geo_mean(fixed_income)
print('Geo-mean quarterly Fixed Income (R) = %f' % gmean_fixed_income_quarterly_R)
#For S&P 500
gmean_sp500_quarterly_R = geo_mean(sp500)
print('Geo-mean quarterly S&P 500 (R) = %f' % gmean_sp500_quarterly_R)

#calculate the quarterly geometric means (r)
gmean_pvt_equity_quarterly_r = gmean_pvt_equity_quarterly_R - 1
print('Geo-mean quarterly Private Equities (r) = %f' % gmean_pvt_equity_quarterly_r)
gmean_ind_return_quarterly_r = gmean_ind_return_quarterly_R - 1
print('Geo-mean quarterly Independent Return (r) = %f' % gmean_ind_return_quarterly_r)
gmean_real_assets_quarterly_r = gmean_real_assets_quarterly_R - 1
print('Geo-mean quarterly Real Assets (r) = %f' % gmean_real_assets_quarterly_r)
gmean_us_domestic_quarterly_r = gmean_us_domestic_quarterly_R - 1
print('Geo-mean quarterly US Domestic (r) = %f' % gmean_us_domestic_quarterly_r)
gmean_int_equity_dev_quarterly_r = gmean_int_equity_dev_quarterly_R - 1
print('Geo-mean quarterly International Equity Developed (r) = %f' % gmean_int_equity_dev_quarterly_r)
gmean_int_equity_em_quarterly_r = gmean_int_equity_em_quarterly_R - 1
print('Geo-mean quarterly International Equity Emerging (r) = %f' % gmean_int_equity_em_quarterly_r)
gmean_fixed_income_quarterly_r = gmean_fixed_income_quarterly_R - 1
print('Geo-mean quarterly Fixed Income (r) = %f' % gmean_fixed_income_quarterly_r)
#For S&P 500
gmean_sp500_quarterly_r = gmean_sp500_quarterly_R - 1
print('Geo-mean quarterly S&P 500 (r) = %f' % gmean_sp500_quarterly_r)

#now, calculate annual geometric means (R)
gmean_pvt_equity_annually_R = math.pow(gmean_pvt_equity_quarterly_R, 4)
print('Geo-mean annually Private Equities (R) = %f' % gmean_pvt_equity_annually_R)
gmean_ind_return_annually_R = math.pow(gmean_ind_return_quarterly_R, 4)
print('Geo-mean annually Independent Return (R) = %f' % gmean_ind_return_annually_R)
gmean_real_assets_annually_R = math.pow(gmean_real_assets_quarterly_R, 4)
print('Geo-mean annually Real Assets (R) = %f' % gmean_real_assets_annually_R)
gmean_us_domestic_annually_R = math.pow(gmean_us_domestic_quarterly_R, 4)
print('Geo-mean annually US Domestic (R) = %f' % gmean_us_domestic_annually_R)
gmean_int_equity_dev_annually_R = math.pow(gmean_int_equity_dev_quarterly_R, 4)
print('Geo-mean annually International Equity Developed (R) = %f' % gmean_int_equity_dev_annually_R)
gmean_int_equity_em_annually_R = math.pow(gmean_int_equity_em_quarterly_R, 4)
print('Geo-mean annually International Equity Emerging (R) = %f' % gmean_int_equity_em_annually_R)
gmean_fixed_income_annually_R = math.pow(gmean_fixed_income_quarterly_R, 4)
print('Geo-mean annually Fixed Income (R) = %f' % gmean_fixed_income_annually_R)
#For S&P 500
gmean_sp500_annually_R = math.pow(gmean_sp500_quarterly_R, 4)
print('Geo-mean annually S&P 500 (R) = %f' % gmean_sp500_annually_R)

#now, calculate annual geometric means (r)
gmean_pvt_equity_annually_r = gmean_pvt_equity_annually_R - 1
print('Geo-mean annually Private Equities (r) = %f' % gmean_pvt_equity_annually_r)
gmean_ind_return_annually_r = gmean_ind_return_annually_R - 1
print('Geo-mean annually Independent Return (r) = %f' % gmean_ind_return_annually_r)
gmean_real_assets_annually_r = gmean_real_assets_annually_R - 1
print('Geo-mean annually Real Assets (r) = %f' % gmean_real_assets_annually_r)
gmean_us_domestic_annually_r = gmean_us_domestic_annually_R - 1
print('Geo-mean annually US Domestic (r) = %f' % gmean_us_domestic_annually_r)
gmean_int_equity_dev_annually_r = gmean_int_equity_dev_annually_R - 1
print('Geo-mean annually International Equity Developed (r) = %f' % gmean_int_equity_dev_annually_r)
gmean_int_equity_em_annually_r = gmean_int_equity_em_annually_R - 1
print('Geo-mean annually International Equity Emerging (r) = %f' % gmean_int_equity_em_annually_r)
gmean_fixed_income_annually_r = gmean_fixed_income_annually_R - 1
print('Geo-mean annually Fixed Income (r) = %f' % gmean_fixed_income_annually_r)
#For S&P 500
gmean_sp500_annually_r = gmean_sp500_annually_R - 1
print('Geo-mean annually S&P 500 (r) = %f' % gmean_sp500_annually_r)

#now, calculate quarterly standard deviation
pvt_equity_std_quarterly = np.std(np.array(pvt_equity))
print("Std for Private Equities quarterly = %f" % pvt_equity_std_quarterly)
ind_return_std_quarterly = np.std(np.array(ind_return))
print("Std for Independent Return quarterly = %f" % ind_return_std_quarterly)
real_assets_std_quarterly = np.std(np.array(real_assets))
print("Std for Real Assets quarterly = %f" % real_assets_std_quarterly)
us_domestic_std_quarterly = np.std(np.array(us_domestic))
print("Std for US Domestic quarterly = %f" % us_domestic_std_quarterly)
int_equity_dev_std_quarterly = np.std(np.array(int_equity_dev))
print("Std for International Equity Developed quarterly = %f" % int_equity_dev_std_quarterly)
int_equity_em_std_quarterly = np.std(np.array(int_equity_em))
print("Std for International Equity Emerging quarterly = %f" % int_equity_em_std_quarterly)
fixed_income_std_quarterly = np.std(np.array(fixed_income))
print("Std for Fixed Income quarterly = %f" % fixed_income_std_quarterly)
#For S&P 500
sp500_std_quarterly = np.std(np.array(sp500))
print("Std for S&P 500 quarterly = %f" % sp500_std_quarterly)

#now, calculate quarterly variance (volatility)
pvt_equity_var_quarterly = math.pow(pvt_equity_std_quarterly, 2)
print("Variance (volatility) for Private Equities quarterly = %f" % pvt_equity_var_quarterly)
ind_return_var_quarterly = math.pow(ind_return_std_quarterly, 2)
print("Variance (volatility) for Independent Return quarterly = %f" % ind_return_var_quarterly)
real_assets_var_quarterly = math.pow(real_assets_std_quarterly, 2)
print("Variance (volatility) for Real Assets quarterly = %f" % real_assets_var_quarterly)
us_domestic_var_quarterly = math.pow(us_domestic_std_quarterly, 2)
print("Variance (volatility) for US Domestic quarterly = %f" % us_domestic_var_quarterly)
int_equity_dev_var_quarterly = math.pow(int_equity_dev_std_quarterly, 2)
print("Variance (volatility) for International Equity Developed quarterly = %f" % int_equity_dev_var_quarterly)
int_equity_em_var_quarterly = math.pow(int_equity_em_std_quarterly, 2)
print("Variance (volatility) for International Equity Emerging quarterly = %f" % int_equity_em_var_quarterly)
fixed_income_var_quarterly = math.pow(fixed_income_std_quarterly, 2)
print("Variance (volatility) for Fixed Income quarterly = %f" % fixed_income_var_quarterly)
#For S&P 500
sp500_var_quarterly = math.pow(sp500_std_quarterly, 2)
print("Variance (volatility) for S&P 500 quarterly = %f" % sp500_var_quarterly)

#now, calculate annualized standard deviation
pvt_equity_std_annually = math.sqrt(4) * pvt_equity_std_quarterly
print("Std for Private Equities annually = %f" % pvt_equity_std_annually)
ind_return_std_annually = math.sqrt(4) * ind_return_std_quarterly
print("Std for Independent Return annually = %f" % ind_return_std_annually)
real_assets_std_annually = math.sqrt(4) * real_assets_std_quarterly
print("Std for Real Assets annually = %f" % real_assets_std_annually)
us_domestic_std_annually = math.sqrt(4) * us_domestic_std_quarterly
print("Std for US Domestic annually = %f" % us_domestic_std_annually)
int_equity_dev_std_annually = math.sqrt(4) * int_equity_dev_std_quarterly
print("Std for International Equity Developed annually = %f" % int_equity_dev_std_annually)
int_equity_em_std_annually = math.sqrt(4) * int_equity_em_std_quarterly
print("Std for International Equity Emerging annually = %f" % int_equity_em_std_annually)
fixed_income_std_annually = math.sqrt(4) * fixed_income_std_quarterly
print("Std for Fixed Income annually = %f" % fixed_income_std_annually)
#For S&P 500
sp500_std_annually = math.sqrt(4) * sp500_std_quarterly
print("Std for Fixed Income annually = %f" % sp500_std_annually)

#now, calculate annualized volatility (variance)
pvt_equity_var_annually = math.pow(pvt_equity_std_annually, 2)
print("Variance (volatility) for Private Equities annually = %f" % pvt_equity_var_annually)
ind_return_var_annually = math.pow(ind_return_std_annually, 2)
print("Variance (volatility) for Independent Return annually = %f" % ind_return_var_annually)
real_assets_var_annually = math.pow(real_assets_std_annually, 2)
print("Variance (volatility) for Real Assets annually = %f" % real_assets_var_annually)
us_domestic_var_annually = math.pow(us_domestic_std_annually, 2)
print("Variance (volatility) for US Domestic annually = %f" % us_domestic_var_annually)
int_equity_dev_var_annually = math.pow(int_equity_dev_std_annually, 2)
print("Variance (volatility) for International Equity Developed annually = %f" % int_equity_dev_var_annually)
int_equity_em_var_annually = math.pow(int_equity_em_std_annually, 2)
print("Variance (volatility) for International Equity Emerging annually = %f" % int_equity_em_var_annually)
fixed_income_var_annually = math.pow(fixed_income_std_annually, 2)
print("Variance (volatility) for Fixed Income annually = %f" % fixed_income_var_annually)
#For S&P 500
sp500_var_annually = math.pow(sp500_std_annually, 2)
print("Variance (volatility) for S&P 500 annually = %f" % sp500_var_annually)

#now, make the quarterly correlation matrix
combine = list()
combine.append(np.array(pvt_equity))
combine.append(np.array(ind_return))
combine.append(np.array(real_assets))
combine.append(np.array(us_domestic))
combine.append(np.array(int_equity_dev))
combine.append(np.array(int_equity_em))
combine.append(np.array(fixed_income))
correlation_mat_quarterly = np.corrcoef(np.array(combine))
'''print('Printing the quarterly correlation matrix.....')
print(correlation_mat_quarterly)'''
#annualized correlation matrix
correlation_mat_annually = 4 * correlation_mat_quarterly
'''print('Printing the annualized correlation matrix.....')
print(correlation_mat_annually)'''

#now, make the quarterly covariance matrix
covariance_mat_quarterly = np.cov(np.array(combine))
'''print('Printing the quarterly covariance matrix.....')
print(covariance_mat_quarterly)'''
#annualized covariance matrix
covariance_mat_annually = 4 * covariance_mat_quarterly
'''print('Printing the annualized covariance matrix.....')
print(covariance_mat_annually)'''

#store annualized historical means (R) for the 7 assets in an array
mu_R = np.array([gmean_pvt_equity_annually_R, gmean_ind_return_annually_R,
                 gmean_real_assets_annually_R, gmean_us_domestic_annually_R,
                 gmean_int_equity_dev_annually_R, gmean_int_equity_em_annually_R,
                 gmean_fixed_income_annually_R])
#print(mu_R)
#store annualized historical means (r) for the 7 assets in an array
mu_r = np.array([gmean_pvt_equity_annually_r, gmean_ind_return_annually_r,
                 gmean_real_assets_annually_r, gmean_us_domestic_annually_r,
                 gmean_int_equity_dev_annually_r, gmean_int_equity_em_annually_r,
                 gmean_fixed_income_annually_r])
#print(mu_r)

#now, generate 1000 scenarios for 7 assets based on anualized R and covariance annually
scenario_matrix_R = np.random.multivariate_normal(mu_R, covariance_mat_annually, 1000)
#print(scenario_matrix_R.shape)
#now, generate 1000 scenarios for 7 assets based on anualized r and covariance annually
scenario_matrix_r = np.random.multivariate_normal(mu_r, covariance_mat_annually, 1000)
#1000 scenarios x 7 assets (shape of the matrix)
#print(scenario_matrix_r.shape)
#print(scenario_matrix_R)
#print(scenario_matrix_r)


#Convex optimization solver for trend filtering
y1 = np.array(sp500) #S&P 500 R's - since changed
#print(y1)
n = y1.size #56
np_ones_y = np.ones(n) #an array of 1's to get back r's from R's y1
#print(np_ones_y)
y = y1 - np_ones_y #S&P 500 r's
#print(y)

#print(n)
#Form first difference matrix
e = np.mat(np.ones((1, n)))
#print(e)
D = scipy.sparse.spdiags(np.vstack((e, -e)), range(2), n - 1, n)
'''D_coo = D.tocoo()
D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())'''

#set the regularization parameter
vlambda = 0.1
#solve the l1 trent filtering problem
x = cvx.Variable(n)
obj = cvx.Minimize(cvx.sum_squares(y - x) + vlambda * cvx.norm(D*x, 1))
prob = cvx.Problem(obj)
#prob.solve(solver=cvx.CVXOPT,verbose=True)
prob.solve()
'''print('Solver status: ', prob.status)
print(x.value)'''

# Plot estimated trend with original signal.
'''plt.plot(np.arange(1,n+1), y, 'k:', linewidth=1.0)
plt.plot(np.arange(1,n+1), np.array(x.value), 'b-', linewidth=2.0)
plt.xlabel('Quarter No.')
plt.ylabel('SP500 returns')
plt.show()'''

#now, get the indices(quarters) of growth and crash
y_growth_indices = list()
y_crash_indices = list()
for i in range(x.size[0]):
    if(x.value[i][0] >= 0):
        y_growth_indices.append(i) #if growth is positive put in growth list
    else:
        y_crash_indices.append(i)  # if growth is negative put in crash list
print(y_growth_indices)
print(y_crash_indices)

#now, that we have the indices(quarters), for each asset, get a crash and growth version of each and then
#calculate the stats - quarterly mean followed by annualized mean, std, var and then covariance for both
#growth and crash periods
#growth
pvt_equity_growth = list()
ind_return_growth = list()
real_assets_growth = list()
us_domestic_growth = list()
int_equity_dev_growth = list()
int_equity_em_growth = list()
fixed_income_growth = list()
#now make the growth numbers for each quarter of growth for each asset
for i in range(len(y_growth_indices)):
    pvt_equity_growth.append(pvt_equity[y_growth_indices[i]])
    ind_return_growth.append(ind_return[y_growth_indices[i]])
    real_assets_growth.append(real_assets[y_growth_indices[i]])
    us_domestic_growth.append(us_domestic[y_growth_indices[i]])
    int_equity_dev_growth.append(int_equity_dev[y_growth_indices[i]])
    int_equity_em_growth.append(int_equity_em[y_growth_indices[i]])
    fixed_income_growth.append(fixed_income[y_growth_indices[i]]) #in R's
#print(len(int_equity_dev_growth))


#crash
pvt_equity_crash = list()
ind_return_crash = list()
real_assets_crash = list()
us_domestic_crash = list()
int_equity_dev_crash = list()
int_equity_em_crash = list()
fixed_income_crash = list()
#now make the growth numbers for each quarter of crash for each asset
for i in range(len(y_crash_indices)):
    pvt_equity_crash.append(pvt_equity[y_crash_indices[i]])
    ind_return_crash.append(ind_return[y_crash_indices[i]])
    real_assets_crash.append(real_assets[y_crash_indices[i]])
    us_domestic_crash.append(us_domestic[y_crash_indices[i]])
    int_equity_dev_crash.append(int_equity_dev[y_crash_indices[i]])
    int_equity_em_crash.append(int_equity_em[y_crash_indices[i]])
    fixed_income_crash.append(fixed_income[y_crash_indices[i]]) #in R's
#print(fixed_income_crash)

#now, we have the growth to crash ratio
total_scenarios = 10000
growth_scenarios = int((len(fixed_income_growth) / len(fixed_income)) * total_scenarios)
crash_scenarios = total_scenarios - growth_scenarios
'''print(growth_scenarios)
print(crash_scenarios)'''


#calculate the quarterly geometric means (R) for growth periods
gmean_pvt_equity_growth_quarterly_R = geo_mean(pvt_equity_growth)
print('Geo-mean quarterly Private Equities (R) during growth periods = %f' %
      gmean_pvt_equity_growth_quarterly_R)
gmean_ind_return_growth_quarterly_R = geo_mean(ind_return_growth)
print('Geo-mean quarterly Independent Return (R) during growth periods = %f' %
      gmean_ind_return_growth_quarterly_R)
gmean_real_assets_growth_quarterly_R = geo_mean(real_assets_growth)
print('Geo-mean quarterly Real Assets (R) during growth periods = %f' %
      gmean_real_assets_growth_quarterly_R)
gmean_us_domestic_growth_quarterly_R = geo_mean(us_domestic_growth)
print('Geo-mean quarterly US Domestic (R) during growth periods = %f' %
      gmean_us_domestic_growth_quarterly_R)
gmean_int_equity_dev_growth_quarterly_R = geo_mean(int_equity_dev_growth)
print('Geo-mean quarterly International Equity Developed (R) during growth periods = %f' %
      gmean_int_equity_dev_growth_quarterly_R)
gmean_int_equity_em_growth_quarterly_R = geo_mean(int_equity_em_growth)
print('Geo-mean quarterly International Equity Emerging (R) during growth periods = %f' %
      gmean_int_equity_em_growth_quarterly_R)
gmean_fixed_income_growth_quarterly_R = geo_mean(fixed_income_growth)
print('Geo-mean quarterly Fixed Income (R) during growth periods = %f' %
      gmean_fixed_income_growth_quarterly_R)

#calculate the quarterly geometric means (r) for growth periods
gmean_pvt_equity_growth_quarterly_r = gmean_pvt_equity_growth_quarterly_R - 1
print('Geo-mean quarterly Private Equities (r) during growth periods = %f' %
      gmean_pvt_equity_growth_quarterly_r)
gmean_ind_return_growth_quarterly_r = gmean_ind_return_growth_quarterly_R - 1
print('Geo-mean quarterly Independent Return (r) during growth periods = %f'
      % gmean_ind_return_growth_quarterly_r)
gmean_real_assets_growth_quarterly_r = gmean_real_assets_growth_quarterly_R - 1
print('Geo-mean quarterly Real Assets (r) during growth periods = %f'
      % gmean_real_assets_growth_quarterly_r)
gmean_us_domestic_growth_quarterly_r = gmean_us_domestic_growth_quarterly_R - 1
print('Geo-mean quarterly US Domestic (r) during growth periods = %f' %
      gmean_us_domestic_growth_quarterly_r)
gmean_int_equity_dev_growth_quarterly_r = gmean_int_equity_dev_growth_quarterly_R - 1
print('Geo-mean quarterly International Equity Developed (r) during growth periods = %f'
      % gmean_int_equity_dev_growth_quarterly_r)
gmean_int_equity_em_growth_quarterly_r = gmean_int_equity_em_growth_quarterly_R - 1
print('Geo-mean quarterly International Equity Emerging (r) during growth periods = %f'
      % gmean_int_equity_em_growth_quarterly_r)
gmean_fixed_income_growth_quarterly_r = gmean_fixed_income_growth_quarterly_R - 1
print('Geo-mean quarterly Fixed Income (r) during growth periods = %f' %
      gmean_fixed_income_growth_quarterly_r)

#now, calculate annual geometric means (R) for growth periods
gmean_pvt_equity_growth_annually_R = math.pow(gmean_pvt_equity_growth_quarterly_R, 4)
print('Geo-mean annually Private Equities (R) during growth periods = %f'
      % gmean_pvt_equity_growth_annually_R)
gmean_ind_return_growth_annually_R = math.pow(gmean_ind_return_growth_quarterly_R, 4)
print('Geo-mean annually Independent Return (R) during growth periods = %f'
      % gmean_ind_return_growth_annually_R)
gmean_real_assets_growth_annually_R = math.pow(gmean_real_assets_growth_quarterly_R, 4)
print('Geo-mean annually Real Assets (R) during growth periods = %f'
      % gmean_real_assets_growth_annually_R)
gmean_us_domestic_growth_annually_R = math.pow(gmean_us_domestic_growth_quarterly_R, 4)
print('Geo-mean annually US Domestic (R) during growth periods = %f'
      % gmean_us_domestic_growth_annually_R)
gmean_int_equity_dev_growth_annually_R = math.pow(gmean_int_equity_dev_growth_quarterly_R, 4)
print('Geo-mean annually International Equity Developed (R) during growth periods = %f'
      % gmean_int_equity_dev_growth_annually_R)
gmean_int_equity_em_growth_annually_R = math.pow(gmean_int_equity_em_growth_quarterly_R, 4)
print('Geo-mean annually International Equity Emerging (R) during growth periods = %f' %
      gmean_int_equity_em_growth_annually_R)
gmean_fixed_income_growth_annually_R = math.pow(gmean_fixed_income_growth_quarterly_R, 4)
print('Geo-mean annually Fixed Income (R) during growth periods = %f' %
      gmean_fixed_income_growth_annually_R)

#now, calculate annual geometric means (r) for growth periods
gmean_pvt_equity_growth_annually_r = gmean_pvt_equity_growth_annually_R - 1
print('Geo-mean annually Private Equities (r) during growth periods = %f' %
      gmean_pvt_equity_growth_annually_r)
gmean_ind_return_growth_annually_r = gmean_ind_return_growth_annually_R - 1
print('Geo-mean annually Independent Return (r) during growth periods = %f' %
      gmean_ind_return_growth_annually_r)
gmean_real_assets_growth_annually_r = gmean_real_assets_growth_annually_R - 1
print('Geo-mean annually Real Assets (r) during growth periods = %f' %
      gmean_real_assets_growth_annually_r)
gmean_us_domestic_growth_annually_r = gmean_us_domestic_growth_annually_R - 1
print('Geo-mean annually US Domestic (r) during growth periods = %f' %
      gmean_us_domestic_growth_annually_r)
gmean_int_equity_dev_growth_annually_r = gmean_int_equity_dev_growth_annually_R - 1
print('Geo-mean annually International Equity Developed (r) during growth periods = %f' %
      gmean_int_equity_dev_growth_annually_r)
gmean_int_equity_em_growth_annually_r = gmean_int_equity_em_growth_annually_R - 1
print('Geo-mean annually International Equity Emerging (r) during growth periods = %f' %
      gmean_int_equity_em_growth_annually_r)
gmean_fixed_income_growth_annually_r = gmean_fixed_income_growth_annually_R - 1
print('Geo-mean annually Fixed Income (r) during growth periods = %f' %
      gmean_fixed_income_growth_annually_r)

#now, calculate quarterly standard deviation for growth periods
pvt_equity_growth_std_quarterly = np.std(np.array(pvt_equity_growth))
print("Std for Private Equities quarterly during growth periods = %f" %
      pvt_equity_growth_std_quarterly)
ind_return_growth_std_quarterly = np.std(np.array(ind_return_growth))
print("Std for Independent Return quarterly during growth periods = %f" %
      ind_return_growth_std_quarterly)
real_assets_growth_std_quarterly = np.std(np.array(real_assets_growth))
print("Std for Real Assets quarterly during growth periods = %f" %
      real_assets_growth_std_quarterly)
us_domestic_growth_std_quarterly = np.std(np.array(us_domestic_growth))
print("Std for US Domestic quarterly during growth periods = %f" %
      us_domestic_growth_std_quarterly)
int_equity_dev_growth_std_quarterly = np.std(np.array(int_equity_dev_growth))
print("Std for International Equity Developed quarterly during growth periods = %f" %
      int_equity_dev_growth_std_quarterly)
int_equity_em_growth_std_quarterly = np.std(np.array(int_equity_em_growth))
print("Std for International Equity Emerging quarterly during growth periods = %f" %
      int_equity_em_growth_std_quarterly)
fixed_income_growth_std_quarterly = np.std(np.array(fixed_income_growth))
print("Std for Fixed Income quarterly during growth periods = %f" %
      fixed_income_growth_std_quarterly)

#now, calculate quarterly variance (volatility) for growth periods
pvt_equity_growth_var_quarterly = math.pow(pvt_equity_growth_std_quarterly, 2)
print("Variance (volatility) for Private Equities quarterly during growth periods = %f" %
      pvt_equity_growth_var_quarterly)
ind_return_growth_var_quarterly = math.pow(ind_return_growth_std_quarterly, 2)
print("Variance (volatility) for Independent Return quarterly during growth periods = %f" %
      ind_return_growth_var_quarterly)
real_assets_growth_var_quarterly = math.pow(real_assets_growth_std_quarterly, 2)
print("Variance (volatility) for Real Assets quarterly during growth periods = %f" %
      real_assets_growth_var_quarterly)
us_domestic_growth_var_quarterly = math.pow(us_domestic_growth_std_quarterly, 2)
print("Variance (volatility) for US Domestic quarterly during growth periods = %f" %
      us_domestic_growth_var_quarterly)
int_equity_dev_growth_var_quarterly = math.pow(int_equity_dev_growth_std_quarterly, 2)
print("Variance (volatility) for International Equity Developed quarterly during growth periods = %f" %
      int_equity_dev_growth_var_quarterly)
int_equity_em_growth_var_quarterly = math.pow(int_equity_em_growth_std_quarterly, 2)
print("Variance (volatility) for International Equity Emerging quarterly during growth periods = %f" %
      int_equity_em_growth_var_quarterly)
fixed_income_growth_var_quarterly = math.pow(fixed_income_growth_std_quarterly, 2)
print("Variance (volatility) for Fixed Income quarterly during growth periods = %f" %
      fixed_income_growth_var_quarterly)

#now, calculate annualized standard deviation for growth periods
pvt_equity_growth_std_annually = math.sqrt(4) * pvt_equity_growth_std_quarterly
print("Std for Private Equities annually during growth periods = %f" %
      pvt_equity_growth_std_annually)
ind_return_growth_std_annually = math.sqrt(4) * ind_return_growth_std_quarterly
print("Std for Independent Return annually during growth periods = %f" %
      ind_return_growth_std_annually)
real_assets_growth_std_annually = math.sqrt(4) * real_assets_growth_std_quarterly
print("Std for Real Assets annually during growth periods = %f" %
      real_assets_growth_std_annually)
us_domestic_growth_std_annually = math.sqrt(4) * us_domestic_growth_std_quarterly
print("Std for US Domestic annually during growth periods = %f" %
      us_domestic_growth_std_annually)
int_equity_dev_growth_std_annually = math.sqrt(4) * int_equity_dev_growth_std_quarterly
print("Std for International Equity Developed annually during growth periods = %f" %
      int_equity_dev_growth_std_annually)
int_equity_em_growth_std_annually = math.sqrt(4) * int_equity_em_growth_std_quarterly
print("Std for International Equity Emerging annually during growth periods = %f" %
      int_equity_em_growth_std_annually)
fixed_income_growth_std_annually = math.sqrt(4) * fixed_income_growth_std_quarterly
print("Std for Fixed Income annually during growth periods = %f" %
      fixed_income_growth_std_annually)

#now, calculate annualized volatility (variance) for growth periods
pvt_equity_growth_var_annually = math.pow(pvt_equity_growth_std_annually, 2)
print("Variance (volatility) for Private Equities annually during growth periods = %f" %
      pvt_equity_growth_var_annually)
ind_return_growth_var_annually = math.pow(ind_return_growth_std_annually, 2)
print("Variance (volatility) for Independent Return annually during growth periods = %f" %
      ind_return_growth_var_annually)
real_assets_growth_var_annually = math.pow(real_assets_growth_std_annually, 2)
print("Variance (volatility) for Real Assets annually during growth periods = %f" %
      real_assets_growth_var_annually)
us_domestic_growth_var_annually = math.pow(us_domestic_growth_std_annually, 2)
print("Variance (volatility) for US Domestic annually during growth periods = %f" %
      us_domestic_growth_var_annually)
int_equity_dev_growth_var_annually = math.pow(int_equity_dev_growth_std_annually, 2)
print("Variance (volatility) for International Equity Developed annually during growth periods = %f" %
      int_equity_dev_growth_var_annually)
int_equity_em_growth_var_annually = math.pow(int_equity_em_growth_std_annually, 2)
print("Variance (volatility) for International Equity Emerging annually during growth periods = %f" %
      int_equity_em_growth_var_annually)
fixed_income_growth_var_annually = math.pow(fixed_income_growth_std_annually, 2)
print("Variance (volatility) for Fixed Income annually during growth periods = %f" %
      fixed_income_growth_var_annually)

#now, make the quarterly correlation matrix for growth periods
combine_growth = list()
combine_growth.append(np.array(pvt_equity_growth))
combine_growth.append(np.array(ind_return_growth))
combine_growth.append(np.array(real_assets_growth))
combine_growth.append(np.array(us_domestic_growth))
combine_growth.append(np.array(int_equity_dev_growth))
combine_growth.append(np.array(int_equity_em_growth))
combine_growth.append(np.array(fixed_income_growth))
correlation_mat_quarterly_growth = np.corrcoef(np.array(combine_growth))
'''print('Printing the quarterly correlation matrix.....')
print(correlation_mat_quarterly)'''
#annualized correlation matrix for growth periods
correlation_mat_annually_growth = 4 * correlation_mat_quarterly_growth
'''print('Printing the annualized correlation matrix.....')
print(correlation_mat_annually)'''

#now, make the quarterly covariance matrix for growth periods
covariance_mat_quarterly_growth = np.cov(np.array(combine_growth))
'''print('Printing the quarterly covariance matrix.....')
print(covariance_mat_quarterly)'''
#annualized covariance matrix for growth periods
covariance_mat_annually_growth = 4 * covariance_mat_quarterly_growth
'''print('Printing the annualized covariance matrix.....')
print(covariance_mat_annually)'''

#store annualized historical means (R) for the 7 assets in an array for growth periods
mu_R_growth = np.array([gmean_pvt_equity_growth_annually_R, gmean_ind_return_growth_annually_R,
                 gmean_real_assets_growth_annually_R, gmean_us_domestic_growth_annually_R,
                 gmean_int_equity_dev_growth_annually_R, gmean_int_equity_em_growth_annually_R,
                 gmean_fixed_income_growth_annually_R])
#print(mu_R)
#store annualized historical means (r) for the 7 assets in an array for growth periods
mu_r_growth = np.array([gmean_pvt_equity_growth_annually_r, gmean_ind_return_growth_annually_r,
                 gmean_real_assets_growth_annually_r, gmean_us_domestic_growth_annually_r,
                 gmean_int_equity_dev_growth_annually_r, gmean_int_equity_em_growth_annually_r,
                 gmean_fixed_income_growth_annually_r])
#print(mu_r)

#now, generate 8571 scenarios for 7 assets based on anualized R and covariance annually for growth periods
scenario_matrix_R_growth = np.random.multivariate_normal(mu_R_growth, covariance_mat_annually_growth, growth_scenarios)
#print(scenario_matrix_R.shape)
#now, generate 8571 scenarios for 7 assets based on anualized r and covariance annually
scenario_matrix_r_growth = np.random.multivariate_normal(mu_r_growth, covariance_mat_annually_growth, growth_scenarios)
#8571 scenarios x 7 assets (shape of the matrix)
#print(scenario_matrix_r.shape)
#print(scenario_matrix_R)
#print(scenario_matrix_r)


#calculate the quarterly geometric means (R) for crash periods
gmean_pvt_equity_crash_quarterly_R = geo_mean(pvt_equity_crash)
print('Geo-mean quarterly Private Equities (R) during crash periods = %f' %
      gmean_pvt_equity_crash_quarterly_R)
gmean_ind_return_crash_quarterly_R = geo_mean(ind_return_crash)
print('Geo-mean quarterly Independent Return (R) during crash periods = %f' %
      gmean_ind_return_crash_quarterly_R)
gmean_real_assets_crash_quarterly_R = geo_mean(real_assets_crash)
print('Geo-mean quarterly Real Assets (R) during crash periods = %f' %
      gmean_real_assets_crash_quarterly_R)
gmean_us_domestic_crash_quarterly_R = geo_mean(us_domestic_crash)
print('Geo-mean quarterly US Domestic (R) during crash periods = %f' %
      gmean_us_domestic_crash_quarterly_R)
gmean_int_equity_dev_crash_quarterly_R = geo_mean(int_equity_dev_crash)
print('Geo-mean quarterly International Equity Developed (R) during crash periods = %f' %
      gmean_int_equity_dev_crash_quarterly_R)
gmean_int_equity_em_crash_quarterly_R = geo_mean(int_equity_em_crash)
print('Geo-mean quarterly International Equity Emerging (R) during crash periods = %f' %
      gmean_int_equity_em_crash_quarterly_R)
gmean_fixed_income_crash_quarterly_R = geo_mean(fixed_income_crash)
print('Geo-mean quarterly Fixed Income (R) during crash periods = %f' %
      gmean_fixed_income_crash_quarterly_R)

#calculate the quarterly geometric means (r) for crash periods
gmean_pvt_equity_crash_quarterly_r = gmean_pvt_equity_crash_quarterly_R - 1
print('Geo-mean quarterly Private Equities (r) during crash periods = %f' %
      gmean_pvt_equity_crash_quarterly_r)
gmean_ind_return_crash_quarterly_r = gmean_ind_return_crash_quarterly_R - 1
print('Geo-mean quarterly Independent Return (r) during crash periods = %f'
      % gmean_ind_return_crash_quarterly_r)
gmean_real_assets_crash_quarterly_r = gmean_real_assets_crash_quarterly_R - 1
print('Geo-mean quarterly Real Assets (r) during crash periods = %f'
      % gmean_real_assets_crash_quarterly_r)
gmean_us_domestic_crash_quarterly_r = gmean_us_domestic_crash_quarterly_R - 1
print('Geo-mean quarterly US Domestic (r) during crash periods = %f' %
      gmean_us_domestic_crash_quarterly_r)
gmean_int_equity_dev_crash_quarterly_r = gmean_int_equity_dev_crash_quarterly_R - 1
print('Geo-mean quarterly International Equity Developed (r) during crash periods = %f'
      % gmean_int_equity_dev_crash_quarterly_r)
gmean_int_equity_em_crash_quarterly_r = gmean_int_equity_em_crash_quarterly_R - 1
print('Geo-mean quarterly International Equity Emerging (r) during crash periods = %f'
      % gmean_int_equity_em_crash_quarterly_r)
gmean_fixed_income_crash_quarterly_r = gmean_fixed_income_crash_quarterly_R - 1
print('Geo-mean quarterly Fixed Income (r) during crash periods = %f' %
      gmean_fixed_income_crash_quarterly_r)

#now, calculate annual geometric means (R) for crash periods
gmean_pvt_equity_crash_annually_R = math.pow(gmean_pvt_equity_crash_quarterly_R, 4)
print('Geo-mean annually Private Equities (R) during crash periods = %f'
      % gmean_pvt_equity_crash_annually_R)
gmean_ind_return_crash_annually_R = math.pow(gmean_ind_return_crash_quarterly_R, 4)
print('Geo-mean annually Independent Return (R) during crash periods = %f'
      % gmean_ind_return_crash_annually_R)
gmean_real_assets_crash_annually_R = math.pow(gmean_real_assets_crash_quarterly_R, 4)
print('Geo-mean annually Real Assets (R) during crash periods = %f'
      % gmean_real_assets_crash_annually_R)
gmean_us_domestic_crash_annually_R = math.pow(gmean_us_domestic_crash_quarterly_R, 4)
print('Geo-mean annually US Domestic (R) during crash periods = %f'
      % gmean_us_domestic_crash_annually_R)
gmean_int_equity_dev_crash_annually_R = math.pow(gmean_int_equity_dev_crash_quarterly_R, 4)
print('Geo-mean annually International Equity Developed (R) during crash periods = %f'
      % gmean_int_equity_dev_crash_annually_R)
gmean_int_equity_em_crash_annually_R = math.pow(gmean_int_equity_em_crash_quarterly_R, 4)
print('Geo-mean annually International Equity Emerging (R) during crash periods = %f' %
      gmean_int_equity_em_crash_annually_R)
gmean_fixed_income_crash_annually_R = math.pow(gmean_fixed_income_crash_quarterly_R, 4)
print('Geo-mean annually Fixed Income (R) during crash periods = %f' %
      gmean_fixed_income_crash_annually_R)

#now, calculate annual geometric means (r) for crash periods
gmean_pvt_equity_crash_annually_r = gmean_pvt_equity_crash_annually_R - 1
print('Geo-mean annually Private Equities (r) during crash periods = %f' %
      gmean_pvt_equity_crash_annually_r)
gmean_ind_return_crash_annually_r = gmean_ind_return_crash_annually_R - 1
print('Geo-mean annually Independent Return (r) during crash periods = %f' %
      gmean_ind_return_crash_annually_r)
gmean_real_assets_crash_annually_r = gmean_real_assets_crash_annually_R - 1
print('Geo-mean annually Real Assets (r) during crash periods = %f' %
      gmean_real_assets_crash_annually_r)
gmean_us_domestic_crash_annually_r = gmean_us_domestic_crash_annually_R - 1
print('Geo-mean annually US Domestic (r) during crash periods = %f' %
      gmean_us_domestic_crash_annually_r)
gmean_int_equity_dev_crash_annually_r = gmean_int_equity_dev_crash_annually_R - 1
print('Geo-mean annually International Equity Developed (r) during crash periods = %f' %
      gmean_int_equity_dev_crash_annually_r)
gmean_int_equity_em_crash_annually_r = gmean_int_equity_em_crash_annually_R - 1
print('Geo-mean annually International Equity Emerging (r) during crash periods = %f' %
      gmean_int_equity_em_crash_annually_r)
gmean_fixed_income_crash_annually_r = gmean_fixed_income_crash_annually_R - 1
print('Geo-mean annually Fixed Income (r) during crash periods = %f' %
      gmean_fixed_income_crash_annually_r)

#now, calculate quarterly standard deviation for crash periods
pvt_equity_crash_std_quarterly = np.std(np.array(pvt_equity_crash))
print("Std for Private Equities quarterly during crash periods = %f" %
      pvt_equity_crash_std_quarterly)
ind_return_crash_std_quarterly = np.std(np.array(ind_return_crash))
print("Std for Independent Return quarterly during crash periods = %f" %
      ind_return_crash_std_quarterly)
real_assets_crash_std_quarterly = np.std(np.array(real_assets_crash))
print("Std for Real Assets quarterly during crash periods = %f" %
      real_assets_crash_std_quarterly)
us_domestic_crash_std_quarterly = np.std(np.array(us_domestic_crash))
print("Std for US Domestic quarterly during crash periods = %f" %
      us_domestic_crash_std_quarterly)
int_equity_dev_crash_std_quarterly = np.std(np.array(int_equity_dev_crash))
print("Std for International Equity Developed quarterly during crash periods = %f" %
      int_equity_dev_crash_std_quarterly)
int_equity_em_crash_std_quarterly = np.std(np.array(int_equity_em_crash))
print("Std for International Equity Emerging quarterly during crash periods = %f" %
      int_equity_em_crash_std_quarterly)
fixed_income_crash_std_quarterly = np.std(np.array(fixed_income_crash))
print("Std for Fixed Income quarterly during crash periods = %f" %
      fixed_income_crash_std_quarterly)

#now, calculate quarterly variance (volatility) for crash periods
pvt_equity_crash_var_quarterly = math.pow(pvt_equity_crash_std_quarterly, 2)
print("Variance (volatility) for Private Equities quarterly during crash periods = %f" %
      pvt_equity_crash_var_quarterly)
ind_return_crash_var_quarterly = math.pow(ind_return_crash_std_quarterly, 2)
print("Variance (volatility) for Independent Return quarterly during crash periods = %f" %
      ind_return_crash_var_quarterly)
real_assets_crash_var_quarterly = math.pow(real_assets_crash_std_quarterly, 2)
print("Variance (volatility) for Real Assets quarterly during crash periods = %f" %
      real_assets_crash_var_quarterly)
us_domestic_crash_var_quarterly = math.pow(us_domestic_crash_std_quarterly, 2)
print("Variance (volatility) for US Domestic quarterly during crash periods = %f" %
      us_domestic_crash_var_quarterly)
int_equity_dev_crash_var_quarterly = math.pow(int_equity_dev_crash_std_quarterly, 2)
print("Variance (volatility) for International Equity Developed quarterly during crash periods = %f" %
      int_equity_dev_crash_var_quarterly)
int_equity_em_crash_var_quarterly = math.pow(int_equity_em_crash_std_quarterly, 2)
print("Variance (volatility) for International Equity Emerging quarterly during crash periods = %f" %
      int_equity_em_crash_var_quarterly)
fixed_income_crash_var_quarterly = math.pow(fixed_income_crash_std_quarterly, 2)
print("Variance (volatility) for Fixed Income quarterly during crash periods = %f" %
      fixed_income_crash_var_quarterly)

#now, calculate annualized standard deviation for crash periods
pvt_equity_crash_std_annually = math.sqrt(4) * pvt_equity_crash_std_quarterly
print("Std for Private Equities annually during crash periods = %f" %
      pvt_equity_crash_std_annually)
ind_return_crash_std_annually = math.sqrt(4) * ind_return_crash_std_quarterly
print("Std for Independent Return annually during crash periods = %f" %
      ind_return_crash_std_annually)
real_assets_crash_std_annually = math.sqrt(4) * real_assets_crash_std_quarterly
print("Std for Real Assets annually during crash periods = %f" %
      real_assets_crash_std_annually)
us_domestic_crash_std_annually = math.sqrt(4) * us_domestic_crash_std_quarterly
print("Std for US Domestic annually during crash periods = %f" %
      us_domestic_crash_std_annually)
int_equity_dev_crash_std_annually = math.sqrt(4) * int_equity_dev_crash_std_quarterly
print("Std for International Equity Developed annually during crash periods = %f" %
      int_equity_dev_crash_std_annually)
int_equity_em_crash_std_annually = math.sqrt(4) * int_equity_em_crash_std_quarterly
print("Std for International Equity Emerging annually during crash periods = %f" %
      int_equity_em_crash_std_annually)
fixed_income_crash_std_annually = math.sqrt(4) * fixed_income_crash_std_quarterly
print("Std for Fixed Income annually during crash periods = %f" %
      fixed_income_crash_std_annually)

#now, calculate annualized volatility (variance) for crash periods
pvt_equity_crash_var_annually = math.pow(pvt_equity_crash_std_annually, 2)
print("Variance (volatility) for Private Equities annually during crash periods = %f" %
      pvt_equity_crash_var_annually)
ind_return_crash_var_annually = math.pow(ind_return_crash_std_annually, 2)
print("Variance (volatility) for Independent Return annually during crash periods = %f" %
      ind_return_crash_var_annually)
real_assets_crash_var_annually = math.pow(real_assets_crash_std_annually, 2)
print("Variance (volatility) for Real Assets annually during crash periods = %f" %
      real_assets_crash_var_annually)
us_domestic_crash_var_annually = math.pow(us_domestic_crash_std_annually, 2)
print("Variance (volatility) for US Domestic annually during crash periods = %f" %
      us_domestic_crash_var_annually)
int_equity_dev_crash_var_annually = math.pow(int_equity_dev_crash_std_annually, 2)
print("Variance (volatility) for International Equity Developed annually during crash periods = %f" %
      int_equity_dev_crash_var_annually)
int_equity_em_crash_var_annually = math.pow(int_equity_em_crash_std_annually, 2)
print("Variance (volatility) for International Equity Emerging annually during crash periods = %f" %
      int_equity_em_crash_var_annually)
fixed_income_crash_var_annually = math.pow(fixed_income_crash_std_annually, 2)
print("Variance (volatility) for Fixed Income annually during crash periods = %f" %
      fixed_income_crash_var_annually)

#now, make the quarterly correlation matrix for crash periods
combine_crash = list()
combine_crash.append(np.array(pvt_equity_crash))
combine_crash.append(np.array(ind_return_crash))
combine_crash.append(np.array(real_assets_crash))
combine_crash.append(np.array(us_domestic_crash))
combine_crash.append(np.array(int_equity_dev_crash))
combine_crash.append(np.array(int_equity_em_crash))
combine_crash.append(np.array(fixed_income_crash))
correlation_mat_quarterly_crash = np.corrcoef(np.array(combine_crash))
'''print('Printing the quarterly correlation matrix.....')
print(correlation_mat_quarterly)'''
#annualized correlation matrix for growth periods
correlation_mat_annually_crash = 4 * correlation_mat_quarterly_crash
'''print('Printing the annualized correlation matrix.....')
print(correlation_mat_annually)'''

#now, make the quarterly covariance matrix for growth periods
covariance_mat_quarterly_crash = np.cov(np.array(combine_crash))
'''print('Printing the quarterly covariance matrix.....')
print(covariance_mat_quarterly)'''
#annualized covariance matrix for growth periods
covariance_mat_annually_crash = 4 * covariance_mat_quarterly_crash
'''print('Printing the annualized covariance matrix.....')
print(covariance_mat_annually)'''

#store annualized historical means (R) for the 7 assets in an array for crash periods
mu_R_crash = np.array([gmean_pvt_equity_crash_annually_R, gmean_ind_return_crash_annually_R,
                 gmean_real_assets_crash_annually_R, gmean_us_domestic_crash_annually_R,
                 gmean_int_equity_dev_crash_annually_R, gmean_int_equity_em_crash_annually_R,
                 gmean_fixed_income_crash_annually_R])
#print(mu_R)
#store annualized historical means (r) for the 7 assets in an array for growth periods
mu_r_crash = np.array([gmean_pvt_equity_crash_annually_r, gmean_ind_return_crash_annually_r,
                 gmean_real_assets_crash_annually_r, gmean_us_domestic_crash_annually_r,
                 gmean_int_equity_dev_crash_annually_r, gmean_int_equity_em_crash_annually_r,
                 gmean_fixed_income_crash_annually_r])
#print(mu_r)

#now, generate 1429 scenarios for 7 assets based on anualized R and covariance annually for growth periods
scenario_matrix_R_crash = np.random.multivariate_normal(mu_R_crash, covariance_mat_annually_crash, crash_scenarios)
#print(scenario_matrix_R.shape)
#now, generate 8571 scenarios for 7 assets based on anualized r and covariance annually
scenario_matrix_r_crash = np.random.multivariate_normal(mu_r_crash, covariance_mat_annually_crash, crash_scenarios)
#8571 scenarios x 7 assets (shape of the matrix)
#print(scenario_matrix_r.shape)
#print(scenario_matrix_R)
#print(scenario_matrix_r)

#join the scenario matrices of growth and crash
scenario_matrix_R_market = np.concatenate((scenario_matrix_R_growth, scenario_matrix_R_crash), axis = 0)
#print(scenario_matrix_R_market.shape) #(1000, 7)

#Now we will calculate VaR and CVaR at h = 0.05 for the 2 simulated versions of each asset - based
#on historical returns, growth periods and crash periods
h = 0.05 #the loss tolerance for VaR and CVaR calculation
#first off, we will start with 1000 scenarios generated based on historical quarterly returns
num_scenarios_historical = 1000 #the total no of scenarios for historical returns
VaR_point = int(h * num_scenarios_historical) #get the VaR point index based on sorted list
#print(var_point)
#now, for each asset get the list - Private Equity
pvt_equity_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 0])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
pvt_equity_historical_simiulated_VaR = pvt_equity_historical_simulated_sorted[VaR_point]
print('VaR for Private Equity for simulated scenarios based on historical returns = %f'
      % pvt_equity_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
pvt_equity_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    pvt_equity_historical_simiulated_CVaR = \
        float(pvt_equity_historical_simiulated_CVaR + pvt_equity_historical_simulated_sorted[i])
pvt_equity_historical_simiulated_CVaR = float(pvt_equity_historical_simiulated_CVaR / VaR_point)
print('CVaR for Private Equity for simulated scenarios based on historical returns = %f'
      % pvt_equity_historical_simiulated_CVaR)

#now, for each asset get the list - Independent Returns
ind_return_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 1])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
ind_return_historical_simiulated_VaR = ind_return_historical_simulated_sorted[VaR_point]
print('VaR for Independent Returns for simulated scenarios based on historical returns = %f'
      % ind_return_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
ind_return_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    ind_return_historical_simiulated_CVaR = \
        float(ind_return_historical_simiulated_CVaR + ind_return_historical_simulated_sorted[i])
ind_return_historical_simiulated_CVaR = float(ind_return_historical_simiulated_CVaR / VaR_point)
print('CVaR for Independent Returns for simulated scenarios based on historical returns = %f'
      % ind_return_historical_simiulated_CVaR)

#now, for each asset get the list - Real Assets
real_assets_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 2])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
real_assets_historical_simiulated_VaR = real_assets_historical_simulated_sorted[VaR_point]
print('VaR for Real Assets for simulated scenarios based on historical returns = %f'
      % real_assets_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
real_assets_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    real_assets_historical_simiulated_CVaR = \
        float(real_assets_historical_simiulated_CVaR + real_assets_historical_simulated_sorted[i])
real_assets_historical_simiulated_CVaR = float(real_assets_historical_simiulated_CVaR / VaR_point)
print('CVaR for Real Assets for simulated scenarios based on historical returns = %f'
      % real_assets_historical_simiulated_CVaR)

#now, for each asset get the list - US Domestic Equity
us_domestic_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 3])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
us_domestic_historical_simiulated_VaR = us_domestic_historical_simulated_sorted[VaR_point]
print('VaR for US Domestic for simulated scenarios based on historical returns = %f'
      % us_domestic_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
us_domestic_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    us_domestic_historical_simiulated_CVaR = \
        float(us_domestic_historical_simiulated_CVaR + us_domestic_historical_simulated_sorted[i])
us_domestic_historical_simiulated_CVaR = float(us_domestic_historical_simiulated_CVaR / VaR_point)
print('CVaR for US Domestic for simulated scenarios based on historical returns = %f'
      % us_domestic_historical_simiulated_CVaR)

#now, for each asset get the list - International Equity - Developed
int_equity_dev_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 4])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_dev_historical_simiulated_VaR = int_equity_dev_historical_simulated_sorted[VaR_point]
print('VaR for International Equity Developed for simulated scenarios based on historical returns = %f'
      % int_equity_dev_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_dev_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_dev_historical_simiulated_CVaR = \
        float(int_equity_dev_historical_simiulated_CVaR + int_equity_dev_historical_simulated_sorted[i])
int_equity_dev_historical_simiulated_CVaR = float(int_equity_dev_historical_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Developed for simulated scenarios based on historical returns = %f'
      % int_equity_dev_historical_simiulated_CVaR)

#now, for each asset get the list - International Equity - Emerging
int_equity_em_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 5])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_em_historical_simiulated_VaR = int_equity_em_historical_simulated_sorted[VaR_point]
print('VaR for International Equity Emerging for simulated scenarios based on historical returns = %f'
      % int_equity_em_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_em_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_em_historical_simiulated_CVaR = \
        float(int_equity_em_historical_simiulated_CVaR + int_equity_em_historical_simulated_sorted[i])
int_equity_em_historical_simiulated_CVaR = float(int_equity_em_historical_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Emerging for simulated scenarios based on historical returns = %f'
      % int_equity_em_historical_simiulated_CVaR)

#now, for each asset get the list - Fixed Income
fixed_income_historical_simulated_sorted = np.sort(scenario_matrix_R[:, 6])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
fixed_income_historical_simiulated_VaR = fixed_income_historical_simulated_sorted[VaR_point]
print('VaR for Fixed Income for simulated scenarios based on historical returns = %f'
      % fixed_income_historical_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
fixed_income_historical_simiulated_CVaR = 0
for i in range(VaR_point):
    fixed_income_historical_simiulated_CVaR = \
        float(fixed_income_historical_simiulated_CVaR + fixed_income_historical_simulated_sorted[i])
fixed_income_historical_simiulated_CVaR = float(fixed_income_historical_simiulated_CVaR / VaR_point)
print('CVaR for Fixed Income for simulated scenarios based on historical returns = %f'
      % fixed_income_historical_simiulated_CVaR)


#next, we will compute for scenarios generated based on growth periods
num_scenarios_growth = growth_scenarios #the total no of scenarios for historical returns
VaR_point = int(h * num_scenarios_growth) #get the VaR point index based on sorted list
#print(var_point)
#now, for each asset get the list - Private Equity
pvt_equity_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 0])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
pvt_equity_growth_simiulated_VaR = pvt_equity_growth_simulated_sorted[VaR_point]
print('VaR for Private Equity for simulated scenarios based on growth returns = %f'
      % pvt_equity_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
pvt_equity_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    pvt_equity_growth_simiulated_CVaR = \
        float(pvt_equity_growth_simiulated_CVaR + pvt_equity_growth_simulated_sorted[i])
pvt_equity_growth_simiulated_CVaR = float(pvt_equity_growth_simiulated_CVaR / VaR_point)
print('CVaR for Private Equity for simulated scenarios based on growth returns = %f'
      % pvt_equity_growth_simiulated_CVaR)

#now, for each asset get the list - Independent Returns
ind_return_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 1])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
ind_return_growth_simiulated_VaR = ind_return_growth_simulated_sorted[VaR_point]
print('VaR for Independent Returns for simulated scenarios based on growth returns = %f'
      % ind_return_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
ind_return_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    ind_return_growth_simiulated_CVaR = \
        float(ind_return_growth_simiulated_CVaR + ind_return_growth_simulated_sorted[i])
ind_return_growth_simiulated_CVaR = float(ind_return_growth_simiulated_CVaR / VaR_point)
print('CVaR for Independent Returns for simulated scenarios based on growth returns = %f'
      % ind_return_growth_simiulated_CVaR)

#now, for each asset get the list - Real Assets
real_assets_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 2])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
real_assets_growth_simiulated_VaR = real_assets_growth_simulated_sorted[VaR_point]
print('VaR for Real Assets for simulated scenarios based on growth returns = %f'
      % real_assets_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
real_assets_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    real_assets_growth_simiulated_CVaR = \
        float(real_assets_growth_simiulated_CVaR + real_assets_growth_simulated_sorted[i])
real_assets_growth_simiulated_CVaR = float(real_assets_growth_simiulated_CVaR / VaR_point)
print('CVaR for Real Assets for simulated scenarios based on growth returns = %f'
      % real_assets_growth_simiulated_CVaR)

#now, for each asset get the list - US Domestic Equity
us_domestic_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 3])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
us_domestic_growth_simiulated_VaR = us_domestic_growth_simulated_sorted[VaR_point]
print('VaR for US Domestic for simulated scenarios based on growth returns = %f'
      % us_domestic_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
us_domestic_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    us_domestic_growth_simiulated_CVaR = \
        float(us_domestic_growth_simiulated_CVaR + us_domestic_growth_simulated_sorted[i])
us_domestic_growth_simiulated_CVaR = float(us_domestic_growth_simiulated_CVaR / VaR_point)
print('CVaR for US Domestic for simulated scenarios based on growth returns = %f'
      % us_domestic_growth_simiulated_CVaR)

#now, for each asset get the list - International Equity - Developed
int_equity_dev_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 4])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_dev_growth_simiulated_VaR = int_equity_dev_growth_simulated_sorted[VaR_point]
print('VaR for International Equity Developed for simulated scenarios based on growth returns = %f'
      % int_equity_dev_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_dev_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_dev_growth_simiulated_CVaR = \
        float(int_equity_dev_growth_simiulated_CVaR + int_equity_dev_growth_simulated_sorted[i])
int_equity_dev_growth_simiulated_CVaR = float(int_equity_dev_growth_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Developed for simulated scenarios based on growth returns = %f'
      % int_equity_dev_growth_simiulated_CVaR)

#now, for each asset get the list - International Equity - Emerging
int_equity_em_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 5])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_em_growth_simiulated_VaR = int_equity_em_growth_simulated_sorted[VaR_point]
print('VaR for International Equity Emerging for simulated scenarios based on growth returns = %f'
      % int_equity_em_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_em_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_em_growth_simiulated_CVaR = \
        float(int_equity_em_growth_simiulated_CVaR + int_equity_em_growth_simulated_sorted[i])
int_equity_em_growth_simiulated_CVaR = float(int_equity_em_growth_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Emerging for simulated scenarios based on growth returns = %f'
      % int_equity_em_growth_simiulated_CVaR)

#now, for each asset get the list - Fixed Income
fixed_income_growth_simulated_sorted = np.sort(scenario_matrix_R_growth[:, 6])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
fixed_income_growth_simiulated_VaR = fixed_income_growth_simulated_sorted[VaR_point]
print('VaR for Fixed Income for simulated scenarios based on growth returns = %f'
      % fixed_income_growth_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
fixed_income_growth_simiulated_CVaR = 0
for i in range(VaR_point):
    fixed_income_growth_simiulated_CVaR = \
        float(fixed_income_growth_simiulated_CVaR + fixed_income_growth_simulated_sorted[i])
fixed_income_growth_simiulated_CVaR = float(fixed_income_growth_simiulated_CVaR / VaR_point)
print('CVaR for Fixed Income for simulated scenarios based on growth returns = %f'
      % fixed_income_growth_simiulated_CVaR)


#next, we will compute for scenarios generated based on crash periods
num_scenarios_crash = growth_scenarios #the total no of scenarios for historical returns
VaR_point = int(h * num_scenarios_crash) #get the VaR point index based on sorted list
#print(var_point)
#now, for each asset get the list - Private Equity
pvt_equity_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 0])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
pvt_equity_crash_simiulated_VaR = pvt_equity_crash_simulated_sorted[VaR_point]
print('VaR for Private Equity for simulated scenarios based on crash returns = %f'
      % pvt_equity_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
pvt_equity_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    pvt_equity_crash_simiulated_CVaR = \
        float(pvt_equity_crash_simiulated_CVaR + pvt_equity_crash_simulated_sorted[i])
pvt_equity_crash_simiulated_CVaR = float(pvt_equity_crash_simiulated_CVaR / VaR_point)
print('CVaR for Private Equity for simulated scenarios based on crash returns = %f'
      % pvt_equity_crash_simiulated_CVaR)

#now, for each asset get the list - Independent Returns
ind_return_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 1])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
ind_return_crash_simiulated_VaR = ind_return_crash_simulated_sorted[VaR_point]
print('VaR for Independent Returns for simulated scenarios based on crash returns = %f'
      % ind_return_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
ind_return_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    ind_return_crash_simiulated_CVaR = \
        float(ind_return_crash_simiulated_CVaR + ind_return_crash_simulated_sorted[i])
ind_return_crash_simiulated_CVaR = float(ind_return_crash_simiulated_CVaR / VaR_point)
print('CVaR for Independent Returns for simulated scenarios based on crash returns = %f'
      % ind_return_crash_simiulated_CVaR)

#now, for each asset get the list - Real Assets
real_assets_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 2])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
real_assets_crash_simiulated_VaR = real_assets_crash_simulated_sorted[VaR_point]
print('VaR for Real Assets for simulated scenarios based on crash returns = %f'
      % real_assets_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
real_assets_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    real_assets_crash_simiulated_CVaR = \
        float(real_assets_crash_simiulated_CVaR + real_assets_crash_simulated_sorted[i])
real_assets_crash_simiulated_CVaR = float(real_assets_crash_simiulated_CVaR / VaR_point)
print('CVaR for Real Assets for simulated scenarios based on crash returns = %f'
      % real_assets_crash_simiulated_CVaR)

#now, for each asset get the list - US Domestic Equity
us_domestic_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 3])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
us_domestic_crash_simiulated_VaR = us_domestic_crash_simulated_sorted[VaR_point]
print('VaR for US Domestic for simulated scenarios based on crash returns = %f'
      % us_domestic_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
us_domestic_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    us_domestic_crash_simiulated_CVaR = \
        float(us_domestic_crash_simiulated_CVaR + us_domestic_crash_simulated_sorted[i])
us_domestic_crash_simiulated_CVaR = float(us_domestic_crash_simiulated_CVaR / VaR_point)
print('CVaR for US Domestic for simulated scenarios based on crash returns = %f'
      % us_domestic_crash_simiulated_CVaR)

#now, for each asset get the list - International Equity - Developed
int_equity_dev_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 4])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_dev_crash_simiulated_VaR = int_equity_dev_crash_simulated_sorted[VaR_point]
print('VaR for International Equity Developed for simulated scenarios based on crash returns = %f'
      % int_equity_dev_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_dev_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_dev_crash_simiulated_CVaR = \
        float(int_equity_dev_crash_simiulated_CVaR + int_equity_dev_crash_simulated_sorted[i])
int_equity_dev_crash_simiulated_CVaR = float(int_equity_dev_crash_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Developed for simulated scenarios based on crash returns = %f'
      % int_equity_dev_crash_simiulated_CVaR)

#now, for each asset get the list - International Equity - Emerging
int_equity_em_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 5])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_em_crash_simiulated_VaR = int_equity_em_crash_simulated_sorted[VaR_point]
print('VaR for International Equity Emerging for simulated scenarios based on crash returns = %f'
      % int_equity_em_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_em_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_em_crash_simiulated_CVaR = \
        float(int_equity_em_crash_simiulated_CVaR + int_equity_em_crash_simulated_sorted[i])
int_equity_em_crash_simiulated_CVaR = float(int_equity_em_crash_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Emerging for simulated scenarios based on crash returns = %f'
      % int_equity_em_crash_simiulated_CVaR)

#now, for each asset get the list - Fixed Income
fixed_income_crash_simulated_sorted = np.sort(scenario_matrix_R_crash[:, 6])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
fixed_income_crash_simiulated_VaR = fixed_income_crash_simulated_sorted[VaR_point]
print('VaR for Fixed Income for simulated scenarios based on crash returns = %f'
      % fixed_income_crash_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
fixed_income_crash_simiulated_CVaR = 0
for i in range(VaR_point):
    fixed_income_crash_simiulated_CVaR = \
        float(fixed_income_crash_simiulated_CVaR + fixed_income_crash_simulated_sorted[i])
fixed_income_crash_simiulated_CVaR = float(fixed_income_crash_simiulated_CVaR / VaR_point)
print('CVaR for Fixed Income for simulated scenarios based on crash returns = %f'
      % fixed_income_crash_simiulated_CVaR)


#next, we will compute for scenarios generated for the entire market data - growth + crash combined
num_scenarios_market = total_scenarios #the total no of scenarios for historical returns
VaR_point = int(h * num_scenarios_market) #get the VaR point index based on sorted list
#print(VaR_point)
#now, for each asset get the list - Private Equity
pvt_equity_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 0])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
pvt_equity_market_simiulated_VaR = pvt_equity_market_simulated_sorted[VaR_point]
print('VaR for Private Equity for simulated scenarios based on market returns = %f'
      % pvt_equity_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
pvt_equity_market_simiulated_CVaR = 0
for i in range(VaR_point):
    pvt_equity_market_simiulated_CVaR = \
        float(pvt_equity_market_simiulated_CVaR + pvt_equity_market_simulated_sorted[i])
pvt_equity_market_simiulated_CVaR = float(pvt_equity_market_simiulated_CVaR / VaR_point)
print('CVaR for Private Equity for simulated scenarios based on market returns = %f'
      % pvt_equity_market_simiulated_CVaR)

#now, for each asset get the list - Independent Returns
ind_return_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 1])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
ind_return_market_simiulated_VaR = ind_return_market_simulated_sorted[VaR_point]
print('VaR for Independent Returns for simulated scenarios based on market returns = %f'
      % ind_return_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
ind_return_market_simiulated_CVaR = 0
for i in range(VaR_point):
    ind_return_market_simiulated_CVaR = \
        float(ind_return_market_simiulated_CVaR + ind_return_market_simulated_sorted[i])
ind_return_market_simiulated_CVaR = float(ind_return_market_simiulated_CVaR / VaR_point)
print('CVaR for Independent Returns for simulated scenarios based on market returns = %f'
      % ind_return_market_simiulated_CVaR)

#now, for each asset get the list - Real Assets
real_assets_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 2])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
real_assets_market_simiulated_VaR = real_assets_market_simulated_sorted[VaR_point]
print('VaR for Real Assets for simulated scenarios based on market returns = %f'
      % real_assets_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
real_assets_market_simiulated_CVaR = 0
for i in range(VaR_point):
    real_assets_market_simiulated_CVaR = \
        float(real_assets_market_simiulated_CVaR + real_assets_market_simulated_sorted[i])
real_assets_market_simiulated_CVaR = float(real_assets_market_simiulated_CVaR / VaR_point)
print('CVaR for Real Assets for simulated scenarios based on market returns = %f'
      % real_assets_market_simiulated_CVaR)

#now, for each asset get the list - US Domestic Equity
us_domestic_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 3])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
us_domestic_market_simiulated_VaR = us_domestic_market_simulated_sorted[VaR_point]
print('VaR for US Domestic for simulated scenarios based on market returns = %f'
      % us_domestic_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
us_domestic_market_simiulated_CVaR = 0
for i in range(VaR_point):
    us_domestic_market_simiulated_CVaR = \
        float(us_domestic_market_simiulated_CVaR + us_domestic_market_simulated_sorted[i])
us_domestic_market_simiulated_CVaR = float(us_domestic_market_simiulated_CVaR / VaR_point)
print('CVaR for US Domestic for simulated scenarios based on market returns = %f'
      % us_domestic_market_simiulated_CVaR)

#now, for each asset get the list - International Equity - Developed
int_equity_dev_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 4])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_dev_market_simiulated_VaR = int_equity_dev_market_simulated_sorted[VaR_point]
print('VaR for International Equity Developed for simulated scenarios based on market returns = %f'
      % int_equity_dev_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_dev_market_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_dev_market_simiulated_CVaR = \
        float(int_equity_dev_market_simiulated_CVaR + int_equity_dev_market_simulated_sorted[i])
int_equity_dev_market_simiulated_CVaR = float(int_equity_dev_market_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Developed for simulated scenarios based on market returns = %f'
      % int_equity_dev_market_simiulated_CVaR)

#now, for each asset get the list - International Equity - Emerging
int_equity_em_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 5])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
int_equity_em_market_simiulated_VaR = int_equity_em_market_simulated_sorted[VaR_point]
print('VaR for International Equity Emerging for simulated scenarios based on market returns = %f'
      % int_equity_em_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
int_equity_em_market_simiulated_CVaR = 0
for i in range(VaR_point):
    int_equity_em_market_simiulated_CVaR = \
        float(int_equity_em_market_simiulated_CVaR + int_equity_em_market_simulated_sorted[i])
int_equity_em_market_simiulated_CVaR = float(int_equity_em_market_simiulated_CVaR / VaR_point)
print('CVaR for International Equity Emerging for simulated scenarios based on market returns = %f'
      % int_equity_em_market_simiulated_CVaR)

#now, for each asset get the list - Fixed Income
fixed_income_market_simulated_sorted = np.sort(scenario_matrix_R_market[:, 6])
#print(len(pvt_equity_historical_simulated_sorted))
#now the VaR point will be directly indexed into from the sorted list
fixed_income_market_simiulated_VaR = fixed_income_market_simulated_sorted[VaR_point]
print('VaR for Fixed Income for simulated scenarios based on market returns = %f'
      % fixed_income_market_simiulated_VaR)
#now, CVaR calculation will be expection upto the VaR point
fixed_income_market_simiulated_CVaR = 0
for i in range(VaR_point):
    fixed_income_market_simiulated_CVaR = \
        float(fixed_income_market_simiulated_CVaR + fixed_income_market_simulated_sorted[i])
fixed_income_market_simiulated_CVaR = float(fixed_income_market_simiulated_CVaR / VaR_point)
print('CVaR for Fixed Income for simulated scenarios based on market returns = %f'
      % fixed_income_market_simiulated_CVaR)

#Mean-variance optimization for historically simulated scenarios
np.random.seed(1)
n = 7 #the number of assets in our portfolio
p = np.asmatrix(np.mean(scenario_matrix_R.T, axis=1)) #take the mean of all scenarios per asset
#print(np.mean(scenario_matrix_R.T, axis=1))
w = cvx.Variable(n) #the variable of weights for each asset
exp_total_returns = 1.08 #the returns needed annually for one period
ret = p * w #the formula for total expected return anually
risk = cvx.quad_form(w, covariance_mat_annually) #w.T * Q * w
prob = cvx.Problem(cvx.Minimize(risk),
               [cvx.sum_entries(w) == 1,
                w >= 0,
                ret >= exp_total_returns])
prob.solve() #solve the problem for the particular target expected return
#print(prob)
#print(w.value)
optimized_weights_simulated_historical = w.value
'''print('Optimized mean-variance weights for simulated historical data = ')
print(optimized_weights_simulated_historical)'''
optimized_returns_simulated_historical = ret.value
print('Optimized mean-variance returns for simulated historical data = %f'
      % optimized_returns_simulated_historical)
optimized_risk_simulated_historical = cvx.sqrt(risk).value
print('Optimized mean-variance risk for simulated historical data = %f'
      % optimized_risk_simulated_historical)


#Mean-variance optimization for simulated growth scenarios
np.random.seed(1)
p = np.asmatrix(np.mean(scenario_matrix_R_growth.T, axis=1)) #take the mean of all scenarios per asset
#print(np.mean(scenario_matrix_R.T, axis=1))
w = cvx.Variable(n) #the variable of weights for each asset
exp_total_returns = 1.08 #the returns needed annually for one period
ret = p * w #the formula for total expected return anually
risk = cvx.quad_form(w, covariance_mat_annually_growth) #w.T * Q * w
prob = cvx.Problem(cvx.Minimize(risk),
               [cvx.sum_entries(w) == 1,
                w >= 0,
                ret >= exp_total_returns])
prob.solve() #solve the problem for the particular target expected return
#print(prob)
optimized_weights_simulated_growth = w.value
'''print('Optimized mean-variance weights for simulated growth data = ')
print(optimized_weights_simulated_growth)'''
optimized_returns_simulated_growth = ret.value
print('Optimized mean-variance returns for simulated growth data = %f'
      % optimized_returns_simulated_growth)
optimized_risk_simulated_growth = cvx.sqrt(risk).value
print('Optimized mean-variance risk for simulated growth data = %f'
      % optimized_risk_simulated_growth)


#Mean-variance optimization for simulated crash scenarios
np.random.seed(1)
p = np.asmatrix(np.mean(scenario_matrix_R_crash.T, axis=1)) #take the mean of all scenarios per asset
#print(np.mean(scenario_matrix_R.T, axis=1))
w = cvx.Variable(n) #the variable of weights for each asset
exp_total_returns = 1.08 #the returns needed annually for one period
ret = p * w #the formula for total expected return anually
risk = cvx.quad_form(w, covariance_mat_annually_crash) #w.T * Q * w
prob = cvx.Problem(cvx.Minimize(risk),
               [cvx.sum_entries(w) == 1,
                w >= 0,
                ret >= exp_total_returns])
prob.solve() #solve the problem for the particular target expected return
#print(prob)
optimized_weights_simulated_crash = w.value
'''print('Optimized mean-variance weights for simulated crash data = ')
print(optimized_weights_simulated_crash)'''
optimized_returns_simulated_crash = ret.value
print('Optimized mean-variance returns for simulated crash data = %f'
      % optimized_returns_simulated_crash)
optimized_risk_simulated_crash = cvx.sqrt(risk).value
print('Optimized mean-variance risk for simulated crash data = %f'
      % optimized_risk_simulated_crash)

#now, make the annualized correlation matrix for simulated market periods
combine_market = list()
combine_market.append(np.array(scenario_matrix_R_market[:, 0]))
combine_market.append(np.array(scenario_matrix_R_market[:, 1]))
combine_market.append(np.array(scenario_matrix_R_market[:, 2]))
combine_market.append(np.array(scenario_matrix_R_market[:, 3]))
combine_market.append(np.array(scenario_matrix_R_market[:, 4]))
combine_market.append(np.array(scenario_matrix_R_market[:, 5]))
combine_market.append(np.array(scenario_matrix_R_market[:, 6]))
#print(combine_market)
#annualized correlation matrix for growth periods
correlation_mat_annually_market = np.corrcoef(np.array(combine_market))
'''print('Printing the annualized correlation matrix for simulated market.....')
print(correlation_mat_annually_market)'''
#now, make the annualized covariance matrix for growth periods
#annualized covariance matrix for growth periods
covariance_mat_annually_market = np.cov(np.array(combine_market))
'''print('Printing the annualized covariance matrix for simulated market.....')
print(covariance_mat_annually_market)'''
#print(covariance_mat_annually_market.shape)

#Mean-variance optimization for simulated market scenarios - combination of growth + crash
np.random.seed(1)
p = np.asmatrix(np.mean(scenario_matrix_R_market.T, axis=1)) #take the mean of all scenarios per asset
#print(np.mean(scenario_matrix_R.T, axis=1))
w = cvx.Variable(n) #the variable of weights for each asset
exp_total_returns = 1.08 #the returns needed annually for one period
ret = p * w #the formula for total expected return anually
risk = cvx.quad_form(w, covariance_mat_annually_market) #w.T * Q * w
prob = cvx.Problem(cvx.Minimize(risk),
               [cvx.sum_entries(w) == 1,
                w >= 0,
                ret >= exp_total_returns])
prob.solve() #solve the problem for the particular target expected return
#print(prob)
optimized_weights_simulated_market = w.value
'''print('Optimized mean-variance weights for simulated market data = ')
print(optimized_weights_simulated_market)'''
optimized_returns_simulated_market = ret.value
print('Optimized mean-variance returns for simulated market data = %f'
      % optimized_returns_simulated_market)
optimized_risk_simulated_market = cvx.sqrt(risk).value
print('Optimized mean-variance risk for simulated market data = %f'
      % optimized_risk_simulated_market)




#Mean-CVaR optimization - market simulated
w = cvx.Variable(n) #the weights variable for 7 assets
V = cvx.Variable(1) #The VaR optimizer variable which will also optimize out CVaR
denominator = int(total_scenarios * h) #Prob_s = (1/10000), h = 0.05, denominator = 500
#print(denominator)
#print(w.size) #(7, 1)
#print(scenario_matrix_R_market.shape) #(10000, 7)
v = V * np.ones((total_scenarios, 1)) #(10000, 1)
#print(v.size)
zero_matrix = np.zeros((total_scenarios, 1))
R = scenario_matrix_R_market * w #returns for each scenario
#print(R.size) #(10000, 1)
R_min_v = -R - v #X - V
#print(R_min_v.size) #(10000, 1)
max_with_zero = cvx.max_elemwise(R_min_v, zero_matrix) #max(0, X - V)
#print(max_with_zero.size) #(10000, 1)
CVaR = V + (cvx.sum_entries(max_with_zero) * (1 / denominator)) #CVaR convex formula
ret_exp = cvx.sum_entries((1 / total_scenarios) * R) #expected returns of all scenarios
#print(ret_exp.size) #(1, 1)
exp_total_returns = 1.08
prob = cvx.Problem(cvx.Minimize(CVaR),
               [cvx.sum_entries(w) == 1,
                w >= 0,
                ret_exp >= exp_total_returns])
prob.solve()
mean_CVaR_opt = CVaR.value #the mean CVaR calculated by this optimization problem
VaR_opt = V.value #the VaR corresponding to CVaR optimization
optimized_returns_simulated_market_CVaR_optimization = ret_exp.value #expected returns for CVaR optimization
optimized_weights_simulated_market_CVaR_optimization = w.value #portfolio weights for CVaR optimization
print('Mean CVaR value for simulated market returns for CVaR optimization = %f' % mean_CVaR_opt)
print('VaR value for simulated market returns for CVaR optimization = %f' % VaR_opt)
print('Expected returns  for simulated market returns for CVaR optimization = %f'
      % optimized_returns_simulated_market_CVaR_optimization)
'''print('Optimized mean-variance weights for simulated market data for CVaR optimization = ')
print(optimized_weights_simulated_market_CVaR_optimization)'''





#now read the test data from the 2nd sheet
df_test = pandas.read_excel('Homework_7.xlsx', sheetname="Test")
#print(df_test)

#get all 7 assets - test
pvt_equity_test = df_test['Private Equity'].values
ind_return_test = df_test['Independent Return'].values
real_assets_test = df_test['Real Assets'].values
us_domestic_test = df_test['US Domestic Equity'].values
int_equity_dev_test = df_test['International Equity - Developed'].values
int_equity_em_test = df_test['International Equity - Emerging'].values
fixed_income_test = df_test['Fixed Income'].values
#print(int_equity_em_test)

#convert r to R by adding 1 - test
for i in range(len(pvt_equity_test)):
    pvt_equity_test[i] = pvt_equity_test[i] + 1
    ind_return_test[i] = ind_return_test[i] + 1
    real_assets_test[i] = real_assets_test[i] + 1
    us_domestic_test[i] = us_domestic_test[i] + 1
    int_equity_dev_test[i] = int_equity_dev_test[i] + 1
    int_equity_em_test[i] = int_equity_em_test[i] + 1
    fixed_income_test[i] = fixed_income_test[i] + 1


#wealth path for portfolio 1 - based on 1000 scenarios historical returns mean-variance optimized
#under testing data
wealth = 1 #we assume initial wealth to be 1 - this can be anything and does not matter
wealth_each_quarter_simulated_historical = list() #this will store the returns at each test quarter
#24 entries
for i in range(len(pvt_equity_test)): #for each test quarter
    #for each of the 7 assets find the contribution to the wealth by each asset
    # wealth * weight * return
    pvt_equity_return_quarter = optimized_weights_simulated_historical[0] * \
                                pvt_equity_test[i] * wealth
    ind_return_return_quarter = optimized_weights_simulated_historical[1] * \
                                ind_return_test[i] * wealth
    real_assets_return_quarter = optimized_weights_simulated_historical[2] * \
                                 real_assets_test[i] * wealth
    us_domestic_return_quarter = optimized_weights_simulated_historical[3] * \
                                 us_domestic_test[i] * wealth
    int_equity_dev_return_quarter = optimized_weights_simulated_historical[4] * \
                                    int_equity_dev_test[i] * wealth
    int_equity_em_return_quarter = optimized_weights_simulated_historical[5] * \
                                   int_equity_em_test[i] * wealth
    fixed_income_return_quarter = optimized_weights_simulated_historical[6] * \
                                  fixed_income_test[i] * wealth

    #now, add up the wealth contributions by each asset which would be the total wealth available
    #at the beginning of the next quarter (Time Period), which is also the wealth at the this quarter
    wealth = pvt_equity_return_quarter + ind_return_return_quarter + real_assets_return_quarter + \
             us_domestic_return_quarter + int_equity_dev_return_quarter + \
             int_equity_em_return_quarter + fixed_income_return_quarter
    #add this wealth at the end of this quarter to the list storing returns
    wealth_each_quarter_simulated_historical.append(wealth)
#now, find the mean and volatility
mean_each_quarter_simulated_historical = geo_mean(wealth_each_quarter_simulated_historical) #geomean
var_each_quarter_simulated_historical = math.pow(np.std(np.array(wealth_each_quarter_simulated_historical)), 2)
#volatility
#print(optimized_weights_simulated_historical)
#print(wealth_each_quarter_simulated_historical)
print('Mean for testing on historical simulated mean-variance optimized portfolio = %f'
      % mean_each_quarter_simulated_historical)
print('Volatility for testing on historical simulated mean-variance optimized portfolio = %f'
      % var_each_quarter_simulated_historical)
#quarter_no  = list(range(len(pvt_equity_test))) #list storing the quarter number
quarter_no  = np.zeros(len(pvt_equity_test))
wealth_vals = np.zeros(len(pvt_equity_test))
for i in range(len(pvt_equity_test)):
    quarter_no[i] = i + 1
    wealth_vals[i] = wealth_each_quarter_simulated_historical[i][0]
#print(quarter_no)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(quarter_no, wealth_vals, 'bs-',)
plt.xlabel('Quarter Number')
plt.ylabel('Wealth Return')
plt.suptitle('Mean-variance optimized historical portfolio')
#plt.show()




#wealth path for portfolio 2 - based on 10000 scenarios of market returns
# (combination of growth + crash) mean-variance optimized
#under testing data
wealth = 1 #we assume initial wealth to be 1 - this can be anything and does not matter
wealth_each_quarter_simulated_market = list() #this will store the returns at each test quarter
#24 entries
for i in range(len(pvt_equity_test)): #for each test quarter
    #for each of the 7 assets find the contribution to the wealth by each asset
    # wealth * weight * return
    pvt_equity_return_quarter = optimized_weights_simulated_market[0] * \
                                pvt_equity_test[i] * wealth
    ind_return_return_quarter = optimized_weights_simulated_market[1] * \
                                ind_return_test[i] * wealth
    real_assets_return_quarter = optimized_weights_simulated_market[2] * \
                                 real_assets_test[i] * wealth
    us_domestic_return_quarter = optimized_weights_simulated_market[3] * \
                                 us_domestic_test[i] * wealth
    int_equity_dev_return_quarter = optimized_weights_simulated_market[4] * \
                                    int_equity_dev_test[i] * wealth
    int_equity_em_return_quarter = optimized_weights_simulated_market[5] * \
                                   int_equity_em_test[i] * wealth
    fixed_income_return_quarter = optimized_weights_simulated_market[6] * \
                                  fixed_income_test[i] * wealth

    #now, add up the wealth contributions by each asset which would be the total wealth available
    #at the beginning of the next quarter (Time Period), which is also the wealth at the this quarter
    wealth = pvt_equity_return_quarter + ind_return_return_quarter + real_assets_return_quarter + \
             us_domestic_return_quarter + int_equity_dev_return_quarter + \
             int_equity_em_return_quarter + fixed_income_return_quarter
    #add this wealth at the end of this quarter to the list storing returns
    wealth_each_quarter_simulated_market.append(wealth)
#now, find the mean and volatility
mean_each_quarter_simulated_market = geo_mean(wealth_each_quarter_simulated_market) #geomean
var_each_quarter_simulated_market = math.pow(np.std(np.array(wealth_each_quarter_simulated_market)), 2)
#volatility
#print(optimized_weights_simulated_market)
#print(wealth_each_quarter_simulated_market)
print('Mean for testing on market simulated mean-variance optimized portfolio = %f'
      % mean_each_quarter_simulated_market)
print('Volatility for testing on market simulated mean-variance optimized portfolio = %f'
      % var_each_quarter_simulated_market)
for i in range(len(pvt_equity_test)):
    quarter_no[i] = i + 1
    wealth_vals[i] = wealth_each_quarter_simulated_market[i][0]
#print(quarter_no)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(quarter_no, wealth_vals, 'bs-',)
plt.xlabel('Quarter Number')
plt.ylabel('Wealth Return')
plt.suptitle('Mean_Variance optimized scenario market portfolio')
#plt.show()


#wealth path for portfolio 3 - based on 10000 scenarios of market returns
# (combination of growth + crash) mean-CVaR optimized
#under testing data
wealth = 1 #we assume initial wealth to be 1 - this can be anything and does not matter
wealth_each_quarter_simulated_market_CVaR_optimization = list()
#this will store the returns at each test quarter
#24 entries
for i in range(len(pvt_equity_test)): #for each test quarter
    #for each of the 7 assets find the contribution to the wealth by each asset
    # wealth * weight * return
    pvt_equity_return_quarter = optimized_weights_simulated_market_CVaR_optimization[0] * \
                                pvt_equity_test[i] * wealth
    ind_return_return_quarter = optimized_weights_simulated_market_CVaR_optimization[1] * \
                                ind_return_test[i] * wealth
    real_assets_return_quarter = optimized_weights_simulated_market_CVaR_optimization[2] * \
                                 real_assets_test[i] * wealth
    us_domestic_return_quarter = optimized_weights_simulated_market_CVaR_optimization[3] * \
                                 us_domestic_test[i] * wealth
    int_equity_dev_return_quarter = optimized_weights_simulated_market_CVaR_optimization[4] * \
                                    int_equity_dev_test[i] * wealth
    int_equity_em_return_quarter = optimized_weights_simulated_market_CVaR_optimization[5] * \
                                   int_equity_em_test[i] * wealth
    fixed_income_return_quarter = optimized_weights_simulated_market_CVaR_optimization[6] * \
                                  fixed_income_test[i] * wealth

    #now, add up the wealth contributions by each asset which would be the total wealth available
    #at the beginning of the next quarter (Time Period), which is also the wealth at the this quarter
    wealth = pvt_equity_return_quarter + ind_return_return_quarter + real_assets_return_quarter + \
             us_domestic_return_quarter + int_equity_dev_return_quarter + \
             int_equity_em_return_quarter + fixed_income_return_quarter
    #add this wealth at the end of this quarter to the list storing returns
    wealth_each_quarter_simulated_market_CVaR_optimization.append(wealth)
#now, find the mean and volatility
mean_each_quarter_simulated_market_CVaR_optimization = geo_mean\
    (wealth_each_quarter_simulated_market_CVaR_optimization) #geomean
var_each_quarter_simulated_market_CVaR_optimization = math.pow(np.std
                                                               (np.array
                                                                (wealth_each_quarter_simulated_market_CVaR_optimization))
                                                               , 2)
#volatility
#print(optimized_weights_simulated_market_CVaR_optimization)
#print(wealth_each_quarter_simulated_market_CVaR_optimization)
print('Mean for testing on market simulated mean-CVaR optimized portfolio = %f'
      % mean_each_quarter_simulated_market_CVaR_optimization)
print('Volatility for testing on market simulated mean-CVaR optimized portfolio = %f'
      % var_each_quarter_simulated_market_CVaR_optimization)
for i in range(len(pvt_equity_test)):
    quarter_no[i] = i + 1
    wealth_vals[i] = wealth_each_quarter_simulated_market_CVaR_optimization[i][0]
#print(quarter_no)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(quarter_no, wealth_vals, 'bs-',)
plt.xlabel('Quarter Number')
plt.ylabel('Wealth Return')
plt.suptitle('CVaR optimized scenario market portfolio')
plt.show()